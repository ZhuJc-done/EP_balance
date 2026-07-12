#include <cstdio>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>
#include <nccl_device.h>
#include <nccl_device/impl/comm__types.h>
#include <nccl_device/impl/gin__funcs.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess)                                               \
      throw std::runtime_error(std::string("CUDA error ") +              \
                               cudaGetErrorString(err) + " at " +        \
                               __FILE__ + ":" + std::to_string(__LINE__));\
  } while (0)

#define NCCL_CHECK(call)                                                  \
  do {                                                                    \
    ncclResult_t err = (call);                                            \
    if (err != ncclSuccess)                                               \
      throw std::runtime_error(std::string("NCCL error ") +              \
                               ncclGetErrorString(err) + " at " +        \
                               __FILE__ + ":" + std::to_string(__LINE__));\
  } while (0)

static constexpr int BARRIER_COUNT = 16;
static constexpr int GIN_SIGNAL_COUNT = 64;
static constexpr size_t GIN_TILE_BYTES = 1ULL << 25; // 32 MB
static constexpr size_t GIN_MIN_BYTES_PER_CTX = 1ULL << 15; // 32 KB

struct WindowInfo {
  ncclWindow_t win;
  void* ptr;
  size_t size;
};

struct NcclGinState {
  ncclComm_t comm = nullptr;
  ncclDevComm devComm;
  bool hasDevComm = false;
  bool ownsComm = true;
  int rank = -1;
  int nRanks = -1;
  bool initialized = false;
  std::mutex mu;

  std::unordered_map<uintptr_t, WindowInfo> windows;
  std::vector<void*> allocations;
};

static NcclGinState g_state;

static const char* getenv_or_unset(const char* name) {
  const char* value = std::getenv(name);
  return (value && value[0] != '\0') ? value : "<unset>";
}

static int read_positive_int_env_or_default(const char* name, int default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return default_value;

  errno = 0;
  char* end = nullptr;
  long value = std::strtol(raw, &end, 10);
  if (errno != 0 || end == raw || *end != '\0' || value > INT_MAX) {
    throw std::runtime_error(std::string("Invalid integer env ") + name + "=" + raw);
  }
  if (value <= 0) return default_value;
  return static_cast<int>(value);
}

static void configure_gin_requirements(ncclDevCommRequirements* reqs,
                                       const char* caller) {
  reqs->barrierCount = BARRIER_COUNT;
  reqs->ginSignalCount = GIN_SIGNAL_COUNT;
  reqs->ginConnectionType = NCCL_GIN_CONNECTION_FULL;
  reqs->ginContextCount = read_positive_int_env_or_default(
      "NCCL_GIN_NCONTEXTS", reqs->ginContextCount);

  fprintf(stderr,
          "[nccl_gin_ext] %s: DevComm requirements: "
          "ginConnectionType=FULL, ginContextCount=%d, ginQueueDepth=%d, "
          "env NCCL_GIN_NCONNECTIONS=%s, NCCL_GIN_NCONTEXTS=%s, "
          "NCCL_GIN_GDAKI_QP_DEPTH=%s\n",
          caller, reqs->ginContextCount, reqs->ginQueueDepth,
          getenv_or_unset("NCCL_GIN_NCONNECTIONS"),
          getenv_or_unset("NCCL_GIN_NCONTEXTS"),
          getenv_or_unset("NCCL_GIN_GDAKI_QP_DEPTH"));
}

static void log_created_dev_comm(const char* caller) {
  fprintf(stderr,
          "[nccl_gin_ext] %s: DevComm created successfully, rank=%d, "
          "ginConnectionCount=%u, ginContextCount=%u, ginContextBase=%u\n",
          caller, g_state.rank,
          static_cast<unsigned>(g_state.devComm.ginConnectionCount),
          static_cast<unsigned>(g_state.devComm.ginContextCount),
          static_cast<unsigned>(g_state.devComm.ginContextBase));
}

// ---------------------------------------------------------------------------
// GIN Put Kernel (device-side)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned active_gin_contexts(size_t numBytes,
                                                        unsigned maxCtx) {
  unsigned activeCtx = static_cast<unsigned>(
      (numBytes + GIN_MIN_BYTES_PER_CTX - 1) / GIN_MIN_BYTES_PER_CTX);
  if (activeCtx == 0) activeCtx = 1;
  if (activeCtx > maxCtx) activeCtx = maxCtx;
  return activeCtx;
}

__global__ void GinPutKernel(ncclWindow_t srcWin, size_t srcOffset,
                             ncclWindow_t dstWin, size_t dstOffset,
                             size_t numBytes, int peer,
                             struct ncclDevComm devComm) {
  if (blockIdx.x != 0) return;
  if (numBytes == 0) return;

  const unsigned numCtx = devComm.ginContextCount;
  const unsigned numWarps = blockDim.x / warpSize;
  if (numCtx == 0 || numWarps == 0) return;

  const unsigned warpId = threadIdx.x / warpSize;
  unsigned activeCtx = active_gin_contexts(numBytes, numCtx);

  ncclCoopWarp coop;
  ncclTeam team = ncclTeamWorld(devComm);

  for (unsigned ginCtx = warpId; ginCtx < activeCtx; ginCtx += numWarps) {
    const size_t stripeBytes = numBytes / activeCtx;
    const size_t stripeRemainder = numBytes % activeCtx;
    const size_t stripeOff =
        stripeBytes * ginCtx + min((size_t)ginCtx, stripeRemainder);
    const size_t myBytes = stripeBytes + (ginCtx < stripeRemainder ? 1 : 0);
    if (myBytes == 0) continue;

    ncclGin gin{devComm, (int)ginCtx};
    for (size_t tileOff = 0; tileOff < myBytes; tileOff += GIN_TILE_BYTES) {
      const size_t tileBytes = min(GIN_TILE_BYTES, myBytes - tileOff);
      gin.put(team, peer,
              dstWin, dstOffset + stripeOff + tileOff,
              srcWin, srcOffset + stripeOff + tileOff,
              tileBytes,
              ncclGin_None{}, ncclGin_None{}, coop);
      gin.flush(coop);
    }
  }
}

// Multi-block variant: each block is an independent SM doing its own chunk
__global__ void GinPutKernelMultiBlock(ncclWindow_t srcWin, size_t srcOffset,
                                       ncclWindow_t dstWin, size_t dstOffset,
                                       size_t numBytes, int peer,
                                       struct ncclDevComm devComm) {
  if (numBytes == 0) return;

  const unsigned numCtx = devComm.ginContextCount;
  const unsigned numWarps = blockDim.x / warpSize;
  if (numCtx == 0 || numWarps == 0) return;

  const int nBlocks = gridDim.x;
  const size_t blockChunk = numBytes / nBlocks;
  const size_t blockRemainder = numBytes % nBlocks;
  const size_t blockOff =
      blockChunk * blockIdx.x + min((size_t)blockIdx.x, blockRemainder);
  const size_t blockSize = blockChunk + (blockIdx.x < blockRemainder ? 1 : 0);
  if (blockSize == 0) return;

  const unsigned warpId = threadIdx.x / warpSize;
  unsigned activeCtx = active_gin_contexts(blockSize, numCtx);

  ncclCoopWarp coop;
  ncclTeam team = ncclTeamWorld(devComm);

  for (unsigned ginCtx = warpId; ginCtx < activeCtx; ginCtx += numWarps) {
    const size_t stripeBytes = blockSize / activeCtx;
    const size_t stripeRemainder = blockSize % activeCtx;
    const size_t stripeOff =
        stripeBytes * ginCtx + min((size_t)ginCtx, stripeRemainder);
    const size_t myBytes = stripeBytes + (ginCtx < stripeRemainder ? 1 : 0);
    if (myBytes == 0) continue;

    ncclGin gin{devComm, (int)ginCtx};
    for (size_t tileOff = 0; tileOff < myBytes; tileOff += GIN_TILE_BYTES) {
      const size_t tileBytes = min(GIN_TILE_BYTES, myBytes - tileOff);
      gin.put(team, peer,
              dstWin, dstOffset + blockOff + stripeOff + tileOff,
              srcWin, srcOffset + blockOff + stripeOff + tileOff,
              tileBytes,
              ncclGin_None{}, ncclGin_None{}, coop);
      gin.flush(coop);
    }
  }
}

__global__ void GinPutSignalKernel(ncclWindow_t srcWin, size_t srcOffset,
                                   ncclWindow_t dstWin, size_t dstOffset,
                                   size_t numBytes, int peer,
                                   int sigIdx,
                                   struct ncclDevComm devComm) {
  if (blockIdx.x != 0) return;
  if (numBytes == 0) return;

  const unsigned numCtx = devComm.ginContextCount;
  const unsigned numWarps = blockDim.x / warpSize;
  if (numCtx == 0 || numWarps == 0) return;

  const unsigned warpId = threadIdx.x / warpSize;
  const unsigned activeCtx = active_gin_contexts(numBytes, numCtx);
  ncclCoopWarp coop;
  ncclTeam team = ncclTeamWorld(devComm);

  for (unsigned ginCtx = warpId; ginCtx < activeCtx; ginCtx += numWarps) {
    const size_t stripeBytes = numBytes / activeCtx;
    const size_t stripeRemainder = numBytes % activeCtx;
    const size_t stripeOff =
        stripeBytes * ginCtx + min((size_t)ginCtx, stripeRemainder);
    const size_t myBytes = stripeBytes + (ginCtx < stripeRemainder ? 1 : 0);
    if (myBytes == 0) continue;

    ncclGin gin{devComm, (int)ginCtx};
    for (size_t tileOff = 0; tileOff < myBytes; tileOff += GIN_TILE_BYTES) {
      const size_t tileBytes = min(GIN_TILE_BYTES, myBytes - tileOff);
      const bool lastTile = tileOff + tileBytes == myBytes;
      if (lastTile) {
        gin.put(team, peer,
                dstWin, dstOffset + stripeOff + tileOff,
                srcWin, srcOffset + stripeOff + tileOff,
                tileBytes,
                ncclGin_SignalInc{static_cast<ncclGinSignal_t>(sigIdx)},
                ncclGin_None{}, coop);
      } else {
        gin.put(team, peer,
                dstWin, dstOffset + stripeOff + tileOff,
                srcWin, srcOffset + stripeOff + tileOff,
                tileBytes,
                ncclGin_None{}, ncclGin_None{}, coop);
      }
      gin.flush(coop);
    }
  }
}

__global__ void GinWaitSignalMeetShadowKernel(size_t numBytes, int sigIdx,
                                              struct ncclDevComm devComm) {
  if (blockIdx.x != 0) return;
  if (numBytes == 0) return;

  const unsigned numCtx = devComm.ginContextCount;
  const unsigned numWarps = blockDim.x / warpSize;
  if (numCtx == 0 || numWarps == 0) return;

  const unsigned warpId = threadIdx.x / warpSize;
  const unsigned activeCtx = active_gin_contexts(numBytes, numCtx);
  ncclCoopWarp coop;

  for (unsigned ginCtx = warpId; ginCtx < activeCtx; ginCtx += numWarps) {
    ncclGin gin{devComm, (int)ginCtx};
    ncclGinSignal_t signal = static_cast<ncclGinSignal_t>(sigIdx);
    if (coop.thread_rank() == 0) {
      gin.increaseSignalShadow(signal, 1);
    }
    gin.waitSignalMeetShadow(coop, signal, 64);
  }
}

__global__ void GinTestSignalKernel(size_t numBytes, int sigIdx,
                                    int* readyOut, bool consume,
                                    struct ncclDevComm devComm) {
  if (blockIdx.x != 0) return;
  if (numBytes == 0) {
    if (threadIdx.x == 0) readyOut[0] = 1;
    return;
  }

  const unsigned numCtx = devComm.ginContextCount;
  const unsigned numWarps = blockDim.x / warpSize;
  if (numCtx == 0 || numWarps == 0) {
    if (threadIdx.x == 0) readyOut[0] = 0;
    return;
  }

  __shared__ int allReady;
  if (threadIdx.x == 0) allReady = 1;
  __syncthreads();

  const unsigned warpId = threadIdx.x / warpSize;
  const unsigned lane = threadIdx.x % warpSize;
  const unsigned activeCtx = active_gin_contexts(numBytes, numCtx);
  ncclGinSignal_t signal = static_cast<ncclGinSignal_t>(sigIdx);

  for (unsigned ginCtx = warpId; ginCtx < activeCtx; ginCtx += numWarps) {
    if (lane == 0) {
      ncclGin gin{devComm, (int)ginCtx};
      const uint64_t least = *gin.getSignalShadowPtr(signal) + 1;
      const uint64_t got = gin.readSignal(signal, 64);
      if (!nccl::utility::rollingLessEq(least, got, 64)) {
        atomicExch(&allReady, 0);
      }
    }
  }
  __syncthreads();

  const bool ready = allReady != 0;
  if (consume && ready) {
    for (unsigned ginCtx = warpId; ginCtx < activeCtx; ginCtx += numWarps) {
      if (lane == 0) {
        ncclGin gin{devComm, (int)ginCtx};
        gin.increaseSignalShadow(signal, 1);
      }
    }
  }

  if (threadIdx.x == 0) {
    readyOut[0] = ready ? 1 : 0;
  }
}

// ---------------------------------------------------------------------------
// GIN Get Kernel (device-side) — new in NCCL 2.30
// get() submits an RDMA read but does NOT wait for data arrival.
// Must call flushAsync() + wait() afterward to ensure data is in local memory.
// Fix: Call flush() instead, since flushAsync() + wait() = flush()
// ---------------------------------------------------------------------------
__global__ void GinGetKernel(ncclWindow_t remoteWin, size_t remoteOffset,
                             ncclWindow_t localWin, size_t localOffset,
                             size_t numBytes, int peer,
                             struct ncclDevComm devComm) {
  if (blockIdx.x != 0) return;
  if (numBytes == 0) return;

  const unsigned numCtx = devComm.ginContextCount;
  const unsigned numWarps = blockDim.x / warpSize;
  if (numCtx == 0 || numWarps == 0) return;

  const unsigned warpId = threadIdx.x / warpSize;
  unsigned activeCtx = active_gin_contexts(numBytes, numCtx);

  ncclCoopWarp coop;
  ncclTeam team = ncclTeamWorld(devComm);

  for (unsigned ginCtx = warpId; ginCtx < activeCtx; ginCtx += numWarps) {
    const size_t stripeBytes = numBytes / activeCtx;
    const size_t stripeRemainder = numBytes % activeCtx;
    const size_t stripeOff =
        stripeBytes * ginCtx + min((size_t)ginCtx, stripeRemainder);
    const size_t myBytes = stripeBytes + (ginCtx < stripeRemainder ? 1 : 0);
    if (myBytes == 0) continue;

    ncclGin gin{devComm, (int)ginCtx};
    for (size_t tileOff = 0; tileOff < myBytes; tileOff += GIN_TILE_BYTES) {
      const size_t tileBytes = min(GIN_TILE_BYTES, myBytes - tileOff);
      gin.get(team, peer,
              remoteWin, remoteOffset + stripeOff + tileOff,
              localWin, localOffset + stripeOff + tileOff,
              tileBytes, coop);
      gin.flush(coop);
    }
  }
}

__global__ void GinGetKernelMultiBlock(ncclWindow_t remoteWin, size_t remoteOffset,
                                       ncclWindow_t localWin, size_t localOffset,
                                       size_t numBytes, int peer,
                                       struct ncclDevComm devComm) {
  if (numBytes == 0) return;

  const unsigned numCtx = devComm.ginContextCount;
  const unsigned numWarps = blockDim.x / warpSize;
  if (numCtx == 0 || numWarps == 0) return;

  const int nBlocks = gridDim.x;
  const size_t blockChunk = numBytes / nBlocks;
  const size_t blockRemainder = numBytes % nBlocks;
  const size_t blockOff =
      blockChunk * blockIdx.x + min((size_t)blockIdx.x, blockRemainder);
  const size_t blockSize = blockChunk + (blockIdx.x < blockRemainder ? 1 : 0);
  if (blockSize == 0) return;

  const unsigned warpId = threadIdx.x / warpSize;
  unsigned activeCtx = active_gin_contexts(blockSize, numCtx);

  ncclCoopWarp coop;
  ncclTeam team = ncclTeamWorld(devComm);

  for (unsigned ginCtx = warpId; ginCtx < activeCtx; ginCtx += numWarps) {
    const size_t stripeBytes = blockSize / activeCtx;
    const size_t stripeRemainder = blockSize % activeCtx;
    const size_t stripeOff =
        stripeBytes * ginCtx + min((size_t)ginCtx, stripeRemainder);
    const size_t myBytes = stripeBytes + (ginCtx < stripeRemainder ? 1 : 0);
    if (myBytes == 0) continue;

    ncclGin gin{devComm, (int)ginCtx};
    for (size_t tileOff = 0; tileOff < myBytes; tileOff += GIN_TILE_BYTES) {
      const size_t tileBytes = min(GIN_TILE_BYTES, myBytes - tileOff);
      gin.get(team, peer,
              remoteWin, remoteOffset + blockOff + stripeOff + tileOff,
              localWin, localOffset + blockOff + stripeOff + tileOff,
              tileBytes, coop);
      gin.flush(coop);
    }
  }
}

// ---------------------------------------------------------------------------
// Batched descriptor-driven GIN put/get (device-resident schedule)
//
// One launch services K transfers whose (peer, offset, size) live in device arrays,
// so the caller never reads the schedule to host: no per-transfer host loop and no D2H.
// Grid is dim3(K, nblk): blockIdx.x selects the descriptor, blockIdx.y sub-stripes that
// descriptor's payload across nblk blocks (matching the single-transfer multiblock path).
// Descriptors with peer < 0 are skipped on device, so empty / local slots need no host
// branch. Offsets are window-relative bytes; *Base folds in any view/data_ptr delta.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void gin_get_range(const ncclDevComm& devComm, int peer,
                                               ncclWindow_t rWin, size_t rOff,
                                               ncclWindow_t lWin, size_t lOff,
                                               size_t numBytes) {
  const unsigned numCtx = devComm.ginContextCount;
  const unsigned numWarps = blockDim.x / warpSize;
  if (numCtx == 0 || numWarps == 0 || numBytes == 0) return;
  const unsigned warpId = threadIdx.x / warpSize;
  const unsigned activeCtx = active_gin_contexts(numBytes, numCtx);
  ncclCoopWarp coop;
  ncclTeam team = ncclTeamWorld(devComm);
  for (unsigned ginCtx = warpId; ginCtx < activeCtx; ginCtx += numWarps) {
    const size_t stripeBytes = numBytes / activeCtx;
    const size_t stripeRemainder = numBytes % activeCtx;
    const size_t stripeOff = stripeBytes * ginCtx + min((size_t)ginCtx, stripeRemainder);
    const size_t myBytes = stripeBytes + (ginCtx < stripeRemainder ? 1 : 0);
    if (myBytes == 0) continue;
    ncclGin gin{devComm, (int)ginCtx};
    for (size_t tileOff = 0; tileOff < myBytes; tileOff += GIN_TILE_BYTES) {
      const size_t tileBytes = min(GIN_TILE_BYTES, myBytes - tileOff);
      gin.get(team, peer, rWin, rOff + stripeOff + tileOff,
              lWin, lOff + stripeOff + tileOff, tileBytes, coop);
      gin.flush(coop);
    }
  }
}

__device__ __forceinline__ void gin_put_range(const ncclDevComm& devComm, int peer,
                                               ncclWindow_t sWin, size_t sOff,
                                               ncclWindow_t dWin, size_t dOff,
                                               size_t numBytes) {
  const unsigned numCtx = devComm.ginContextCount;
  const unsigned numWarps = blockDim.x / warpSize;
  if (numCtx == 0 || numWarps == 0 || numBytes == 0) return;
  const unsigned warpId = threadIdx.x / warpSize;
  const unsigned activeCtx = active_gin_contexts(numBytes, numCtx);
  ncclCoopWarp coop;
  ncclTeam team = ncclTeamWorld(devComm);
  for (unsigned ginCtx = warpId; ginCtx < activeCtx; ginCtx += numWarps) {
    const size_t stripeBytes = numBytes / activeCtx;
    const size_t stripeRemainder = numBytes % activeCtx;
    const size_t stripeOff = stripeBytes * ginCtx + min((size_t)ginCtx, stripeRemainder);
    const size_t myBytes = stripeBytes + (ginCtx < stripeRemainder ? 1 : 0);
    if (myBytes == 0) continue;
    ncclGin gin{devComm, (int)ginCtx};
    for (size_t tileOff = 0; tileOff < myBytes; tileOff += GIN_TILE_BYTES) {
      const size_t tileBytes = min(GIN_TILE_BYTES, myBytes - tileOff);
      gin.put(team, peer, dWin, dOff + stripeOff + tileOff,
              sWin, sOff + stripeOff + tileOff, tileBytes,
              ncclGin_None{}, ncclGin_None{}, coop);
      gin.flush(coop);
    }
  }
}

__device__ __forceinline__ void block_subrange(size_t numBytes, size_t* off, size_t* size) {
  const int nsub = gridDim.y;
  const int sub = blockIdx.y;
  const size_t chunk = numBytes / nsub;
  const size_t rem = numBytes % nsub;
  *off = chunk * sub + min((size_t)sub, rem);
  *size = chunk + (sub < rem ? 1 : 0);
}

__global__ void GinGetBatchedKernel(ncclWindow_t remoteWin, size_t remoteBase,
                                    ncclWindow_t localWin, size_t localBase,
                                    const int64_t* remoteOff, const int64_t* localOff,
                                    const int64_t* nbytes, const int* peers,
                                    int K, struct ncclDevComm devComm) {
  const int k = blockIdx.x;
  if (k >= K) return;
  const int peer = peers[k];
  if (peer < 0) return;                       // empty / local slot: nothing to fetch
  const size_t total = (size_t)nbytes[k];
  if (total == 0) return;
  size_t bOff, bSize;
  block_subrange(total, &bOff, &bSize);
  if (bSize == 0) return;
  gin_get_range(devComm, peer,
                remoteWin, remoteBase + (size_t)remoteOff[k] + bOff,
                localWin, localBase + (size_t)localOff[k] + bOff,
                bSize);
}

__global__ void GinPutBatchedKernel(ncclWindow_t srcWin, size_t srcBase,
                                    ncclWindow_t dstWin, size_t dstBase,
                                    const int64_t* srcOff, const int64_t* dstOff,
                                    const int64_t* nbytes, const int* peers,
                                    int K, struct ncclDevComm devComm) {
  const int k = blockIdx.x;
  if (k >= K) return;
  const int peer = peers[k];
  if (peer < 0) return;                       // empty / local slot: no remote push
  const size_t total = (size_t)nbytes[k];
  if (total == 0) return;
  size_t bOff, bSize;
  block_subrange(total, &bOff, &bSize);
  if (bSize == 0) return;
  gin_put_range(devComm, peer,
                srcWin, srcBase + (size_t)srcOff[k] + bOff,
                dstWin, dstBase + (size_t)dstOff[k] + bOff,
                bSize);
}

// ---------------------------------------------------------------------------
// Host API
// ---------------------------------------------------------------------------

py::bytes get_unique_id() {
  ncclUniqueId id;
  NCCL_CHECK(ncclGetUniqueId(&id));
  return py::bytes(reinterpret_cast<const char*>(&id), sizeof(id));
}

void init(py::bytes unique_id_bytes, int rank, int nRanks) {
  std::lock_guard<std::mutex> lock(g_state.mu);
  if (g_state.initialized)
    throw std::runtime_error("nccl_gin already initialized");

  std::string uid_str = unique_id_bytes;
  if ((int)uid_str.size() != sizeof(ncclUniqueId))
    throw std::runtime_error("Invalid unique_id size");

  ncclUniqueId uid;
  std::memcpy(&uid, uid_str.data(), sizeof(uid));

  NCCL_CHECK(ncclCommInitRank(&g_state.comm, nRanks, uid, rank));
  g_state.ownsComm = true;
  g_state.rank = rank;
  g_state.nRanks = nRanks;

  ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(g_state.comm, &props));
  fprintf(stderr, "[nccl_gin_ext] init(): rank=%d, nRanks=%d, ginType=%d, ownsComm=true\n",
          rank, nRanks, (int)props.ginType);
  if (props.ginType == NCCL_GIN_TYPE_NONE)
    throw std::runtime_error("[nccl_gin_ext] ginType=NONE");

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  configure_gin_requirements(&reqs, "init()");
  NCCL_CHECK(ncclDevCommCreate(g_state.comm, &reqs, &g_state.devComm));
  g_state.hasDevComm = true;

  g_state.initialized = true;
  log_created_dev_comm("init()");
}

void init_from_comm(int64_t comm_handle) {
  std::lock_guard<std::mutex> lock(g_state.mu);
  if (g_state.initialized)
    throw std::runtime_error("nccl_gin already initialized");

  g_state.comm = reinterpret_cast<ncclComm_t>(comm_handle);
  g_state.ownsComm = false;

  NCCL_CHECK(ncclCommUserRank(g_state.comm, &g_state.rank));
  NCCL_CHECK(ncclCommCount(g_state.comm, &g_state.nRanks));

  ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(g_state.comm, &props));
  fprintf(stderr, "[nccl_gin_ext] init_from_comm(): rank=%d, nRanks=%d, ginType=%d, "
          "ownsComm=false, comm_handle=0x%lx\n",
          g_state.rank, g_state.nRanks, (int)props.ginType, (long)comm_handle);
  if (props.ginType == NCCL_GIN_TYPE_NONE)
    throw std::runtime_error("[nccl_gin_ext] ginType=NONE");

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  configure_gin_requirements(&reqs, "init_from_comm()");
  NCCL_CHECK(ncclDevCommCreate(g_state.comm, &reqs, &g_state.devComm));
  g_state.hasDevComm = true;

  g_state.initialized = true;
  log_created_dev_comm("init_from_comm()");
}

int get_rank() {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");
  return g_state.rank;
}

int get_world_size() {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");
  return g_state.nRanks;
}

py::dict get_comm_properties() {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");

  ncclCommProperties props = NCCL_COMM_PROPERTIES_INITIALIZER;
  NCCL_CHECK(ncclCommQueryProperties(g_state.comm, &props));

  py::dict out;
  out["rank"] = props.rank;
  out["nRanks"] = props.nRanks;
  out["cudaDev"] = props.cudaDev;
  out["nvmlDev"] = props.nvmlDev;
  out["deviceApiSupport"] = props.deviceApiSupport;
  out["multimemSupport"] = props.multimemSupport;
  out["ginType"] = (int)props.ginType;
  out["nLsaTeams"] = props.nLsaTeams;
  out["hostRmaSupport"] = props.hostRmaSupport;
  out["railedGinType"] = (int)props.railedGinType;
  out["devGinConnectionCount"] =
      g_state.hasDevComm ? (int)g_state.devComm.ginConnectionCount : 0;
  out["devGinContextCount"] =
      g_state.hasDevComm ? (int)g_state.devComm.ginContextCount : 0;
  out["devGinContextBase"] =
      g_state.hasDevComm ? (int)g_state.devComm.ginContextBase : 0;
  return out;
}

torch::Tensor create_tensor(int64_t numel, py::object dtype_obj) {
  std::lock_guard<std::mutex> lock(g_state.mu);
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");

  auto dtype = torch::python::detail::py_object_to_dtype(dtype_obj);
  int64_t elem_size = torch::elementSize(dtype);
  size_t total_bytes = (size_t)numel * elem_size;

  // Allocate symmetric memory
  void* ptr = nullptr;
  NCCL_CHECK(ncclMemAlloc(&ptr, total_bytes));
  g_state.allocations.push_back(ptr);

  // Register as symmetric window
  ncclWindow_t win = nullptr;
  NCCL_CHECK(ncclCommWindowRegister(g_state.comm, ptr, total_bytes,
                                    &win, NCCL_WIN_COLL_SYMMETRIC));

  g_state.windows[reinterpret_cast<uintptr_t>(ptr)] = {win, ptr, total_bytes};

  // Wrap in torch::Tensor (the tensor does NOT own the memory)
  auto options = torch::TensorOptions()
                     .dtype(dtype)
                     .device(torch::kCUDA, at::cuda::current_device());
  auto tensor = torch::from_blob(ptr, {numel}, options);
  return tensor;
}

static ncclWindow_t lookup_window(const torch::Tensor& t) {
  uintptr_t ptr = reinterpret_cast<uintptr_t>(t.data_ptr());
  auto it = g_state.windows.find(ptr);
  if (it != g_state.windows.end()) return it->second.win;

  // Might be a view/slice: search for enclosing allocation
  for (auto& [base, info] : g_state.windows) {
    uintptr_t end = base + info.size;
    if (ptr >= base && ptr < end) return info.win;
  }
  throw std::runtime_error(
      "Tensor not backed by nccl_gin symmetric memory. "
      "Use nccl_gin.create_tensor() to allocate.");
}

bool is_symmetric_tensor(torch::Tensor t) {
  if (!g_state.initialized) return false;
  try {
    (void)lookup_window(t);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

static uintptr_t get_window_base(ncclWindow_t win) {
  for (auto& [base, info] : g_state.windows) {
    if (info.win == win) return base;
  }
  throw std::runtime_error("Window not found");
}

void put(torch::Tensor src_buffer, torch::Tensor dst_buffer,
         int64_t src_offset, int64_t dst_offset,
         int64_t num_bytes, int peer,
         int grid_size,
         c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");
  if (!g_state.hasDevComm)
    throw std::runtime_error("DevComm not created");

  ncclWindow_t srcWin = lookup_window(src_buffer);
  ncclWindow_t dstWin = lookup_window(dst_buffer);

  // Compute window-relative offsets
  uintptr_t srcBase = get_window_base(srcWin);
  uintptr_t dstBase = get_window_base(dstWin);
  size_t srcWinOff = (reinterpret_cast<uintptr_t>(src_buffer.data_ptr()) - srcBase) + src_offset;
  size_t dstWinOff = (reinterpret_cast<uintptr_t>(dst_buffer.data_ptr()) - dstBase) + dst_offset;

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  if (grid_size <= 1) {
    GinPutKernel<<<1, 512, 0, cuda_stream>>>(
        srcWin, srcWinOff, dstWin, dstWinOff,
        (size_t)num_bytes, peer, g_state.devComm);
  } else {
    GinPutKernelMultiBlock<<<grid_size, 512, 0, cuda_stream>>>(
        srcWin, srcWinOff, dstWin, dstWinOff,
        (size_t)num_bytes, peer, g_state.devComm);
  }
}

void put_signal_device(torch::Tensor src_buffer, torch::Tensor dst_buffer,
                       int64_t src_offset, int64_t dst_offset,
                       int64_t num_bytes, int peer,
                       int sig_idx,
                       c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");
  if (!g_state.hasDevComm)
    throw std::runtime_error("DevComm not created");
  if (sig_idx < 0 || sig_idx >= GIN_SIGNAL_COUNT)
    throw std::runtime_error("sig_idx out of range");

  ncclWindow_t srcWin = lookup_window(src_buffer);
  ncclWindow_t dstWin = lookup_window(dst_buffer);

  uintptr_t srcBase = get_window_base(srcWin);
  uintptr_t dstBase = get_window_base(dstWin);
  size_t srcWinOff = (reinterpret_cast<uintptr_t>(src_buffer.data_ptr()) - srcBase) + src_offset;
  size_t dstWinOff = (reinterpret_cast<uintptr_t>(dst_buffer.data_ptr()) - dstBase) + dst_offset;

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  GinPutSignalKernel<<<1, 512, 0, cuda_stream>>>(
      srcWin, srcWinOff, dstWin, dstWinOff,
      (size_t)num_bytes, peer, sig_idx, g_state.devComm);
}

void wait_signal_device(int sig_idx, int64_t num_bytes,
                        c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");
  if (!g_state.hasDevComm)
    throw std::runtime_error("DevComm not created");
  if (sig_idx < 0 || sig_idx >= GIN_SIGNAL_COUNT)
    throw std::runtime_error("sig_idx out of range");

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  GinWaitSignalMeetShadowKernel<<<1, 512, 0, cuda_stream>>>(
      (size_t)num_bytes, sig_idx, g_state.devComm);
}

void test_signal_device(int sig_idx, int64_t num_bytes,
                        torch::Tensor ready,
                        bool consume,
                        c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");
  if (!g_state.hasDevComm)
    throw std::runtime_error("DevComm not created");
  if (sig_idx < 0 || sig_idx >= GIN_SIGNAL_COUNT)
    throw std::runtime_error("sig_idx out of range");
  if (!ready.is_cuda())
    throw std::runtime_error("ready tensor must be CUDA");
  if (ready.scalar_type() != torch::kInt32)
    throw std::runtime_error("ready tensor must be torch.int32");
  if (ready.numel() < 1)
    throw std::runtime_error("ready tensor must have at least one element");
  if (!ready.is_contiguous())
    throw std::runtime_error("ready tensor must be contiguous");

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  GinTestSignalKernel<<<1, 512, 0, cuda_stream>>>(
      (size_t)num_bytes, sig_idx, ready.data_ptr<int>(), consume,
      g_state.devComm);
}

void get(torch::Tensor remote_buffer, torch::Tensor local_buffer,
         int64_t remote_offset, int64_t local_offset,
         int64_t num_bytes, int peer,
         int grid_size,
         c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");
  if (!g_state.hasDevComm)
    throw std::runtime_error("DevComm not created");

  ncclWindow_t remoteWin = lookup_window(remote_buffer);
  ncclWindow_t localWin = lookup_window(local_buffer);

  uintptr_t remoteBase = get_window_base(remoteWin);
  uintptr_t localBase = get_window_base(localWin);
  size_t remoteWinOff = (reinterpret_cast<uintptr_t>(remote_buffer.data_ptr()) - remoteBase) + remote_offset;
  size_t localWinOff = (reinterpret_cast<uintptr_t>(local_buffer.data_ptr()) - localBase) + local_offset;

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  if (grid_size <= 1) {
    GinGetKernel<<<1, 512, 0, cuda_stream>>>(
        remoteWin, remoteWinOff, localWin, localWinOff,
        (size_t)num_bytes, peer, g_state.devComm);
  } else {
    GinGetKernelMultiBlock<<<grid_size, 512, 0, cuda_stream>>>(
        remoteWin, remoteWinOff, localWin, localWinOff,
        (size_t)num_bytes, peer, g_state.devComm);
  }
}

static void check_desc(const torch::Tensor& t, torch::ScalarType st, int64_t K,
                       const char* name) {
  if (!t.is_cuda())
    throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
  if (t.scalar_type() != st)
    throw std::runtime_error(std::string(name) + " has wrong dtype");
  if (!t.is_contiguous())
    throw std::runtime_error(std::string(name) + " must be contiguous");
  if (t.numel() < K)
    throw std::runtime_error(std::string(name) + " shorter than K descriptors");
}

void get_batched(torch::Tensor remote_buffer, torch::Tensor local_buffer,
                 torch::Tensor remote_off, torch::Tensor local_off,
                 torch::Tensor nbytes, torch::Tensor peers, int64_t K,
                 int64_t blocks_per_desc,
                 c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");
  if (!g_state.hasDevComm)
    throw std::runtime_error("DevComm not created");
  if (K <= 0) return;
  check_desc(remote_off, torch::kInt64, K, "remote_off");
  check_desc(local_off, torch::kInt64, K, "local_off");
  check_desc(nbytes, torch::kInt64, K, "nbytes");
  check_desc(peers, torch::kInt32, K, "peers");

  ncclWindow_t remoteWin = lookup_window(remote_buffer);
  ncclWindow_t localWin = lookup_window(local_buffer);
  size_t remoteBase = reinterpret_cast<uintptr_t>(remote_buffer.data_ptr()) - get_window_base(remoteWin);
  size_t localBase = reinterpret_cast<uintptr_t>(local_buffer.data_ptr()) - get_window_base(localWin);

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  int nblk = blocks_per_desc > 0 ? (int)blocks_per_desc : 1;
  dim3 grid((unsigned)K, (unsigned)nblk);
  GinGetBatchedKernel<<<grid, 512, 0, cuda_stream>>>(
      remoteWin, remoteBase, localWin, localBase,
      remote_off.data_ptr<int64_t>(), local_off.data_ptr<int64_t>(),
      nbytes.data_ptr<int64_t>(), peers.data_ptr<int>(),
      (int)K, g_state.devComm);
}

void put_batched(torch::Tensor src_buffer, torch::Tensor dst_buffer,
                 torch::Tensor src_off, torch::Tensor dst_off,
                 torch::Tensor nbytes, torch::Tensor peers, int64_t K,
                 int64_t blocks_per_desc,
                 c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");
  if (!g_state.hasDevComm)
    throw std::runtime_error("DevComm not created");
  if (K <= 0) return;
  check_desc(src_off, torch::kInt64, K, "src_off");
  check_desc(dst_off, torch::kInt64, K, "dst_off");
  check_desc(nbytes, torch::kInt64, K, "nbytes");
  check_desc(peers, torch::kInt32, K, "peers");

  ncclWindow_t srcWin = lookup_window(src_buffer);
  ncclWindow_t dstWin = lookup_window(dst_buffer);
  size_t srcBase = reinterpret_cast<uintptr_t>(src_buffer.data_ptr()) - get_window_base(srcWin);
  size_t dstBase = reinterpret_cast<uintptr_t>(dst_buffer.data_ptr()) - get_window_base(dstWin);

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  int nblk = blocks_per_desc > 0 ? (int)blocks_per_desc : 1;
  dim3 grid((unsigned)K, (unsigned)nblk);
  GinPutBatchedKernel<<<grid, 512, 0, cuda_stream>>>(
      srcWin, srcBase, dstWin, dstBase,
      src_off.data_ptr<int64_t>(), dst_off.data_ptr<int64_t>(),
      nbytes.data_ptr<int64_t>(), peers.data_ptr<int>(),
      (int)K, g_state.devComm);
}

void put_signal(torch::Tensor src_buffer, torch::Tensor dst_buffer,
                int64_t src_offset, int64_t dst_offset,
                int64_t num_bytes, int peer,
                int sig_idx, int ctx,
                c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");

  ncclWindow_t dstWin = lookup_window(dst_buffer);
  uintptr_t dstBase = get_window_base(dstWin);
  size_t dstWinOff = (reinterpret_cast<uintptr_t>(dst_buffer.data_ptr()) - dstBase) + dst_offset;

  void* src_ptr = reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(src_buffer.data_ptr()) + src_offset);

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  NCCL_CHECK(ncclPutSignal(src_ptr, (size_t)num_bytes, ncclChar,
                           peer, dstWin, dstWinOff,
                           sig_idx, ctx, 0,
                           g_state.comm, cuda_stream));
}

void signal(int peer, int sig_idx, int ctx,
            c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  NCCL_CHECK(ncclSignal(peer, sig_idx, ctx, 0, g_state.comm, cuda_stream));
}

void wait_signal(int peer, int sig_idx, int op_cnt, int ctx,
                 c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  ncclWaitSignalDesc_t desc;
  desc.opCnt = op_cnt;
  desc.peer = peer;
  desc.sigIdx = sig_idx;
  desc.ctx = ctx;
  NCCL_CHECK(ncclWaitSignal(1, &desc, g_state.comm, cuda_stream));
}

void destroy() {
  std::lock_guard<std::mutex> lock(g_state.mu);
  if (!g_state.initialized) return;

  for (auto& [base, info] : g_state.windows) {
    ncclCommWindowDeregister(g_state.comm, info.win);
  }
  g_state.windows.clear();

  if (g_state.hasDevComm) {
    ncclDevCommDestroy(g_state.comm, &g_state.devComm);
    g_state.hasDevComm = false;
  }

  if (g_state.comm && g_state.ownsComm) {
    ncclCommDestroy(g_state.comm);
  }
  g_state.comm = nullptr;
  g_state.ownsComm = true;

  for (void* p : g_state.allocations) {
    ncclMemFree(p);
  }
  g_state.allocations.clear();

  g_state.initialized = false;
  g_state.rank = -1;
  g_state.nRanks = -1;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_nccl_gin_C, m) {
  m.doc() = "NCCL GIN P2P put/get library";
  m.def("get_unique_id", &get_unique_id,
        "Generate ncclUniqueId (call on rank 0 only)");
  m.def("init", &init, "Initialize NCCL GIN communicator",
        py::arg("unique_id"), py::arg("rank"), py::arg("nranks"));
  m.def("init_from_comm", &init_from_comm,
        "Initialize from an existing ncclComm_t pointer (int64)",
        py::arg("comm_handle"));
  m.def("get_rank", &get_rank);
  m.def("get_world_size", &get_world_size);
  m.def("get_comm_properties", &get_comm_properties,
        "Return ncclCommQueryProperties for the GIN communicator");
  m.def("create_tensor", &create_tensor,
        "Allocate symmetric memory and register as window",
        py::arg("numel"), py::arg("dtype"));
  m.def("is_symmetric_tensor", &is_symmetric_tensor,
        "Return whether a tensor is backed by a registered NCCL GIN window",
        py::arg("tensor"));
  m.def("put", &put,
        "P2P put via device-side GIN kernel",
        py::arg("src_buffer"), py::arg("dst_buffer"),
        py::arg("src_offset"), py::arg("dst_offset"),
        py::arg("num_bytes"), py::arg("peer"),
        py::arg("grid_size") = 1,
        py::arg("stream") = py::none());
  m.def("put_signal_device", &put_signal_device,
        "P2P put via device-side GIN kernel with ncclGin_SignalInc",
        py::arg("src_buffer"), py::arg("dst_buffer"),
        py::arg("src_offset"), py::arg("dst_offset"),
        py::arg("num_bytes"), py::arg("peer"),
        py::arg("sig_idx") = 0,
        py::arg("stream") = py::none());
  m.def("wait_signal_device", &wait_signal_device,
        "Wait for one device-side GIN signal increment per active context",
        py::arg("sig_idx"),
        py::arg("num_bytes"),
        py::arg("stream") = py::none());
  m.def("test_signal_device", &test_signal_device,
        "Test whether one device-side GIN signal increment is ready",
        py::arg("sig_idx"),
        py::arg("num_bytes"),
        py::arg("ready"),
        py::arg("consume") = true,
        py::arg("stream") = py::none());
  m.def("get", &get,
        "P2P get via device-side GIN kernel",
        py::arg("remote_buffer"), py::arg("local_buffer"),
        py::arg("remote_offset"), py::arg("local_offset"),
        py::arg("num_bytes"), py::arg("peer"),
        py::arg("grid_size") = 1,
        py::arg("stream") = py::none());
  m.def("get_batched", &get_batched,
        "Batched P2P get: K (peer, offset, size) descriptors resident on device",
        py::arg("remote_buffer"), py::arg("local_buffer"),
        py::arg("remote_off"), py::arg("local_off"),
        py::arg("nbytes"), py::arg("peers"), py::arg("k"),
        py::arg("blocks_per_desc") = 1,
        py::arg("stream") = py::none());
  m.def("put_batched", &put_batched,
        "Batched P2P put: K (peer, offset, size) descriptors resident on device",
        py::arg("src_buffer"), py::arg("dst_buffer"),
        py::arg("src_off"), py::arg("dst_off"),
        py::arg("nbytes"), py::arg("peers"), py::arg("k"),
        py::arg("blocks_per_desc") = 1,
        py::arg("stream") = py::none());
  m.def("put_signal", &put_signal,
        "P2P put via host-side ncclPutSignal",
        py::arg("src_buffer"), py::arg("dst_buffer"),
        py::arg("src_offset"), py::arg("dst_offset"),
        py::arg("num_bytes"), py::arg("peer"),
        py::arg("sig_idx") = 0,
        py::arg("ctx") = 0,
        py::arg("stream") = py::none());
  m.def("signal", &signal,
        "Send a host-side ncclSignal without payload",
        py::arg("peer"), py::arg("sig_idx") = 0,
        py::arg("ctx") = 0,
        py::arg("stream") = py::none());
  m.def("wait_signal", &wait_signal,
        "Wait for signal from peer",
        py::arg("peer"), py::arg("sig_idx") = 0,
        py::arg("op_cnt") = 1, py::arg("ctx") = 0,
        py::arg("stream") = py::none());
  m.def("destroy", &destroy, "Destroy all resources");
}
