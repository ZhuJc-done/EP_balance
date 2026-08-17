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
static constexpr size_t LSA_TMA_ALIGN_BYTES = 16;
static constexpr int LSA_TMA_COPY_WARPS = 4;
static constexpr int LSA_TMA_STAGES = 2;
static constexpr size_t LSA_TMA_STAGE_BYTES = 1ULL << 12; // 4 KB
static constexpr size_t LSA_TMA_MIN_BYTES = LSA_TMA_STAGE_BYTES;
static constexpr size_t LSA_TMA_WARP_SMEM_BYTES =
    LSA_TMA_STAGES * LSA_TMA_STAGE_BYTES +
    LSA_TMA_STAGES * sizeof(uint64_t);
static constexpr size_t LSA_TMA_SMEM_BYTES =
    LSA_TMA_COPY_WARPS * LSA_TMA_WARP_SMEM_BYTES;

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

static bool read_bool_env_or_default(const char* name, bool default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return default_value;
  if (std::strcmp(raw, "1") == 0 || std::strcmp(raw, "true") == 0 ||
      std::strcmp(raw, "yes") == 0 || std::strcmp(raw, "on") == 0)
    return true;
  if (std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0 ||
      std::strcmp(raw, "no") == 0 || std::strcmp(raw, "off") == 0)
    return false;
  throw std::runtime_error(std::string("Invalid boolean env ") + name + "=" + raw);
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
          "ginConnectionCount=%u, ginContextCount=%u\n",
          caller, g_state.rank,
          static_cast<unsigned>(g_state.devComm.ginConnectionCount),
          static_cast<unsigned>(g_state.devComm.ginContextCount));
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
//
// Transport is chosen per descriptor. GIN is the network path (InfiniBand / RoCE); peers
// inside this rank's LSA team are reachable by TMA copies over NVLink through the same
// window registration, which is one collective registration covering both -- ncclGinRegister
// for the RDMA side, cuMemMap + cuMemSetAccess for the LSA side. Routing an intra-node peer
// through gin.put() would send it out to the NIC and back, so those descriptors take the
// TMA path instead, the same split NCCL's own hybrid all-to-all example makes.
// ---------------------------------------------------------------------------

// True when `peer` (a world rank of the GIN communicator) sits in this rank's LSA team, i.e. its
// memory is mapped into our address space. Asks NCCL's team arithmetic rather than testing the
// window [rank - lsaRank, ...): `ncclTeam` carries a stride, and under a rank ordering that
// interleaves nodes a contiguity test would call a genuinely remote peer load/store reachable and
// hand the copy a pointer into nothing.
__device__ __forceinline__ bool peer_is_lsa(const ncclDevComm& devComm, int peer) {
  return devComm.lsaSize > 1 &&
         ncclTeamRankIsMember(ncclTeamLsa(devComm), ncclTeamWorld(devComm), peer);
}

// Fallback copy for a small or misaligned LSA range. The normal aligned path below uses
// Hopper/Blackwell TMA to move peer global -> shared -> local global for get, and the reverse for put.
__device__ __forceinline__ void lsa_copy_bytes(char* __restrict__ dst,
                                               const char* __restrict__ src,
                                               size_t numBytes) {
  const uintptr_t mask = (reinterpret_cast<uintptr_t>(dst) |
                          reinterpret_cast<uintptr_t>(src) | (uintptr_t)numBytes);
  const unsigned tid = threadIdx.x, nthr = blockDim.x;
  if ((mask & 0xF) == 0) {
    int4* d4 = reinterpret_cast<int4*>(dst);
    const int4* s4 = reinterpret_cast<const int4*>(src);
    for (size_t i = tid; i < (numBytes >> 4); i += nthr) d4[i] = s4[i];
  } else {
    for (size_t i = tid; i < numBytes; i += nthr) dst[i] = src[i];
  }
}

// Minimal SM90+ TMA wrappers. The extension is built only for sm_90/sm_100; keeping the PTX local
// avoids taking a build-time dependency on DeepEP while matching its 1-D cp.async.bulk path.
__device__ __forceinline__ bool lsa_tma_elect_one_sync() {
  int pred = 0;
  asm volatile(
      "{\n"
      ".reg .b32 %%rx;\n"
      ".reg .pred %%px;\n"
      "elect.sync %%rx|%%px, %1;\n"
      "@%%px mov.s32 %0, 1;\n"
      "}\n"
      : "+r"(pred)
      : "r"(0xffffffff));
  return pred != 0;
}

__device__ __forceinline__ uint32_t lsa_tma_smem_addr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void lsa_tma_mbarrier_init(uint64_t* ptr) {
  asm volatile("mbarrier.init.shared::cta.b64 [%1], %0;" ::
               "r"(1), "r"(lsa_tma_smem_addr(ptr)));
  asm volatile("fence.mbarrier_init.release.cluster;" ::);
}

__device__ __forceinline__ void lsa_tma_mbarrier_arrive_expect_tx(
    uint64_t* ptr, int numBytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%1], %0;" ::
               "r"(numBytes), "r"(lsa_tma_smem_addr(ptr)));
}

__device__ __forceinline__ void lsa_tma_mbarrier_wait(uint64_t* ptr,
                                                       uint32_t* phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LSA_TMA_WAIT:\n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra LSA_TMA_DONE;\n\t"
      "bra LSA_TMA_WAIT;\n\t"
      "LSA_TMA_DONE:\n\t"
      "}" ::
      "r"(lsa_tma_smem_addr(ptr)), "r"(*phase), "r"(0x989680));
  *phase ^= 1;
}

static constexpr uint64_t LSA_TMA_EVICT_FIRST = 0x12f0000000000000ULL;
static constexpr uint64_t LSA_TMA_EVICT_NORMAL = 0x1000000000000000ULL;

__device__ __forceinline__ void lsa_tma_load(void* smemDst,
                                             const void* globalSrc,
                                             uint64_t* barrier,
                                             int numBytes) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1], %2, [%3], %4;" ::
      "r"(lsa_tma_smem_addr(smemDst)), "l"(globalSrc), "r"(numBytes),
      "r"(lsa_tma_smem_addr(barrier)), "l"(LSA_TMA_EVICT_FIRST)
      : "memory");
}

__device__ __forceinline__ void lsa_tma_store(void* globalDst,
                                              const void* smemSrc,
                                              int numBytes) {
  asm volatile(
      "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint "
      "[%0], [%1], %2, %3;" ::
      "l"(globalDst), "r"(lsa_tma_smem_addr(smemSrc)), "r"(numBytes),
      "l"(LSA_TMA_EVICT_NORMAL)
      : "memory");
}

__device__ __forceinline__ void lsa_tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int Remaining>
__device__ __forceinline__ void lsa_tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" :: "n"(Remaining) : "memory");
}

// One elected lane owns a two-stage TMA pipeline for its aligned warp subrange. Loads for the next
// use of a stage are issued only after the oldest store group has finished reading that stage.
__device__ __forceinline__ void lsa_tma_copy_aligned(
    char* __restrict__ dst, const char* __restrict__ src, size_t numBytes,
    unsigned char* warpSmem) {
  auto* barriers = reinterpret_cast<uint64_t*>(
      warpSmem + LSA_TMA_STAGES * LSA_TMA_STAGE_BYTES);
  uint32_t phases[LSA_TMA_STAGES] = {0, 0};
  #pragma unroll
  for (int s = 0; s < LSA_TMA_STAGES; ++s)
    lsa_tma_mbarrier_init(barriers + s);

  const size_t numIters =
      (numBytes + LSA_TMA_STAGE_BYTES - 1) / LSA_TMA_STAGE_BYTES;
  const size_t preload = min(numIters, (size_t)LSA_TMA_STAGES);
  for (size_t i = 0; i < preload; ++i) {
    const int stage = static_cast<int>(i % LSA_TMA_STAGES);
    const size_t off = i * LSA_TMA_STAGE_BYTES;
    const int bytes = static_cast<int>(
        min(LSA_TMA_STAGE_BYTES, numBytes - off));
    lsa_tma_load(warpSmem + stage * LSA_TMA_STAGE_BYTES,
                 src + off, barriers + stage, bytes);
    lsa_tma_mbarrier_arrive_expect_tx(barriers + stage, bytes);
  }

  for (size_t i = 0; i < numIters; ++i) {
    const int stage = static_cast<int>(i % LSA_TMA_STAGES);
    const size_t off = i * LSA_TMA_STAGE_BYTES;
    const int bytes = static_cast<int>(
        min(LSA_TMA_STAGE_BYTES, numBytes - off));
    lsa_tma_mbarrier_wait(barriers + stage, phases + stage);
    lsa_tma_store(dst + off, warpSmem + stage * LSA_TMA_STAGE_BYTES, bytes);
    lsa_tma_store_commit();

    const size_t next = i + 1;
    if (next >= (size_t)LSA_TMA_STAGES && next < numIters) {
      // At this point two stages have produced stores. Retire the oldest before reusing its buffer,
      // while leaving the newer store in flight.
      lsa_tma_store_wait<LSA_TMA_STAGES - 1>();
      const int nextStage = static_cast<int>(next % LSA_TMA_STAGES);
      const size_t nextOff = next * LSA_TMA_STAGE_BYTES;
      const int nextBytes = static_cast<int>(
          min(LSA_TMA_STAGE_BYTES, numBytes - nextOff));
      lsa_tma_load(warpSmem + nextStage * LSA_TMA_STAGE_BYTES,
                   src + nextOff, barriers + nextStage, nextBytes);
      lsa_tma_mbarrier_arrive_expect_tx(barriers + nextStage, nextBytes);
    }
  }
  lsa_tma_store_wait<0>();
}

template <bool SystemFence>
__device__ __forceinline__ void lsa_tma_copy_bytes(
    char* __restrict__ dst, const char* __restrict__ src, size_t numBytes,
    unsigned char* tmaSmem) {
  const unsigned warp = threadIdx.x / warpSize;
  if (warp >= LSA_TMA_COPY_WARPS) return;
  const unsigned lane = threadIdx.x % warpSize;

  const size_t units = numBytes / LSA_TMA_ALIGN_BYTES;
  const size_t chunk = units / LSA_TMA_COPY_WARPS;
  const size_t rem = units % LSA_TMA_COPY_WARPS;
  const size_t warpOff =
      (chunk * warp + min((size_t)warp, rem)) * LSA_TMA_ALIGN_BYTES;
  const size_t warpBytes =
      (chunk + (warp < rem ? 1 : 0)) * LSA_TMA_ALIGN_BYTES;

  if (warpBytes != 0) {
    const bool elected = lsa_tma_elect_one_sync();
    if (elected) {
      lsa_tma_copy_aligned(
          dst + warpOff, src + warpOff, warpBytes,
          tmaSmem + warp * LSA_TMA_WARP_SMEM_BYTES);
    }
  }
  __syncwarp();

  // TMA requires a 16B-multiple transaction. Only the final copy warp owns the short tail.
  if (warp == LSA_TMA_COPY_WARPS - 1) {
    const size_t tailOff = units * LSA_TMA_ALIGN_BYTES;
    for (size_t i = lane; i < numBytes - tailOff; i += warpSize)
      dst[tailOff + i] = src[tailOff + i];
  }
  if constexpr (SystemFence) __threadfence_system();
}

// 16B-aligned variant of block_subrange, so the LSA sub-ranges keep the vector path.
// Any trailing bytes below one unit are handed to the last sub-block.
__device__ __forceinline__ void block_subrange_16b(size_t numBytes, size_t* off, size_t* size) {
  const size_t nsub = gridDim.y, sub = blockIdx.y;
  const size_t units = numBytes >> 4;
  const size_t chunk = units / nsub, rem = units % nsub;
  *off = (chunk * sub + min(sub, rem)) << 4;
  *size = (chunk + (sub < rem ? 1 : 0)) << 4;
  if (sub == nsub - 1) *size += numBytes & 0xF;
}
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
                                    int K, int useLsa, int useTma,
                                    struct ncclDevComm devComm) {
  extern __shared__ __align__(LSA_TMA_ALIGN_BYTES) unsigned char lsaTmaSmem[];
  const int k = blockIdx.x;
  if (k >= K) return;
  const int peer = peers[k];
  if (peer < 0) return;                       // empty / local slot: nothing to fetch
  const size_t total = (size_t)nbytes[k];
  if (total == 0) return;
  const bool lsa = useLsa && peer_is_lsa(devComm, peer);
  size_t bOff, bSize;
  if (lsa) block_subrange_16b(total, &bOff, &bSize);
  else block_subrange(total, &bOff, &bSize);
  if (bSize == 0) return;
  if (lsa) {
    // peer's window is mapped here: read it over NVLink instead of pulling it through the NIC.
    // `peer` is a world rank, so name the team -- the two-argument overload's rank space is not
    // visible in the headers, and on a single-node run the spaces coincide and hide a mix-up.
    const char* src = (const char*)ncclGetPeerPointer(
        remoteWin, remoteBase + (size_t)remoteOff[k] + bOff, ncclTeamWorld(devComm), peer);
    char* dst = (char*)ncclGetLocalPointer(localWin, localBase + (size_t)localOff[k] + bOff);
    const uintptr_t mask =
        reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);
    if (useTma && bSize >= LSA_TMA_MIN_BYTES &&
        (mask & (LSA_TMA_ALIGN_BYTES - 1)) == 0)
      lsa_tma_copy_bytes<false>(dst, src, bSize, lsaTmaSmem);
    else
      lsa_copy_bytes(dst, src, bSize);
    return;
  }
  gin_get_range(devComm, peer,
                remoteWin, remoteBase + (size_t)remoteOff[k] + bOff,
                localWin, localBase + (size_t)localOff[k] + bOff,
                bSize);
}

__global__ void GinPutBatchedKernel(ncclWindow_t srcWin, size_t srcBase,
                                    ncclWindow_t dstWin, size_t dstBase,
                                    const int64_t* srcOff, const int64_t* dstOff,
                                    const int64_t* nbytes, const int* peers,
                                    int K, int useLsa, int useTma,
                                    struct ncclDevComm devComm) {
  extern __shared__ __align__(LSA_TMA_ALIGN_BYTES) unsigned char lsaTmaSmem[];
  const int k = blockIdx.x;
  if (k >= K) return;
  const int peer = peers[k];
  if (peer < 0) return;                       // empty / local slot: no remote push
  const size_t total = (size_t)nbytes[k];
  if (total == 0) return;
  const bool lsa = useLsa && peer_is_lsa(devComm, peer);
  size_t bOff, bSize;
  if (lsa) block_subrange_16b(total, &bOff, &bSize);
  else block_subrange(total, &bOff, &bSize);
  if (bSize == 0) return;
  if (lsa) {
    char* dst = (char*)ncclGetPeerPointer(
        dstWin, dstBase + (size_t)dstOff[k] + bOff, ncclTeamWorld(devComm), peer);
    const char* src = (const char*)ncclGetLocalPointer(
        srcWin, srcBase + (size_t)srcOff[k] + bOff);
    const uintptr_t mask =
        reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src);
    if (useTma && bSize >= LSA_TMA_MIN_BYTES &&
        (mask & (LSA_TMA_ALIGN_BYTES - 1)) == 0) {
      lsa_tma_copy_bytes<true>(dst, src, bSize, lsaTmaSmem);
    } else {
      lsa_copy_bytes(dst, src, bSize);
      // These stores land in another device's memory over the fabric, where completing the
      // kernel is not by itself enough to make them visible. The caller's fence orders the
      // ranks; this makes sure there is nothing left in flight when it runs.
      __threadfence_system();
    }
    return;
  }
  gin_put_range(devComm, peer,
                srcWin, srcBase + (size_t)srcOff[k] + bOff,
                dstWin, dstBase + (size_t)dstOff[k] + bOff,
                bSize);
}

// ---------------------------------------------------------------------------
// Local gradient staging and sparse owner reduction
//
// The symmetric scratch allocation is fixed at [max_mains, world, weight_bytes], but only the
// columns named by the current placement table contain this iteration's gradients. Predicating the
// loads here removes both the full-buffer memset and the dense world-way torch.sum. The owner's own
// gradient bypasses scratch, which also removes the fixed-shape index_copy_ and its dump row.
// ---------------------------------------------------------------------------
static constexpr int LOCAL_GRAD_THREADS = 256;
static constexpr int LOCAL_GRAD_MAX_BLOCKS = 4096;
static constexpr int MASKED_COPY_BYTES = 16;

__global__ void CopyRowsMaskedKernel(const uint8_t* __restrict__ src,
                                     uint8_t* __restrict__ dst,
                                     const bool* __restrict__ active,
                                     int64_t rows, int64_t rowBytes) {
  const int64_t chunksPerRow =
      (rowBytes + MASKED_COPY_BYTES - 1) / MASKED_COPY_BYTES;
  const int64_t tasks = rows * chunksPerRow;
  for (int64_t task = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       task < tasks;
       task += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = task / chunksPerRow;
    if (!active[row]) continue;
    const int64_t inRow = (task - row * chunksPerRow) * MASKED_COPY_BYTES;
    const int64_t bytes =
        min((int64_t)MASKED_COPY_BYTES, rowBytes - inRow);
    const int64_t off = row * rowBytes + inRow;
    const uint8_t* s = src + off;
    uint8_t* d = dst + off;
    const uintptr_t alignment =
        reinterpret_cast<uintptr_t>(s) | reinterpret_cast<uintptr_t>(d);
    if (bytes == MASKED_COPY_BYTES &&
        (alignment & (MASKED_COPY_BYTES - 1)) == 0) {
      *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(s);
    } else {
      #pragma unroll
      for (int i = 0; i < MASKED_COPY_BYTES; ++i)
        if (i < bytes) d[i] = s[i];
    }
  }
}

template <typename scalar_t>
__global__ void CopyRowsMaskedStridedKernel(
    const scalar_t* __restrict__ src,
    scalar_t* __restrict__ dst,
    const bool* __restrict__ active,
    int64_t rows, int64_t rowElems, int64_t innerDim,
    int64_t stride0, int64_t stride1, int64_t stride2) {
  const int64_t tasks = rows * rowElems;
  for (int64_t task = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       task < tasks;
       task += (int64_t)gridDim.x * blockDim.x) {
    const int64_t row = task / rowElems;
    if (!active[row]) continue;
    const int64_t elem = task - row * rowElems;
    const int64_t outer = elem / innerDim;
    const int64_t inner = elem - outer * innerDim;
    dst[task] = src[row * stride0 + outer * stride1 + inner * stride2];
  }
}

template <typename scalar_t>
__global__ void MaskedSumRowsKernel(
    const scalar_t* __restrict__ scratch,
    const scalar_t* __restrict__ local,
    const bool* __restrict__ active,
    const int64_t* __restrict__ localRows,
    scalar_t* __restrict__ output,
    int64_t experts, int64_t world, int64_t rowElems, int localRank,
    int localContiguous, int64_t localInnerDim,
    int64_t localStride0, int64_t localStride1, int64_t localStride2) {
  const int64_t tasks = experts * rowElems;
  for (int64_t task = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       task < tasks;
       task += (int64_t)gridDim.x * blockDim.x) {
    const int64_t expert = task / rowElems;
    const int64_t elem = task - expert * rowElems;
    float sum = 0.0f;
    for (int64_t rank = 0; rank < world; ++rank) {
      if (!active[expert * world + rank]) continue;
      int64_t localOff = localRows[expert] * rowElems + elem;
      if (!localContiguous) {
        const int64_t outer = elem / localInnerDim;
        const int64_t inner = elem - outer * localInnerDim;
        localOff = localRows[expert] * localStride0 +
                   outer * localStride1 + inner * localStride2;
      }
      const scalar_t value =
          rank == localRank
              ? local[localOff]
              : scratch[(expert * world + rank) * rowElems + elem];
      sum += static_cast<float>(value);
    }
    output[task] = static_cast<scalar_t>(sum);
  }
}

// ---------------------------------------------------------------------------
// World fence
//
// ncclSignal posts to a GIN connection, and connections only exist for peers this rank reaches
// over the network -- signalling a same-node peer fails outright, so a signal/wait mesh cannot
// order an EP group that lives inside one node. The barrier session splits the same way the
// transfers above do: an LSA barrier across the node, a GIN rail barrier across nodes, composed
// with release/acquire so it orders every peer whatever the topology. Both halves are provisioned
// by `barrierCount` at devComm creation.
// ---------------------------------------------------------------------------
__global__ void WorldFenceKernel(uint32_t index, struct ncclDevComm devComm) {
  ncclGin gin{devComm, 0};
  ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld{}, gin, index};
  bar.sync(ncclCoopCta(), cuda::memory_order_seq_cst, ncclGinFenceLevel::Relaxed);
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
  // LSA team = the peers whose windows are mapped here (NVLink / P2P). Size 1 means no peer is
  // load/store reachable, so every transfer takes the network path.
  out["lsaRank"] = g_state.hasDevComm ? g_state.devComm.lsaRank : 0;
  out["lsaSize"] = g_state.hasDevComm ? g_state.devComm.lsaSize : 1;
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
                 int64_t blocks_per_desc, bool use_lsa,
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
  const bool enableLsa = use_lsa && g_state.devComm.lsaSize > 1;
  const bool enableTma = enableLsa &&
      read_bool_env_or_default("EPLB_GIN_LSA_TMA", true);
  const size_t smemBytes = enableTma ? LSA_TMA_SMEM_BYTES : 0;
  GinGetBatchedKernel<<<grid, 512, smemBytes, cuda_stream>>>(
      remoteWin, remoteBase, localWin, localBase,
      remote_off.data_ptr<int64_t>(), local_off.data_ptr<int64_t>(),
      nbytes.data_ptr<int64_t>(), peers.data_ptr<int>(),
      (int)K, enableLsa ? 1 : 0, enableTma ? 1 : 0, g_state.devComm);
}

void put_batched(torch::Tensor src_buffer, torch::Tensor dst_buffer,
                 torch::Tensor src_off, torch::Tensor dst_off,
                 torch::Tensor nbytes, torch::Tensor peers, int64_t K,
                 int64_t blocks_per_desc, bool use_lsa,
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
  const bool enableLsa = use_lsa && g_state.devComm.lsaSize > 1;
  const bool enableTma = enableLsa &&
      read_bool_env_or_default("EPLB_GIN_LSA_TMA", true);
  const size_t smemBytes = enableTma ? LSA_TMA_SMEM_BYTES : 0;
  GinPutBatchedKernel<<<grid, 512, smemBytes, cuda_stream>>>(
      srcWin, srcBase, dstWin, dstBase,
      src_off.data_ptr<int64_t>(), dst_off.data_ptr<int64_t>(),
      nbytes.data_ptr<int64_t>(), peers.data_ptr<int>(),
      (int)K, enableLsa ? 1 : 0, enableTma ? 1 : 0, g_state.devComm);
}

void copy_rows_masked(torch::Tensor src, torch::Tensor dst,
                      torch::Tensor active,
                      c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!src.is_cuda() || !dst.is_cuda() || !active.is_cuda())
    throw std::runtime_error("copy_rows_masked tensors must be CUDA tensors");
  if (!dst.is_contiguous() || !active.is_contiguous())
    throw std::runtime_error("copy_rows_masked destination and mask must be contiguous");
  if (src.scalar_type() != dst.scalar_type())
    throw std::runtime_error("copy_rows_masked source and destination dtypes differ");
  if (src.sizes() != dst.sizes())
    throw std::runtime_error("copy_rows_masked source and destination shapes differ");
  if (src.dim() < 1)
    throw std::runtime_error("copy_rows_masked source must have a row dimension");
  if (active.scalar_type() != torch::kBool || active.dim() != 1 ||
      active.numel() != src.size(0))
    throw std::runtime_error("copy_rows_masked active must be bool [rows]");
  if (src.device() != dst.device() || src.device() != active.device())
    throw std::runtime_error("copy_rows_masked tensors must share a CUDA device");

  const int64_t rows = src.size(0);
  if (rows == 0) return;
  const int64_t totalBytes = src.numel() * src.element_size();
  if (totalBytes == 0) return;
  const int64_t rowBytes = totalBytes / rows;
  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  if (src.is_contiguous()) {
    const int64_t chunksPerRow =
        (rowBytes + MASKED_COPY_BYTES - 1) / MASKED_COPY_BYTES;
    const int64_t tasks = rows * chunksPerRow;
    int64_t blocks64 =
        (tasks + LOCAL_GRAD_THREADS - 1) / LOCAL_GRAD_THREADS;
    if (blocks64 > LOCAL_GRAD_MAX_BLOCKS) blocks64 = LOCAL_GRAD_MAX_BLOCKS;
    CopyRowsMaskedKernel<<<(int)blocks64, LOCAL_GRAD_THREADS, 0, cuda_stream>>>(
        reinterpret_cast<const uint8_t*>(src.data_ptr()),
        reinterpret_cast<uint8_t*>(dst.data_ptr()),
        active.data_ptr<bool>(), rows, rowBytes);
    CUDA_CHECK(cudaGetLastError());
    return;
  }

  if (src.dim() != 2 && src.dim() != 3)
    throw std::runtime_error(
        "copy_rows_masked non-contiguous source must be [rows, n] or [rows, m, n]");
  const int64_t rowElems = src.numel() / rows;
  const int64_t innerDim = src.dim() == 3 ? src.size(2) : 1;
  const int64_t stride0 = src.stride(0);
  const int64_t stride1 = src.stride(1);
  const int64_t stride2 = src.dim() == 3 ? src.stride(2) : 0;
  const int64_t tasks = rows * rowElems;
  int64_t blocks64 =
      (tasks + LOCAL_GRAD_THREADS - 1) / LOCAL_GRAD_THREADS;
  if (blocks64 > LOCAL_GRAD_MAX_BLOCKS) blocks64 = LOCAL_GRAD_MAX_BLOCKS;

#define LAUNCH_MASKED_COPY(scalar_t)                                             \
  CopyRowsMaskedStridedKernel<scalar_t>                                          \
      <<<(int)blocks64, LOCAL_GRAD_THREADS, 0, cuda_stream>>>(                    \
          src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>(),                    \
          active.data_ptr<bool>(), rows, rowElems, innerDim,                     \
          stride0, stride1, stride2)

  switch (src.scalar_type()) {
    case torch::kFloat32:
      LAUNCH_MASKED_COPY(float);
      break;
    case torch::kFloat16:
      LAUNCH_MASKED_COPY(at::Half);
      break;
    case torch::kBFloat16:
      LAUNCH_MASKED_COPY(at::BFloat16);
      break;
    default:
      throw std::runtime_error(
          "copy_rows_masked strided source supports float32, float16, and bfloat16");
  }
#undef LAUNCH_MASKED_COPY
  CUDA_CHECK(cudaGetLastError());
}

torch::Tensor masked_sum_rows(
    torch::Tensor scratch, torch::Tensor local, torch::Tensor active,
    torch::Tensor local_rows, int64_t local_rank,
    c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!scratch.is_cuda() || !local.is_cuda() || !active.is_cuda() ||
      !local_rows.is_cuda())
    throw std::runtime_error("masked_sum_rows tensors must be CUDA tensors");
  if (!scratch.is_contiguous() || !active.is_contiguous() ||
      !local_rows.is_contiguous())
    throw std::runtime_error(
        "masked_sum_rows scratch, active, and local_rows must be contiguous");
  if (scratch.dim() < 3)
    throw std::runtime_error("masked_sum_rows scratch must be [experts, world, ...]");
  if (local.dim() + 1 != scratch.dim())
    throw std::runtime_error("masked_sum_rows local tensor has the wrong rank");
  if (scratch.scalar_type() != local.scalar_type())
    throw std::runtime_error("masked_sum_rows scratch and local dtypes differ");
  if (scratch.device() != local.device() || scratch.device() != active.device() ||
      scratch.device() != local_rows.device())
    throw std::runtime_error("masked_sum_rows tensors must share a CUDA device");

  const int64_t experts = scratch.size(0);
  const int64_t world = scratch.size(1);
  if (active.scalar_type() != torch::kBool || active.dim() != 2 ||
      active.size(0) != experts || active.size(1) != world)
    throw std::runtime_error("masked_sum_rows active must be bool [experts, world]");
  if (local_rows.scalar_type() != torch::kInt64 || local_rows.dim() != 1 ||
      local_rows.numel() != experts)
    throw std::runtime_error("masked_sum_rows local_rows must be int64 [experts]");
  if (local_rank < 0 || local_rank >= world)
    throw std::runtime_error("masked_sum_rows local_rank is out of range");
  if (!local.is_contiguous() && local.dim() != 2 && local.dim() != 3)
    throw std::runtime_error(
        "masked_sum_rows non-contiguous local must be [slots, n] or [slots, m, n]");
  for (int64_t dim = 2; dim < scratch.dim(); ++dim)
    if (scratch.size(dim) != local.size(dim - 1))
      throw std::runtime_error("masked_sum_rows trailing shapes differ");

  std::vector<int64_t> outputShape;
  outputShape.reserve(scratch.dim() - 1);
  outputShape.push_back(experts);
  for (int64_t dim = 2; dim < scratch.dim(); ++dim)
    outputShape.push_back(scratch.size(dim));
  torch::Tensor output = torch::empty(outputShape, scratch.options());
  if (output.numel() == 0) return output;

  const int64_t rowElems = output.numel() / experts;
  int64_t blocks64 =
      (output.numel() + LOCAL_GRAD_THREADS - 1) / LOCAL_GRAD_THREADS;
  if (blocks64 > LOCAL_GRAD_MAX_BLOCKS) blocks64 = LOCAL_GRAD_MAX_BLOCKS;
  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();
  const int localContiguous = local.is_contiguous() ? 1 : 0;
  const int64_t localInnerDim =
      local.dim() == 3 ? local.size(2) : 1;
  const int64_t localStride0 = localContiguous ? 0 : local.stride(0);
  const int64_t localStride1 = localContiguous ? 0 : local.stride(1);
  const int64_t localStride2 =
      localContiguous || local.dim() != 3 ? 0 : local.stride(2);

#define LAUNCH_MASKED_SUM(scalar_t)                                              \
  MaskedSumRowsKernel<scalar_t><<<(int)blocks64, LOCAL_GRAD_THREADS, 0,           \
                                  cuda_stream>>>(                                 \
      scratch.data_ptr<scalar_t>(), local.data_ptr<scalar_t>(),                   \
      active.data_ptr<bool>(), local_rows.data_ptr<int64_t>(),                    \
      output.data_ptr<scalar_t>(), experts, world, rowElems, (int)local_rank,     \
      localContiguous, localInnerDim, localStride0, localStride1, localStride2)

  switch (scratch.scalar_type()) {
    case torch::kFloat32:
      LAUNCH_MASKED_SUM(float);
      break;
    case torch::kFloat16:
      LAUNCH_MASKED_SUM(at::Half);
      break;
    case torch::kBFloat16:
      LAUNCH_MASKED_SUM(at::BFloat16);
      break;
    default:
      throw std::runtime_error(
          "masked_sum_rows supports float32, float16, and bfloat16");
  }
#undef LAUNCH_MASKED_SUM
  CUDA_CHECK(cudaGetLastError());
  return output;
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

void world_fence(int64_t index, c10::optional<c10::cuda::CUDAStream> stream_opt) {
  if (!g_state.initialized)
    throw std::runtime_error("nccl_gin not initialized");
  if (!g_state.hasDevComm)
    throw std::runtime_error("nccl_gin has no devComm");

  cudaStream_t cuda_stream = stream_opt.has_value()
                                 ? stream_opt->stream()
                                 : at::cuda::getCurrentCUDAStream().stream();

  WorldFenceKernel<<<1, 128, 0, cuda_stream>>>(
      (uint32_t)(index % BARRIER_COUNT), g_state.devComm);
  CUDA_CHECK(cudaGetLastError());
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
        "Batched P2P get: K (peer, offset, size) descriptors resident on device; "
        "LSA-team peers are read over NVLink, the rest over GIN",
        py::arg("remote_buffer"), py::arg("local_buffer"),
        py::arg("remote_off"), py::arg("local_off"),
        py::arg("nbytes"), py::arg("peers"), py::arg("k"),
        py::arg("blocks_per_desc") = 1,
        py::arg("use_lsa") = true,
        py::arg("stream") = py::none());
  m.def("put_batched", &put_batched,
        "Batched P2P put: K (peer, offset, size) descriptors resident on device; "
        "LSA-team peers are written over NVLink, the rest over GIN",
        py::arg("src_buffer"), py::arg("dst_buffer"),
        py::arg("src_off"), py::arg("dst_off"),
        py::arg("nbytes"), py::arg("peers"), py::arg("k"),
        py::arg("blocks_per_desc") = 1,
        py::arg("use_lsa") = true,
        py::arg("stream") = py::none());
  m.def("copy_rows_masked", &copy_rows_masked,
        "Copy only active rows into a symmetric staging tensor",
        py::arg("src"), py::arg("dst"), py::arg("active"),
        py::arg("stream") = py::none());
  m.def("masked_sum_rows", &masked_sum_rows,
        "Sum live per-rank scratch rows, taking the local rank directly from slot gradients",
        py::arg("scratch"), py::arg("local"), py::arg("active"),
        py::arg("local_rows"), py::arg("local_rank"),
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
  m.def("world_fence", &world_fence,
        "Stream-ordered barrier over all ranks (LSA barrier inside the node, GIN rail across)",
        py::arg("index") = 0,
        py::arg("stream") = py::none());
  m.def("wait_signal", &wait_signal,
        "Wait for signal from peer",
        py::arg("peer"), py::arg("sig_idx") = 0,
        py::arg("op_cnt") = 1, py::arg("ctx") = 0,
        py::arg("stream") = py::none());
  m.def("destroy", &destroy, "Destroy all resources");
}
