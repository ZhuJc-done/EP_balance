#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

constexpr int kMaxStage2Threads = 1024;

__device__ __forceinline__ int64_t min_i64(int64_t a, int64_t b) {
  return a < b ? a : b;
}

__device__ __forceinline__ int64_t max_i64(int64_t a, int64_t b) {
  return a > b ? a : b;
}

__device__ __forceinline__ void atomic_add_i64(int64_t* address, int64_t value) {
  atomicAdd(
      reinterpret_cast<unsigned long long*>(address),
      static_cast<unsigned long long>(value));
}

__global__ void stage1_admit_kernel(
    int8_t* x,
    const int64_t* dom,
    const int64_t* candidate_expert,
    const int64_t* candidate_domain,
    const int64_t* candidate_valid,
    int64_t* slot_used,
    int num_ranks,
    int64_t num_candidates,
    int64_t num_slots) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  // Admissions are intentionally ordered by the pre-sorted benefit list.  This
  // loop is small (E * actual_num_domains), while routing is fully parallel.
  for (int64_t c = 0; c < num_candidates; ++c) {
    if (candidate_valid[c] == 0) {
      continue;
    }
    const int expert = static_cast<int>(candidate_expert[c]);
    const int64_t domain = candidate_domain[c];
    bool already_present = false;
    int target = -1;
    int64_t target_slots = std::numeric_limits<int64_t>::max();

    for (int rank = 0; rank < num_ranks; ++rank) {
      if (dom[rank] != domain) {
        continue;
      }
      if (x[static_cast<int64_t>(expert) * num_ranks + rank] != 0) {
        already_present = true;
      }
      const int64_t used = slot_used[rank];
      if (used < num_slots &&
          (used < target_slots || (used == target_slots && rank < target))) {
        target = rank;
        target_slots = used;
      }
    }

    if (!already_present && target >= 0) {
      x[static_cast<int64_t>(expert) * num_ranks + target] = 1;
      ++slot_used[target];
    }
  }
}

__global__ void parallel_route_kernel(
    const int64_t* omega,
    const int8_t* x,
    const int64_t* cost,
    const int64_t* dom,
    int64_t* q,
    int64_t* expert_rank_load,
    int64_t* rank_load,
    int num_ranks,
    int num_experts,
    int64_t quota_floor) {
  const int pair = static_cast<int>(blockIdx.x);
  const int src = pair / num_experts;
  const int expert = pair - src * num_experts;
  const int dst = static_cast<int>(threadIdx.x);

  __shared__ int has_local;
  __shared__ int active_count;
  __shared__ int active_prefix[1024];
  if (threadIdx.x == 0) {
    has_local = 0;
  }
  __syncthreads();

  bool host = false;
  bool local = false;
  if (dst < num_ranks) {
    host = x[static_cast<int64_t>(expert) * num_ranks + dst] != 0;
    local = host && dom[dst] == dom[src];
    if (local) {
      atomicExch(&has_local, 1);
    }
  }
  __syncthreads();

  const bool active = dst < num_ranks && host && (!has_local || local);
  active_prefix[threadIdx.x] = active ? 1 : 0;
  __syncthreads();
  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    const int prior =
        threadIdx.x >= offset ? active_prefix[threadIdx.x - offset] : 0;
    __syncthreads();
    if (threadIdx.x >= offset) {
      active_prefix[threadIdx.x] += prior;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    active_count = active_prefix[blockDim.x - 1];
  }
  __syncthreads();

  if (dst >= num_ranks) {
    return;
  }

  const int64_t need =
      omega[static_cast<int64_t>(src) * num_experts + expert];
  int64_t assigned = 0;
  if (active && need > 0 && active_count > 0) {
    // Rank active destinations by id using a block scan.
    // Every (src, expert) pair is independent, so the grid exposes R*E blocks
    // instead of serialising all pairs in one persistent block.
    const int position = active_prefix[dst] - 1;

    int destinations = active_count;
    if (quota_floor > 1 && need >= quota_floor) {
      destinations = static_cast<int>(
          min_i64(active_count, need / quota_floor));
    } else if (need < quota_floor) {
      // Input validation normally rejects this case.  Keep token conservation
      // when validate=False, matching the exact solver's best-effort behavior.
      destinations = 1;
    }

    if (position < destinations) {
      assigned = need / destinations + (position < need % destinations ? 1 : 0);
    }
  }

  const int64_t q_offset =
      (static_cast<int64_t>(src) * num_experts + expert) * num_ranks + dst;
  q[q_offset] = assigned;
  if (assigned > 0) {
    atomic_add_i64(
        expert_rank_load + static_cast<int64_t>(expert) * num_ranks + dst,
        assigned);
    atomic_add_i64(rank_load + dst, assigned);
  }
}

struct Candidate {
  int64_t delta;
  int64_t load;
  int64_t cost;
  int expert;
  int target;
};

__device__ __forceinline__ Candidate invalid_candidate() {
  return Candidate{
      0,
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int>::max(),
      std::numeric_limits<int>::max()};
}

__device__ __forceinline__ bool better_candidate(
    const Candidate& lhs,
    const Candidate& rhs) {
  if (lhs.delta != rhs.delta) {
    return lhs.delta > rhs.delta;
  }
  if (lhs.load != rhs.load) {
    return lhs.load < rhs.load;
  }
  if (lhs.cost != rhs.cost) {
    return lhs.cost < rhs.cost;
  }
  if (lhs.expert != rhs.expert) {
    return lhs.expert < rhs.expert;
  }
  return lhs.target < rhs.target;
}

__device__ __forceinline__ int64_t safe_move(
    int64_t source_quota,
    int64_t target_quota,
    int64_t remaining,
    int64_t quota_floor) {
  const int64_t upper = min_i64(source_quota, remaining);
  if (upper <= 0) {
    return 0;
  }
  if (upper == source_quota && target_quota + upper >= quota_floor) {
    return upper;
  }
  const int64_t partial_cap =
      min_i64(upper, max_i64(source_quota - quota_floor, 0));
  const int64_t minimum_partial = target_quota > 0 ? 1 : quota_floor;
  return partial_cap >= minimum_partial ? partial_cap : 0;
}

__global__ void parallel_stage2_kernel(
    int8_t* x,
    const int64_t* cost,
    const int64_t* dom,
    int64_t* q,
    int64_t* expert_rank_load,
    int64_t* slot_used,
    int64_t* rank_load,
    uint8_t* stuck,
    int num_ranks,
    int num_experts,
    int64_t num_slots,
    int max_iterations,
    int64_t quota_floor) {
  __shared__ Candidate candidates[kMaxStage2Threads];
  __shared__ int busiest_rank;
  __shared__ int64_t busiest_load;
  __shared__ int finished;

  const int tid = static_cast<int>(threadIdx.x);
  if (tid < num_ranks) {
    stuck[tid] = 0;
  }
  __syncthreads();

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    if (tid == 0) {
      busiest_rank = -1;
      busiest_load = -1;
      for (int rank = 0; rank < num_ranks; ++rank) {
        if (stuck[rank] == 0 &&
            (rank_load[rank] > busiest_load ||
             (rank_load[rank] == busiest_load && rank < busiest_rank))) {
          busiest_rank = rank;
          busiest_load = rank_load[rank];
        }
      }
      finished = busiest_rank < 0 || busiest_load <= 0;
    }
    __syncthreads();
    if (finished) {
      break;
    }

    const int source_rank = busiest_rank;
    const int64_t source_domain = dom[source_rank];
    Candidate local_best = invalid_candidate();

    // Experts are distributed over the block; each thread only scans target
    // ranks in the overloaded rank's domain.
    for (int expert = tid; expert < num_experts; expert += blockDim.x) {
      const int64_t available =
          expert_rank_load[static_cast<int64_t>(expert) * num_ranks + source_rank];
      if (available <= 0) {
        continue;
      }
      for (int target = 0; target < num_ranks; ++target) {
        if (target == source_rank || dom[target] != source_domain ||
            rank_load[target] >= busiest_load) {
          continue;
        }
        const bool is_host =
            x[static_cast<int64_t>(expert) * num_ranks + target] != 0;
        if (!is_host && slot_used[target] >= num_slots) {
          continue;
        }
        const int64_t gap = busiest_load - rank_load[target];
        const int64_t delta = min_i64(available, gap / 2);
        if (delta < quota_floor) {
          continue;
        }
        const Candidate candidate{
            delta,
            rank_load[target],
            cost[static_cast<int64_t>(source_rank) * num_ranks + target],
            expert,
            target};
        if (better_candidate(candidate, local_best)) {
          local_best = candidate;
        }
      }
    }

    candidates[tid] = local_best;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride &&
          better_candidate(candidates[tid + stride], candidates[tid])) {
        candidates[tid] = candidates[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const Candidate best = candidates[0];
      if (best.delta <= 0) {
        stuck[source_rank] = 1;
      } else {
        int64_t transfer_limit = best.delta;
        int64_t remaining = transfer_limit;
        int64_t actual = 0;
        for (int src = 0; src < num_ranks; ++src) {
          const int64_t from_offset =
              (static_cast<int64_t>(src) * num_experts + best.expert) *
                  num_ranks +
              source_rank;
          const int64_t to_offset =
              (static_cast<int64_t>(src) * num_experts + best.expert) *
                  num_ranks +
              best.target;
          const int64_t move =
              safe_move(q[from_offset], q[to_offset], remaining, quota_floor);
          remaining -= move;
          actual += move;
        }

        if (actual < quota_floor) {
          transfer_limit = busiest_load - best.load - 1;
          remaining = transfer_limit;
          actual = 0;
          for (int src = 0; src < num_ranks; ++src) {
            const int64_t from_offset =
                (static_cast<int64_t>(src) * num_experts + best.expert) *
                    num_ranks +
                source_rank;
            const int64_t to_offset =
                (static_cast<int64_t>(src) * num_experts + best.expert) *
                    num_ranks +
                best.target;
            const int64_t move =
                safe_move(q[from_offset], q[to_offset], remaining, quota_floor);
            remaining -= move;
            actual += move;
          }
        }

        if (actual < quota_floor) {
          stuck[source_rank] = 1;
        } else {
          const int64_t placement_offset =
              static_cast<int64_t>(best.expert) * num_ranks + best.target;
          if (x[placement_offset] == 0) {
            x[placement_offset] = 1;
            ++slot_used[best.target];
          }

          remaining = transfer_limit;
          for (int src = 0; src < num_ranks; ++src) {
            const int64_t from_offset =
                (static_cast<int64_t>(src) * num_experts + best.expert) *
                    num_ranks +
                source_rank;
            const int64_t to_offset =
                (static_cast<int64_t>(src) * num_experts + best.expert) *
                    num_ranks +
                best.target;
            const int64_t source_quota = q[from_offset];
            const int64_t target_quota = q[to_offset];
            const int64_t move =
                safe_move(source_quota, target_quota, remaining, quota_floor);
            q[from_offset] = source_quota - move;
            q[to_offset] = target_quota + move;
            remaining -= move;
          }

          const int64_t source_u_offset =
              static_cast<int64_t>(best.expert) * num_ranks + source_rank;
          const int64_t target_u_offset =
              static_cast<int64_t>(best.expert) * num_ranks + best.target;
          expert_rank_load[source_u_offset] -= actual;
          expert_rank_load[target_u_offset] += actual;
          rank_load[source_rank] -= actual;
          rank_load[best.target] += actual;

          for (int rank = 0; rank < num_ranks; ++rank) {
            if (dom[rank] == source_domain) {
              stuck[rank] = 0;
            }
          }
        }
      }
    }
    __syncthreads();
  }
}

void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void fast_solve_cuda(
    torch::Tensor omega,
    torch::Tensor x,
    torch::Tensor cost,
    torch::Tensor dom,
    torch::Tensor q,
    torch::Tensor expert_rank_load,
    torch::Tensor candidate_expert,
    torch::Tensor candidate_domain,
    torch::Tensor candidate_valid,
    torch::Tensor slot_used,
    torch::Tensor rank_load,
    torch::Tensor stuck,
    int64_t num_slots,
    int64_t max_stage2_iterations,
    int64_t quota_floor,
    bool allow_cross_domain) {
  check_cuda_contiguous(omega, "omega");
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(cost, "cost");
  check_cuda_contiguous(dom, "dom");
  check_cuda_contiguous(q, "q");
  check_cuda_contiguous(expert_rank_load, "expert_rank_load");
  check_cuda_contiguous(candidate_expert, "candidate_expert");
  check_cuda_contiguous(candidate_domain, "candidate_domain");
  check_cuda_contiguous(candidate_valid, "candidate_valid");
  check_cuda_contiguous(slot_used, "slot_used");
  check_cuda_contiguous(rank_load, "rank_load");
  check_cuda_contiguous(stuck, "stuck");

  TORCH_CHECK(omega.scalar_type() == torch::kInt64, "omega must be int64");
  TORCH_CHECK(x.scalar_type() == torch::kInt8, "x must be int8");
  TORCH_CHECK(q.scalar_type() == torch::kInt64, "q must be int64");
  TORCH_CHECK(stuck.scalar_type() == torch::kUInt8, "stuck must be uint8");
  TORCH_CHECK(omega.dim() == 2, "omega must have shape [R, E]");

  const int num_ranks = static_cast<int>(omega.size(0));
  const int num_experts = static_cast<int>(omega.size(1));
  TORCH_CHECK(num_ranks > 0 && num_ranks <= 1024, "fast CUDA solver requires 1 <= R <= 1024");
  TORCH_CHECK(num_experts > 0, "fast CUDA solver requires E > 0");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({num_experts, num_ranks}), "x shape mismatch");
  TORCH_CHECK(
      q.sizes() == torch::IntArrayRef({num_ranks, num_experts, num_ranks}),
      "q shape mismatch");
  TORCH_CHECK(num_slots > 0, "num_slots must be positive");
  TORCH_CHECK(max_stage2_iterations > 0, "max_stage2_iterations must be positive");
  TORCH_CHECK(quota_floor > 0, "quota_floor must be positive");

  const c10::cuda::CUDAGuard device_guard(omega.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(omega.get_device());

  if (allow_cross_domain && candidate_expert.numel() > 0) {
    stage1_admit_kernel<<<1, 1, 0, stream>>>(
        x.data_ptr<int8_t>(),
        dom.data_ptr<int64_t>(),
        candidate_expert.data_ptr<int64_t>(),
        candidate_domain.data_ptr<int64_t>(),
        candidate_valid.data_ptr<int64_t>(),
        slot_used.data_ptr<int64_t>(),
        num_ranks,
        candidate_expert.numel(),
        num_slots);
  }

  int route_threads = 32;
  while (route_threads < num_ranks) {
    route_threads <<= 1;
  }
  parallel_route_kernel<<<num_ranks * num_experts, route_threads, 0, stream>>>(
      omega.data_ptr<int64_t>(),
      x.data_ptr<int8_t>(),
      cost.data_ptr<int64_t>(),
      dom.data_ptr<int64_t>(),
      q.data_ptr<int64_t>(),
      expert_rank_load.data_ptr<int64_t>(),
      rank_load.data_ptr<int64_t>(),
      num_ranks,
      num_experts,
      quota_floor);

  int stage2_threads = 32;
  const int stage2_width = std::max(num_ranks, std::min(num_experts, 1024));
  while (stage2_threads < stage2_width) {
    stage2_threads <<= 1;
  }
  parallel_stage2_kernel<<<1, stage2_threads, 0, stream>>>(
      x.data_ptr<int8_t>(),
      cost.data_ptr<int64_t>(),
      dom.data_ptr<int64_t>(),
      q.data_ptr<int64_t>(),
      expert_rank_load.data_ptr<int64_t>(),
      slot_used.data_ptr<int64_t>(),
      rank_load.data_ptr<int64_t>(),
      stuck.data_ptr<uint8_t>(),
      num_ranks,
      num_experts,
      num_slots,
      static_cast<int>(max_stage2_iterations),
      quota_floor);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fast_solve",
      &fast_solve_cuda,
      "Parallel Scale-EPLB CUDA solver (CUDA)");
}
