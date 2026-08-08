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
constexpr int kWarpSize = 32;
constexpr int kMaxWarps = kMaxStage2Threads / kWarpSize;
constexpr unsigned kFullWarpMask = 0xffffffffu;

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
    int num_experts,
    int64_t num_slots) {
  if (threadIdx.x >= kWarpSize) {
    return;
  }

  const int lane = static_cast<int>(threadIdx.x);
  const int64_t candidate_begin =
      static_cast<int64_t>(blockIdx.x) * num_experts;

  // One warp owns one topology domain. Domain-local slot states are disjoint,
  // so admissions commute across blocks while remaining ordered within a domain.
  for (int index = 0; index < num_experts; ++index) {
    const int64_t c = candidate_begin + index;
    if (candidate_valid[c] == 0) {
      // _stage1_candidates sorts all valid entries before invalid entries.
      break;
    }
    const int expert = static_cast<int>(candidate_expert[c]);
    const int64_t domain = candidate_domain[c];
    bool local_present = false;
    int local_target = -1;
    int64_t local_slots = std::numeric_limits<int64_t>::max();

    for (int rank = lane; rank < num_ranks; rank += kWarpSize) {
      if (dom[rank] != domain) {
        continue;
      }
      if (x[static_cast<int64_t>(expert) * num_ranks + rank] != 0) {
        local_present = true;
      }
      const int64_t used = slot_used[rank];
      if (used < num_slots &&
          (used < local_slots ||
           (used == local_slots &&
            (local_target < 0 || rank < local_target)))) {
        local_target = rank;
        local_slots = used;
      }
    }

    const unsigned present_mask =
        __ballot_sync(kFullWarpMask, local_present);
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
      const int other_target =
          __shfl_down_sync(kFullWarpMask, local_target, offset);
      const auto other_slots = static_cast<int64_t>(__shfl_down_sync(
          kFullWarpMask,
          static_cast<unsigned long long>(local_slots),
          offset));
      if (lane < offset && other_target >= 0 &&
          (local_target < 0 || other_slots < local_slots ||
           (other_slots == local_slots && other_target < local_target))) {
        local_target = other_target;
        local_slots = other_slots;
      }
    }

    if (lane == 0 && present_mask == 0 && local_target >= 0) {
      x[static_cast<int64_t>(expert) * num_ranks + local_target] = 1;
      ++slot_used[local_target];
    }
    __syncwarp(kFullWarpMask);
  }
}

__global__ void update_routing_kernel(
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

__device__ __forceinline__ Candidate warp_reduce_candidate(Candidate value) {
  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    const Candidate other{
        static_cast<int64_t>(__shfl_down_sync(
            kFullWarpMask,
            static_cast<unsigned long long>(value.delta),
            offset)),
        static_cast<int64_t>(__shfl_down_sync(
            kFullWarpMask,
            static_cast<unsigned long long>(value.load),
            offset)),
        static_cast<int64_t>(__shfl_down_sync(
            kFullWarpMask,
            static_cast<unsigned long long>(value.cost),
            offset)),
        __shfl_down_sync(kFullWarpMask, value.expert, offset),
        __shfl_down_sync(kFullWarpMask, value.target, offset)};
    if (lane < offset && better_candidate(other, value)) {
      value = other;
    }
  }
  return value;
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

template <bool EnableGlobalStop>
__global__ void parallel_stage2_kernel(
    int8_t* x,
    const int64_t* cost,
    const int64_t* dom,
    int64_t* q,
    int64_t* expert_rank_load,
    int64_t* slot_used,
    int64_t* rank_load,
    uint8_t* stuck,
    const int64_t* stage2_control,
    int num_ranks,
    int num_experts,
    int64_t num_slots,
    int max_iterations,
    int stagnation_patience,
    int64_t quota_floor) {
  __shared__ Candidate warp_candidates[kMaxWarps];
  __shared__ int domain_ranks[kMaxStage2Threads];
  __shared__ int64_t domain_rank_load[kMaxStage2Threads];
  __shared__ int64_t domain_slot_used[kMaxStage2Threads];
  __shared__ uint8_t domain_stuck[kMaxStage2Threads];
  __shared__ int domain_rank_count;
  __shared__ int busiest_index;
  __shared__ int busiest_rank;
  __shared__ int64_t busiest_load;
  __shared__ int64_t best_domain_theta;
  __shared__ int stagnant_iterations;
  __shared__ int64_t source_warp_prefix[kMaxWarps];
  __shared__ int selected_target_index;
  __shared__ int apply_parallel_move;
  __shared__ int finished;

  const int tid = static_cast<int>(threadIdx.x);
  if (EnableGlobalStop && stage2_control[2] != 0) {
    return;
  }
  const int64_t domain_id = static_cast<int64_t>(blockIdx.x);
  if (tid == 0) {
    domain_rank_count = 0;
    for (int rank = 0; rank < num_ranks; ++rank) {
      if (dom[rank] == domain_id) {
        const int index = domain_rank_count++;
        domain_ranks[index] = rank;
        domain_rank_load[index] = rank_load[rank];
        domain_slot_used[index] = slot_used[rank];
        domain_stuck[index] = EnableGlobalStop ? stuck[rank] : 0;
      }
    }
  }
  __syncthreads();
  if (domain_rank_count == 0) {
    return;
  }

  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    if (tid == 0) {
      busiest_index = -1;
      busiest_rank = -1;
      busiest_load = -1;
      int64_t domain_theta = -1;
      for (int index = 0; index < domain_rank_count; ++index) {
        const int rank = domain_ranks[index];
        const int64_t load = domain_rank_load[index];
        if (EnableGlobalStop) {
          domain_theta = max_i64(domain_theta, load);
        }
        if (domain_stuck[index] == 0 &&
            (load > busiest_load ||
             (load == busiest_load && rank < busiest_rank))) {
          busiest_index = index;
          busiest_rank = rank;
          busiest_load = load;
        }
      }
      if (EnableGlobalStop) {
        if (iteration == 0) {
          best_domain_theta = domain_theta;
          stagnant_iterations = 0;
        } else if (domain_theta < best_domain_theta) {
          best_domain_theta = domain_theta;
          stagnant_iterations = 0;
        } else {
          ++stagnant_iterations;
        }
      }
      finished =
          (EnableGlobalStop &&
           stagnant_iterations >= stagnation_patience) ||
          busiest_rank < 0 ||
          busiest_load <= 0;
    }
    __syncthreads();
    if (finished) {
      break;
    }

    const int source_rank = busiest_rank;
    Candidate local_best = invalid_candidate();

    // Each block owns one domain. Experts are distributed over its threads and
    // all mutable rank-level state stays in shared memory until the final flush.
    for (int expert = tid; expert < num_experts; expert += blockDim.x) {
      const int64_t available =
          expert_rank_load[static_cast<int64_t>(expert) * num_ranks + source_rank];
      if (available <= 0) {
        continue;
      }
      for (int target_index = 0;
           target_index < domain_rank_count;
           ++target_index) {
        const int target = domain_ranks[target_index];
        const int64_t target_load = domain_rank_load[target_index];
        if (target == source_rank || target_load >= busiest_load) {
          continue;
        }
        const bool is_host =
            x[static_cast<int64_t>(expert) * num_ranks + target] != 0;
        if (!is_host && domain_slot_used[target_index] >= num_slots) {
          continue;
        }
        const int64_t gap = busiest_load - target_load;
        const int64_t delta = min_i64(available, gap / 2);
        if (delta < quota_floor) {
          continue;
        }
        const Candidate candidate{
            delta,
            target_load,
            cost[static_cast<int64_t>(source_rank) * num_ranks + target],
            expert,
            target};
        if (better_candidate(candidate, local_best)) {
          local_best = candidate;
        }
      }
    }

    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;
    const int num_warps = static_cast<int>(blockDim.x) / kWarpSize;
    const Candidate warp_best = warp_reduce_candidate(local_best);
    if (lane == 0) {
      warp_candidates[warp] = warp_best;
    }
    __syncthreads();

    if (warp == 0) {
      Candidate block_best =
          lane < num_warps ? warp_candidates[lane] : invalid_candidate();
      block_best = warp_reduce_candidate(block_best);
      if (lane == 0) {
        warp_candidates[0] = block_best;
      }
    }
    __syncthreads();

    if (quota_floor == 1) {
      if (tid == 0) {
        const Candidate best = warp_candidates[0];
        selected_target_index = -1;
        for (int index = 0; index < domain_rank_count; ++index) {
          if (domain_ranks[index] == best.target) {
            selected_target_index = index;
            break;
          }
        }
        apply_parallel_move =
            best.delta > 0 && selected_target_index >= 0;
        if (!apply_parallel_move) {
          domain_stuck[busiest_index] = 1;
        } else {
          const int64_t placement_offset =
              static_cast<int64_t>(best.expert) * num_ranks + best.target;
          if (x[placement_offset] == 0) {
            x[placement_offset] = 1;
            ++domain_slot_used[selected_target_index];
          }
        }
      }
      __syncthreads();

      if (apply_parallel_move) {
        const Candidate best = warp_candidates[0];
        int64_t source_quota = 0;
        int64_t target_quota = 0;
        int64_t from_offset = 0;
        int64_t to_offset = 0;
        if (tid < num_ranks) {
          from_offset =
              (static_cast<int64_t>(tid) * num_experts + best.expert) *
                  num_ranks +
              source_rank;
          to_offset =
              (static_cast<int64_t>(tid) * num_experts + best.expert) *
                  num_ranks +
              best.target;
          source_quota = q[from_offset];
          target_quota = q[to_offset];
        }

        int64_t inclusive = source_quota;
        for (int offset = 1; offset < kWarpSize; offset <<= 1) {
          const int64_t other = static_cast<int64_t>(__shfl_up_sync(
              kFullWarpMask,
              static_cast<unsigned long long>(inclusive),
              offset));
          if (lane >= offset) {
            inclusive += other;
          }
        }
        if (lane == kWarpSize - 1) {
          source_warp_prefix[warp] = inclusive;
        }
        __syncthreads();

        if (warp == 0) {
          int64_t warp_inclusive =
              lane < num_warps ? source_warp_prefix[lane] : 0;
          for (int offset = 1; offset < kWarpSize; offset <<= 1) {
            const int64_t other = static_cast<int64_t>(__shfl_up_sync(
                kFullWarpMask,
                static_cast<unsigned long long>(warp_inclusive),
                offset));
            if (lane >= offset) {
              warp_inclusive += other;
            }
          }
          if (lane < num_warps) {
            source_warp_prefix[lane] = warp_inclusive;
          }
        }
        __syncthreads();

        if (tid < num_ranks) {
          const int64_t prior_warps =
              warp == 0 ? 0 : source_warp_prefix[warp - 1];
          const int64_t prefix_before =
              prior_warps + inclusive - source_quota;
          const int64_t move = min_i64(
              source_quota,
              max_i64(best.delta - prefix_before, 0));
          q[from_offset] = source_quota - move;
          q[to_offset] = target_quota + move;
        }
      }
      __syncthreads();

      if (tid == 0 && apply_parallel_move) {
        const Candidate best = warp_candidates[0];
        const int64_t source_u_offset =
            static_cast<int64_t>(best.expert) * num_ranks + source_rank;
        const int64_t target_u_offset =
            static_cast<int64_t>(best.expert) * num_ranks + best.target;
        expert_rank_load[source_u_offset] -= best.delta;
        expert_rank_load[target_u_offset] += best.delta;
        domain_rank_load[busiest_index] -= best.delta;
        domain_rank_load[selected_target_index] += best.delta;
        for (int index = 0; index < domain_rank_count; ++index) {
          domain_stuck[index] = 0;
        }
      }
    } else if (tid == 0) {
      const Candidate best = warp_candidates[0];
      int target_index = -1;
      for (int index = 0; index < domain_rank_count; ++index) {
        if (domain_ranks[index] == best.target) {
          target_index = index;
          break;
        }
      }
      if (best.delta <= 0) {
        domain_stuck[busiest_index] = 1;
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
          domain_stuck[busiest_index] = 1;
        } else {
          const int64_t placement_offset =
              static_cast<int64_t>(best.expert) * num_ranks + best.target;
          if (x[placement_offset] == 0) {
            x[placement_offset] = 1;
            ++domain_slot_used[target_index];
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
          domain_rank_load[busiest_index] -= actual;
          domain_rank_load[target_index] += actual;
          for (int index = 0; index < domain_rank_count; ++index) {
            domain_stuck[index] = 0;
          }
        }
      }
    }
    __syncthreads();
  }

  for (int index = tid;
       index < domain_rank_count;
       index += static_cast<int>(blockDim.x)) {
    const int rank = domain_ranks[index];
    rank_load[rank] = domain_rank_load[index];
    slot_used[rank] = domain_slot_used[index];
    stuck[rank] = domain_stuck[index];
  }
}

__global__ void initialize_stage2_control_kernel(
    const int64_t* rank_load,
    int64_t* stage2_control,
    int num_ranks) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int64_t theta = -1;
  for (int rank = 0; rank < num_ranks; ++rank) {
    theta = max_i64(theta, rank_load[rank]);
  }
  stage2_control[0] = theta;
  stage2_control[1] = 0;
  stage2_control[2] = 0;
}

__global__ void update_stage2_control_kernel(
    const int64_t* rank_load,
    int64_t* stage2_control,
    int num_ranks,
    int completed_iterations,
    int stagnation_patience) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || stage2_control[2] != 0) {
    return;
  }
  int64_t theta = -1;
  for (int rank = 0; rank < num_ranks; ++rank) {
    theta = max_i64(theta, rank_load[rank]);
  }
  if (theta < stage2_control[0]) {
    stage2_control[0] = theta;
    stage2_control[1] = 0;
  } else {
    stage2_control[1] += completed_iterations;
    if (stage2_control[1] >= stagnation_patience) {
      stage2_control[2] = 1;
    }
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
    torch::Tensor stage2_control,
    int64_t num_domains,
    int64_t num_slots,
    int64_t max_stage2_iterations,
    int64_t stage2_stagnation_patience,
    bool stage2_patience_all_scales,
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
  check_cuda_contiguous(stage2_control, "stage2_control");

  TORCH_CHECK(omega.scalar_type() == torch::kInt64, "omega must be int64");
  TORCH_CHECK(x.scalar_type() == torch::kInt8, "x must be int8");
  TORCH_CHECK(q.scalar_type() == torch::kInt64, "q must be int64");
  TORCH_CHECK(stuck.scalar_type() == torch::kUInt8, "stuck must be uint8");
  TORCH_CHECK(
      stage2_control.scalar_type() == torch::kInt64,
      "stage2_control must be int64");
  TORCH_CHECK(
      stage2_control.numel() >= 3,
      "stage2_control must contain at least three values");
  TORCH_CHECK(omega.dim() == 2, "omega must have shape [R, E]");

  const int num_ranks = static_cast<int>(omega.size(0));
  const int num_experts = static_cast<int>(omega.size(1));
  TORCH_CHECK(num_ranks > 0 && num_ranks <= 1024, "fast CUDA solver requires 1 <= R <= 1024");
  TORCH_CHECK(num_experts > 0, "fast CUDA solver requires E > 0");
  TORCH_CHECK(
      num_domains > 0 && num_domains <= num_ranks,
      "fast CUDA solver requires 1 <= num_domains <= R");
  TORCH_CHECK(x.sizes() == torch::IntArrayRef({num_experts, num_ranks}), "x shape mismatch");
  TORCH_CHECK(
      q.sizes() == torch::IntArrayRef({num_ranks, num_experts, num_ranks}),
      "q shape mismatch");
  TORCH_CHECK(num_slots > 0, "num_slots must be positive");
  TORCH_CHECK(max_stage2_iterations > 0, "max_stage2_iterations must be positive");
  TORCH_CHECK(
      stage2_stagnation_patience > 0,
      "stage2_stagnation_patience must be positive");
  TORCH_CHECK(quota_floor > 0, "quota_floor must be positive");

  const c10::cuda::CUDAGuard device_guard(omega.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(omega.get_device());

  if (allow_cross_domain && candidate_expert.numel() > 0) {
    stage1_admit_kernel<<<static_cast<int>(num_domains), kWarpSize, 0, stream>>>(
        x.data_ptr<int8_t>(),
        dom.data_ptr<int64_t>(),
        candidate_expert.data_ptr<int64_t>(),
        candidate_domain.data_ptr<int64_t>(),
        candidate_valid.data_ptr<int64_t>(),
        slot_used.data_ptr<int64_t>(),
        num_ranks,
        num_experts,
        num_slots);
  }

  int update_routing_threads = 32;
  while (update_routing_threads < num_ranks) {
    update_routing_threads <<= 1;
  }
  update_routing_kernel<<<
      num_ranks * num_experts,
      update_routing_threads,
      0,
      stream>>>(
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
  const bool use_stagnation_probe =
      stage2_patience_all_scales &&
      max_stage2_iterations > stage2_stagnation_patience + 1;
  if (use_stagnation_probe) {
    initialize_stage2_control_kernel<<<1, 1, 0, stream>>>(
        rank_load.data_ptr<int64_t>(),
        stage2_control.data_ptr<int64_t>(),
        num_ranks);
    const auto launch_stage2_probe = [&](int iterations) {
      parallel_stage2_kernel<true><<<
          static_cast<int>(num_domains),
          stage2_threads,
          0,
          stream>>>(
          x.data_ptr<int8_t>(),
          cost.data_ptr<int64_t>(),
          dom.data_ptr<int64_t>(),
          q.data_ptr<int64_t>(),
          expert_rank_load.data_ptr<int64_t>(),
          slot_used.data_ptr<int64_t>(),
          rank_load.data_ptr<int64_t>(),
          stuck.data_ptr<uint8_t>(),
          stage2_control.data_ptr<int64_t>(),
          num_ranks,
          num_experts,
          num_slots,
          iterations,
          static_cast<int>(stage2_stagnation_patience),
          quota_floor);
    };

    // The first repair round can sharply reduce theta after Update Routing, so it
    // establishes the baseline and does not count as a stagnant round.
    launch_stage2_probe(1);
    int64_t remaining_iterations = max_stage2_iterations - 1;
    initialize_stage2_control_kernel<<<1, 1, 0, stream>>>(
        rank_load.data_ptr<int64_t>(),
        stage2_control.data_ptr<int64_t>(),
        num_ranks);
    const int probe_iterations = static_cast<int>(std::min<int64_t>(
        stage2_stagnation_patience,
        remaining_iterations));
    launch_stage2_probe(probe_iterations);
    remaining_iterations -= probe_iterations;
    if (remaining_iterations > 0) {
      update_stage2_control_kernel<<<1, 1, 0, stream>>>(
          rank_load.data_ptr<int64_t>(),
          stage2_control.data_ptr<int64_t>(),
          num_ranks,
          probe_iterations,
          static_cast<int>(stage2_stagnation_patience));
      launch_stage2_probe(static_cast<int>(remaining_iterations));
    }
  } else {
    parallel_stage2_kernel<false><<<
        static_cast<int>(num_domains),
        stage2_threads,
        0,
        stream>>>(
        x.data_ptr<int8_t>(),
        cost.data_ptr<int64_t>(),
        dom.data_ptr<int64_t>(),
        q.data_ptr<int64_t>(),
        expert_rank_load.data_ptr<int64_t>(),
        slot_used.data_ptr<int64_t>(),
        rank_load.data_ptr<int64_t>(),
        stuck.data_ptr<uint8_t>(),
        stage2_control.data_ptr<int64_t>(),
        num_ranks,
        num_experts,
        num_slots,
        static_cast<int>(max_stage2_iterations),
        static_cast<int>(stage2_stagnation_patience),
        quota_floor);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "fast_solve",
      &fast_solve_cuda,
      "Parallel Scale-EPLB CUDA solver (CUDA)");
}
