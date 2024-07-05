#define INTERLEAVE_CATCH(idx, stride) idx += stride;

#define WARP_DYNAMIC_CATCH(idx, stride) \
  if (thread_lane == 0)                 \
    idx = atomicAdd(stride, 1);         \
  __syncwarp();                         \
  idx = __shfl_sync(0xffffffff, idx, 0);

#define BLOCK_DYNAMIC_CATCH(shared_idx, local_idx, stride) \
  if (threadIdx.x == 0)                                    \
    shared_idx = atomicAdd(stride, 1);                     \
  __syncthreads();                                         \
  local_idx = shared_idx;

#define DYNAMIC_MODE
#ifdef DYNAMIC_MODE
#define WARP_NEXT_WORK_CATCH(idx, stride1, stride2) \
  WARP_DYNAMIC_CATCH(idx, stride1);
#define BLOCK_NEXT_WORK_CATCH(shared_idx, local_idx, stride1, stride2) \
  BLOCK_DYNAMIC_CATCH(shared_idx, local_idx, stride1);
#define NEXT_WORK_CATCH(idx, stride1, stride2) \
  WARP_NEXT_WORK_CATCH(idx, stride1, stride2)
#else
#define WARP_NEXT_WORK_CATCH(idx, stride1, stride2) \
  INTERLEAVE_CATCH(idx, stride2);
#define BLOCK_NEXT_WORK_CATCH(shared_idx, local_idx, stride1, stride2) \
  INTERLEAVE_CATCH(local_idx, stride2);
#define NEXT_WORK_CATCH(idx, stride1, stride2) \
  WARP_NEXT_WORK_CATCH(idx, stride1, stride2)
#endif

// #define FREQ_PROFILE
#ifdef FREQ_PROFILE
#define PROFILE(result, size_a, size_b) \
  if (thread_lane == 0)                 \
    (result) += (size_a) * (size_b);
#else
#define PROFILE(result, size_a, size_b)
#endif

#define DEG_THD 128

#define SUB_BLOCK_HASH_LOOKUP
// #define WHOLE_BLOCK_HASH_LOOKUP

#define SUB_WARP_BINARYSEARCH_LOOKUP
// #define WHOLE_WARP_BINARYSEARCH_LOOKUP