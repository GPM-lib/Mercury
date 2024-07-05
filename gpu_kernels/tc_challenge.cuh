#include "tuning_schedules.cuh"

// Hash Table for large degree
__global__ void __launch_bounds__(256)
    hashIndex_for_large_degree(GraphGPU g, AccType *total, int *global_index,
                               int *large_tasks, int n_large_task, vidType bucket_num, vidType bucket_size, vidType *hash_bucket)
{
  __shared__ vidType bucket_count[1024];
  extern __shared__ vidType shared_bucket[];

  vidType *hash_bucket_cta;
  hash_bucket_cta = hash_bucket + blockIdx.x * bucket_num * bucket_size;
  unsigned long long __shared__ G_counter;
  if (threadIdx.x == 0)
    G_counter = 0;

  unsigned long long P_counter = 0;

  vidType task_id;
  __shared__ vidType shared_idx;

  task_id = blockIdx.x;
  while (task_id < n_large_task)
  {

    vidType vertex_id = large_tasks[task_id];
    eidType start = g.edge_begin(vertex_id);
    eidType end = g.edge_end(vertex_id);
    eidType degree = end - start;

    eidType now = start + threadIdx.x;

    for (vidType i = threadIdx.x; i < bucket_num; i += blockDim.x)
      bucket_count[i] = 0;
    __syncthreads();
    // Buiding Hash Table (block co-work)
    for (auto eid = threadIdx.x; eid < degree; eid += blockDim.x)
    {
      auto uid = g.getEdgeDst(eid + start);
      vidType bin = uid % bucket_num;
      vidType index = atomicAdd(&bucket_count[bin], 1);
      hash_bucket_cta[bin * bucket_size + index] = uid;
    }
    __syncthreads();

    // Prefix sum
    // bucket_count: the number of each slot.
    // shared_bucket: the prefix sum of each slot.
    // bucket_count[i] = shared_bucket[i+1] - shared_bucket[i]
    vidType pout = 0, pin = 1;
    for (vidType i = threadIdx.x; i < bucket_num; i += blockDim.x)
    {
      shared_bucket[i] = (i > 0) ? bucket_count[i - 1] : 0;
    }
    __syncthreads();

    for (vidType offset = 1; offset < bucket_num; offset *= 2)
    {
      pout = 1 - pout;
      pin = 1 - pout;
      for (vidType i = threadIdx.x; i < bucket_num; i += blockDim.x)
      {
        if (i >= offset)
        {
          shared_bucket[pout * bucket_num + i] = shared_bucket[pin * bucket_num + i - offset] + shared_bucket[pin * bucket_num + i];
        }
        else
        {
          shared_bucket[pout * bucket_num + i] = shared_bucket[pin * bucket_num + i];
        }
      }
      __syncthreads();
    }
    __syncthreads();

    // High 16 bits store offset index.
    // Low 16 bits store slot size.
    for (vidType i = threadIdx.x; i < bucket_num; i += blockDim.x)
    {
      bucket_count[i] = (bucket_count[i] & 0xFFFF) | ((shared_bucket[pout * bucket_num + i] & 0xFFFF) << 16);
    }

    // Load global hash table to shared array.
    // Compress shared array for hash table with offset and size.
    for (vidType i = threadIdx.x; i < bucket_num; i += blockDim.x)
    {
      vidType i_beg = bucket_count[i] >> 16;
      vidType i_num = bucket_count[i] & 0xFFFF;
      for (vidType index = 0; index < i_num; index++)
      {
        shared_bucket[i_beg + index] = hash_bucket_cta[i * bucket_size + index];
      }
    }

    __syncthreads();

    auto u_start = g.large_edge_begin(vertex_id);
    auto u_end = g.large_edge_end(vertex_id);

#ifdef WHOLE_BLOCK_HASH_LOOKUP
    for (auto u_idx = u_start; u_idx < u_end; u_idx++)
    {
      auto uid = g.get_large_EdgeDst(u_idx);
      auto w_start = g.edge_begin(uid);
      auto u_degree = g.edge_end(uid) - w_start;
      auto w_end = g.edge_end(uid);
      for (auto w_idx = w_start + threadIdx.x; w_idx < w_end; w_idx += blockDim.x)
      {
        vidType target = g.getEdgeDst(w_idx);
        vidType bin = target % bucket_num;

        vidType slot_offset = bucket_count[bin] >> 16;
        vidType slot_idx = 0;
        vidType slot_num = bucket_count[bin] & 0xFFFF;

        while (slot_idx < slot_num)
        {
          if (shared_bucket[slot_offset] == target)
          {
            P_counter++;
            slot_idx = slot_num;
          }
          slot_offset++;
          slot_idx++;
        }
      }
    }

#endif
#ifdef SUB_BLOCK_HASH_LOOKUP
    vidType nwarps_per_block = blockDim.x / WARP_SIZE;
    vidType warp_lane = threadIdx.x / WARP_SIZE;
    vidType thread_lane = threadIdx.x % WARP_SIZE;
    vidType workid = thread_lane;
    auto u_idx = u_start + warp_lane;

    vidType uid;
    eidType w_start;
    eidType u_degree;

    if (u_idx < u_end)
    {
      uid = g.get_large_EdgeDst(u_idx);
      w_start = g.edge_begin(uid);
      u_degree = g.edge_end(uid) - w_start;
    }
    int lc = 0;
    // Hash table lookup (warp co-work).
    // Each thread get one w as key to look up.
    while (u_idx < u_end)
    {
      while (u_idx < u_end && workid >= u_degree)
      {
        u_idx += nwarps_per_block;
        workid -= u_degree;
        uid = g.get_large_EdgeDst(u_idx);
        w_start = g.edge_begin(uid);
        u_degree = g.edge_end(uid) - w_start;
      }

      if (u_idx < u_end)
      {
        vidType target = g.getEdgeDst(w_start + workid);
        vidType bin = target % bucket_num;

        vidType slot_offset = bucket_count[bin] >> 16;
        vidType slot_idx = 0;
        vidType slot_num = bucket_count[bin] & 0xFFFF;

        while (slot_idx < slot_num)
        {
          if (shared_bucket[slot_offset] == target)
          {
            P_counter++;
            slot_idx = slot_num;
          }
          slot_offset++;
          slot_idx++;
        }
      }
      workid += WARP_SIZE;
    }
#endif
    __syncthreads();
    BLOCK_NEXT_WORK_CATCH(shared_idx, task_id, global_index, blockDim.x);
  }

  atomicAdd(&G_counter, P_counter);
  __syncthreads();
  if (threadIdx.x == 0)
    atomicAdd(total, G_counter);
}

// Binary search for small degree
__global__ void __launch_bounds__(256)
    binarySearch_for_small_degree(GraphGPU g, AccType *total, int *global_index, int *small_tasks, int n_small_tasks)
{
  __shared__ vidType shd_src[256 / WARP_SIZE * DEG_THD];
  unsigned long long __shared__ G_counter;
  if (threadIdx.x == 0)
  {
    G_counter = 0;
  }
  unsigned long long P_counter = 0;

  int num_warps = gridDim.x * blockDim.x / WARP_SIZE;
  vidType warp_id = threadIdx.x / WARP_SIZE;
  vidType global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  ;
  eidType thread_lane = threadIdx.x % WARP_SIZE;
  vidType task_id;

  vidType *shared_search_list = shd_src + warp_id * DEG_THD;
  __syncwarp();

  task_id = global_warp_id;
  while (task_id < n_small_tasks)
  {

    vidType vid = small_tasks[task_id];
    int start = g.edge_begin(vid);
    int end = g.edge_end(vid);
    int degree = end - start;

    // load V'neighbours to shared memory
    if (degree <= DEG_THD)
      for (auto eid = thread_lane; eid < degree; eid += WARP_SIZE)
      {
        auto uid = g.getEdgeDst(eid + start);
        shared_search_list[eid] = uid;
      }
    __syncwarp();

    auto u_start = g.small_edge_begin(vid);
    auto u_end = g.small_edge_end(vid);

#ifdef WHOLE_WARP_BINARYSEARCH_LOOKUP
    for (auto u_idx = u_start; u_idx < u_end; u_idx++)
    {
      vidType uid = g.get_small_EdgeDst(u_idx);
      eidType w_start = g.edge_begin(uid);
      eidType w_end = g.edge_end(uid);
      eidType u_degree = w_end - w_start;

      for (auto w_idx = thread_lane + w_start; w_idx < w_end; w_idx += WARP_SIZE)
      {
        vidType wid = g.getEdgeDst(w_idx);
        P_counter += binary_search(shared_search_list, wid, degree);
      }
    }
#endif

#ifdef SUB_WARP_BINARYSEARCH_LOOKUP
    eidType workid = thread_lane;
    eidType l = 0;
    eidType r = degree - 1;

    auto u_idx = u_start;
    while (u_idx < u_end)
    {
      vidType uid = g.get_small_EdgeDst(u_idx);
      eidType w_start = g.edge_begin(uid);
      eidType w_end = g.edge_end(uid);
      eidType u_degree = w_end - w_start;

      // If warp_size >= dregee
      // Use sub_warp to process different vertices.
      while (u_idx < u_end && workid >= u_degree)
      {
        u_idx++;
        workid -= u_degree;
        uid = g.get_small_EdgeDst(u_idx);
        w_start = g.edge_begin(uid);
        u_degree = g.edge_end(uid) - w_start;
        l = 0;
      }

      r = degree - 1;
      if (u_idx < u_end)
      {
        vidType wid = g.getEdgeDst(w_start + workid);
        P_counter += binary_search(shared_search_list, wid, degree);
      }
      __syncwarp();

      l = __shfl_sync(0xffffffff, l, WARP_SIZE - 1);
      u_idx = __shfl_sync(0xffffffff, u_idx, WARP_SIZE - 1);
      workid = __shfl_sync(0xffffffff, workid, WARP_SIZE - 1);

      workid += thread_lane + 1;
    }
#endif
    __syncwarp();
    WARP_NEXT_WORK_CATCH(task_id, global_index, num_warps);
  }

  atomicAdd(&G_counter, P_counter);
  __syncthreads();
  if (threadIdx.x == 0)
  {
    atomicAdd(total, G_counter);
  }
}
