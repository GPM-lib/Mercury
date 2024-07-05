#include <cub/cub.cuh>
#include <mpi.h>
#include "timer.h"
#include "edgelist.h"
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "tc_challenge.cuh"

std::vector<eidType> round_robin(int n, int *total_task, int n_tasks, std::vector<vidType *> &task_ptrs, int stride)
{
  task_ptrs.resize(n);
  if (n_tasks == 0)
  {
    std::vector<eidType> lens(n, 0);
    return lens;  
  }
  eidType total_num_chunks = (n_tasks - 1) / stride + 1;
  eidType nchunks_per_queue = total_num_chunks / n;
  std::vector<eidType> lens(n, stride * nchunks_per_queue);
  if (total_num_chunks % n != 0)
  {
    for (int i = 0; i < n; i++)
    {
      if (i + 1 == int(total_num_chunks % n))
      {
        lens[i] += n_tasks % stride == 0 ? stride : n_tasks % stride;
      }
      else if (i + 1 < int(total_num_chunks % n))
      {
        lens[i] += stride;
      }
    }
  }
  else
  {
    lens[n - 1] = lens[n - 1] + n_tasks % stride - stride;
  }
  for (int i = 0; i < n; i++)
  {
    task_ptrs[i] = new vidType[lens[i]];
  }
#pragma omp parallel for
  for (eidType chunk_id = 0; chunk_id < nchunks_per_queue; chunk_id++)
  {
    eidType begin = chunk_id * n * stride;
    for (int qid = 0; qid < n; qid++)
    {
      eidType pos = begin + qid * stride;
      int size = stride;
      if ((total_num_chunks % n == 0) && (chunk_id == nchunks_per_queue - 1) && (qid == n - 1))
        size = n_tasks % stride;
      std::copy(total_task + pos, total_task + pos + size, task_ptrs[qid] + chunk_id * stride);
    }
  }
  eidType begin = nchunks_per_queue * n * stride;
  for (int i = 0; i < n; i++)
  {
    eidType pos = begin + i * stride;
    if (i + 1 == int(total_num_chunks % n))
    {
      std::copy(total_task + pos, total_task + n_tasks, task_ptrs[i] + nchunks_per_queue * stride);
    }
    else if (i + 1 < int(total_num_chunks % n))
    {
      std::copy(total_task + pos, total_task + pos + stride, task_ptrs[i] + nchunks_per_queue * stride);
    }
  }
  return lens;
}

void TCSolverMultiGPU(Graph &g, uint64_t &total, MPI_Comm &mpi_comm, int rank, int nranks, int deviceCount)
{
  //==============Set GPU config====================
  // int gpu_id = rank;
  // int n_gpus = nranks;
  // CUDA_SAFE_CALL(cudaSetDevice(gpu_id));

  if (deviceCount == 0)
  {
    cudaGetDeviceCount(&deviceCount);
  }
  cudaSetDevice(rank % deviceCount);

  //==============Preprocessing=====================
  // Step1: Divide graph into large degree and small degree.
  g.orientation_with_division(DEG_THD);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();

  int n_small_tasks = 0, n_large_tasks = 0;
  int *small_tasks = (int *)malloc(sizeof(int) * nv);
  int *large_tasks = (int *)malloc(sizeof(int) * nv);
#pragma omp parallel for
  for (int i = 0; i < nv; i++)
  {
    auto e_start = g.small_edge_begin(i);
    auto e_end = g.small_edge_end(i);
    auto e_deg = e_end - e_start;
    if (e_deg != 0)
    {
      small_tasks[n_small_tasks] = i;
      n_small_tasks++;
    }
  }
#pragma omp parallel for
  for (int i = 0; i < nv; i++)
  {
    auto e_start = g.large_edge_begin(i);
    auto e_end = g.large_edge_end(i);
    auto e_deg = e_end - e_start;

    if (e_deg != 0)
    {
      large_tasks[n_large_tasks] = i;
      n_large_tasks++;
    }
  }

  // Step2: task split with round-robin among GPUS.
  std::vector<vidType *> large_task_ptrs, small_task_ptrs;
  int chunk_size = 1024;
  auto large_tasks_per_gpu = round_robin(nranks, large_tasks, n_large_tasks, large_task_ptrs, chunk_size);
  auto small_tasks_per_gpu = round_robin(nranks, small_tasks, n_small_tasks, small_task_ptrs, chunk_size);

  int *d_small_tasks;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_small_tasks, sizeof(int) * small_tasks_per_gpu[rank]));
  CUDA_SAFE_CALL(cudaMemcpy(d_small_tasks, small_task_ptrs[rank], sizeof(int) * small_tasks_per_gpu[rank], cudaMemcpyHostToDevice));

  int *d_large_tasks;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_large_tasks, sizeof(int) * large_tasks_per_gpu[rank]));
  CUDA_SAFE_CALL(cudaMemcpy(d_large_tasks, large_task_ptrs[rank], sizeof(int) * large_tasks_per_gpu[rank], cudaMemcpyHostToDevice));

  Timer t;
  bool divide = true;
  GraphGPU gg(g, divide);

  // Step3: GPU thread setting.
  size_t nthreads = 256;
  size_t nblocks = 1024;
  size_t nwarps = nthreads * nblocks / WARP_SIZE;

  // Step4: Bucket preparing.
  vidType bucket_size = 64;
  eidType bucket_num = 1019;
  AccType h_total = 0, *d_total;

  vidType *hash_bucket;
  auto buckets_mem = nblocks * (bucket_num + 5) * bucket_size * sizeof(vidType);
  std::cout << "Bucket memory allocation: " << buckets_mem / 1024 / 1024 << " MB\n";
  CUDA_SAFE_CALL(cudaMalloc((void **)&hash_bucket, buckets_mem));
  CUDA_SAFE_CALL(cudaMemset(hash_bucket, 0, buckets_mem));

  // Step5: Result preparing.
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  int block_index = nblocks;
  int warp_index = nwarps;
  int *d_block_index, *d_warp_index;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_block_index, sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_warp_index, sizeof(int)));
  CUDA_SAFE_CALL(cudaMemcpy(d_block_index, &block_index, sizeof(int), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_warp_index, &warp_index, sizeof(int), cudaMemcpyHostToDevice));

  //==============Kernel Launching==================
  // Step6: Processing large degree subgraph.
  //       Using Hash Table.
  t.Start();
  CUDA_SAFE_CALL(cudaMemcpy(d_block_index, &block_index, sizeof(int), cudaMemcpyHostToDevice));

  hashIndex_for_large_degree<<<nblocks, nthreads, max((eidType)4096, bucket_num * 2) * sizeof(vidType), 0>>>(
      gg, d_total, d_block_index, d_large_tasks, large_tasks_per_gpu[rank], bucket_num, bucket_size, hash_bucket);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  // Step7: Processing small degree subgraph.
  //        Using Binary search.
  CUDA_SAFE_CALL(cudaMemcpy(d_warp_index, &warp_index, sizeof(int), cudaMemcpyHostToDevice));
  binarySearch_for_small_degree<<<nblocks, nthreads, 0, 0>>>(gg, d_total, d_warp_index, d_small_tasks, small_tasks_per_gpu[rank]);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();

  //==============Post processing====================
  //       Obtain Result, and clear buckets.
  float local_time = t.Seconds();
  float global_time = 0;
  std::cout << "Rank:" << rank << " runtime  = " << local_time << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));

  CUDA_SAFE_CALL(cudaFree(d_total));
  CUDA_SAFE_CALL(cudaFree(hash_bucket));

  AccType total_count = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&h_total, &total_count, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_time, &global_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

  total = total_count;
  if (rank == 0)
    std::cout << "Total runtime  = " << global_time << " sec\n";
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
