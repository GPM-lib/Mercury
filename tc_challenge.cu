#include <cub/cub.cuh>
#include "timer.h"
#include "edgelist.h"
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "tc_challenge.cuh"

void TCSolver(Graph &g, uint64_t &total)
{
  //==============Preprocessing=====================
  // Step1: Divide graph into large degree and small degree.
  g.orientation_with_division(DEG_THD);
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();

  int n_small_tasks = 0, n_large_tasks = 0;
  ;
  int *small_tasks = (int *)malloc(sizeof(int) * nv);
  int *large_tasks = (int *)malloc(sizeof(int) * nv);
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
  int *d_small_tasks;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_small_tasks, sizeof(int) * nv));
  CUDA_SAFE_CALL(cudaMemcpy(d_small_tasks, small_tasks, sizeof(int) * nv, cudaMemcpyHostToDevice));

  int *d_large_tasks;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_large_tasks, sizeof(int) * nv));
  CUDA_SAFE_CALL(cudaMemcpy(d_large_tasks, large_tasks, sizeof(int) * nv, cudaMemcpyHostToDevice));

  g.print_meta_data();
  // size_t memsize = print_device_info(0);

  // size_t mem_graph = size_t(nv + 1) * sizeof(eidType) + size_t(2) * size_t(ne) * sizeof(vidType);
  // std::cout << "GPU_total_mem = " << memsize << " graph_mem = " << mem_graph << "\n";

  Timer t;

  bool divide = true;
  GraphGPU gg(g, divide);

  // Step2: GPU thread setting.
  size_t nthreads = 256;
  size_t nblocks = 1024;
  size_t nwarps = nthreads * nblocks / WARP_SIZE;

  // Step3: Bucket preparing.
  vidType bucket_size = 64;
  eidType bucket_num = 1019;
  AccType h_total = 0, *d_total;

  vidType *hash_bucket;
  auto buckets_mem = nblocks * (bucket_num + 5) * bucket_size * sizeof(vidType);
  std::cout << "Bucket memory allocation: " << buckets_mem / 1024 / 1024 << " MB\n";
  CUDA_SAFE_CALL(cudaMalloc((void **)&hash_bucket, buckets_mem));
  CUDA_SAFE_CALL(cudaMemset(hash_bucket, 0, buckets_mem));

  // Step4: Result preparing.
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
  // Step1: Processing large degree subgraph.
  //       Using Hash Table.
  t.Start();
  CUDA_SAFE_CALL(cudaMemcpy(d_block_index, &block_index, sizeof(int), cudaMemcpyHostToDevice));

  hashIndex_for_large_degree<<<nblocks, nthreads, max((eidType)4096, bucket_num * 2) * sizeof(vidType), 0>>>(
      gg, d_total, d_block_index, d_large_tasks, n_large_tasks, bucket_num, bucket_size, hash_bucket);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  // Step2: Processing small degree subgraph.
  //        Using Binary search.
  CUDA_SAFE_CALL(cudaMemcpy(d_warp_index, &warp_index, sizeof(int), cudaMemcpyHostToDevice));
  binarySearch_for_small_degree<<<nblocks, nthreads, 0, 0>>>(gg, d_total, d_warp_index, d_small_tasks, n_small_tasks);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  //==============Postprocessing====================
  //       Obtain Result, and clear buckets.
  std::cout << "Runtime  = " << t.Seconds() << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  total = h_total;
  CUDA_SAFE_CALL(cudaFree(d_total));
  CUDA_SAFE_CALL(cudaFree(hash_bucket));
}
