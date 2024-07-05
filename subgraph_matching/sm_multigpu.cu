#include <cub/cub.cuh>
#include <mpi.h>
#include "timer.h"
#include "edgelist.h"
#include "graph_gpu.h"
#include "operations.cuh"
#include "cuda_launch_config.hpp"

typedef cub::BlockReduce<AccType, BLOCK_SIZE> BlockReduce;
#include "pattern_kernels.cuh"

std::vector<eidType> coo_round_robin(int n, Graph &hg,
                                     std::vector<int *> &src_ptrs,
                                     std::vector<int *> &dst_ptrs, int stride)
{
  // auto nnz = hg.getNNZ();
  auto nnz = hg.num_edges();
  assert(nnz > 8192); // if edgelist is too small, no need to split
  std::cout << "split edgelist with chunk size of " << stride
            << " using chunked round robin\n";
  // resize workload list to n
  src_ptrs.resize(n);
  dst_ptrs.resize(n);

  int total_num_chunks = (nnz - 1) / stride + 1;
  int nchunks_per_queue = total_num_chunks / n;
  // every GPU queue basic size(except the border): every chunksize(edges) *
  // n_chunks
  std::vector<eidType> lens(n, stride * nchunks_per_queue);
  if (total_num_chunks % n != 0)
  {
    for (int i = 0; i < n; i++)
    {
      if (i + 1 == int(total_num_chunks % n))
      {
        lens[i] += nnz % stride == 0 ? stride : nnz % stride;
      }
      else if (i + 1 < int(total_num_chunks % n))
      {
        lens[i] += stride;
      }
    }
  }
  else
  {
    lens[n - 1] = lens[n - 1] + nnz % stride - stride;
  }

  for (int i = 0; i < n; i++)
  {
    src_ptrs[i] = new int[lens[i]];
    dst_ptrs[i] = new int[lens[i]];
  }

  auto src_list = hg.get_src_ptr();
  auto dst_list = hg.get_dst_ptr();
#pragma omp parallel for
  for (int chunk_id = 0; chunk_id < nchunks_per_queue; chunk_id++)
  {
    int begin = chunk_id * n * stride;
    for (int qid = 0; qid < n; qid++)
    {
      int pos = begin + qid * stride;
      int size = stride;
      if ((total_num_chunks % n == 0) &&
          (chunk_id == nchunks_per_queue - 1) && (qid == n - 1))
        size = nnz % stride;
      std::copy(src_list + pos, src_list + pos + size,
                src_ptrs[qid] + chunk_id * stride);
      std::copy(dst_list + pos, dst_list + pos + size,
                dst_ptrs[qid] + chunk_id * stride);
    }
  }

  int begin = nchunks_per_queue * n * stride;
  for (int i = 0; i < n; i++)
  {
    int pos = begin + i * stride;
    if (i + 1 == int(total_num_chunks % n))
    {
      std::copy(src_list + pos, src_list + nnz,
                src_ptrs[i] + nchunks_per_queue * stride);
      std::copy(dst_list + pos, dst_list + nnz,
                dst_ptrs[i] + nchunks_per_queue * stride);
    }
    else if (i + 1 < int(total_num_chunks % n))
    {
      std::copy(src_list + pos, src_list + pos + stride,
                src_ptrs[i] + nchunks_per_queue * stride);
      std::copy(dst_list + pos, dst_list + pos + stride,
                dst_ptrs[i] + nchunks_per_queue * stride);
    }
  }
  return lens;
}

void SMSolverMultiGPU(Graph &g, uint64_t &total, MPI_Comm &mpi_comm, int rank, int nranks, std::string pattern, int deviceCount)
{
  //==============Set GPU config====================
  if (deviceCount == 0)
  {
    cudaGetDeviceCount(&deviceCount);
  }
  cudaSetDevice(rank % deviceCount);

  //==============Preprocessing=====================
  auto nv = g.num_vertices();
  auto ne = g.num_edges();
  auto md = g.get_max_degree();
  int nblocks = 1024;
  int nthreads = 256;
  int list_num = 8;

  g.print_meta_data();

  // Immediated data buffer building.
  size_t per_block_vlist_size = WARPS_PER_BLOCK * list_num * size_t(md) * sizeof(int);
  AccType nowindex = (nblocks * nthreads / WARP_SIZE);
  size_t flist_size = nblocks * per_block_vlist_size;
  int *d_frontier_list;
  int *freq_list;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_frontier_list, flist_size));

  // Thread index.
  AccType *G_INDEX, *G_INDEX1;
  CUDA_SAFE_CALL(cudaMalloc((void **)&G_INDEX, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&G_INDEX1, sizeof(AccType)));
  CUDA_SAFE_CALL(cudaMemcpy(G_INDEX, &nowindex, sizeof(AccType), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(G_INDEX1, &nowindex, sizeof(AccType), cudaMemcpyHostToDevice));

  // Result data.
  int count_length = 6;
  std::vector<AccType> h_total(count_length, 0);
  AccType *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType) * count_length));
  CUDA_SAFE_CALL(cudaMemcpy(d_total, &h_total[0], sizeof(AccType) * count_length, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaDeviceSynchronize());

  //===========Task disision=========================
  g.init_edgelist();
  std::vector<int *> src_ptrs, dst_ptrs;
  int chunk_size = 1024;
  auto tasks = coo_round_robin(nranks, g, src_ptrs, dst_ptrs, chunk_size);

  GraphGPU gg(g);
  gg.copy_edgelist_to_device(tasks, src_ptrs, dst_ptrs, rank);

  Timer t;
  t.Start();
  if (pattern == "P1")
  {

    P1_frequency_count<<<nblocks, nthreads>>>(
        nv, gg, d_frontier_list, md, d_total, G_INDEX);

    P1_count_correction<<<nblocks, nthreads>>>(
        tasks[rank], gg, d_frontier_list, md, d_total, G_INDEX1);
  }
  else if (pattern == "P2")
  {
    P2_subgraph_matching<<<nblocks, nthreads>>>(
        tasks[rank], gg, d_frontier_list, md, d_total, G_INDEX);
  }
  else if (pattern == "P3")
  {
    P3_subgraph_matching<<<nblocks, nthreads>>>(
        tasks[rank], gg, d_frontier_list, md, d_total, G_INDEX);
  }
  else if (pattern == "P4")
  {
    P4_subgraph_matching<<<nblocks, nthreads>>>(
        tasks[rank], gg, d_frontier_list, md, d_total, G_INDEX);
  }
  else if (pattern == "P5")
  {
    P5_subgraph_matching<<<nblocks, nthreads>>>(
        tasks[rank], gg, d_frontier_list, md, d_total, G_INDEX);
  }
  else if (pattern == "P6")
  {
    P6_subgraph_matching<<<nblocks, nthreads>>>(
        tasks[rank], gg, d_frontier_list, md, d_total, G_INDEX);
  }
  else if (pattern == "P7")
  {
    P7_subgraph_matching<<<nblocks, nthreads>>>(
        tasks[rank], gg, d_frontier_list, md, d_total, G_INDEX);
  }
  else if (pattern == "P8")
  {
    P8_subgraph_matching<<<nblocks, nthreads>>>(
        tasks[rank], gg, d_frontier_list, md, d_total, G_INDEX);
  }
  else
  {
    std::cout << "Not support." << std::endl;
    exit(-1);
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  t.Stop();
  float local_time = t.Seconds();
  float global_time = 0;
  std::cout << "Rank:" << rank << " runtime  = " << local_time << " sec\n";
  CUDA_SAFE_CALL(cudaMemcpy(&h_total[0], d_total, sizeof(AccType) * count_length, cudaMemcpyDeviceToHost));

  AccType result0 = 0;
  AccType result1 = 0;

  if (pattern == "P4" || pattern == "P6")
  {
    result0 = h_total[1];
  }
  else if (pattern == "P1")
  {
    result0 = h_total[2];
    result1 = h_total[1];
  }
  else
  {
    result0 = h_total[1] - h_total[2];
  }

  //==============Post processing====================
  //       Obtain Result, and clear buckets.

  AccType total_count = 0;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Reduce(&result0, &total_count, 1, MPI_INT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_time, &global_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

  if (pattern == "P1")
    total_count = result1 - total_count;

  total = total_count;
  if (rank == 0)
    std::cout << "Total runtime  = " << global_time << " sec\n";
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
}
