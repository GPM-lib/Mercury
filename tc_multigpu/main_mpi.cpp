#include "graph.h"
#include <mpi.h>
void TCSolverMultiGPU(Graph &g, uint64_t &total, MPI_Comm &mpi_comm, int rank, int nranks, int deviceCount);

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    std::cout << "Usage:  mpirun -n np" << argv[0] << " <graph> <devices per node>\n";
    std::cout << "Example: mpirun -n 4" << argv[0] << " /graph_inputs/mico/graph 4\n";
    exit(1);
  }

  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int rank, nranks;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  int deviceCount = 0;
  if (argc == 3)
  {
    deviceCount = atoi(argv[2]);
  }

  std::cout << "Triangle Counting\n";
  // if (USE_DAG) std::cout << "Using DAG (static orientation)\n";
  // Graph g(argv[1], USE_DAG); // use DAG
  Graph g(argv[1]);

  // g.print_meta_data();
  uint64_t total = 0;
  TCSolverMultiGPU(g, total, mpi_comm, rank, nranks, deviceCount);
  if (rank == 0)
    std::cout << "total_num_triangles = " << total << "\n";
  return 0;
}
