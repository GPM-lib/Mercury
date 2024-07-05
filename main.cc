#include "graph.h"

void TCSolver(Graph &g, uint64_t &total);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <graph>\n";
    std::cout << "Example: " << argv[0] << " /graph_inputs/mico/graph\n";
    exit(1);
  }
  std::cout << "Triangle Counting\n";
  //if (USE_DAG) std::cout << "Using DAG (static orientation)\n";
  //Graph g(argv[1], USE_DAG); // use DAG
  Graph g(argv[1]);
  // g.print_meta_data();
  uint64_t total = 0;
  TCSolver(g, total);
  std::cout << "total_num_triangles = " << total << "\n";
  return 0;
}

