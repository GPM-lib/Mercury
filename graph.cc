#include "graph.h"
#include "scan.h"

#define REVERSE_DAG

Graph::Graph(std::string prefix, bool use_dag, bool directed,
             bool use_vlabel, bool use_elabel) :
    is_directed_(directed), vlabels(NULL), elabels(NULL), nnz(0) {
  // parse file name
  size_t i = prefix.rfind('/', prefix.length());
  if (i != string::npos) inputfile_path = prefix.substr(0, i);
  i = inputfile_path.rfind('/', inputfile_path.length());
  if (i != string::npos) name = inputfile_path.substr(i+1);
  std::cout << "input file path: " << inputfile_path << ", graph name: " << name << "\n";

  // read meta information
  VertexSet::release_buffers();
  std::ifstream f_meta((prefix + ".meta.txt").c_str());
  assert(f_meta);
  int vid_size = 0, eid_size = 0, vlabel_size = 0, elabel_size = 0;
  //f_meta >> n_vertices >> n_edges >> vid_size >> max_degree;
  f_meta >> n_vertices >> n_edges >> vid_size >> eid_size >> vlabel_size >> elabel_size
        >> max_degree >> feat_len >> num_vertex_classes >> num_edge_classes;
  assert(sizeof(vidType) == vid_size);
  assert(sizeof(eidType) == eid_size);
  assert(sizeof(vlabel_t) == vlabel_size);
  assert(sizeof(elabel_t) == elabel_size);
  assert(max_degree > 0 && max_degree < n_vertices);
  f_meta.close();
  // read row pointers
  if (map_vertices) map_file(prefix + ".vertex.bin", vertices, n_vertices+1);
  else read_file(prefix + ".vertex.bin", vertices, n_vertices+1);
  // read column indices
  if (map_edges) map_file(prefix + ".edge.bin", edges, n_edges);
  else read_file(prefix + ".edge.bin", edges, n_edges);
  // read vertex labels
  if (use_vlabel) {
    assert (num_vertex_classes > 0);
    assert (num_vertex_classes < 255); // we use 8-bit vertex label dtype
    std::string vlabel_filename = prefix + ".vlabel.bin";
    ifstream f_vlabel(vlabel_filename.c_str());
    if (f_vlabel.good()) {
      if (map_vlabels) map_file(vlabel_filename, vlabels, n_vertices);
      else read_file(vlabel_filename, vlabels, n_vertices);
      std::set<vlabel_t> labels;
      for (vidType v = 0; v < n_vertices; v++)
        labels.insert(vlabels[v]);
      std::cout << "# distinct vertex labels: " << labels.size() << "\n";
      assert(size_t(num_vertex_classes) == labels.size());
    } else {
      std::cout << "WARNING: vertex label file not exist; generating random labels\n";
      vlabels = new vlabel_t[n_vertices];
      for (vidType v = 0; v < n_vertices; v++) {
        vlabels[v] = rand() % num_vertex_classes + 1;
      }
    }
    auto max_vlabel = unsigned(*(std::max_element(vlabels, vlabels+n_vertices)));
    std::cout << "maximum vertex label: " << max_vlabel << "\n";
  }
  if (use_elabel) {
    assert (num_edge_classes > 0);
    assert (num_edge_classes < 65535); // we use 16-bit edge label dtype
    std::string elabel_filename = prefix + ".elabel.bin";
    ifstream f_elabel(elabel_filename.c_str());
    if (f_elabel.good()) {
      if (map_elabels) map_file(elabel_filename, elabels, n_edges);
      else read_file(elabel_filename, elabels, n_edges);
      std::set<elabel_t> labels;
      for (eidType e = 0; e < n_edges; e++)
        labels.insert(elabels[e]);
      std::cout << "# distinct edge labels: " << labels.size() << "\n";
      assert(size_t(num_edge_classes) >= labels.size());
    } else {
      std::cout << "WARNING: edge label file not exist; generating random labels\n";
      elabels = new elabel_t[n_edges];
      for (eidType e = 0; e < n_edges; e++) {
        elabels[e] = rand() % num_edge_classes + 1;
      }
    }
    auto max_elabel = unsigned(*(std::max_element(elabels, elabels+n_edges)));
    std::cout << "maximum edge label: " << max_elabel << "\n";
  }
  // orientation: convert the undirected graph into directed. Only for k-cliques. This may change max_degree.
  if (use_dag) {
    assert(!directed); // must be undirected before orientation
    orientation();
  }
  // compute maximum degree
  VertexSet::MAX_DEGREE = std::max(max_degree, VertexSet::MAX_DEGREE);

  labels_frequency_.clear();
}

Graph::~Graph() {
  if (map_edges) munmap(edges, n_edges*sizeof(vidType));
  else custom_free(edges, n_edges);
  if (map_vertices) {
    munmap(vertices, (n_vertices+1)*sizeof(eidType));
  } else custom_free(vertices, n_vertices+1);
  if (vlabels != NULL) delete [] vlabels;
}

VertexSet Graph::N(vidType vid) const {
  assert(vid >= 0);
  assert(vid < n_vertices);
  eidType begin = vertices[vid], end = vertices[vid+1];
  if (begin > end) {
    fprintf(stderr, "vertex %u DEG_THDs error: [%lu, %lu)\n", vid, begin, end);
    exit(1);
  }
  assert(end <= n_edges);
  return VertexSet(edges + begin, end - begin, vid);
}

void Graph::allocateFrom(vidType nv, eidType ne) {
  n_vertices = nv;
  n_edges    = ne;
  vertices = new eidType[nv+1];
  edges = new vidType[ne];
  vertices[0] = 0;
}

vidType Graph::compute_max_degree() {
  std::cout << "computing the maximum degree\n";
  Timer t;
  t.Start();
  std::vector<vidType> degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = vertices[v+1] - vertices[v];
  }
  vidType max_degree = *(std::max_element(degrees.begin(), degrees.end()));
  t.Start();
  return max_degree;
}

void Graph::orientation() {
  std::cout << "Orientation enabled, using DAG\n";
  Timer t;
  t.Start();
  std::vector<vidType> degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = get_degree(v);
    //if(v==0) printf("@@@@@@@@@@@@@@ vid:%d deg:%d\n",v, get_degree(v));
    //if(degrees[v]>50000) printf("@@@@@@@@@@@@@@ vid:%d deg:%d\n",v, get_degree(v));
  }
  std::vector<vidType> new_degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_degrees[src]++;
      }
    }
  }



  // for (vidType v = 0; v < n_vertices; v++) {
  //   degrees[v] = get_degree(v);
  //   //if(v==0) printf("@@@@@@@@@@@@@@ vid:%d deg:%d\n",v, get_degree(v));
  //   if(degrees[v]>50000) printf("@@@@@@@@@@@@@@ vid:%d deg:%d  new:%d\n",v, get_degree(v), new_degrees[v]);
  // }

  max_degree = *(std::max_element(new_degrees.begin(), new_degrees.end()));
  eidType *old_vertices = vertices;
  vidType *old_edges = edges;
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  //prefix_sum<vidType,eidType>(new_degrees, new_vertices);
  parallel_prefix_sum<vidType,eidType>(new_degrees, new_vertices);
  auto num_edges = new_vertices[n_vertices];
  vidType *new_edges = custom_alloc_global<vidType>(num_edges);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    auto begin = new_vertices[src];
    eidType offset = 0;
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_edges[begin+offset] = dst;
        offset ++;
      }
    }
  }
  vertices = new_vertices;
  edges = new_edges;
  custom_free<eidType>(old_vertices, n_vertices);
  custom_free<vidType>(old_edges, n_edges);
  n_edges = num_edges;
  t.Stop();
  std::cout << "Time on generating the DAG: " << t.Seconds() << " sec\n";
}

void Graph::orientation_with_division(int DEG_THD) {
  std::cout << "Orientation enabled, using DAG\n";
  Timer t;
  t.Start();
  std::vector<vidType> degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType v = 0; v < n_vertices; v++) {
    degrees[v] = get_degree(v);
  }
  std::vector<vidType> new_degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_degrees[src]++;
      }
    }
  }

  max_degree = *(std::max_element(new_degrees.begin(), new_degrees.end()));
  eidType *old_vertices = vertices;
  vidType *old_edges = edges;
  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  parallel_prefix_sum<vidType,eidType>(new_degrees, new_vertices);
  auto num_edges = new_vertices[n_vertices];
  vidType *new_edges = custom_alloc_global<vidType>(num_edges);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    //if(degrees[src]==max_degree) printf("!!!!!!!!!!!!!!!vid:%d\n",src);
    auto begin = new_vertices[src];
    eidType offset = 0;
    for (auto dst : N(src)) {
      if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
        new_edges[begin+offset] = dst;
        offset ++;
      }
    }
  }

  //Start division graph....
  std::vector<vidType> small_degrees(n_vertices, 0);
  std::vector<vidType> large_degrees(n_vertices, 0);
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    auto new_src_deg = new_degrees[src];
    for (auto dst : N(src)) {
      auto new_dst_deg = new_degrees[dst];
      //auto dst_deg = degrees[dst];
      if(std::min(new_src_deg, new_dst_deg) < 1) continue;
      if(std::max(new_src_deg, new_dst_deg) <= DEG_THD){
        if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
          small_degrees[src]++;
        }
      }else if(new_src_deg>=DEG_THD || new_dst_deg>=DEG_THD){
        #ifdef REVERSE_DAG
        if (new_degrees[src] > new_degrees[dst] ||
          (new_degrees[dst] == new_degrees[src] && src > dst)) {
          large_degrees[src]++;
        }
        #else
         if (new_degrees[dst] > new_degrees[src] ||
          (new_degrees[dst] == new_degrees[src] && dst > src)) {
          large_degrees[src]++;
        }
        #endif 
        
      }
    }
  }
  small_vertices = custom_alloc_global<eidType>(n_vertices+1);
  large_vertices = custom_alloc_global<eidType>(n_vertices+1);
  parallel_prefix_sum<vidType,eidType>(small_degrees, small_vertices);
  parallel_prefix_sum<vidType,eidType>(large_degrees, large_vertices);
  auto num_small_edges = small_vertices[n_vertices];
  auto num_large_edges = large_vertices[n_vertices];
  small_edges = custom_alloc_global<vidType>(num_small_edges);
  large_edges = custom_alloc_global<vidType>(num_large_edges);
  n_small_edges = num_small_edges;
  n_large_edges = num_large_edges;
  #pragma omp parallel for
  for (vidType src = 0; src < n_vertices; src ++) {
    auto new_src_deg = new_degrees[src];
    auto small_begin = small_vertices[src];
    auto large_begin = large_vertices[src];
    eidType small_offset = 0;
    eidType large_offset = 0;
    for (auto dst : N(src)) {
      auto new_dst_deg = new_degrees[dst];
      //auto dst_deg = degrees[dst];
      if(std::min(new_src_deg, new_dst_deg) < 1) continue;
      if(std::max(new_src_deg, new_dst_deg) <= DEG_THD){
        if (degrees[dst] > degrees[src] ||
          (degrees[dst] == degrees[src] && dst > src)) {
          small_edges[small_begin + small_offset] = dst;
          small_offset ++;
        }
      }else if(new_src_deg>=DEG_THD || new_dst_deg>=DEG_THD){
        #ifdef REVERSE_DAG
        if (new_degrees[src] > new_degrees[dst] ||
          (new_degrees[dst] == new_degrees[src] && src > dst)) {
          large_edges[large_begin + large_offset] = dst;
          large_offset ++;
        }
        #else
         if (new_degrees[dst] > new_degrees[src] ||
          (new_degrees[dst] == new_degrees[src] && dst > src)) {
           large_edges[large_begin + large_offset] = dst;
          large_offset ++;
        }
        #endif
        
      }
    }
  }

  vertices = new_vertices;
  edges = new_edges;
  custom_free<eidType>(old_vertices, n_vertices);
  custom_free<vidType>(old_edges, n_edges);
  n_edges = num_edges;
  t.Stop();
  std::cout << "Time on generating the DAG: " << t.Seconds() << " sec\n";
}



void Graph::print_graph() const {
  std::cout << "Printing the graph: \n";
  for (vidType n = 0; n < n_vertices; n++) {
    std::cout << "vertex " << n << ": degree = "
      << get_degree(n) << " edgelist = [ ";
    for (auto e = edge_begin(n); e != edge_end(n); e++)
      std::cout << getEdgeDst(e) << " ";
    std::cout << "]\n";
  }
}

eidType Graph::init_edgelist(bool sym_break, bool ascend) {
  Timer t;
  t.Start();
  if (nnz != 0) return nnz; // already initialized
  nnz = E();
  if (sym_break) nnz = nnz/2;
  sizes.resize(V());
  src_list = new vidType[nnz];
  if (sym_break) dst_list = new vidType[nnz];
  else dst_list = edges;
  size_t i = 0;
  for (vidType v = 0; v < V(); v ++) {
    for (auto u : N(v)) {
      if (u == v) continue; // no selfloops
      if (ascend) {
        if (sym_break && v > u) continue;  
      } else {
        if (sym_break && v < u) break;  
      }
      src_list[i] = v;
      if (sym_break) dst_list[i] = u;
      sizes[v] ++;
      i ++;
    }
  }
  //assert(i == nnz);
  t.Stop();
  std::cout << "Time on generating the edgelist: " << t.Seconds() << " sec\n";
  return nnz;
}

bool Graph::is_connected(vidType v, vidType u) const {
  auto v_deg = get_degree(v);
  auto u_deg = get_degree(u);
  bool found;
  if (v_deg < u_deg) {
    found = binary_search(u, edge_begin(v), edge_end(v));
  } else {
    found = binary_search(v, edge_begin(u), edge_end(u));
  }
  return found;
}

bool Graph::is_connected(std::vector<vidType> sg) const {
  return false;
}

bool Graph::binary_search(vidType key, eidType begin, eidType end) const {
  auto l = begin;
  auto r = end-1;
  while (r >= l) { 
    auto mid = l + (r - l) / 2;
    auto value = getEdgeDst(mid);
    if (value == key) return true;
    if (value < key) l = mid + 1; 
    else r = mid - 1; 
  } 
  return false;
}

vidType Graph::intersect_num(vidType v, vidType u, vlabel_t label) {
  vidType num = 0;
  vidType idx_l = 0, idx_r = 0;
  vidType v_size = get_degree(v);
  vidType u_size = get_degree(u);
  vidType* v_ptr = &edges[vertices[v]];
  vidType* u_ptr = &edges[vertices[u]];
  while (idx_l < v_size && idx_r < u_size) {
    vidType a = v_ptr[idx_l];
    vidType b = u_ptr[idx_r];
    if (a <= b) idx_l++;
    if (b <= a) idx_r++;
    if (a == b && vlabels[a] == label) num++;
  }
  return num;
}

void Graph::BuildReverseIndex() {
  if (labels_frequency_.empty()) computeLabelsFrequency();
  int nl = num_vertex_classes;
  if (max_label == num_vertex_classes) nl += 1;
  reverse_index_.resize(size());
  reverse_index_offsets_.resize(nl+1);
  reverse_index_offsets_[0] = 0;
  vidType total = 0;
  for (int i = 0; i < nl; ++i) {
    total += labels_frequency_[i];
    reverse_index_offsets_[i+1] = total;
    //std::cout << "label " << i << " frequency: " << labels_frequency_[i] << "\n";
  }
  std::vector<eidType> start(nl);
  for (int i = 0; i < nl; ++i) {
    start[i] = reverse_index_offsets_[i];
    //std::cout << "label " << i << " start: " << start[i] << "\n";
  }
  for (vidType i = 0; i < size(); ++i) {
    auto vl = vlabels[i];
    reverse_index_[start[vl]++] = i;
  }
}

#pragma omp declare reduction(vec_plus : std::vector<int> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<int>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
void Graph::computeLabelsFrequency() {
  printf("=========================step1:%d   %d = = == \n",num_vertex_classes, max_label);
  labels_frequency_.resize(num_vertex_classes+1);
  std::fill(labels_frequency_.begin(), labels_frequency_.end(), 0);
  //max_label = int(*std::max_element(vlabels, vlabels+size()));
  #pragma omp parallel for reduction(max:max_label)
  for (int i = 0; i < size(); ++i) {
    max_label = max_label > vlabels[i] ? max_label : vlabels[i];
  }

  printf("=========================step2\n");
  #pragma omp parallel for reduction(vec_plus:labels_frequency_)
  for (vidType v = 0; v < size(); ++v) {
    int label = int(get_vlabel(v));
    assert(label <= num_vertex_classes);
    labels_frequency_[label] += 1;
  }
  max_label_frequency_ = int(*std::max_element(labels_frequency_.begin(), labels_frequency_.end()));

  printf("=========================step3\n");
  //std::cout << "max_label = " << max_label << "\n";
  //std::cout << "max_label_frequency_ = " << max_label_frequency_ << "\n";
  //for (size_t i = 0; i < labels_frequency_.size(); ++i)
  //  std::cout << "label " << i << " vertex frequency: " << labels_frequency_[i] << "\n";
}

int Graph::get_frequent_labels(int threshold) {
  int num = 0;
  for (size_t i = 0; i < labels_frequency_.size(); ++i)
    if (labels_frequency_[i] > threshold)
      num++;
  return num;
}

bool Graph::is_freq_vertex(vidType v, int threshold) {
  assert(v >= 0 && v < size());
  auto label = int(vlabels[v]);
  assert(label <= num_vertex_classes);
  if (labels_frequency_[label] >= threshold) return true;
  return false;
}

// NLF: neighborhood label frequency
void Graph::BuildNLF() {
  //std::cout << "Building NLF map for the data graph\n";
  nlf_.resize(size());
  #pragma omp parallel for
  for (vidType v = 0; v < size(); ++v) {
    for (auto u : N(v)) {
      auto vl = get_vlabel(u);
      if (nlf_[v].find(vl) == nlf_[v].end())
        nlf_[v][vl] = 0;
      nlf_[v][vl] += 1;
    }
  }
}

void Graph::print_meta_data() const {
  std::cout << "|V|: " << n_vertices << ", |E|: " << n_edges << ", Max Degree: " << max_degree << "\n";
  if (num_vertex_classes > 0) {
    std::cout << "vertex-|\u03A3|: " << num_vertex_classes;
    if (!labels_frequency_.empty()) 
      std::cout << ", Max Label Frequency: " << max_label_frequency_;
    std::cout << "\n";
  } else {
    std::cout  << "This graph does not have vertex labels\n";
  }
  if (num_edge_classes > 0) {
    std::cout << "edge-|\u03A3|: " << num_edge_classes << "\n";
  } else {
    std::cout  << "This graph does not have edge labels\n";
  }
  if (feat_len > 0) {
    std::cout << "Vertex feature vector length: " << feat_len << "\n";
  } else {
    std::cout  << "This graph has no input vertex features\n";
  }
}

void Graph::sort_graph()  {
  std::vector<int> index(n_vertices);
  std::vector<int> r_index(n_vertices);
  for(int i=0; i<n_vertices; i++) index[i] = i;
  

  // for(int i=0; i<10; i++){
  //   printf("index[i]:%d  deg:%d\n",index[i],get_degree(index[i]));
  // }

  std::sort(index.begin(), index.end(), [&](int a, int b){
    return get_degree(a) > get_degree(b);
  });

  // printf("After\n");
  // for(int i=0; i<10; i++){
  //   printf("index[i]:%d  deg:%d\n",index[i],get_degree(index[i]));
  // }

  eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
  vidType *new_edges = custom_alloc_global<vidType>(n_edges);
  std::vector<vidType> new_degrees(n_vertices, 0);

  for (vidType src = 0; src < n_vertices; src ++) {
    vidType v = index[src];
    // if(src==133 || src==1835 || src==2114){
    //   printf("vid:%d  index:%d\n",src,v);
    // }
    r_index[v] = src;
  }

  for (vidType src = 0; src < n_vertices; src ++) {
    vidType v = index[src];
    new_degrees[src] = get_degree(v);
  }
  parallel_prefix_sum<vidType,eidType>(new_degrees, new_vertices);


  for (vidType src = 0; src < n_vertices; src ++) {
    auto begin = new_vertices[src];
    eidType offset = 0;
    vidType v = index[src];
    for (auto dst : N(v)) {
        new_edges[begin+offset] = r_index[dst];
        offset ++;
    }
    std::sort(&new_edges[begin], &new_edges[begin + offset]);
  }

  eidType *old_vertices = vertices;
  vidType *old_edges = edges;

  vertices = new_vertices;
  edges = new_edges;
  custom_free<eidType>(old_vertices, n_vertices);
  custom_free<vidType>(old_edges, n_edges);
}

// void Graph::sort_graph_with_division()  {
//   std::vector<int> index(n_vertices);
//   std::vector<int> r_index(n_vertices);
//   for(int i=0; i<index.size(); i++) index[i] = i;
//   std::stable_sort(index.begin(), index.end(), [&](int a, int b){
//     return get_degree(a) > get_degree(b);
//   });

//   eidType *new_vertices = custom_alloc_global<eidType>(n_vertices+1);
//   vidType *new_edges = custom_alloc_global<vidType>(n_edges);
//   std::vector<vidType> new_degrees(n_vertices, 0);


//   vidType *new_small_edges = custom_alloc_global<vidType>(n_small_edges);
//   vidType *new_large_edges = custom_alloc_global<vidType>(n_large_edges);
//   std::vector<vidType> new_small_degrees(n_vertices, 0);
//   std::vector<vidType> new_large_degrees(n_vertices, 0);

//   for (vidType src = 0; src < n_vertices; src ++) {
//     vidType v = index[src];
//     r_index[v] = src;
//   }

//   for (vidType src = 0; src < n_vertices; src ++) {
//     vidType v = index[src];
//     new_degrees[src] = get_degree(v);
//     new_small_degrees[src] = get_small_degree(v);
//     new_large_degrees[src] = get_large_degree(v);
//   }
//   parallel_prefix_sum<vidType,eidType>(new_degrees, new_vertices);


//   for (vidType src = 0; src < n_vertices; src ++) {
//     auto begin = new_vertices[src];
//     eidType offset = 0;
//     vidType v = index[src];
//     for (auto dst : N(v)) {
//         new_edges[begin+offset] = r_index[dst];
//         offset ++;
//     }
//     std::sort(&new_edges[begin], &new_edges[begin + offset]);
//   }

//   eidType *new_small_vertices = custom_alloc_global<eidType>(n_vertices+1);
//   eidType *new_large_vertices = custom_alloc_global<eidType>(n_vertices+1);

//   parallel_prefix_sum<vidType,eidType>(new_small_degrees, new_small_vertices);
//   parallel_prefix_sum<vidType,eidType>(new_large_degrees, new_large_vertices);


//   for (vidType src = 0; src < n_vertices; src ++) {
//     auto begin = new_small_vertices[src];
//     eidType offset = 0;
//     vidType v = index[src];
//     // for (auto dst : N(v)) {
//     auto start = small_edge_begin(v);
//     auto end = small_edge_end(v);
//     for(auto eid = start; eid<end; eid++){
//         auto dst = get_small_EdgeDst(eid);

//         new_small_edges[begin+offset] = r_index[dst];
//         offset ++;
//     }
//     std::sort(&new_small_edges[begin], &new_small_edges[begin + offset]);
//   }

//   for (vidType src = 0; src < n_vertices; src ++) {
//     auto begin = new_large_vertices[src];
//     eidType offset = 0;
//     vidType v = index[src];
//     // for (auto dst : N(v)) {
//     auto start = large_edge_begin(v);
//     auto end = large_edge_end(v);
//     for(auto eid = start; eid<end; eid++){
//         auto dst = get_large_EdgeDst(eid);
//         new_large_edges[begin+offset] = r_index[dst];
//         offset ++;
//     }
//     std::sort(&new_large_edges[begin], &new_large_edges[begin + offset]);
//   }


//   eidType *old_vertices = vertices;
//   vidType *old_edges = edges;

//   eidType *old_small_vertices = small_vertices;
//   vidType *old_small_edges = small_edges;

//   eidType *old_large_vertices = large_vertices;
//   vidType *old_large_edges = large_edges;

//   vertices = new_vertices;
//   edges = new_edges;
//   custom_free<eidType>(old_vertices, n_vertices);
//   custom_free<vidType>(old_edges, n_edges);

//   small_vertices = new_small_vertices;
//   small_edges = new_small_edges;
//   custom_free<eidType>(old_small_vertices, n_vertices);
//   custom_free<vidType>(old_small_edges, n_small_edges);

//   large_vertices = new_large_vertices;
//   large_edges = new_large_edges;
//   custom_free<eidType>(old_large_vertices, n_vertices);
//   custom_free<vidType>(old_large_edges, n_large_edges);
  
// }

void Graph::dump_graph(std::string const& coo_path, std::string const& csr_path){
    std::ofstream coo_fout(coo_path);
    std::ofstream csr_fout(csr_path);

    // coo_fout << n_vertices <<"  "<< n_vertices<<"  "<< n_edges<<"\n";
    // int idx = 0;
    // while(idx<n_edges){
    //   coo_fout << src_list[idx] <<" "<< dst_list[idx]<<"\n";
    //   idx++;
    // }


    // idx = 0;
    // while(idx<n_vertices){
    //   csr_fout << get_degree(idx)<<"\n";
    //   idx++;
    // }
    // idx = 0;
    // while(idx<n_vertices+1){
    //   csr_fout << vertices[idx]<<"\n";
    //   idx++;
    // }
    // idx = 0;
    // while(idx<n_edges){
    //   csr_fout << edges[idx]<<"\n";
    //   idx++;
    // }

    for(int idx=0; idx<n_vertices; idx++){
      csr_fout << idx;
      int deg = get_degree(idx);
      if(deg==0)  {csr_fout << "\n"; continue;}
      //printf("\n v:%d adj \n",idx);
      for(int i=0; i<deg; i++)
        csr_fout <<" " <<N(idx, i);
      csr_fout << "\n";
      //if(idx>10) break;
      // printf("\n");
    }

    coo_fout.close();
    csr_fout.close();
}