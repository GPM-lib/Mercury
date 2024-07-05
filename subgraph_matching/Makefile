include ../common.mk

#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I../gpu_kernels
INCLUDES+=-I../include
all: sm_multigpu

MPI_INCLUDES+=-I$(MPI_HOME)/include


VertexSet.o: ../VertexSet.cc 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

graph.o: ../graph.cc 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

sm_multigpu.o : sm_multigpu.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) ${MPI_INCLUDES} -c ${TESTS} $<

main_mpi.o: main_mpi.cpp 
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $<

sm_multigpu:  main_mpi.o VertexSet.o graph.o  sm_multigpu.o
	$(NVCC) -L$(MPI_HOME)/lib  -lmpi $(NVFLAGS) $(INCLUDES) ${MPI_INCLUDES} VertexSet.o graph.o main_mpi.o sm_multigpu.o -o $@ $(LIBS)

clean:
	rm *.o
