include common.mk

#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./gpu_kernels
INCLUDES+=-I./include
all: tc_challenge

MPI_INCLUDES+=-I$(MPI_HOME)/include

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c ${TESTS} $<

%.o: %.cpp
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) ${MPI_INCLUDES} -c $<

tc_challenge: $(OBJS) tc_challenge.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) tc_challenge.o -o $@ $(LIBS)

clean:
	rm *.o
