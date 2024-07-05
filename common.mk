DEBUG ?= 0
MPI_HOME := $(shell dirname $(shell dirname $(shell which mpicxx)))
CUDA_HOME := $(shell dirname $(shell dirname $(shell which nvcc)))

CC=gcc
CXX=g++
ICC=$(ICC_HOME)/icc
ICPC=$(ICC_HOME)/icpc
MPICC=mpicc
MPICXX=mpicxx
NVCC=nvcc
CUDA_ARCH := -gencode arch=compute_80,code=sm_80
CXXFLAGS=-Wall -fopenmp -std=c++11 -march=native
ICPCFLAGS=-O3 -Wall -qopenmp
NVFLAGS=$(CUDA_ARCH)
NVFLAGS+=-DUSE_GPU

ifeq ($(VTUNE), 1)
	CXXFLAGS += -g
endif
ifeq ($(NVPROF), 1)
	NVFLAGS += -lineinfo
endif

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0
	NVFLAGS += -G
else
	CXXFLAGS += -O3
	NVFLAGS += -O3 -w
endif

ifeq ($(PAPI), 1)
CXXFLAGS += -DENABLE_PAPI
INCLUDES += -I$(PAPI_HOME)/include
LIBS += -L$(PAPI_HOME)/lib -lpapi
endif


LIBS += -lgomp

VPATH += ../common
OBJS=main.o VertexSet.o graph.o

# CUDA vertex parallel
ifneq ($(VPAR),)
NVFLAGS += -DVERTEX_PAR
endif

# CUDA CTA centric
ifneq ($(CTA),)
NVFLAGS += -DCTA_CENTRIC
endif

ifneq ($(PROFILE),)
CXXFLAGS += -DPROFILING
endif

ifneq ($(USE_SET_OPS),)
CXXFLAGS += -DUSE_MERGE
endif

ifneq ($(USE_SIMD),)
CXXFLAGS += -DSI=0
endif

# counting or listing
ifneq ($(COUNT),)
NVFLAGS += -DDO_COUNT
endif

# GPU vertex/edge parallel 
ifeq ($(VERTEX_PAR),)
NVFLAGS += -DEDGE_PAR
endif

# CUDA unified memory
ifneq ($(USE_UM),)
NVFLAGS += -DUSE_UM
endif

# kernel fission
ifneq ($(FISSION),)
NVFLAGS += -DFISSION
endif

