EXECUTABLE := cudaSpikeSort

CU_FILES   := interpolate_cluster.cu

CU_DEPS    :=

CC_FILES   := spike_sort.cpp

HPP_FILES  := spike_sort.h cycle_timer.h work_queue.h

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -g  -Wall
ifeq ($(ARCH), Darwin)
# Building on mac
LDFLAGS=-L/usr/local/depot/cuda-6.5/lib/ -lcudart
else
# Building on Linux
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart -lpthread
endif
NVCC=nvcc
NVCCFLAGS=-O3 -m64


OBJS=$(OBJDIR)/spike_sort.o  $(OBJDIR)/interpolate_cluster.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *.ppm *~ $(EXECUTABLE)

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp $(HPP_FILES)
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu $(HPP_FILES)
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
