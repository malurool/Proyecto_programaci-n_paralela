CXX      := /usr/bin/mpicxx.mpich
NVCC     := /usr/local/cuda-13.1/bin/nvcc
CXXFLAGS := -std=c++17 -O3 -march=native -funroll-loops -fopenmp -Isrc
NVFLAGS  := -std=c++17 -O3 -Isrc -arch=sm_100 --expt-relaxed-constexpr
LDFLAGS  := -fopenmp
TARGET   := build/hadar_kmeans
TARGET_G := build/hadar_kmeans_gpu
SRC      := src/main.cpp
SRC_G    := src/main_gpu.cu

.PHONY: all cpu gpu clean
all: cpu
cpu: build $(TARGET)
gpu: build $(TARGET_G)
build:
	mkdir -p build results

$(TARGET): $(SRC) src/kmeans.hpp src/io.hpp src/timer.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) $(LDFLAGS)
	@echo "Build OK → $@"

$(TARGET_G): $(SRC_G) src/kmeans.hpp src/io.hpp src/timer.hpp
	$(NVCC) $(NVFLAGS) -o $@ $(SRC_G)
	@echo "Build OK → $@"

clean:
	rm -rf build
