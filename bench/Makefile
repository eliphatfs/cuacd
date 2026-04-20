# Standalone benchmark build. Not part of the Python extension build.
#
# Usage:
#   make                  # build with sm_89 (RTX 4090)
#   make ARCH=86          # other arch
#   make clean

ARCH ?= 89
NVCC ?= nvcc
CXXSTD ?= c++17
OPT ?= -O3

CUDA_INC ?= ../cuda

NVCC_FLAGS := -std=$(CXXSTD) $(OPT) -arch=sm_$(ARCH) \
              -I$(CUDA_INC) \
              --expt-relaxed-constexpr \
              -Xcompiler -Wno-unused-variable

BIN := warp_sort_bench
SRC := warp_sort_bench.cu

all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run: $(BIN)
	./$(BIN) | tee results.csv

clean:
	rm -f $(BIN) results.csv

.PHONY: all run clean
