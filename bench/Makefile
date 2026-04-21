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

# Build with ptxas verbose output (register / smem / spill per kernel).
# Parse with: python3 parse_ptxas.py ptxas.log
verbose: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -Xptxas=-v $< -o $(BIN) 2> ptxas.log
	@echo "ptxas report written to ptxas.log"

run: $(BIN)
	./$(BIN) | tee results.csv

clean:
	rm -f $(BIN) results.csv ptxas.log summary.md summary.txt

.PHONY: all verbose run clean
