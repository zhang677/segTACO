all: matrix-eval-gpu
NVCC = /home/eva_share/opt/cuda-11.6/bin/nvcc
CXX = icpc
OPTS = -O3 -DNDEBUG -march=native -mtune=native -ffast-math -fopenmp
CC_FLAGS = -Xcompiler -fPIC -shared --std=c++14
EIGEN_INC = -I ./eigen-3.4.0
CUDA_INC = -I /home/eva_share/opt/cuda-11.6/lib64


matrix-eval-gpu: matrix_main_gpu.o Makefile
	$(NVCC) -o matrix-eval-gpu ${CUDA_INC} -I . ${EIGEN_INC} -L /home/eva_share/opt/cuda-11.6/lib64 matrix_main_gpu.o gpu_kernels.cu -lpthread -lm -ldl -lcusparse -lcublas -llapack -lblas

matrix_main_gpu.o: matrix_main.cpp Makefile 

	$(NVCC) -o matrix_main_gpu.o ${CC_FLAGS} ${EIGEN_INC} ${CUDA_INC} -I .  matrix_main.cpp

.PHONY: clean run

clean:
	rm -f matrix-eval-gpu
	rm -f *.o

run:
	./run_gpu.sh

# ds.h timers.h mmio.h 
# $(shell find matrix_kernels -type f) $(shell find matrix_experiments -type f)