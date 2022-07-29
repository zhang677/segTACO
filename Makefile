all: matrix-eval-gpu
NVCC = /home/eva_share/opt/cuda-11.6/bin/nvcc
CXX = icpc
OPTS = -O3 -DNDEBUG -march=native -mtune=native -ffast-math -fopenmp
#GPU_OPTS = -O3 --std=c++14 -gencode arch=compute_86,code=sm_86 --use_fast_math
CC_FLAGS = -Xcompiler -fPIC -shared --std=c++14
EIGEN_INC = -I /home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/eigen-3.4.0
SPLATT_INC = -I /home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/splatt/include
CUDA_INC = -I /home/eva_share/opt/cuda-11.6/lib64
SPLATT_LIN = -L /home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/splatt/build/Linux-x86_64/lib
#export LIBRARY_PATH=/opt/cuda-11.1/lib64:${LIBRARY_PATH}


matrix-eval-gpu: matrix_main_gpu.o Makefile
	$(NVCC) -o matrix-eval-gpu ${CUDA_INC} -I . ${EIGEN_INC} -L /home/eva_share/opt/cuda-11.6/lib64 ${SPLATT_LIN} matrix_main_gpu.o gpu_kernels.cu -lpthread -lm -ldl -lcusparse -lcublas -llapack -lblas

matrix_main_gpu.o: matrix_main.cpp ds.h timers.h mmio.h Makefile $(shell find matrix_kernels -type f) $(shell find matrix_experiments -type f)

	$(NVCC) -o matrix_main_gpu.o ${CC_FLAGS} ${EIGEN_INC} ${CUDA_INC} -I .  matrix_main.cpp
#g++ -o matrix_main_gpu.o -c -std=c++14 ${OPTS} -DGPU -Wno-deprecated-declarations -I . ${EIGEN_INC} -lcuda -lcudart -I/opt/cuda-11.1/lib64 matrix_main.cpp

.PHONY: clean runm runt

clean:
	rm -f matrix-eval
	rm -f matrix-eval-gpu
	rm -f *.o

runm:
	./run_gpu.sh

runt:
	sudo ./run_gpu_t.sh