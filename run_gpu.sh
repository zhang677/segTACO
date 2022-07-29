#!/bin/bash

#MATRICES_DIR=/home/nfs_data/datasets/SuiteSparse
MATRICES_DIR=/home/nfs_data/datasets/sparse_mat
#MATRICES_DIR=/home/nfs_data/zhanggh/mytaco/learn-taco/matrices
#MATRICES_DIR=/home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/matrix_subset
#MATRICES_DIR=/home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/matrix_faster
RESULTS_DIR=/home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/matrix_results
export LD_LIBRARY_PATH="/home/eva_share/opt/cuda-11.6/lib64:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="/home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/splatt/build/Linux-x86_64/lib:$LD_LIBRARY_PATH"
#OMP_NUM_THREADS=4 ./taco-eval.py $MATRICES_DIR spmv_csr_gpu   $RESULTS_DIR
CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 ./taco-eval.py $MATRICES_DIR spmm_csr_gpu $RESULTS_DIR $2
#OMP_NUM_THREADS=4 ./taco-eval.py $MATRICES_DIR spmspv_csr_gpu   $RESULTS_DIR
#OMP_NUM_THREADS=4 ./taco-eval.py $MATRICES_DIR sddmm_csr_gpu   $RESULTS_DIR 



