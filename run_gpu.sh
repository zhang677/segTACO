#!/bin/bash
MATRICES_DIR=./matrix_subset
RESULTS_DIR=./matrix_results
export LD_LIBRARY_PATH="/home/eva_share/opt/cuda-11.6/lib64:$LD_LIBRARY_PATH"
#OMP_NUM_THREADS=4 ./taco-eval.py $MATRICES_DIR spmv_csr_gpu   $RESULTS_DIR
CUDA_VISIBLE_DEVICES=$1 OMP_NUM_THREADS=4 ./taco-eval.py $MATRICES_DIR spmm_csr_gpu $RESULTS_DIR $2
#OMP_NUM_THREADS=4 ./taco-eval.py $MATRICES_DIR spmspv_csr_gpu   $RESULTS_DIR
#OMP_NUM_THREADS=4 ./taco-eval.py $MATRICES_DIR sddmm_csr_gpu   $RESULTS_DIR 



