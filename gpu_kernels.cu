#include "gpu_kernels.cuh"
#include "gpu_library.h"
/*
#include "matrix_kernels/spmv_csr_gpu_taco.h"
#include "matrix_kernels/spmv_csr_gpu_cusparse.h"
#include "matrix_kernels/spmv_csr_gpu_merge.h"
*/
#include "matrix_kernels/spmm_csr_gpu_taco.h"
#include "matrix_kernels/spmm_csr_gpu_cusparse.h"
#include "matrix_kernels/spmm_eb_sr.h"
#include "matrix_kernels/spmm_eb_pr.h"
#include "matrix_kernels/spmm_rb_sr.h"
#include "matrix_kernels/spmm_rb_pr.h"
//#include "matrix_kernels/spmm_csr_reduction.h"
//#include "matrix_kernels/sddmm_csr_gpu_taco.h"
//#include "matrix_kernels/spmspv_csr_gpu_taco.h"

GPUTimer gpu_timer;

float get_gpu_timer_result() {
    return gpu_timer.get_result();
}
