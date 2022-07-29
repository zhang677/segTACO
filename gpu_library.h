#ifndef GPU_LIBRARY_H
#define GPU_LIBRARY_H

#include <cuda_runtime.h>
#include "taco_tensor_t.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <thrust/complex.h>
#include "cusparse.h"
#include "cublas.h"
#include <cooperative_groups.h>

#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#define FULLMASK 0xffffffff
#define CEIL(x, y) (((x) + (y)-1) / (y))
#define DIV_UP(x, y) (((x) + (y)-1) / (y))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define CUBLAS_CHECK(x) {cublasStatus_t _c=x; if (_c != CUBLAS_STATUS_SUCCESS) {printf("cublas fail: %d, %s, line: %d\n", (int)_c, _cublasGetErrorEnum(_c), __LINE__);  exit(-1);}}
using namespace cooperative_groups;
const int RefThreadPerBlock = 256;


/***************************/
/* CUBLAS ERROR CHECKING */
/***************************/
static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {

        case CUBLAS_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, %s, line: %d\n", (int)_c, _cusparseGetErrorEnum(_c), __LINE__);  exit(-1);}}

/***************************/
/* CUSPARSE ERROR CHECKING */
/***************************/
static const char *_cusparseGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {

        case CUSPARSE_STATUS_SUCCESS:
            return "CUSPARSE_STATUS_SUCCESS";

        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return "CUSPARSE_STATUS_NOT_INITIALIZED";

        case CUSPARSE_STATUS_ALLOC_FAILED:
            return "CUSPARSE_STATUS_ALLOC_FAILED";

        case CUSPARSE_STATUS_INVALID_VALUE:
            return "CUSPARSE_STATUS_INVALID_VALUE";

        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return "CUSPARSE_STATUS_ARCH_MISMATCH";

        case CUSPARSE_STATUS_MAPPING_ERROR:
            return "CUSPARSE_STATUS_MAPPING_ERROR";

        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return "CUSPARSE_STATUS_EXECUTION_FAILED";

        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return "CUSPARSE_STATUS_INTERNAL_ERROR";

        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

        case CUSPARSE_STATUS_ZERO_PIVOT:
            return "CUSPARSE_STATUS_ZERO_PIVOT";
    }

    return "<unknown>";
}

#define SHFL_DOWN_REDUCE(v) \
v += __shfl_down_sync(FULLMASK, v, 16);\
v += __shfl_down_sync(FULLMASK, v, 8);\
v += __shfl_down_sync(FULLMASK, v, 4);\
v += __shfl_down_sync(FULLMASK, v, 2);\
v += __shfl_down_sync(FULLMASK, v, 1);

#define SEG_SHFL_SCAN(v, tmpv, segid, tmps) \
tmpv = __shfl_down_sync(FULLMASK, v, 1); tmps = __shfl_down_sync(FULLMASK, segid, 1); if (tmps == segid && lane_id < 31) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 2); tmps = __shfl_down_sync(FULLMASK, segid, 2); if (tmps == segid && lane_id < 30) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 4); tmps = __shfl_down_sync(FULLMASK, segid, 4); if (tmps == segid && lane_id < 28) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 8); tmps = __shfl_down_sync(FULLMASK, segid, 8); if (tmps == segid && lane_id < 24) v += tmpv;\
tmpv = __shfl_down_sync(FULLMASK, v, 16); tmps = __shfl_down_sync(FULLMASK, segid, 16); if (tmps == segid && lane_id < 16) v += tmpv;


template <typename index_t>
__device__ __forceinline__ index_t
binary_search_segment_number(const index_t *seg_offsets, const index_t n_seg,
                             const index_t n_elem, const index_t elem_id) {
  index_t lo = 1, hi = n_seg, mid;
  while (lo < hi) {
    mid = (lo + hi) >> 1;
    if (seg_offsets[mid] <= elem_id) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return (hi - 1);
}

template <typename T>
__device__ __forceinline__ T __guard_load_default_one(const T *base,
                                                      int offset) {
  if (base != nullptr)
    return base[offset];
  else
    return static_cast<T>(1);
}

///////////////////////
__device__ __host__ int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target) {
    return arrayStart;
  }
  int lowerBound = arrayStart; // always < target
  int upperBound = arrayEnd; // always >= target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return upperBound;
}
__device__ __host__ int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}
__global__ void taco_binarySearchBeforeBlock(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int values_per_block, int num_blocks) {
  int thread = threadIdx.x;
  int block = blockIdx.x;
  int idx = block * blockDim.x + thread;
  if (idx >= num_blocks+1) {
    return;
  }

  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, idx * values_per_block);
}

__global__ void taco_vectorizeSearchBeforeBlock(uint32_t * __restrict__ array, uint32_t * __restrict__ results, int len_ori, int len_vec){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= len_vec) {
    return;
  }
  int p_start = 2 * idx;
  int p_end = p_start + 1;
  if (p_end >= len_ori) {
    results[idx] = array[p_start] << 16;
  } else {
    results[idx] = array[p_start] << 16 | array[p_end];
  }
}

__host__ uint32_t* taco_vectorizeSearchBeforeBlockLaunch(uint32_t * __restrict__ array, uint32_t * __restrict__ results, int len_ori, int len_vec){
  taco_vectorizeSearchBeforeBlock<<<CEIL(len_vec,256), 256>>>(array, results, len_ori, len_vec);
  return results;
}

__host__ int * taco_binarySearchBeforeBlockLaunch(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int values_per_block, int block_size, int num_blocks){
  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;
  taco_binarySearchBeforeBlock<<<num_search_blocks, block_size>>>(array, results, arrayStart, arrayEnd, values_per_block, num_blocks);
  return results;
}
__global__ void taco_binarySearchIndirectBeforeBlock(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int * __restrict__ targets, int num_blocks) {
  int thread = threadIdx.x;
  int block = blockIdx.x;
  int idx = block * blockDim.x + thread;
  if (idx >= num_blocks+1) {
    return;
  }

  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, targets[idx]);
}

__host__ int * taco_binarySearchIndirectBeforeBlockLaunch(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int * __restrict__ targets, int block_size, int num_blocks){
  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;
  taco_binarySearchIndirectBeforeBlock<<<num_search_blocks, block_size>>>(array, results, arrayStart, arrayEnd, targets, num_blocks);
  return results;
}
template<typename T>
__device__ inline void atomicAddWarp(T *array, int32_t index, T val)
{
  int leader_index = __shfl_sync(-1, index, 0);
  int mask = __ballot_sync(-1, leader_index == index);
  if(mask == -1) {
    val += __shfl_down_sync(-1, val, 16);
    val += __shfl_down_sync(-1, val, 8);
    val += __shfl_down_sync(-1, val, 4);
    val += __shfl_down_sync(-1, val, 2);
    val += __shfl_down_sync(-1, val, 1);
    if(threadIdx.x % 32 == 0) {
      atomicAdd(&array[index], val);
    }
  } else {
    atomicAdd(&array[index], val);
  }
}
template<typename T>
__device__ inline void segReduceWarp(T *array, int32_t index, T val)
{
  int row_intv = __shfl_sync(FULLMASK, index, (32 - 1)) - __shfl_sync(FULLMASK, index, 0);
  if (row_intv == 0) {
    SHFL_DOWN_REDUCE(val);
    if(threadIdx.x % 32 == 0) {
      atomicAdd(&array[index], val);
    }
  }else {
    bool is_seg_start = ((__shfl_up_sync(FULLMASK, index, 1) != index) || (threadIdx.x % 32 == 0));
    T tmpv;
    int tmpr;
    int lane_id = threadIdx.x % 32;
    SEG_SHFL_SCAN(val, tmpv, index, tmpr);
    if (is_seg_start) {
      atomicAdd(&array[index], val);
    }
  } 
}
template<typename T, int group_size>
__device__ inline void segReduceGroup(T *array, int32_t index, T val) {
  thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());
  int row_intv = group.shfl(index, group.size()-1) - group.shfl(index, 0);
  if (row_intv == 0) {
    for (int k = group.size() >> 1; k > 0; k = k >> 1) {
      val += group.shfl_down(val, k);
    }
    if (group.thread_rank() == 0) {
      atomicAdd(&array[index], val);
    }
  } else {
    bool is_seg_start = ((group.shfl_up(index,1) != index)|| (group.thread_rank() == 0));
    T tmpv;
    int tmpr;
    for (int k = 1; k < group.size(); k = k << 1) {
      tmpv = group.shfl_down(val,k);
      tmpr = group.shfl_down(index,k);
      if (tmpr == index && group.thread_rank() < (group.size()-k)) {
        val += tmpv;
      }
    }
    if (is_seg_start) {
      atomicAdd(&array[index], val);
    }
  }
}

template<typename T, int group_size>
__device__ inline void atomicAddGroup(T *array, int32_t index, T val) {
  thread_block_tile<group_size,thread_block> group = tiled_partition<group_size>(this_thread_block());
  for (int k = group.size()>>1 ; k>0; k = k >> 1) {
    val += group.shfl_down(val, k);
  }
  if (group.thread_rank() == 0) {
    atomicAdd(&array[index], val);
  }
}



__device__ int global_y1_dimension;

double * global_y_vals_host_copy;
__device__ double * global_y_vals_device;

__constant__ int global_A1_dimension;

int * global_A2_pos_host_copy;
__device__ int * global_A2_pos_device;

int * global_A2_crd_host_copy;
__device__ int * global_A2_crd_device;

float * global_A_vals_host_copy;
__device__ float * global_A_vals_device;

__device__ int global_x_dimension;

double * global_x_vals_host_copy;
__device__ double *global_x_vals_device;

int * global_x_pos_host_copy;
__device__ int * global_x_pos_device;

int * global_x_crd_host_copy;
__device__ int * global_x_crd_device;


void copy_to_device_spmspv(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
  int*  x_pos = (int*)(x->indices[0][0]);
  int*  x_crd = (int*)(x->indices[0][1]);

  int y1_dimension = (int)(y->dimensions[0]);
  double*  y_vals = (double*)(y->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int*  A2_pos = (int*)(A->indices[1][0]);
  int*  A2_crd = (int*)(A->indices[1][1]);
  double*  A_vals = (double*)(A->vals);
  double*  x_vals = (double*)(x->vals);

  int num_rows = (int)(y->dimensions[0]);
  int num_cols = (int)(x->dimensions[0]);
  int nnz = A2_pos[A1_dimension];
  int nx = x_pos[1];

  // allocate host_copy
  gpuErrchk(cudaMalloc(&global_y_vals_host_copy, (num_rows * sizeof(double))));
  gpuErrchk(cudaMalloc(&global_A2_pos_host_copy, (A1_dimension + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_A2_crd_host_copy, nnz * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_A_vals_host_copy, nnz * sizeof(double)));
  gpuErrchk(cudaMalloc(&global_x_vals_host_copy, nx * sizeof(double)));

  gpuErrchk(cudaMalloc(&global_x_crd_host_copy, nx * sizeof(double)));
  gpuErrchk(cudaMalloc(&global_x_pos_host_copy, 2 * sizeof(double)));

  // copy data from host to host_copy
  gpuErrchk(cudaMemcpyAsync(global_y_vals_host_copy, y_vals, (num_rows * sizeof(double)), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_A2_pos_host_copy, A2_pos, (A1_dimension + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_A2_crd_host_copy, A2_crd, nnz * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_A_vals_host_copy, A_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpyAsync(global_x_vals_host_copy, x_vals, nx * sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpyAsync(global_x_pos_host_copy, x_pos, 2 * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_x_crd_host_copy, x_crd, nx * sizeof(int), cudaMemcpyHostToDevice));

}

void free_tensors_spmspv() {
  cudaFree(global_y_vals_host_copy);
  cudaFree(global_A2_pos_host_copy);
  cudaFree(global_A2_crd_host_copy);
  cudaFree(global_A_vals_host_copy);
  cudaFree(global_x_vals_host_copy);
  cudaFree(global_x_pos_host_copy);
  cudaFree(global_x_crd_host_copy);
}


void copy_to_device_spmv(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
  int y1_dimension = (int)(y->dimensions[0]);
  double*  y_vals = (double*)(y->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int*  A2_pos = (int*)(A->indices[1][0]);
  int*  A2_crd = (int*)(A->indices[1][1]);
  double*  A_vals = (double*)(A->vals);
  double*  x_vals = (double*)(x->vals);

  int num_rows = (int)(y->dimensions[0]);
  int num_cols = (int)(x->dimensions[0]);
  int nnz = A2_pos[A1_dimension];

  // allocate host_copy
  gpuErrchk(cudaMalloc(&global_y_vals_host_copy, (num_rows * sizeof(double))));
  gpuErrchk(cudaMalloc(&global_A2_pos_host_copy, (num_rows + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_A2_crd_host_copy, nnz * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_A_vals_host_copy, nnz * sizeof(double)));
  gpuErrchk(cudaMalloc(&global_x_vals_host_copy, num_cols * sizeof(double)));

  // copy data from host to host_copy
  gpuErrchk(cudaMemcpyAsync(global_y_vals_host_copy, y_vals, (num_rows * sizeof(double)), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_A2_pos_host_copy, A2_pos, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_A2_crd_host_copy, A2_crd, nnz * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_A_vals_host_copy, A_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpyAsync(global_x_vals_host_copy, x_vals, num_cols * sizeof(double), cudaMemcpyHostToDevice));

}



void copy_from_device_spmv(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
  int y1_dimension = (int)(y->dimensions[0]);
  double*  y_vals = (double*)(y->vals);

  gpuErrchk(cudaMemcpyAsync(y_vals, global_y_vals_host_copy, (y1_dimension * sizeof(double)), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
}

void free_tensors_spmv() {
  cudaFree(global_y_vals_host_copy);
  cudaFree(global_A2_pos_host_copy);
  cudaFree(global_A2_crd_host_copy);
  cudaFree(global_A_vals_host_copy);
  cudaFree(global_x_vals_host_copy);
}

__constant__ int global_C1_dimension;
__constant__ int global_C2_dimension;

float * global_C_vals_host_copy;
__device__ float * global_C_vals_device;

__constant__ int global_B2_dimension;

float * global_B_vals_host_copy;
__device__ float *global_B_vals_device;

void copy_to_device_spmm(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  float*  C_vals = (float*)(C->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int*  A2_pos = (int*)(A->indices[1][0]);
  int*  A2_crd = (int*)(A->indices[1][1]);
  float*  A_vals = (float*)(A->vals);
  float*  B_vals = (float*)(B->vals);
  int32_t* C_dimensions = (int32_t*)(C->dimensions);
  int32_t* B_dimensions = (int32_t*)(B->dimensions);
  int32_t* A_dimensions = (int32_t*)(A->dimensions);
  int num_i = (int)(C->dimensions[0]);
  int num_k = (int)(C->dimensions[1]);
  int num_j = (int)(B->dimensions[0]);
  int nnz = A2_pos[A1_dimension];

  // allocate host_copy
  gpuErrchk(cudaMalloc(&global_C_vals_host_copy, (num_i * num_k * sizeof(float))));
  gpuErrchk(cudaMalloc(&global_A2_pos_host_copy, (num_i + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_A2_crd_host_copy, nnz * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_A_vals_host_copy, nnz * sizeof(float)));
  gpuErrchk(cudaMalloc(&global_B_vals_host_copy, num_j * num_k * sizeof(float)));


  // copy data from host to host_copy
  gpuErrchk(cudaMemcpyAsync(global_C_vals_host_copy, C_vals, (num_i * num_k * sizeof(float)), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_A2_pos_host_copy, A2_pos, (num_i + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_A2_crd_host_copy, A2_crd, nnz * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_A_vals_host_copy, A_vals, nnz * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B_vals_host_copy, B_vals, num_j * num_k * sizeof(float), cudaMemcpyHostToDevice));


  // copy from host_copy to symbol device

  gpuErrchk(cudaMemcpyToSymbol(global_C_vals_device, &global_C_vals_host_copy, sizeof(float*)));
  gpuErrchk(cudaMemcpyToSymbol(global_A2_pos_device, &global_A2_pos_host_copy, sizeof(int*)));
  gpuErrchk(cudaMemcpyToSymbol(global_A2_crd_device, &global_A2_crd_host_copy, sizeof(int*)));
  gpuErrchk(cudaMemcpyToSymbol(global_A_vals_device, &global_A_vals_host_copy, sizeof(float*)));
  gpuErrchk(cudaMemcpyToSymbol(global_B_vals_device, &global_B_vals_host_copy, sizeof(float*)));
  gpuErrchk(cudaMemcpyToSymbol(global_C1_dimension, &num_i, sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbol(global_A1_dimension, &num_i, sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbol(global_B2_dimension, &num_k, sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbol(global_C2_dimension, &num_k, sizeof(int)));

  
}

void copy_from_device_spmm(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {

  float*  C_vals = (float*)(C->vals);
  int num_i = (int)(C->dimensions[0]);
  int num_k = (int)(C->dimensions[1]);


  gpuErrchk(cudaMemcpyAsync(C_vals, global_C_vals_host_copy, (num_i * num_k * sizeof(float)), cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();
  C->vals = C_vals;
  

}

__device__ int global_B1_dimension;
__device__ int global_D1_dimension;
//__device__ int global_C2_dimension;

int * global_B2_pos_host_copy;
__device__ int * global_B2_pos_device;

int * global_B2_crd_host_copy;
__device__ int * global_B2_crd_device;

double * global_D_vals_host_copy;
__device__ double * global_D_vals_device;

__device__ int global_D2_dimension;



void copy_to_device_sddmm(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
  int C1_dimension = (int)(C->dimensions[0]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B1_dimension = (int)(B->dimensions[0]);
  int*  B2_pos = (int*)(B->indices[1][0]);
  int*  B2_crd = (int*)(B->indices[1][1]);
  double*  A_vals = (double*)(A->vals);
  double*  B_vals = (double*)(B->vals);
  double*  C_vals = (double*)(C->vals);
  double*  D_vals = (double*)(D->vals);

  int num_i = (int)(D->dimensions[0]);
  int num_k = (int)(D->dimensions[1]);
  int num_j = (int)(C->dimensions[0]);
  int nnz = B2_pos[B1_dimension];

  // allocate host_copy
  gpuErrchk(cudaMalloc(&global_D_vals_host_copy, (num_i * num_k * sizeof(double))));
  gpuErrchk(cudaMalloc(&global_B2_pos_host_copy, (num_j + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_B2_crd_host_copy, nnz * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_B_vals_host_copy, nnz * sizeof(double)));
  gpuErrchk(cudaMalloc(&global_C_vals_host_copy, num_j * num_k * sizeof(double)));

  gpuErrchk(cudaMalloc(&global_A_vals_host_copy, nnz * sizeof(double)));

  // copy data from host to host_copy
  gpuErrchk(cudaMemcpyAsync(global_D_vals_host_copy, D_vals, (num_i * num_k * sizeof(double)), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B2_pos_host_copy, B2_pos, (num_j + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B2_crd_host_copy, B2_crd, nnz * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B_vals_host_copy, B_vals, nnz * sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_C_vals_host_copy, C_vals, num_j * num_k * sizeof(double), cudaMemcpyHostToDevice));

}

void copy_from_device_sddmm(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
  int B1_dimension = (int)(B->dimensions[0]);
  int*  B2_pos = (int*)(B->indices[1][0]);
  int*  B2_crd = (int*)(B->indices[1][1]);
  double*  A_vals = (double*)(A->vals);

  int nnz = B2_pos[B1_dimension];
  
   // copy data from host_copy to host
  gpuErrchk(cudaMemcpyAsync(A_vals, global_A_vals_host_copy, (nnz * sizeof(double)), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
}




void free_tensors_spmm() {
  
  gpuErrchk(cudaFree(global_C_vals_host_copy));
  gpuErrchk(cudaFree(global_A2_pos_host_copy));
  gpuErrchk(cudaFree(global_A2_crd_host_copy));
  gpuErrchk(cudaFree(global_A_vals_host_copy));
  gpuErrchk(cudaFree(global_B_vals_host_copy));
  
}

void free_tensors_sddmm() {
  cudaFree(global_D_vals_host_copy);
  cudaFree(global_B2_pos_host_copy);
  cudaFree(global_B2_crd_host_copy);
  cudaFree(global_B_vals_host_copy);
  cudaFree(global_C_vals_host_copy);
  cudaFree(global_A_vals_host_copy);
}


__device__ int global_A2_dimension;

int * global_B1_pos_host_copy;
__device__ int * global_B1_pos_device;

int * global_B1_crd_host_copy;
__device__ int * global_B1_crd_device;

int * global_B3_pos_host_copy;
__device__ int * global_B3_pos_device;

int * global_B3_crd_host_copy;
__device__ int * global_B3_crd_device;

float * global_A_vals_host_copy_float;
__device__ float *global_A_vals_device_float;

float * global_B_vals_host_copy_float;
__device__ float *global_B_vals_device_float;

float * global_C_vals_host_copy_float;
__device__ float *global_C_vals_device_float;

float * global_D_vals_host_copy_float;
__device__ float *global_D_vals_device_float;

void copy_to_device_mttkrp(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  float* A_vals = (float*)(A->vals);
  int* B1_pos = (int*)(B->indices[0][0]);
  int* B1_crd = (int*)(B->indices[0][1]);
  int* B2_pos = (int*)(B->indices[1][0]);
  int* B2_crd = (int*)(B->indices[1][1]);
  int* B3_pos = (int*)(B->indices[2][0]);
  int* B3_crd = (int*)(B->indices[2][1]);
  float* B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  float* C_vals = (float*)(C->vals);
  int D1_dimension = (int)(D->dimensions[0]);
  int D2_dimension = (int)(D->dimensions[1]);
  float* D_vals = (float*)(D->vals);

  int num_i = (int)(A->dimensions[0]);
  int num_j = (int)(A->dimensions[1]);
  int num_k = (int)(C->dimensions[0]);
  int num_l = (int)(D->dimensions[0]);
  int nnz = B3_pos[B2_pos[B1_pos[1]]];

  // allocate host_copy
  gpuErrchk(cudaMalloc(&global_B1_pos_host_copy, 2 * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_B1_crd_host_copy, B1_pos[1] * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_B2_pos_host_copy, (B1_pos[1] + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_B2_crd_host_copy, (B2_pos[B1_pos[1]]) * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_B3_pos_host_copy, (B2_pos[B1_pos[1]] + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_B3_crd_host_copy, nnz * sizeof(int)));

  gpuErrchk(cudaMalloc(&global_A_vals_host_copy_float, num_i * num_j * sizeof(float)));
  gpuErrchk(cudaMalloc(&global_B_vals_host_copy_float, (nnz * sizeof(float))));
  gpuErrchk(cudaMalloc(&global_C_vals_host_copy_float, num_k * num_j * sizeof(float)));
  gpuErrchk(cudaMalloc(&global_D_vals_host_copy_float, num_l * num_j * sizeof(float)));

  // copy data from host to host_copy
  gpuErrchk(cudaMemcpyAsync(global_A_vals_host_copy_float, A_vals, (num_i * num_j * sizeof(float)), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B_vals_host_copy_float, B_vals, (nnz * sizeof(float)), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_C_vals_host_copy_float, C_vals, (num_k * num_j * sizeof(float)), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_D_vals_host_copy_float, D_vals, (num_l * num_j * sizeof(float)), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpyAsync(global_B1_pos_host_copy, B1_pos, 2 * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B1_crd_host_copy, B1_crd, B1_pos[1]  * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B2_pos_host_copy, B2_pos, (B1_pos[1] + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B2_crd_host_copy, B2_crd, (B2_pos[B1_pos[1]]) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B3_pos_host_copy, B3_pos, (B2_pos[B1_pos[1]] + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B3_crd_host_copy, B3_crd, nnz * sizeof(int), cudaMemcpyHostToDevice));

  // copy from host_copy to symbol device
  
  gpuErrchk(cudaMemcpyToSymbolAsync(global_A_vals_device_float, &global_A_vals_host_copy_float, sizeof(float*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B_vals_device_float, &global_B_vals_host_copy_float, sizeof(float*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_C_vals_device_float, &global_C_vals_host_copy_float, sizeof(float*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_D_vals_device_float, &global_D_vals_host_copy_float, sizeof(float*)));

  gpuErrchk(cudaMemcpyToSymbolAsync(global_B1_pos_device, &global_B1_pos_host_copy, sizeof(int*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B1_crd_device, &global_B1_crd_host_copy, sizeof(int*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B2_pos_device, &global_B2_pos_host_copy, sizeof(int*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B2_crd_device, &global_B2_crd_host_copy, sizeof(int*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B3_pos_device, &global_B3_pos_host_copy, sizeof(int*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B3_crd_device, &global_B3_crd_host_copy, sizeof(int*)));

  gpuErrchk(cudaMemcpyToSymbolAsync(global_A2_dimension, &A2_dimension, sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_C2_dimension, &C2_dimension, sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_D2_dimension, &D2_dimension, sizeof(int)));
  
}

void copy_from_device_mttkrp(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D){
  int A1_dimension = (int)(A->dimensions[0]);
  int A2_dimension = (int)(A->dimensions[1]);
  float*  A_vals = (float*)(A->vals);

   // copy data from host_copy to host
  gpuErrchk(cudaMemcpyAsync(A_vals, global_A_vals_host_copy_float, (A1_dimension * A2_dimension * sizeof(float)), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
}


void copy_to_device_ttv(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *c) {

  int A1_dimension = (int)(A->dimensions[0]);
  int B1_dimension = (int)(B->dimensions[0]);
  int C1_dimension = (int)(c->dimensions[0]);
  int* B2_pos = (int*)(B->indices[1][0]);
  int* B2_crd = (int*)(B->indices[1][1]);
  int* B3_pos = (int*)(B->indices[2][0]);
  int* B3_crd = (int*)(B->indices[2][1]);
  float* A_vals = (float*)(A->vals);
  float* B_vals = (float*)(B->vals);
  float* c_vals = (float*)(c->vals);

  int nnz = B3_pos[B2_pos[B1_dimension]];

  // allocate host_copy
  gpuErrchk(cudaMalloc(&global_B2_pos_host_copy, (B1_dimension + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_B2_crd_host_copy, (B2_pos[B1_dimension]) * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_B3_pos_host_copy, (B2_pos[B1_dimension] + 1) * sizeof(int)));
  gpuErrchk(cudaMalloc(&global_B3_crd_host_copy, nnz * sizeof(int)));

  gpuErrchk(cudaMalloc(&global_A_vals_host_copy_float, A1_dimension * sizeof(float)));
  gpuErrchk(cudaMalloc(&global_B_vals_host_copy_float, (nnz * sizeof(float))));
  gpuErrchk(cudaMalloc(&global_C_vals_host_copy_float, C1_dimension * sizeof(float)));

  // copy data from host to host_copy
  gpuErrchk(cudaMemcpyAsync(global_B_vals_host_copy_float, B_vals, (nnz * sizeof(float)), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_C_vals_host_copy_float, c_vals, (C1_dimension * sizeof(float)), cudaMemcpyHostToDevice));

  gpuErrchk(cudaMemcpyAsync(global_B2_pos_host_copy, B2_pos, (B1_dimension + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B2_crd_host_copy, B2_crd, (B2_pos[B1_dimension]) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B3_pos_host_copy, B3_pos, (B2_pos[B1_dimension] + 1) * sizeof(int), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpyAsync(global_B3_crd_host_copy, B3_crd, nnz * sizeof(int), cudaMemcpyHostToDevice));

  // copy from host_copy to symbol device
  
  gpuErrchk(cudaMemcpyToSymbolAsync(global_A_vals_device_float, &global_A_vals_host_copy_float, sizeof(float*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B_vals_device_float, &global_B_vals_host_copy_float, sizeof(float*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_C_vals_device_float, &global_C_vals_host_copy_float, sizeof(float*)));

  gpuErrchk(cudaMemcpyToSymbolAsync(global_B2_pos_device, &global_B2_pos_host_copy, sizeof(int*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B2_crd_device, &global_B2_crd_host_copy, sizeof(int*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B3_pos_device, &global_B3_pos_host_copy, sizeof(int*)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B3_crd_device, &global_B3_crd_host_copy, sizeof(int*)));

  gpuErrchk(cudaMemcpyToSymbolAsync(global_A1_dimension, &A1_dimension, sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_B1_dimension, &B1_dimension, sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbolAsync(global_C1_dimension, &C1_dimension, sizeof(int)));
  
}


void copy_from_device_ttv(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
  int y1_dimension = (int)(y->dimensions[0]);
  float*  y_vals = (float*)(y->vals);

  gpuErrchk(cudaMemcpyAsync(y_vals, global_A_vals_host_copy_float, (y1_dimension * sizeof(float)), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();
}


void free_tensors_ttv() {
  cudaFree(global_B2_pos_host_copy);
  cudaFree(global_B2_crd_host_copy);
  cudaFree(global_B3_pos_host_copy);
  cudaFree(global_B3_crd_host_copy);
  cudaFree(global_A_vals_host_copy_float);
  cudaFree(global_B_vals_host_copy_float);
  cudaFree(global_C_vals_host_copy_float);
}


void free_tensors_mttkrp() {
  cudaFree(global_B1_pos_host_copy);
  cudaFree(global_B1_crd_host_copy);
  cudaFree(global_B2_pos_host_copy);
  cudaFree(global_B2_crd_host_copy);
  cudaFree(global_B3_pos_host_copy);
  cudaFree(global_B3_crd_host_copy);
  cudaFree(global_A_vals_host_copy_float);
  cudaFree(global_B_vals_host_copy_float);
  cudaFree(global_C_vals_host_copy_float);
  cudaFree(global_D_vals_host_copy_float);
}

/// TIMING

class GPUTimer {
public:
  GPUTimer() {
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);
  }

  ~GPUTimer() {
    cudaEventDestroy(event1);
    cudaEventDestroy(event2);
  }

  void start() {
    //gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaEventRecord(event1,0));
  }

  void stop() {
    gpuErrchk(cudaEventRecord(event2,0));
    gpuErrchk(cudaEventSynchronize(event2));
    gpuErrchk(cudaEventElapsedTime(&res, event1, event2));
    gpuErrchk(cudaDeviceSynchronize());
  }

  double get_result() {
    return res;
  }

private:
  float res = -1;
  cudaEvent_t event1, event2;
};

extern GPUTimer gpu_timer;

#define TIME_GPU(...) { \
    gpu_timer.start();   \
    __VA_ARGS__;         \
    gpu_timer.stop();    \
  }

#endif
