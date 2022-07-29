#include "gpu_library.h"
#include "gpu_kernels.cuh"
#include "stdio.h"


int* count=0;

__global__
//void spmm_csr_reduction_kernel(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C){
 void spmm_csr_reduction_kernel(int A1_dimension, int* __restrict__ A2_pos, int* __restrict__ A2_crd,
    double* __restrict__ A_vals, int B2_dimension, double* __restrict__ B_vals, 
    int C1_dimension, double* __restrict__ C_vals, int* count){

  atomicAdd(&count[0], 1.0);
  int32_t ko = blockIdx.x;
  int32_t jpos1 = (threadIdx.x % (32));
  int32_t ki = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  int32_t io = ko * 8 + ki;
  int32_t i = io / B2_dimension;
  if (i >= A1_dimension)
    return;

  int32_t k = io % B2_dimension;
  int32_t iC = k * C1_dimension + i;
  if (k >= B2_dimension)
    return;

  double res = 0;  // store the reduction result of every thread
  for (int32_t jpos0 = A2_pos[i] / 32; jpos0 < ((A2_pos[(i + 1)] + 31) / 32); jpos0++) {
    int32_t jposA = jpos0 * 32 + jpos1;
    if (jposA < A2_pos[i] || jposA >= A2_pos[(i + 1)])
      break;
    int32_t j = A2_crd[jposA];
    int32_t kB = j * B2_dimension + k;
		res += A_vals[jposA]*B_vals[kB];
  }
  atomicAddWarp<double>(C_vals, iC, res);  // warp reduction of res in every thread 
}

void spmm_csr_reduction(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C2_dimension = (int)(C->dimensions[1]);
  int C1_dimension = (int)(C->dimensions[0]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[0]);
//  double* __restrict__ C_vals = (double*)(C->vals);
  copy_to_device_spmm(C, A, B);
  
  gpuErrchk(cudaMalloc((void**)&count, sizeof(int)));
  gpuErrchk(cudaMemset(count, 0, sizeof(int)));

TIME_GPU(
  gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 8));
  //spmm_csr_gpu_reduction_kernel(int A1_dimension, int* __restrict__ A2_pos, int* __restrict__ A2_crd,
  //  double* __restrict__ A_vals, int B2_dimension, double* __restrict__ B_vals, int C1_dimension, double* __restrict__ C_vals)
  spmm_csr_reduction_kernel<<<(C1_dimension + 7) / 8, 256>>>(A1_dimension, global_A2_pos_host_copy, global_A2_crd_host_copy,
  global_A_vals_host_copy, B2_dimension, global_B_vals_host_copy, C1_dimension, global_C_vals_host_copy,count);
);
/*
  printf("%s\n",cudaGetErrorString(cudaGetLastError()));
  int cnt[1]={-1};
  gpuErrchk(cudaMemcpyAsync(cnt,count,sizeof(int),cudaMemcpyDeviceToHost));

  printf("%d\n",cnt[0]);
  exit(0);
*/
  cudaFree(count);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
//  C->vals = (uint8_t*)C_vals; //transformed to uint_8
}

__global__
void spmm_csr_eb_kernel(int A1_dimension, int* __restrict__ A2_pos, int* __restrict__ A2_crd,
    double* __restrict__ A_vals, int B2_dimension, double* __restrict__ B_vals, int C1_dimension, 
    double* __restrict__ C_vals, int32_t* i_blockStarts){
  
  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) {
    return;
  }
  
  for (int32_t dense_val = 0; dense_val < 4; dense_val++) {
    int32_t k = dense_val * 32 + thread;
    double tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block]; // 256 nnz per block
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 16;
    int32_t fposA = block * 256 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA); // find the start row
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 16; nnz++) {
      int32_t fpos1 = warp * 16 + nnz; // 16 nnz per warp, 16 warp per block
      int32_t fposA = block * 256 + fpos1; // A_vals 
      if (fposA >= A2_pos[A1_dimension])
        break;
      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos; // enumerate the sparse row
      }
      int32_t iC = k * C1_dimension + i; // col-major
      int32_t kB = f * B2_dimension + k; // row-major
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    int32_t iC = k * C1_dimension + i;
    atomicAdd(&C_vals[iC], tnnz_val);
  }
}

void spmm_csr_eb(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C2_dimension = (int)(C->dimensions[1]);
  int C1_dimension = (int)(C->dimensions[0]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[0]);
//  double* __restrict__ C_vals = (double*)(C->vals);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  copy_to_device_spmm(C, A, B);
  

  int32_t* i_blockStarts = 0; // start of each segment
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1))); // element-balance
  cudaMemset(i_blockStarts, 0, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1));

  taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 512, ((A2_pos[A1_dimension] + 255) / 256));
  gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 8));
  spmm_csr_eb_kernel<<<(A2_pos[A1_dimension] + 255) / 256, 32 * 16>>>(A1_dimension, global_A2_pos_host_copy, global_A2_crd_host_copy,
  global_A_vals_host_copy, B2_dimension, global_B_vals_host_copy, 
  C1_dimension, global_C_vals_host_copy,i_blockStarts);

  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  //printf("A1d: %d\n",A1_dimension);
  for(int i=0;i<50;i++){
    printf("ib[%d]:%d\n",i,i_blockStarts[i]);
  }
  cudaFree(i_blockStarts);
  //copy_from_device_spmm(C, A, B);
  double*  C_vals = (double*)(C->vals);
  //cudaDeviceSynchronize();
  gpuErrchk(cudaMemcpy(C_vals, global_C_vals_host_copy, sizeof(double)*C2_dimension*C1_dimension, cudaMemcpyDeviceToHost));
  //C->vals = (uint8_t*)C_vals;
  C->vals = C_vals;
  free_tensors_spmm();
  //C->vals = (uint8_t*)C_vals; //transformed to uint_8
}
