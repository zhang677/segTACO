#include "gpu_library.h"
#include "gpu_kernels.cuh"
#include <iostream>

__global__
void compute_128_512_4DeviceKernel0(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, taco_tensor_t * __restrict__ D, int32_t &kB){
  double* __restrict__ A_vals = global_A_vals_device;
  int B1_dimension = global_B1_dimension;
  int* __restrict__ B2_pos = global_B2_pos_device;
  int* __restrict__ B2_crd = global_B2_crd_device;
  double* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;
  int C1_dimension = global_C1_dimension;
  double* __restrict__ C_vals = global_C_vals_device;
  int D2_dimension = global_D2_dimension;
  double* __restrict__ D_vals = global_D_vals_device;


  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) {
    return;
  }

  for (int32_t warp_row = 0; warp_row < 1; warp_row++) {
    int32_t i1 = warp + warp_row;
    int32_t i = block * 16 + i1;
    if (i >= C1_dimension)
      break;

    for (int32_t kB = B2_pos[i]; kB < B2_pos[(i + 1)]; kB++) {
      int32_t k = B2_crd[kB];
      double tdense_val_val = 0.0;
      for (int32_t dense_val = 0; dense_val < 4; dense_val++) {
        int32_t j = dense_val * 32 + thread;
        int32_t jC = i * C2_dimension + j;
        int32_t jD = k * D2_dimension + j;
        tdense_val_val = tdense_val_val + B_vals[kB] * C_vals[jC] * D_vals[jD];
      }
      atomicAddWarp<double>(A_vals, kB, tdense_val_val);
    }
  }
}

void sddmm_csr_gpu_warp(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
  int B1_dimension = (int)(B->dimensions[0]);
  int C1_dimension = (int)(C->dimensions[0]);
  int* __restrict__ B2_pos = (int*)(B->indices[1][0]);
  double* __restrict__ A_vals = (double*)(A->vals);
  int nnz = B2_pos[B1_dimension];

  int32_t* kB_ptr;
  copy_to_device_sddmm(A, B, C, D);
  gpuErrchk(cudaMallocManaged((void**)&kB_ptr, sizeof(int32_t)));
  int32_t& kB = *kB_ptr;
  kB = 0;

TIME_GPU(
  int32_t status = cudaMemset(global_A_vals_host_copy, 0, nnz * 8);
  compute_128_512_4DeviceKernel0<<<(C1_dimension + 15) / 16, 32 * 16>>>(A, B, C, D, kB);
);
  cudaDeviceSynchronize();

  copy_from_device_sddmm(A, B, C, D);
  free_tensors_sddmm();
  A->vals = (uint8_t*)A_vals;
}

__global__
void sddmm_csr_gpu_taco_kernel(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, taco_tensor_t * __restrict__ D, int32_t* i_blockStarts){
  double* __restrict__ A_vals = global_A_vals_device;
  int B1_dimension = global_B1_dimension;
  int* __restrict__ B2_pos = global_B2_pos_device;
  int* __restrict__ B2_crd = global_B2_crd_device;
  double* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;
  double* __restrict__ C_vals = global_C_vals_device;
  int D2_dimension = global_D2_dimension;
  double* __restrict__ D_vals = global_D_vals_device;

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) {
    return;
  }

  int32_t pB2_begin = i_blockStarts[block];
  int32_t pB2_end = i_blockStarts[(block + 1)];
  int32_t fpos1 = warp * 128;
  int32_t fposB = block * 2048 + fpos1;
  int32_t i_pos = taco_binarySearchBefore(B2_pos, pB2_begin, pB2_end, fposB);
  int32_t i = i_pos;
  for (int32_t nnz = 0; nnz < 128; nnz++) {
    int32_t fpos1 = warp * 128 + nnz;
    int32_t fposB = block * 2048 + fpos1;
    if (fposB >= B2_pos[B1_dimension])
      break;

    int32_t f = B2_crd[fposB];
    while (fposB == B2_pos[(i_pos + 1)]) {
      i_pos = i_pos + 1;
      i = i_pos;
    }
    double tdense_val_val = 0.0;
    #pragma unroll 4
    for (int32_t dense_val = 0; dense_val < 4; dense_val++) {
      int32_t j = dense_val * 32 + thread;
      int32_t jC = i * C2_dimension + j;
      int32_t jD = f * D2_dimension + j;
      tdense_val_val = tdense_val_val + B_vals[fposB] * C_vals[jC] * D_vals[jD];
    }
    atomicAddWarp<double>(A_vals, fposB, tdense_val_val);
  }

}


void sddmm_csr_gpu_taco(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
  int B1_dimension = (int)(B->dimensions[0]);
  int* __restrict__ B2_pos = (int*)(B->indices[1][0]);
  double* __restrict__ A_vals = (double*)(A->vals);

  copy_to_device_sddmm(A, B, C, D);

  int nnz = B2_pos[B1_dimension];

  int32_t* i_blockStarts = 0;
  
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((B2_pos[B1_dimension] + 2047) / 2048 + 1)));
  
TIME_GPU(
  i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_B2_pos_host_copy, i_blockStarts, (int32_t) 0, B1_dimension, (int32_t) 2048, (int32_t) 128, ((B2_pos[B1_dimension] + 2047) / 2048));

  int32_t status = cudaMemset(global_A_vals_host_copy, 0, nnz * 8);
  std::cout<<"Call the kernel"<<std::endl;
  sddmm_csr_gpu_taco_kernel<<<(B2_pos[B1_dimension] + 2047) / 2048, 32 * 16>>>(A, B, D, C, i_blockStarts);

);
  cudaDeviceSynchronize();

  copy_from_device_sddmm(A, B, C, D);
  cudaFree(i_blockStarts);
  free_tensors_sddmm();
  A->vals = (uint8_t*)A_vals;

}


