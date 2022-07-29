#include "gpu_library.h"
#include "gpu_kernels.cuh"
#include "stdio.h"


__global__
void compute_4_512DeviceKernel0(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (((B2_dimension + 3) / 4)));
  int32_t warp = (threadIdx.x / ((B2_dimension + 3) / 4));
  if (threadIdx.x >= (B2_dimension + 3) / 4 * 16) {
    return;
  }

  for (int32_t warp_row = 0; warp_row < 1; warp_row++) {
    int32_t i1 = warp + warp_row;
    int32_t i = block * 16 + i1;
    if (i >= A1_dimension)
      return;

    for (int32_t thread_col = 0; thread_col < 4; thread_col++) {
      int32_t k = thread * 4 + thread_col;
      if (k >= B2_dimension)
        break;

      int32_t kC = i * C2_dimension + k;
      for (int32_t jA = A2_pos[i]; jA < A2_pos[(i + 1)]; jA++) {
        int32_t j = A2_crd[jA];
        int32_t kB = j * B2_dimension + k;
        C_vals[kC] = C_vals[kC] + A_vals[jA] * B_vals[kB];
      }
    }
  }
}

void spmm_csr_gpu_warp(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

TIME_GPU(
  gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
  compute_4_512DeviceKernel0<<<(A1_dimension + 15) / 16, (B2_dimension + 3) / 4 * 16>>>(A, B, C);
);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}


__global__
void spmm_csr_gpu_taco_kernel(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C1_dimension = global_C1_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

  
  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) {
    return;
  }

  for (int32_t dense_val = 0; dense_val < 4; dense_val++) {
    int32_t k = dense_val * 32 + thread;
    float tnnz_val = 0.0;
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

__global__
void spmm_csr_gpu_taco_row(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

  
  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) { // 512 threads per block, 16 warps per block
    return;
  }

  for (int32_t dense_val = 0; dense_val < 4; dense_val++) {
    int32_t k = dense_val * 32 + thread;
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block]; 
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 16; // 16 nnz per warp 16=256/16
    int32_t fposA = block * 256 + fpos1; // 256 nnz per block
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
        i = i_pos; 
      }
      int32_t kC = i * C2_dimension + k; // row-major
      int32_t kB = f * B2_dimension + k; // row-major
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[kC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    int32_t kC = i * C2_dimension + k;
    atomicAdd(&C_vals[kC], tnnz_val);
  }
}

__global__
void spmm_csr_reduction(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C1_dimension = global_C1_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;
  
  int32_t ko = blockIdx.x;
  int32_t jpos1 = (threadIdx.x % (32));
  int32_t ki = (threadIdx.x / 32);
  if (threadIdx.x >= 1024) {
    return;
  }

  int32_t io = ko * 32 + ki;
  int32_t i = io / B2_dimension;
  if (i >= A1_dimension)
    return;

  int32_t k = io % B2_dimension;
  int32_t iC = k * C1_dimension + i;
  if (k >= B2_dimension)
    return;

  float res = 0;
  for (int32_t jpos0 = A2_pos[i] / 32; jpos0 < ((A2_pos[(i + 1)] + 31) / 32); jpos0++) {
    int32_t jposA = jpos0 * 32 + jpos1;
    if (jposA < A2_pos[i] )
        continue; 
    if (jposA < A2_pos[i] || jposA >= A2_pos[(i + 1)])
      break;
    int32_t j = A2_crd[jposA];
    int32_t kB = j * B2_dimension + k;
    res += A_vals[jposA]*B_vals[kB];
  }
  atomicAddWarp<float>(C_vals, iC, res);
}


__global__
void spmm_csr_reduction_warp(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C1_dimension = global_C1_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

  int32_t ko = blockIdx.x;
  int32_t jpos1 = (threadIdx.x % (32));
  int32_t wo = (threadIdx.x / 32);
  if (threadIdx.x >= 1024) {
    return;
  }

  for (int32_t wi = 0; wi < 8; wi++) {
    int32_t ki = wo * 8 + wi;
    int32_t io = ko * 256 + ki;
    int32_t i = io / B2_dimension;
    if (i >= A1_dimension)
      break;

    int32_t k = io % B2_dimension;
    int32_t iC = k * C1_dimension + i;
    if (k >= B2_dimension)
      break;

    float res = 0;
    for (int32_t jpos0 = A2_pos[i] / 32; jpos0 < ((A2_pos[(i + 1)] + 31) / 32); jpos0++) {
      int32_t jposA = jpos0 * 32 + jpos1;
      if (jposA < A2_pos[i] )
        continue;  
      if (jposA < A2_pos[i] || jposA >= A2_pos[(i + 1)])
        break;

      int32_t j = A2_crd[jposA];
      int32_t kB = j * B2_dimension + k;
      res += A_vals[jposA]*B_vals[kB];
      //atomicAddWarp<float>(C_vals, iC,A_vals[jposA]*B_vals[kB]);
    }
    atomicAddWarp<float>(C_vals, iC, res);
  }
}

__global__
void spmm_csr_reduction_sclar(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C1_dimension = global_C1_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

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

  float tjpos1C_val = 0.0;
  for (int32_t jpos0 = A2_pos[i] / 32; jpos0 < ((A2_pos[(i + 1)] + 31) / 32); jpos0++) {
    int32_t jposA = jpos0 * 32 + jpos1;
    if (jposA < A2_pos[i] || jposA >= A2_pos[(i + 1)])
      continue;

    int32_t j = A2_crd[jposA];
    int32_t kB = j * B2_dimension + k;
    tjpos1C_val = tjpos1C_val + A_vals[jposA] * B_vals[kB];
  }
  atomicAddWarp<float>(C_vals, iC, tjpos1C_val);
}

__global__
void spmm_csr_reduction_sclar_row(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

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
  int32_t kC = i * C2_dimension + k;
  if (k >= B2_dimension)
    return;

  float tjpos1C_val = 0.0;
  for (int32_t jpos0 = A2_pos[i] / 32; jpos0 < ((A2_pos[(i + 1)] + 31) / 32); jpos0++) {
    int32_t jposA = jpos0 * 32 + jpos1;
    if (jposA < A2_pos[i] || jposA >= A2_pos[(i + 1)])
      continue;

    int32_t j = A2_crd[jposA];
    int32_t kB = j * B2_dimension + k;
    tjpos1C_val = tjpos1C_val + A_vals[jposA] * B_vals[kB];
  }
  atomicAddWarp<float>(C_vals, kC, tjpos1C_val);
}

__global__
void spmm_csr_eb_sr_1(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

  int32_t block = blockIdx.x;
  int32_t nnz = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  int32_t pA2_begin = i_blockStarts[block];
  int32_t pA2_end = i_blockStarts[(block + 1)];
  int32_t fpos1 = warp * 32;
  int32_t fposA = block * 256 + fpos1;
  int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
  int32_t i = i_pos;
  fpos1 = warp * 32 + nnz; //int32_t fpos1 = warp * 32 + nnz;
  fposA = block * 256 + fpos1; //int32_t fposA = block * 256 + fpos1;
  if (fposA >= A2_pos[A1_dimension])
    return;

  int32_t f = A2_crd[fposA];
  while (fposA == A2_pos[(i_pos + 1)]) {
    i_pos = i_pos + 1;
    i = i_pos;
  }
  for (int32_t k = 0; k < B2_dimension; k++) {
    int32_t kC = i * C2_dimension + k;
    int32_t kB = f * B2_dimension + k;
    atomicAdd(&C_vals[kC], A_vals[fposA] * B_vals[kB]);
  }

}

__global__
void spmm_csr_eb_sr_2(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

  int32_t block = blockIdx.x;
  //int32_t lane = (threadIdx.x % (32));
  int32_t fpos1 = (threadIdx.x % (256));
  //int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  for (int32_t k = 0; k < B2_dimension; k++) {
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    //int32_t fpos1 = warp * 32+ lane;
    //int32_t fposA = block * 256 + fpos1;
    //int32_t fposA = block * 256;
    int32_t fposA = block * 256 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    
    float tmpVal = 0;
    if (fposA >= A2_pos[A1_dimension]){
      tmpVal = 0;
    }
    else{
      int32_t f = A2_crd[fposA];
      int32_t kB = f * B2_dimension + k;

      tmpVal = A_vals[fposA] * B_vals[kB];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
    }
    
    int32_t kC = i * C2_dimension + k;
    //atomicAdd(&C_vals[kC], A_vals[fposA] * B_vals[kB]);
    segReduceWarp<float>(C_vals, kC, tmpVal);
  }

/*
  int32_t block = blockIdx.x;
  int32_t lane = (threadIdx.x % (32));
  if (threadIdx.x >= 32) {
    return;
  }
  int32_t pA2_begin = i_blockStarts[block];
  int32_t pA2_end = i_blockStarts[(block + 1)];
  int32_t fpos1 = lane * 32;
  int32_t fposA = block * 1024 + fpos1;
  int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
  int32_t i = i_pos;
  for (int32_t k = 0; k < B2_dimension; k++) {
    for (int32_t nnz = 0; nnz < 32; nnz++) {
      int32_t fpos1 = lane * 32 + nnz;
      int32_t fposA = block * 1024 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      int32_t kB = f * B2_dimension + k;
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t kC = i * C2_dimension + k;
      atomicAdd(&C_vals[kC], A_vals[fposA] * B_vals[kB]);
    }
  }
*/
/*  
  for (int32_t k = 0; k < B2_dimension; k++) {
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 32;
    int32_t fposA = block * 1024 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 32; nnz++) {
      int32_t fpos1 = warp * 32 + nnz;
      int32_t fposA = block * 1024 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      int32_t kB = f * B2_dimension + k;
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t kC = i * C2_dimension + k;
      atomicAdd(&C_vals[kC], A_vals[fposA] * B_vals[kB]);
    }
  }
 */ 
/*
  int32_t pA2_begin = i_blockStarts[block];
  int32_t pA2_end = i_blockStarts[(block + 1)];
  int32_t fpos1 = lane * 32;
  int32_t fposA = block * 1024 + fpos1;
  int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
  int32_t i = i_pos;
  for (int32_t nnz = 0; nnz < 32; nnz++) {
    int32_t fpos1 = lane * 32 + nnz;
    int32_t fposA = block * 1024 + fpos1;
    if (fposA >= A2_pos[A1_dimension])
      break;

    int32_t f = A2_crd[fposA];
    while (fposA == A2_pos[(i_pos + 1)]) {
      i_pos = i_pos + 1;
      i = i_pos;
    }
    for (int32_t k = 0; k < B2_dimension; k++) {
      int32_t kC = i * C2_dimension + k;
      int32_t kB = f * B2_dimension + k;
      atomicAdd(&C_vals[kC], A_vals[fposA] * B_vals[kB]);
    }
  }
*/
}

__global__
void spmm_csr_eb_sr_3(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

  int32_t block = blockIdx.x;
  int32_t fpos1 = (threadIdx.x % (256));
  if (threadIdx.x >= 256) {
    return;
  }

  for (int32_t k = 0; k < B2_dimension; k++) {
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    //int32_t fposA = block * 256;
    int32_t fposA = block * 256 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    float tmp_val = 0.0;
    if (fposA >= A2_pos[A1_dimension]) tmp_val = 0.0;

    else {
      int32_t fposA = block * 256 + fpos1;
      int32_t f = A2_crd[fposA];
      int32_t kB = f * B2_dimension + k;
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      tmp_val = A_vals[fposA] * B_vals[kB];
    }
    int32_t kC = i * C2_dimension + k;
    //atomicAdd(&C_vals[kC], tmp_val);
    segReduceWarp<float>(C_vals, kC, tmp_val);
  }
}





void spmm_csr_gpu_taco(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  copy_to_device_spmm(C, A, B);

  int32_t* i_blockStarts = 0; // start of each segment
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1))); // element-balance
  //gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 1023) / 1024 + 1))); // element-balance

TIME_GPU(
  i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 512, ((A2_pos[A1_dimension] + 255) / 256)); // spmm_csr_gpu_taco_row
  //i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 256, ((A2_pos[A1_dimension] + 255) / 256)); // spmm_csr_eb_pr
  //i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 1024, (int32_t) 32, ((A2_pos[A1_dimension] + 1023) / 1024)); // spmm_csr_eb_pr

  //i_blockStarts = taco_binarySearchBeforeBlockLaunch((int*)(spmm_sparse_A->indices[1][0]), i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 512, ((A2_pos[A1_dimension] + 255) / 256));
  gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
  spmm_csr_gpu_taco_row<<<(A2_pos[A1_dimension] + 255) / 256, 32 * 16>>>(A, B, C, i_blockStarts);
  //spmm_csr_reduction_<<<(A1_dimension * B2_dimension + 7) / 8, (32 * 8)>>>(A, B, C);
  //computeDeviceKernel0<<<(A1_dimension * B2_dimension + 7) / 8, (32 * 8)>>>(spmm_sparse_A, spmm_dense_B, spmm_dense_C);
  //spmm_csr_eb_sr_2<<<(A2_pos[A1_dimension] + 1023) / 1024, 32>>>(A, B, C, i_blockStarts);
  //spmm_csr_eb_sr_3<<<(A2_pos[A1_dimension] + 255) / 256, (32 * 8)>>>(A, B, C, i_blockStarts);
);
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}

void spmm_csr_pr_eb(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  copy_to_device_spmm(C, A, B);

  int32_t* i_blockStarts = 0; // start of each segment
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1))); // element-balance

TIME_GPU(
  i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 256, ((A2_pos[A1_dimension] + 255) / 256)); // spmm_csr_eb_pr

  gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
  spmm_csr_eb_sr_3<<<(A2_pos[A1_dimension] + 255) / 256, (32 * 8)>>>(A, B, C, i_blockStarts);
);
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}


void spmm_csr_gpu_reduction(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  copy_to_device_spmm(C, A, B);
 

TIME_GPU(
  //gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 8)); // used with spmm_csr_reduction_warp
  //spmm_csr_reduction<<<(A1_dimension * B2_dimension + 31) / 32, (32 * 32)>>>(A, B, C);
  //spmm_csr_reduction_warp<<<(A1_dimension * B2_dimension + 255) / 256, (32 * 32)>>>(A, B, C); // good kernel
  //spmm_csr_reduction_shared<<<(A1_dimension * B2_dimension + 7) / 8, (32 * 8)>>>(A, B, C);
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4)); // used with spmm_csr_reduction_scalar
    spmm_csr_reduction_sclar_row<<<(A1_dimension * B2_dimension + 7) / 8, (32 * 8)>>>(A, B, C);
);
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
