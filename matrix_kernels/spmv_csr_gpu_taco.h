#include "gpu_library.h"
#include "gpu_kernels.cuh"

__global__
void spmv_csr_gpu_taco_kernel(taco_tensor_t * __restrict__ A, int32_t* i_blockStarts, taco_tensor_t * __restrict__ x, taco_tensor_t * __restrict__ y){
  int A1_dimension = global_A1_dimension;
  int* A2_pos = global_A2_pos_device;
  int* A2_crd = global_A2_crd_device;
  double* A_vals = global_A_vals_device;
  double* x_vals = global_x_vals_device;
  double* y_vals = global_y_vals_device;

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) {
    return;
  }

  double precomputed[7];
  for (int32_t pprecomputed = 0; pprecomputed < 7; pprecomputed++) {
    precomputed[pprecomputed] = 0.0;
  }
  int32_t thread_nz = 0;
  int32_t fpos2 = thread * 7 + thread_nz;
  int32_t fpos1 = warp * 224 + fpos2;
  int32_t fposA = block * 3584 + fpos1;
  if (block * 3584 + fpos1 + 7 >= A2_pos[A1_dimension]) {
    for (int32_t thread_nz_pre = 0; thread_nz_pre < 7; thread_nz_pre++) {
      int32_t thread_nz = thread_nz_pre;
      int32_t fpos2 = thread * 7 + thread_nz;
      int32_t fpos1 = warp * 224 + fpos2;
      int32_t fposA = block * 3584 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      precomputed[thread_nz_pre] = A_vals[fposA] * x_vals[f];
    }
  }
  else {
    #pragma unroll 7
    for (int32_t thread_nz_pre = 0; thread_nz_pre < 7; thread_nz_pre++) {
      int32_t thread_nz = thread_nz_pre;
      int32_t fpos2 = thread * 7 + thread_nz;
      int32_t fpos1 = warp * 224 + fpos2;
      int32_t fposA = block * 3584 + fpos1;
      int32_t f = A2_crd[fposA];
      precomputed[thread_nz_pre] = A_vals[fposA] * x_vals[f];
    }
  }
  double tthread_nz_val = 0.0;
  int32_t pA2_begin = i_blockStarts[block];
  int32_t pA2_end = i_blockStarts[(block + 1)];
  int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
  int32_t i = i_pos;
  for (int32_t thread_nz = 0; thread_nz < 7; thread_nz++) {
    int32_t fpos2 = thread * 7 + thread_nz;
    int32_t fpos1 = warp * 224 + fpos2;
    int32_t fposA = block * 3584 + fpos1;
    if (fposA >= A2_pos[A1_dimension])
      return;

    int32_t f = A2_crd[fposA];
    while (fposA == A2_pos[(i_pos + 1)]) {
      i_pos = i_pos + 1;
      i = i_pos;
    }
    tthread_nz_val = tthread_nz_val + precomputed[thread_nz];
    if (fposA + 1 == A2_pos[(i_pos + 1)]) {
      atomicAdd(&y_vals[i], tthread_nz_val);
      tthread_nz_val = 0.0;
    }
  }
  atomicAddWarp<double>(y_vals, i, tthread_nz_val);
}

void spmv_csr_gpu_taco(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
  int y1_dimension = (int)(y->dimensions[0]);
  double* __restrict__ y_vals = (double*)(y->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  copy_to_device_spmv(y, A, x);

  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMalloc((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 3583) / 3584 + 1)));

TIME_GPU(
  i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, 0, A1_dimension, 3584, 512, (A2_pos[A1_dimension] + 3583) / 3584);

  int32_t status = cudaMemset(global_y_vals_host_copy, 0, (size_t) y1_dimension * 8);

  spmv_csr_gpu_taco_kernel<<<(A2_pos[A1_dimension] + 3583) / 3584, 32 * 16>>>(A, i_blockStarts, x, y);
);

  copy_from_device_spmv(y, A, x);
  cudaFree(i_blockStarts);
  free_tensors_spmv();
  y->vals = (uint8_t*)y_vals;
}
