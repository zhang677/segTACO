
#include "gpu_library.h"
#include "gpu_kernels.cuh"

__global__
void spmspv_kernel(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ x, taco_tensor_t * __restrict__ y){
  int* A2_pos = global_A2_pos_device;
  int* A2_crd = global_A2_crd_device;
  double* A_vals = global_A_vals_device;
  double* x_vals = global_x_vals_device;
  double* y_vals = global_y_vals_device;

  int x1_dimension = global_x_dimension;
  int* __restrict__ x1_pos = global_x_pos_device;
  int* __restrict__ x1_crd = global_x_crd_device;


  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (512));
  if (threadIdx.x >= 512) {
    return;
  }

  int32_t jpos_bound = block * 512 + thread;
  if (jpos_bound < x1_pos[0] || jpos_bound >= x1_pos[1])
    return;

  int32_t j = x1_crd[jpos_bound];
  for (int32_t iA = A2_pos[j]; iA < A2_pos[(j + 1)]; iA++) {
    int32_t i = A2_crd[iA];
    atomicAdd(&y_vals[i], A_vals[iA] * x_vals[jpos_bound]);
  }
}

void spmspv_csr_gpu_taco(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
  int *x1_dimension = (int*)(x->indices[0][0]);
  int y1_dimension = (int)(y->dimensions[0]);
  double* __restrict__ y_vals = (double*)(y->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  copy_to_device_spmspv(y, A, x);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMalloc((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 3583) / 3584 + 1)));

TIME_GPU(
  int32_t status = cudaMemset(global_y_vals_host_copy, 0, y1_dimension * sizeof(double));

  spmspv_kernel<<<(x1_dimension[1] + 511) / 512, 512>>>(A, x, y);

);

  copy_from_device_spmv(y, A, x);
  free_tensors_spmspv();
  y->vals = (uint8_t*)y_vals;
}




