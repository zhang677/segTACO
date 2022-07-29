#include "gpu_library.h"
#include <cub/device/device_spmv.cuh>
using namespace cub;

void spmv_csr_gpu_merge(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
  int A_num_rows = (int)(A->dimensions[0]);
  int A_num_cols = (int)(A->dimensions[1]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int A_num_nnz = A2_pos[A_num_rows];

  copy_to_device_spmv(y, A, x);

  // Allocate temporary storage
  size_t temp_storage_bytes = 0;
  void *d_temp_storage = NULL;

  // Get amount of temporary storage needed
  CubDebugExit(DeviceSpmv::CsrMV(
      d_temp_storage, temp_storage_bytes,
      global_A_vals_host_copy, global_A2_pos_host_copy, global_A2_crd_host_copy,
      global_x_vals_host_copy, global_y_vals_host_copy,
      A_num_rows, A_num_cols, A_num_nnz,
      (cudaStream_t) 0, false));

  // Allocate
  gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

TIME_GPU(
  CubDebugExit(DeviceSpmv::CsrMV(
              d_temp_storage, temp_storage_bytes,
              global_A_vals_host_copy, global_A2_pos_host_copy, global_A2_crd_host_copy,
              global_x_vals_host_copy, global_y_vals_host_copy,
              A_num_rows, A_num_cols, A_num_nnz,
              (cudaStream_t) 0, false));
);

  copy_from_device_spmv(y, A, x);
  free_tensors_spmv();
  gpuErrchk(cudaFree(d_temp_storage));
}