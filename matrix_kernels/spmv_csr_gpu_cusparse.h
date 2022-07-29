#include "gpu_library.h"

void spmv_csr_gpu_cusparse(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
  cusparseHandle_t handle=0;
  cusparseMatDescr_t descra=0;

  int A_num_rows = (int)(A->dimensions[0]);
  int A_num_cols = (int)(A->dimensions[1]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int A_num_nnz = A2_pos[A_num_rows];

  copy_to_device_spmv(y, A, x);

  /* initialize cusparse library */
  CUSPARSE_CHECK(cusparseCreate(&handle));
  

  /* create and setup matrix descriptor */
  CUSPARSE_CHECK(cusparseCreateMatDescr(&descra));

  cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  const double alpha=1.0f, beta=0.0f;

  // allocate an external buffer if needed
/*

  size_t bufferSize = 0;
  CUSPARSE_CHECK(cusparseCsrmvEx_bufferSize(
                                handle, (cusparseAlgMode_t) 0, CUSPARSE_OPERATION_NON_TRANSPOSE, A_num_rows, A_num_cols, A_num_nnz,
                                &alpha, CUDA_R_64F, descra, global_A_vals_host_copy, CUDA_R_64F, global_A2_pos_host_copy,
                                global_A2_crd_host_copy, global_x_vals_host_copy, CUDA_R_64F, &beta, CUDA_R_64F, 
                                global_y_vals_host_copy, CUDA_R_64F, CUDA_R_64F, &bufferSize));
  void *dBuffer = NULL;
  gpuErrchk(cudaMalloc(&dBuffer, bufferSize));

TIME_GPU(
  CUSPARSE_CHECK(cusparseCsrmvEx(handle, (cusparseAlgMode_t) 0, CUSPARSE_OPERATION_NON_TRANSPOSE, A_num_rows, A_num_cols, A_num_nnz,
                                &alpha, CUDA_R_64F, descra, global_A_vals_host_copy, CUDA_R_64F, global_A2_pos_host_copy,
                                global_A2_crd_host_copy, global_x_vals_host_copy, CUDA_R_64F, &beta, CUDA_R_64F, 
                                global_y_vals_host_copy, CUDA_R_64F, CUDA_R_64F, dBuffer));
);
*/

  size_t bufferSize = 0;
  CUSPARSE_CHECK(cusparseCsrmvEx_bufferSize(
                                handle, (cusparseAlgMode_t) 1, CUSPARSE_OPERATION_NON_TRANSPOSE, A_num_rows, A_num_cols, A_num_nnz,
                                &alpha, CUDA_R_64F, descra, global_A_vals_host_copy, CUDA_R_64F, global_A2_pos_host_copy,
                                global_A2_crd_host_copy, global_x_vals_host_copy, CUDA_R_64F, &beta, CUDA_R_64F, 
                                global_y_vals_host_copy, CUDA_R_64F, CUDA_R_64F, &bufferSize));
  void *dBuffer = NULL;
  gpuErrchk(cudaMalloc(&dBuffer, bufferSize));

TIME_GPU(
  CUSPARSE_CHECK(cusparseCsrmvEx(handle, (cusparseAlgMode_t) 1, CUSPARSE_OPERATION_NON_TRANSPOSE, A_num_rows, A_num_cols, A_num_nnz,
                                &alpha, CUDA_R_64F, descra, global_A_vals_host_copy, CUDA_R_64F, global_A2_pos_host_copy,
                                global_A2_crd_host_copy, global_x_vals_host_copy, CUDA_R_64F, &beta, CUDA_R_64F, 
                                global_y_vals_host_copy, CUDA_R_64F, CUDA_R_64F, dBuffer));
);

  copy_from_device_spmv(y, A, x);
  free_tensors_spmv();
  gpuErrchk(cudaFree(dBuffer));
  CUSPARSE_CHECK(cusparseDestroyMatDescr(descra));
  CUSPARSE_CHECK(cusparseDestroy(handle));
}
