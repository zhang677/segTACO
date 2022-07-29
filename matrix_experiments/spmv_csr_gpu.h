#ifndef SPMV_CSR_GPU_H
#define SPMV_CSR_GPU_H

#include "ds.h"
#include "timers.h"

#include "gpu_kernels.cuh"

void spmv_csr_gpu(COO input_matrix, const std::string& matrix_name, const bool do_verify) {
  EigenCSR A_eigen = to_eigen_csr(input_matrix);
  taco_tensor_t A_taco = to_taco_tensor(A_eigen);
  
  EigenVector x_eigen = gen_vector(input_matrix.n);
  taco_tensor_t x_taco = to_taco_tensor(x_eigen);

  EigenVector y_eigen = gen_vector(input_matrix.m);
  taco_tensor_t y_taco = to_taco_tensor(y_eigen);

   if (do_verify) {
    EigenVector yref_eigen = gen_vector(input_matrix.m);
    taco_tensor_t yref_taco = to_taco_tensor(yref_eigen);

    spmv_csr_gpu_cusparse(&yref_taco, &A_taco, &x_taco);
    spmv_csr_gpu_taco(&y_taco, &A_taco, &x_taco);
    std::cout << "taco vs cusparse: " << compare_vectors(yref_taco, y_taco) << std::endl;

    spmv_csr_gpu_merge(&y_taco, &A_taco, &x_taco);
    std::cout << "merge-based vs cusparse: " << compare_vectors(yref_taco, y_taco) << std::endl;

  exit(0);
  }

  const int trials = 25;

  RUN_GPU(spmv_csr_gpu_taco(&y_taco, &A_taco, &x_taco);, trials,  
      "spmv", "gpu", "csr", "taco", matrix_name);
  RUN_GPU(spmv_csr_gpu_cusparse(&y_taco, &A_taco, &x_taco);, trials,  
      "spmv", "gpu", "csr", "cusparse", matrix_name);
  RUN_GPU(spmv_csr_gpu_merge(&y_taco, &A_taco, &x_taco);, trials,  
      "spmv", "gpu", "csr", "merge", matrix_name);
}

#endif
