#ifndef SDDMM_CSR_GPU_H
#define SDDMM_CSR_GPU_H

#include "ds.h"
#include "timers.h"

#include "gpu_kernels.cuh"

void sddmm_csr_cpu_taco_ref(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C, taco_tensor_t *D) {
TIME_COLD(
  double* restrict A_vals = (double*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* restrict B2_pos = (int*)(B->indices[1][0]);
  int* restrict B2_crd = (int*)(B->indices[1][1]);
  double* restrict B_vals = (double*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  double* restrict C_vals = (double*)(C->vals);
  int D1_dimension = (int)(D->dimensions[0]);
  int D2_dimension = (int)(D->dimensions[1]);
  double* restrict D_vals = (double*)(D->vals);

  int32_t jB = 0;

  _Pragma("omp parallel for schedule(static)")
  for (int32_t pC = 0; pC < B2_pos[C1_dimension]; pC++) {
    A_vals[pC] = 0.0;
  }

  _Pragma("omp parallel for schedule(dynamic, 1)")
  for (int32_t i0 = 0; i0 < ((C1_dimension + 7) / 8); i0++) {
    for (int32_t i1 = 0; i1 < 8; i1++) {
      int32_t i = i0 * 8 + i1;
      if (i >= C1_dimension)
        continue;


    for (int32_t jB = B2_pos[i]; jB < B2_pos[(i + 1)]; jB++) {
      int32_t j = B2_crd[jB];
      for (int32_t k = 0; k < D2_dimension; k++) {
        int32_t kC = i * C2_dimension + k;
        int32_t kD = j * D2_dimension + k;
        A_vals[jB] = A_vals[jB] + (B_vals[jB] * C_vals[kC]) * D_vals[kD];
      }
    }
  }
  }

  );
}

void sddmm_csr_gpu(COO input_matrix, const std::string& matrix_name, const bool do_verify) {
  EigenCSR A_eigen = to_eigen_csr(input_matrix);
  taco_tensor_t A_taco = to_taco_tensor(A_eigen);
  CSR A_arrays = get_csr_arrays(A_taco);




  EigenCSR B_eigen = to_eigen_csr(input_matrix);
  taco_tensor_t B_taco = to_taco_tensor(B_eigen);


  const int num_cols = 128;
  //const int num_cols = 1024;

  EigenRowMajor D_eigen = gen_row_major_matrix(input_matrix.n, num_cols);
  taco_tensor_t D_taco = to_taco_tensor(D_eigen);

  EigenRowMajor C_eigen = gen_row_major_matrix(input_matrix.m, num_cols);
  taco_tensor_t C_taco = to_taco_tensor(C_eigen);

  if (do_verify) {
	
  EigenCSR A_eigen_ref = to_eigen_csr(input_matrix);
  taco_tensor_t A_taco_ref = to_taco_tensor(A_eigen_ref);

        sddmm_csr_gpu_taco(&A_taco, &B_taco, &C_taco, &D_taco);
        sddmm_csr_cpu_taco_ref(&A_taco_ref, &B_taco, &C_taco, &D_taco);
 
 double* restrict A_vals = (double*)(A_taco_ref.vals);

        std::cout << "taco_cpu vs taco: " << compare_csr_val(A_taco_ref, A_taco) << std::endl;

        sddmm_csr_gpu_warp(&A_taco, &B_taco, &C_taco, &D_taco);
        std::cout << "taco_cpu vs taco_warp: " << compare_csr_val(A_taco_ref, A_taco) << std::endl;

        exit(0);

  }

  const int trials = 25;
  std::cout<<"call the RUN_GPU"<<std::endl;
  RUN_GPU(sddmm_csr_gpu_taco(&A_taco, &B_taco, &C_taco, &D_taco);, trials,  
      "sddmm", "gpu", "csr", "taco", matrix_name);
  RUN_GPU(sddmm_csr_gpu_warp(&A_taco, &B_taco, &C_taco, &D_taco);, trials,  
      "sddmm", "gpu", "csr", "taco_warp", matrix_name);
}

#endif
