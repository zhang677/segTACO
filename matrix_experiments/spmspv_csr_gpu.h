#ifndef SPMSPV_CSR_GPU_H
#define SPMSPV_CSR_GPU_H

#include "ds.h"
#include "timers.h"

#include "gpu_kernels.cuh"

void spmspv_csr_cpu_taco_ref(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
        int y1_dimension = (int)(y->dimensions[0]);
  double* restrict y_vals = (double*)(y->vals);
  int A2_dimension = (int)(A->dimensions[1]);
  int* restrict A2_pos = (int*)(A->indices[1][0]);
  int* restrict A2_crd = (int*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  int* restrict x1_pos = (int*)(x->indices[0][0]);
  int* restrict x1_crd = (int*)(x->indices[0][1]);
  double* restrict x_vals = (double*)(x->vals);

  #pragma omp parallel for schedule(static)
  for (int32_t py = 0; py < y1_dimension; py++) {
    y_vals[py] = 0.0;
  }

  for (int32_t jx = x1_pos[0]; jx < x1_pos[1]; jx++) {
    int32_t j = x1_crd[jx];
    for (int32_t iA = A2_pos[j]; iA < A2_pos[(j + 1)]; iA++) {
      int32_t i = A2_crd[iA];
      y_vals[i] = y_vals[i] + A_vals[iA] * x_vals[jx];
    }
  }
}


void spmspv_csr_gpu(COO input_matrix, const std::string& matrix_name, const bool do_verify) {

  // convert CSR to CSC
  int tmp = input_matrix.n; input_matrix.n = input_matrix.m; input_matrix.m = tmp;
  for(long i=0; i<input_matrix.nnz; i++) {
	  int tmp = input_matrix.rows[i];
	  input_matrix.rows[i] = input_matrix.cols[i];
	  input_matrix.cols[i] = tmp;
  } 

  EigenCSR A_eigen_t = to_eigen_csr(input_matrix);
  taco_tensor_t A_taco_t = to_taco_tensor(A_eigen_t);

  EigenVector y_eigen = gen_vector(input_matrix.n);
  taco_tensor_t y_taco = to_taco_tensor(y_eigen);

   // setup input vector

  taco_tensor_t x_taco;
  x_taco.indices = new uint8_t**[1*4];
  x_taco.indices[0] = new uint8_t*[2*4];
  x_taco.indices[0][0] = (uint8_t*)malloc(sizeof(int)*input_matrix.m);
  x_taco.indices[0][1] = (uint8_t*)malloc(sizeof(int)*input_matrix.m);
  x_taco.vals = (uint8_t*)malloc(sizeof(double)*input_matrix.m);
  int* restrict x1_pos = (int*)(x_taco.indices[0][0]);
  int* restrict x1_crd = (int*)(x_taco.indices[0][1]);
  double* restrict x_vals = (double*)(x_taco.vals);
  for(int i=0;i<input_matrix.m; i++) {
      x1_crd[i] = i;     
  }
  for(int i=0;i<input_matrix.m; i++) {
      int k = rand()%input_matrix.m;
      int tmp = x1_crd[k]; x1_crd[k] = x1_crd[i]; x1_crd[i] = tmp;
      x_vals[i] = (double)rand() / RAND_MAX;
  }
  double sparsity = 0.1;
  x1_pos[0] = 0;
  x1_pos[1] = (double)input_matrix.m * sparsity;

   if (do_verify) {
  
  EigenVector y_eigen_ref = gen_vector(input_matrix.n);
  taco_tensor_t y_taco_ref = to_taco_tensor(y_eigen_ref);

 spmspv_csr_gpu_taco(&y_taco, &A_taco_t, &x_taco);
 spmspv_csr_cpu_taco_ref(&y_taco_ref, &A_taco_t, &x_taco);

    std::cout << "taco_unscheduled vs taco: " << compare_vectors(y_taco_ref, y_taco) << std::endl;
	exit(0);
  
    }

  const int trials = 25;

  RUN_GPU(spmspv_csr_gpu_taco(&y_taco, &A_taco_t, &x_taco);, 
      trials, "spmspv", "gpu", "csr", "taco", matrix_name);

}

#endif
