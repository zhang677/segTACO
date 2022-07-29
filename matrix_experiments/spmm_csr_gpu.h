#ifndef SPMM_CSR_GPU_H
#define SPMM_CSR_GPU_H

#include "ds.h"
#include "timers.h"

#include "gpu_kernels.cuh"


void spmm_csr_cpu(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  double* restrict C_vals = (double*)(C->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int* restrict A2_pos = (int*)(A->indices[1][0]);
  int* restrict A2_crd = (int*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  double* restrict B_vals = (double*)(B->vals);

  #pragma omp parallel for schedule(static)
  for (int32_t pC = 0; pC < (C2_dimension * C1_dimension); pC++) {
    C_vals[pC] = 0.0;
  }

  for (int32_t ko = 0; ko < ((A1_dimension * B2_dimension + 7) / 8); ko++) {
    for (int32_t ki = 0; ki < 8; ki++) {
      int32_t io = ko * 8 + ki;
      int32_t i = io / B2_dimension;
      if (i >= A1_dimension)
        continue;

      int32_t k = io % B2_dimension;
      int32_t iC = k * C1_dimension + i;
      if (k >= B2_dimension)
        continue;

      for (int32_t jpos1 = 0; jpos1 < 32; jpos1++) {
        for (int32_t jpos0 = (A2_pos[i] / 32); jpos0 < ((A2_pos[(i + 1)] + 31) / 32); jpos0++) {
          int32_t jposA = jpos0 * 32 + jpos1;
          if (jposA < A2_pos[i] || jposA >= A2_pos[(i + 1)])
            continue;

          int32_t j = A2_crd[jposA];
          int32_t kB = j * B2_dimension + k;
          C_vals[iC] = C_vals[iC] + A_vals[jposA] * B_vals[kB];
        }
      }
    }
  }
}

void spmm_cpu_row(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  double* restrict C_vals = (double*)(C->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int* restrict A2_pos = (int*)(A->indices[1][0]);
  int* restrict A2_crd = (int*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  double* restrict B_vals = (double*)(B->vals);

  #pragma omp parallel for schedule(static)
  for (int32_t pC = 0; pC < (C1_dimension * C2_dimension); pC++) {
    C_vals[pC] = 0.0;
  }

  #pragma omp parallel for schedule(runtime)
  for (int32_t i0 = 0; i0 < ((A1_dimension + 15) / 16); i0++) {
    for (int32_t i1 = 0; i1 < 16; i1++) {
      int32_t i = i0 * 16 + i1;
      if (i >= A1_dimension)
        continue;

      for (int32_t jpos0 = A2_pos[i] / 8; jpos0 < ((A2_pos[(i + 1)] + 7) / 8); jpos0++) {
        if (jpos0 * 8 < A2_pos[i] || (jpos0 * 8 + 8) + ((jpos0 * 8 + 8) - jpos0 * 8) >= A2_pos[(i + 1)]) {
          for (int32_t k = 0; k < B2_dimension; k++) {
            int32_t kC = i * C2_dimension + k;
            for (int32_t jpos1 = 0; jpos1 < 8; jpos1++) {
              int32_t jposA = jpos0 * 8 + jpos1;
              if (jposA < A2_pos[i] || jposA >= A2_pos[(i + 1)])
                continue;

              int32_t j = A2_crd[jposA];
              int32_t kB = j * B2_dimension + k;
              C_vals[kC] = C_vals[kC] + A_vals[jposA] * B_vals[kB];
            }
          }
        }
        else {
          #pragma clang loop interleave(enable) vectorize(enable)
          for (int32_t k = 0; k < B2_dimension; k++) {
            int32_t kC = i * C2_dimension + k;
            for (int32_t jpos1 = 0; jpos1 < 8; jpos1++) {
              int32_t jposA = jpos0 * 8 + jpos1;
              int32_t j = A2_crd[jposA];
              int32_t kB = j * B2_dimension + k;
              C_vals[kC] = C_vals[kC] + A_vals[jposA] * B_vals[kB];
            }
          }
        }
      }
    }
  }
}

void spmm_cpu_naive(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int M = (int)(C->dimensions[0]);
  int N = (int)(C->dimensions[1]);
  float* restrict C_vals = (float*)(C->vals);
  int* restrict A2_pos = (int*)(A->indices[1][0]);
  int* restrict A2_crd = (int*)(A->indices[1][1]);
  float* restrict A_vals = (float*)(A->vals);
  int K = (int)(B->dimensions[0]);
  float* restrict B_vals = (float*)(B->vals);

  for (int i=0; i<M ; i++) {
    for (int j=0 ; j<N ; j++) {
      for (int pos=A2_pos[i] ; pos<A2_pos[i+1] ; pos++) {
        C_vals[i*N + j] += A_vals[pos]*B_vals[A2_crd[pos]*N + j];
      }
    }
  }
}

void spmm_csr_gpu(COO input_matrix, const std::string& matrix_name, const bool do_verify, const int num_cols) {
  EigenCSR A_eigen = to_eigen_csr(input_matrix);
  taco_tensor_t A_taco = to_taco_tensor(A_eigen);

  EigenRowMajor B_eigen = gen_row_major_matrix(input_matrix.n, num_cols);
  taco_tensor_t B_taco = to_taco_tensor(B_eigen);

  //EigenColMajor C_eigen = gen_col_major_matrix(input_matrix.m, num_cols);
  EigenRowMajor C_eigen = gen_row_major_matrix(input_matrix.m, num_cols);
  taco_tensor_t C_taco = to_taco_tensor(C_eigen);
/*
  EigenRowMajor Bt = gen_row_ones(2,3);
  taco_tensor_t Bta = to_taco_tensor(Bt);
  EigenRowMajor Br = gen_col_ones(2,3);
  taco_tensor_t Bra = to_taco_tensor(Br);
  for (int i=0;i<6;i++)
  {
    std::cout<< "row: "<<((double*)Bta.vals)[i]<<"col: "<<((double*)Bra.vals)[i]<<std::endl;
  }
  std::cout << compare_matrices(Bta, Bra) <<std::endl;

  exit(0);
*/
  if (do_verify) {
    //EigenColMajor Cref_eigen = gen_col_major_matrix(input_matrix.m, num_cols);
    EigenRowMajor Cref_eigen = gen_row_major_matrix(input_matrix.m, num_cols);
    taco_tensor_t Cref_taco = to_taco_tensor(Cref_eigen);
    //std::cout<< "c[0],c[1]" << (int)(C_taco.dimensions[0])<<(int)(C_taco.dimensions[1]) <<std::endl;
    //spmm_csr_gpu_warp(&C_taco, &A_taco, &B_taco);
    //spmm_csr_gpu_taco(&C_taco, &A_taco, &B_taco);
    //spmm_eb_pr(&C_taco, &A_taco, &B_taco); // pass
    //spmm_eb_sr(&C_taco, &A_taco, &B_taco); // pass
    //spmm_rb_pr(&C_taco, &A_taco, &B_taco); //pass
    //spmm_rb_sr(&C_taco, &A_taco, &B_taco); // not exactly. Same very wierd bug
    //spmm_rb_sr_taco(&C_taco, &A_taco, &B_taco); // not exactly. Same very wierd bug
    //spmm_rb_pr_taco(&C_taco, &A_taco, &B_taco); //pass
    //spmm_eb_sr_taco(&C_taco, &A_taco, &B_taco); //pass
    //print_coo_matrix(input_matrix, "/home/nfs_data/zhanggh/mytaco/learn-taco/count-dataset/"+matrix_name+"_info.txt");
    //spmm_eb_sr_search(&C_taco, &A_taco, &B_taco);
    spmm_csr_cusparse_row(&Cref_taco, &A_taco, &B_taco);
    //spmm_eb_pr_256(&C_taco, &A_taco, &B_taco);
    //std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    //spmm_eb_sr_256_4(&C_taco, &A_taco, &B_taco);
    //std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    //spmm_rb_pr_256_4(&C_taco, &A_taco, &B_taco);
    spmm_eb_sr_256_4_1(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_eb_sr_256_4_2(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_eb_sr_256_4_4(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_eb_sr_256_8_1(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_eb_sr_256_8_2(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_eb_sr_256_8_4(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_eb_sr_256_16_1(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_eb_sr_256_16_2(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_eb_sr_256_16_4(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    /*
    spmm_rb_sr_256_1_1(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_rb_sr_256_1_2(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_rb_sr_256_1_4(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_rb_sr_256_2_1(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_rb_sr_256_2_2(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_rb_sr_256_2_4(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_rb_sr_256_4_1(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_rb_sr_256_4_2(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_rb_sr_256_4_4(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 spmm_eb_sr_256_4(&C_taco, &A_taco, &B_taco);std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
 */
    /*
  spmm_rb_pr_256_1_4_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_2_4_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_4_4_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_1_8_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_2_8_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_4_8_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_1_8_8(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_2_8_8(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_4_8_8(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_1_16_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_2_16_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_4_16_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_1_16_8(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_2_16_8(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_4_16_8(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_1_16_16(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_2_16_16(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_4_16_16(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_1_32_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_2_32_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_4_32_4(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_1_32_8(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_2_32_8(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_4_32_8(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_1_32_16(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_2_32_16(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_4_32_16(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_1_32_32(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_2_32_32(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
  spmm_rb_pr_256_4_32_32(&C_taco, &A_taco, &B_taco) ;    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    */
    /*
    spmm_eb_pr_256_32(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_256_16(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_256_8(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_256_4(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_128_32(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_128_16(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_128_8(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_128_4(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_64_32(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_64_16(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_64_8(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    spmm_eb_pr_64_4(&C_taco, &A_taco, &B_taco);
    std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    */
    //spmm_rb_sr_256_4(&C_taco, &A_taco, &B_taco);
    //std::cout << "test vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;

    //spmm_eb_pr_taco(&C_taco, &A_taco, &B_taco); //pass
    //spmm_rb_sr_512_64(&C_taco, &A_taco, &B_taco); // not exactly. Very wierd bug
    //spmm_eb_sr_512_64(&C_taco, &A_taco, &B_taco); //pass
    //spmm_rb_pr_256(&C_taco, &A_taco, &B_taco); //pass
    //spmm_eb_pr_256(&C_taco, &A_taco, &B_taco); //pass
    //spmm_cpu_naive(&C_taco, &A_taco, &B_taco);
    
    
    //spmm_cpu_row(&Cref_taco, &A_taco, &B_taco);

    //spmm_csr_gpu_taco(&C_taco, &A_taco, &B_taco);

    //print_matrix(C_taco);
    //spmm_csr_gpu_warp(&C_taco, &A_taco, &B_taco);
    //std::cout << "taco-warp vs cusparse: " << compare_matrices(Cref_taco, C_taco) << std::endl;
    exit(0);
  }

  const int trials = 25;
  int* __restrict__ A2_pos = (int*)(A_taco.indices[1][0]);
  int A1_dimension = (int)(A_taco.dimensions[0]); //nr
  long nnz = A2_pos[A1_dimension];
  long K = C_taco.dimensions[1];
  std::cout << "nnz: " << nnz << " K: " << K << " nnz*K: " << nnz*K << std::endl;

  //RUN_GPU(spmm_eb_pr_256(&C_taco, &A_taco, &B_taco);, trials,  
  //    "spmm", "eb", "pr", "taco", matrix_name);
RUN_GPU( spmm_rb_pr_256_1_4_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune0", matrix_name);
RUN_GPU( spmm_rb_pr_256_2_4_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune1", matrix_name);
RUN_GPU( spmm_rb_pr_256_4_4_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune2", matrix_name);
RUN_GPU( spmm_rb_pr_256_1_8_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune3", matrix_name);
RUN_GPU( spmm_rb_pr_256_2_8_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune4", matrix_name);
RUN_GPU( spmm_rb_pr_256_4_8_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune5", matrix_name);
RUN_GPU( spmm_rb_pr_256_1_8_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune6", matrix_name);
RUN_GPU( spmm_rb_pr_256_2_8_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune7", matrix_name);
RUN_GPU( spmm_rb_pr_256_4_8_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune8", matrix_name);
RUN_GPU( spmm_rb_pr_256_1_16_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune9", matrix_name);
RUN_GPU( spmm_rb_pr_256_2_16_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune10", matrix_name);
RUN_GPU( spmm_rb_pr_256_4_16_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune11", matrix_name);
RUN_GPU( spmm_rb_pr_256_1_16_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune12", matrix_name);
RUN_GPU( spmm_rb_pr_256_2_16_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune13", matrix_name);
RUN_GPU( spmm_rb_pr_256_4_16_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune14", matrix_name);
RUN_GPU( spmm_rb_pr_256_1_16_16(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune15", matrix_name);
RUN_GPU( spmm_rb_pr_256_2_16_16(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune16", matrix_name);
RUN_GPU( spmm_rb_pr_256_4_16_16(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune17", matrix_name);
RUN_GPU( spmm_rb_pr_256_1_32_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune18", matrix_name);
RUN_GPU( spmm_rb_pr_256_2_32_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune19", matrix_name);
RUN_GPU( spmm_rb_pr_256_4_32_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune20", matrix_name);
RUN_GPU( spmm_rb_pr_256_1_32_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune21", matrix_name);
RUN_GPU( spmm_rb_pr_256_2_32_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune22", matrix_name);
RUN_GPU( spmm_rb_pr_256_4_32_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune23", matrix_name);
RUN_GPU( spmm_rb_pr_256_1_32_16(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune24", matrix_name);
RUN_GPU( spmm_rb_pr_256_2_32_16(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune25", matrix_name);
RUN_GPU( spmm_rb_pr_256_4_32_16(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune26", matrix_name);
RUN_GPU( spmm_rb_pr_256_1_32_32(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune27", matrix_name);
RUN_GPU( spmm_rb_pr_256_2_32_32(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune28", matrix_name);
RUN_GPU( spmm_rb_pr_256_4_32_32(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "pr", "tune29", matrix_name);
RUN_GPU( spmm_eb_pr_256_32(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune0", matrix_name);
RUN_GPU( spmm_eb_pr_256_16(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune1", matrix_name);
RUN_GPU( spmm_eb_pr_256_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune2", matrix_name);
RUN_GPU( spmm_eb_pr_256_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune3", matrix_name);
RUN_GPU( spmm_eb_pr_128_32(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune4", matrix_name);
RUN_GPU( spmm_eb_pr_128_16(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune5", matrix_name);
RUN_GPU( spmm_eb_pr_128_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune6", matrix_name);
RUN_GPU( spmm_eb_pr_128_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune7", matrix_name);
RUN_GPU( spmm_eb_pr_64_32(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune8", matrix_name);
RUN_GPU( spmm_eb_pr_64_16(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune9", matrix_name);
RUN_GPU( spmm_eb_pr_64_8(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune10", matrix_name);
RUN_GPU( spmm_eb_pr_64_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "pr", "tune11", matrix_name);
RUN_GPU( spmm_rb_sr_256_1_1(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "sr", "tune0", matrix_name);
RUN_GPU( spmm_rb_sr_256_1_2(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "sr", "tune1", matrix_name);
RUN_GPU( spmm_rb_sr_256_1_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "sr", "tune2", matrix_name);
RUN_GPU( spmm_rb_sr_256_2_1(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "sr", "tune3", matrix_name);
RUN_GPU( spmm_rb_sr_256_2_2(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "sr", "tune4", matrix_name);
RUN_GPU( spmm_rb_sr_256_2_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "sr", "tune5", matrix_name);
RUN_GPU( spmm_rb_sr_256_4_1(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "sr", "tune6", matrix_name);
RUN_GPU( spmm_rb_sr_256_4_2(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "sr", "tune7", matrix_name);
RUN_GPU( spmm_rb_sr_256_4_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "rb", "sr", "tune8", matrix_name);
RUN_GPU( spmm_eb_sr_256_4_1(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "sr", "tune0", matrix_name);
RUN_GPU( spmm_eb_sr_256_4_2(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "sr", "tune1", matrix_name);
RUN_GPU( spmm_eb_sr_256_4_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "sr", "tune2", matrix_name);
RUN_GPU( spmm_eb_sr_256_8_1(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "sr", "tune3", matrix_name);
RUN_GPU( spmm_eb_sr_256_8_2(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "sr", "tune4", matrix_name);
RUN_GPU( spmm_eb_sr_256_8_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "sr", "tune5", matrix_name);
RUN_GPU( spmm_eb_sr_256_16_1(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "sr", "tune6", matrix_name);
RUN_GPU( spmm_eb_sr_256_16_2(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "sr", "tune7", matrix_name);
RUN_GPU( spmm_eb_sr_256_16_4(&C_taco, &A_taco, &B_taco);, trials,"spmm", "eb", "sr", "tune8", matrix_name);
RUN_GPU( spmm_csr_cusparse_row(&C_taco, &A_taco, &B_taco);,trials,"spmm", "eb", "sr", "alg2", matrix_name)      
}

#endif
