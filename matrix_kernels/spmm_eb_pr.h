#include "gpu_library.h"
#include "stdio.h"

#define  kassert_eq( X,Y,b,e,t ) if ( X!=Y)  \
    printf("tid %d: %d, %d, %d, %d, %d\n", threadIdx.x, X, Y, b, e, t);\

#define  kassert_lt( X,Y ) if (X>=Y)  \
    printf("(tidx %d, tidy %d): %d, %d\n", threadIdx.x, threadIdx.y, X, Y);\
    return ;\

__global__
void spmm_eb_pr_taco_kernel(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
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

void spmm_eb_pr_taco(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
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
  spmm_eb_pr_taco_kernel<<<(A2_pos[A1_dimension] + 255) / 256, (32 * 8)>>>(A, B, C, i_blockStarts);
);
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
/*
template <int CoarsenFactor>
__global__ void csrspmm_non_transpose_parreduce_nnzbalance_kernel(){
    const int M = global_A1_dimension;
    const int N = global_B2_dimension;
    const int K = global_B1_dimension;

    int* __restrict__ csr_indptr = global_A2_pos_device;
    int* __restrict__ csr_indices = global_A2_crd_device;
    float* __restrict__ csr_data = global_A_vals_device;
    float* __restrict__ B = global_B_vals_device;
    float* __restrict__ C = global_C_vals_device;

    const int nnz = csr_indptr[M];

    int lane_id = (threadIdx.x & (32 - 1));
    int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int nz_start = Nnzdim_warp_id * 32;
    int stride = gridDim.x * (blockDim.y * 32);

    // get the dense column offset
    int col_offset = blockIdx.y * CoarsenFactor;
    int ldB = K;
    int ldC = M;
    const float *B_panels[CoarsenFactor];
    float *C_panels[CoarsenFactor];
  #pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      B_panels[i] = B + (col_offset + i) * ldB;
      C_panels[i] = C + (col_offset + i) * ldC;
    }

    int k;
    float v;
    float c[CoarsenFactor] = {0};

    if (col_offset >= N)
      return;
    if (col_offset + CoarsenFactor >= N)
      goto Ndim_Residue;

    for (int nz_id = nz_start + lane_id;
        nz_id < nnz + lane_id; // make sure NO warp loop-divergence
        nz_id += stride) {
      int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);

      if (nz_id < nnz) {
        k = csr_indices[nz_id];
        v = __guard_load_default_one<float>(csr_data, nz_id);
      } else {
        k = 0;
        v = 0.0f;
      }

  // load B-elements in vector-type
  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] = v * B_panels[i][k];
      }

      // reduction
      int row_intv =
          __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
      if (row_intv == 0) {
  // if all non-zeros in this warp belong to the same row, use a simple reduction
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SHFL_DOWN_REDUCE(c[i]);
        }
        if (lane_id == 0) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_panels[i] + row, c[i]);
          }
        }
      } else {
        // if non-zeros belong to different rows, use a parallel-scan primitive
        // thread that holds the start of each segment are responsible for writing
        // results
        bool is_seg_start =
            ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
        float tmpv;
        int tmpr;
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
        }
        if (is_seg_start) {
  // atomic add has no vector-type form.
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_panels[i] + row, c[i]);
          }
        }
      }
    }
    return;
  Ndim_Residue:
    int valid_lane_num = N - col_offset;

    for (int nz_id = nz_start + lane_id;
        nz_id < nnz + lane_id; // make sure NO warp loop-divergence
        nz_id += stride) {
      int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);

      if (nz_id < nnz) {
        k = csr_indices[nz_id];
        v = __guard_load_default_one<float>(csr_data, nz_id);
      } else {
        k = 0;
        v = 0.0f;
      }

  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          c[i] = B_panels[i][k] * v;
        }
      }

      // reduction
      int row_intv =
          __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
      if (row_intv == 0) {
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SHFL_DOWN_REDUCE(c[i]);
        }
        if (lane_id == 0) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_panels[i] + row, c[i]);
            }
          }
        }
      } else {
        bool is_seg_start =
            ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
        float tmpv;
        int tmpr;
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
        }
        if (is_seg_start) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_panels[i] + row, c[i]);
            }
          }
        }
      }
    }
    return;
}


void spmm_eb_pr(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
    const int M = (int)(C->dimensions[0]);
    const int N = (int)(C->dimensions[1]);
    int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
    const int nnz = A2_pos[M];
    copy_to_device_spmm(C, A, B);
    // factor of thread coarsening
    int coarsen_factor = (N >= 128) ? 4 : (N >= 64) ? 2 : 1;
    // number of parallel warps along M-dimension
    const int segreduce_size_per_warp = 32;
    int Nnzdim_worker = CEIL(nnz, segreduce_size_per_warp);
    // partition large-N and map to blockdim.y to help cache performance
    int Ndim_threadblock = CEIL(N, coarsen_factor);

    int ref_warp_per_tb = RefThreadPerBlock / 32;
    int Nnzdim_warp_per_tb = ref_warp_per_tb;

    // total number of warps
    int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
    int gridDimY = Ndim_threadblock;
    dim3 gridDim(gridDimX, gridDimY, 1);
    dim3 blockDim(32, Nnzdim_warp_per_tb, 1);

    if (coarsen_factor == 4) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
            csrspmm_non_transpose_parreduce_nnzbalance_kernel<4><<<gridDim, blockDim>>>();
        );
    } else if (coarsen_factor == 2) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
      csrspmm_non_transpose_parreduce_nnzbalance_kernel<2><<<gridDim, blockDim>>>();
              );
    } else {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
      csrspmm_non_transpose_parreduce_nnzbalance_kernel<1><<<gridDim, blockDim>>>();
              );
    }
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
*/

template <typename access_t>
__global__ void csrspmm_parreduce_nnzbalance_kernel() {
  constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);
    const int M = global_A1_dimension;
    const int N = global_B2_dimension;

    int* __restrict__ csr_indptr = global_A2_pos_device;
    int* __restrict__ csr_indices = global_A2_crd_device;
    float* __restrict__ csr_data = global_A_vals_device;
    float* __restrict__ B = global_B_vals_device;
    float* __restrict__ C = global_C_vals_device;

    const int nnz = csr_indptr[M];

  int lane_id = (threadIdx.x & (32 - 1));
  int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  int nz_start = Nnzdim_warp_id * 32;
  int stride = gridDim.x * (blockDim.y * 32);

  // get the dense column offset
  int col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
  const float *B_panel = B + col_offset;
  float *C_panel = C + col_offset;
  int ldB = N;
  int ldC = N;

  int k;
  float v;
  float c[CoarsenFactor] = {0};
  float buffer[CoarsenFactor] = {0};

  if (col_offset >= N)
    return;
  if (col_offset + CoarsenFactor >= N)
    goto Ndim_Residue;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);

    if (nz_id < nnz) {
      k = csr_indices[nz_id];
      v = __guard_load_default_one<float>(csr_data, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

    // load B-elements in vector-type
    *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      c[i] = buffer[i] * v;
    }

    // reduction
    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
// if all non-zeros in this warp belong to the same row, use a simple reduction
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    } else {
      // if non-zeros belong to different rows, use a parallel-scan primitive
      // thread that holds the start of each segment are responsible for writing
      // results
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
// atomic add has no vector-type form.
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_panel + row * ldC + i, c[i]);
        }
      }
    }
  }
  return;
Ndim_Residue:
  int valid_lane_num = N - col_offset;

  for (int nz_id = nz_start + lane_id;
       nz_id < nnz + lane_id; // make sure NO warp loop-divergence
       nz_id += stride) {
    int row = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);

    if (nz_id < nnz) {
      k = csr_indices[nz_id];
      v = __guard_load_default_one<float>(csr_data, nz_id);
    } else {
      k = 0;
      v = 0.0f;
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        c[i] = B_panel[k * ldB + i] * v;
      }
    }

    // reduction
    int row_intv =
        __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
    if (row_intv == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i]);
      }
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    } else {
      bool is_seg_start =
          ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
      float tmpv;
      int tmpr;
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
      }
      if (is_seg_start) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    }
  }
  return;
}

void spmm_eb_pr(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
    const int M = (int)(C->dimensions[0]);
    const int N = (int)(C->dimensions[1]);
    int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
    const int nnz = A2_pos[M];
    copy_to_device_spmm(C, A, B);  

      // factor of thread coarsening
  int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  // number of parallel warps along M-dimension
  const int segreduce_size_per_warp = 32;
  int Nnzdim_worker = M; // CEIL(spmatA.nnz, segreduce_size_per_warp);
  // partition large-N and map to blockdim.y to help cache performance
  int Ndim_threadblock = CEIL(N, 32);
  int Ndim_warp_per_tb = min(N, 32) / coarsen_factor;

  int ref_warp_per_tb = RefThreadPerBlock / 32;
  int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb); 

  // total number of warps
  int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
  int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(Ndim_warp_per_tb * 32, Nnzdim_warp_per_tb, 1);

      if (coarsen_factor == 4) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
            csrspmm_parreduce_nnzbalance_kernel<float4><<<gridDim, blockDim>>>();
        );
    } else if (coarsen_factor == 2) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
      csrspmm_parreduce_nnzbalance_kernel<float2><<<gridDim, blockDim>>>();
              );
    } else {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
      csrspmm_parreduce_nnzbalance_kernel<float><<<gridDim, blockDim>>>();
              );
    }
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}

template<int Nnz, int c, int group_size>
__global__
void kernel_taco_eb_pr_256(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = global_A1_dimension;
  int* __restrict__ A2_pos = global_A2_pos_device;
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;
  float* __restrict__ C_vals = global_C_vals_device;
  int32_t block = blockIdx.x;
  int32_t fpos1 = (threadIdx.x % (Nnz));
  int32_t ko = (threadIdx.x / Nnz);
  if (threadIdx.x >= 256) {
    return;
  }
  for (int32_t ki = 0; ki < c ; ki++) {
    int32_t k = ko * c + ki;
    if (k >= B2_dimension)
      break;

    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fposA = block * Nnz + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    float tmp_val = 0.0;
    if (fposA >= A2_pos[A1_dimension]) tmp_val = 0.0;
    else {
      int32_t fposA = block * Nnz + fpos1;
      int32_t f = A2_crd[fposA];
      int32_t kB = f * B2_dimension + k;
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      tmp_val = A_vals[fposA] * B_vals[kB];
    }
    int32_t kC = i * C2_dimension + k;
    segReduceGroup<float,group_size>(C_vals, kC, tmp_val);
  }
}

void spmm_eb_pr_256_32(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 256, ((A2_pos[A1_dimension] + 255) / 256));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<256,4,32><<<(A2_pos[A1_dimension] + 255) / 256, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_pr_256_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 256, ((A2_pos[A1_dimension] + 255) / 256));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<256,4,16><<<(A2_pos[A1_dimension] + 255) / 256, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_pr_256_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 256, ((A2_pos[A1_dimension] + 255) / 256));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<256,4,8><<<(A2_pos[A1_dimension] + 255) / 256, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_pr_256_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 256, ((A2_pos[A1_dimension] + 255) / 256));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<256,4,4><<<(A2_pos[A1_dimension] + 255) / 256, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}

void spmm_eb_pr_128_32(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 127) / 128 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 128, (int32_t) 128, ((A2_pos[A1_dimension] + 127) / 128));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<128,2,32><<<(A2_pos[A1_dimension] + 127) / 128, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_pr_128_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 127) / 128 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 128, (int32_t) 128, ((A2_pos[A1_dimension] + 127) / 128));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<128,2,16><<<(A2_pos[A1_dimension] + 127) / 128, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_pr_128_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 127) / 128 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 128, (int32_t) 128, ((A2_pos[A1_dimension] + 127) / 128));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<128,2,8><<<(A2_pos[A1_dimension] + 127) / 128, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_pr_128_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 127) / 128 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 128, (int32_t) 128, ((A2_pos[A1_dimension] + 127) / 128));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<128,2,4><<<(A2_pos[A1_dimension] + 127) / 128, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}

void spmm_eb_pr_64_32(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 63) / 64 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 64, (int32_t) 64, ((A2_pos[A1_dimension] + 63) / 64));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<64,1,32><<<(A2_pos[A1_dimension] + 63) / 64, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_pr_64_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 63) / 64 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 64, (int32_t) 64, ((A2_pos[A1_dimension] + 63) / 64));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<64,1,16><<<(A2_pos[A1_dimension] + 63) / 64, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_pr_64_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 63) / 64 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 64, (int32_t) 64, ((A2_pos[A1_dimension] + 63) / 64));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<64,1,8><<<(A2_pos[A1_dimension] + 63) / 64, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_pr_64_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 63) / 64 + 1)));
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 64, (int32_t) 64, ((A2_pos[A1_dimension] + 63) / 64));
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    kernel_taco_eb_pr_256<64,1,4><<<(A2_pos[A1_dimension] + 63) / 64, 256>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
template <typename access_t, int SearchSeg>
__global__ void pr_eb_search_kernel(int32_t* segStarts) {
    constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);
      const int M = global_A1_dimension;
      const int N = global_B2_dimension;
  
      int* __restrict__ csr_indptr = global_A2_pos_device;
      int* __restrict__ csr_indices = global_A2_crd_device;
      float* __restrict__ csr_data = global_A_vals_device;
      float* __restrict__ B = global_B_vals_device;
      float* __restrict__ C = global_C_vals_device;
  
      const int nnz = csr_indptr[M];
  
    int lane_id = (threadIdx.x & (32 - 1));
    int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int nz_start = Nnzdim_warp_id * 32;
    int stride = gridDim.x * (blockDim.y * 32);
  
    // get the dense column offset
    int col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
    const float *B_panel = B + col_offset;
    float *C_panel = C + col_offset;
    int ldB = N;
    int ldC = N;
  
    int k;
    float v;
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor] = {0};
  
    if (col_offset >= N)
      return;
    if (col_offset + CoarsenFactor >= N)
      goto Ndim_Residue;
  
    for (int nz_id = nz_start + lane_id;
         nz_id < nnz + lane_id; // make sure NO warp loop-divergence
         nz_id += stride) {
        int seg_num = nz_id / SearchSeg ;
        int p_begin = segStarts[seg_num];
        int p_end = segStarts[seg_num+1];
        //int row_tmp = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);
        int row = taco_binarySearchBefore(csr_indptr, p_begin, p_end, nz_id);
        row = TACO_MIN(row,M-1);
        //kassert_eq(row,row_tmp,p_begin,p_end,nz_id);
  
      if (nz_id < nnz) {
        k = csr_indices[nz_id];
        v = __guard_load_default_one<float>(csr_data, nz_id);
      } else {
        k = 0;
        v = 0.0f;
      }
  
      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);
  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] = buffer[i] * v;
      }
  
      // reduction
      int row_intv =
          __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
      if (row_intv == 0) {
  // if all non-zeros in this warp belong to the same row, use a simple reduction
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SHFL_DOWN_REDUCE(c[i]);
        }
        if (lane_id == 0) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      } else {
        // if non-zeros belong to different rows, use a parallel-scan primitive
        // thread that holds the start of each segment are responsible for writing
        // results
        bool is_seg_start =
            ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
        float tmpv;
        int tmpr;
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
        }
        if (is_seg_start) {
  // atomic add has no vector-type form.
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    }
    return;
  Ndim_Residue:
    int valid_lane_num = N - col_offset;
  
    for (int nz_id = nz_start + lane_id;
         nz_id < nnz + lane_id; // make sure NO warp loop-divergence
         nz_id += stride) {
        int seg_num = nz_id / SearchSeg ;
        int p_begin = segStarts[seg_num];
        int p_end = segStarts[seg_num+1];
        //int row_tmp = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);
        int row = taco_binarySearchBefore(csr_indptr, p_begin, p_end, nz_id);
        row = TACO_MIN(row,M-1);
        //kassert_eq(row, row_tmp,p_begin,p_end,nz_id);
        


      if (nz_id < nnz) {
        k = csr_indices[nz_id];
        v = __guard_load_default_one<float>(csr_data, nz_id);
      } else {
        k = 0;
        v = 0.0f;
      }
  
  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          c[i] = B_panel[k * ldB + i] * v;
        }
      }
  
      // reduction
      int row_intv =
          __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
      if (row_intv == 0) {
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SHFL_DOWN_REDUCE(c[i]);
        }
        if (lane_id == 0) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_panel + row * ldC + i, c[i]);
            }
          }
        }
      } else {
        bool is_seg_start =
            ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
        float tmpv;
        int tmpr;
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
        }
        if (is_seg_start) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_panel + row * ldC + i, c[i]);
            }
          }
        }
      }
    }
    return;
  }
template <typename access_t, int SearchSeg>
__global__ void pr_eb_cache_search_kernel(int32_t* segStarts, int seg_share_size) {
    constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);
      const int M = global_A1_dimension;
      const int N = global_B2_dimension;
  
      int* __restrict__ csr_indptr = global_A2_pos_device;
      int* __restrict__ csr_indices = global_A2_crd_device;
      float* __restrict__ csr_data = global_A_vals_device;
      float* __restrict__ B = global_B_vals_device;
      float* __restrict__ C = global_C_vals_device;
  
      const int nnz = csr_indptr[M];
  
    int lane_id = (threadIdx.x & (32 - 1));
    int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int nz_start = Nnzdim_warp_id * 32;
    int stride = gridDim.x * (blockDim.y * 32);

    extern __shared__ int shared_mem[];
    int* workspace = &shared_mem[threadIdx.y*seg_share_size];
    //kassert_lt(threadIdx.y, blockDim.y);
    // get the dense column offset
    int col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
    const float *B_panel = B + col_offset;
    float *C_panel = C + col_offset;
    int ldB = N;
    int ldC = N;
  
    int k;
    float v;
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor] = {0};


    if (col_offset >= N)
      return;
    if (col_offset + CoarsenFactor >= N)
      goto Ndim_Residue;
    
    for (int base_nz_id = nz_start ; base_nz_id < nnz; base_nz_id += stride) {
        int seg_num = base_nz_id / SearchSeg;
        int p_begin = segStarts[seg_num];
        int p_end = segStarts[seg_num+1];
        for (int i=threadIdx.x+p_begin ; i<=threadIdx.x+p_end; i+=blockDim.x){
          workspace[i-p_begin] = csr_indptr[i];
        }
  	    __syncthreads();// [0,nnz/32]; [p_begin, p_end]

        int nz_id = base_nz_id + lane_id;
        int row = taco_binarySearchBefore(workspace, 0, p_end-p_begin, nz_id)+p_begin;
        //int row_valid = taco_binarySearchBefore(csr_indptr, p_begin, p_end, nz_id);
        //int row_tmp = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);
        //row = TACO_MIN(row,M-1);
        //kassert_eq(row,row_tmp,p_begin,p_end,nz_id);
  
      if (nz_id < nnz) {
        k = csr_indices[nz_id];
        v = __guard_load_default_one<float>(csr_data, nz_id);
      } else {
        k = 0;
        v = 0.0f;
      }
  
      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);
  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] = buffer[i] * v;
      }
  
      // reduction
      int row_intv =
          __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
      if (row_intv == 0) {
  // if all non-zeros in this warp belong to the same row, use a simple reduction
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SHFL_DOWN_REDUCE(c[i]);
        }
        if (lane_id == 0) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      } else {
        // if non-zeros belong to different rows, use a parallel-scan primitive
        // thread that holds the start of each segment are responsible for writing
        // results
        bool is_seg_start =
            ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
        float tmpv;
        int tmpr;
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
        }
        if (is_seg_start) {
  // atomic add has no vector-type form.
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    }
    return;
  Ndim_Residue:
    int valid_lane_num = N - col_offset;
  
    for (int base_nz_id = nz_start ; base_nz_id < nnz; base_nz_id += stride) {
        int seg_num = base_nz_id / SearchSeg; 
        int p_begin = segStarts[seg_num];
        int p_end = segStarts[seg_num+1];
      for (int i=threadIdx.x+p_begin ; i<=threadIdx.x+p_end; i+=blockDim.x){
        workspace[i-p_begin] = csr_indptr[i];
      }
  	    __syncthreads();

        int nz_id = base_nz_id + lane_id;
        int row = taco_binarySearchBefore(workspace, 0, p_end-p_begin, nz_id)+p_begin;
        //int row_valid = taco_binarySearchBefore(csr_indptr, p_begin, p_end, nz_id);
        //int row_tmp = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);
        //row = TACO_MIN(row,M-1);
        //kassert_eq(row,row_tmp,p_begin,p_end,nz_id);
        


      if (nz_id < nnz) {
        k = csr_indices[nz_id];
        v = __guard_load_default_one<float>(csr_data, nz_id);
      } else {
        k = 0;
        v = 0.0f;
      }
  
  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          c[i] = B_panel[k * ldB + i] * v;
        }
      }
  
      // reduction
      int row_intv =
          __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
      if (row_intv == 0) {
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SHFL_DOWN_REDUCE(c[i]);
        }
        if (lane_id == 0) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_panel + row * ldC + i, c[i]);
            }
          }
        }
      } else {
        bool is_seg_start =
            ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
        float tmpv;
        int tmpr;
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
        }
        if (is_seg_start) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_panel + row * ldC + i, c[i]);
            }
          }
        }
      }
    }
    return;
  }
template <typename access_t, int SearchSeg>
__global__ void pr_eb_vectorization_search_kernel(uint32_t* VecsegStarts, uint32_t* VecsegStarts_odd) {
    constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);
      const int M = global_A1_dimension;
      const int N = global_B2_dimension;
  
      int* __restrict__ csr_indptr = global_A2_pos_device;
      int* __restrict__ csr_indices = global_A2_crd_device;
      float* __restrict__ csr_data = global_A_vals_device;
      float* __restrict__ B = global_B_vals_device;
      float* __restrict__ C = global_C_vals_device;
  
      const int nnz = csr_indptr[M];
  
    int lane_id = (threadIdx.x & (32 - 1));
    int Nnzdim_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    int nz_start = Nnzdim_warp_id * 32;
    int stride = gridDim.x * (blockDim.y * 32);


    // get the dense column offset
    int col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
    const float *B_panel = B + col_offset;
    float *C_panel = C + col_offset;
    int ldB = N;
    int ldC = N;
  
    int k;
    float v;
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor] = {0};
    unsigned int vec;

    if (col_offset >= N)
      return;
    if (col_offset + CoarsenFactor >= N)
      goto Ndim_Residue;
    
    for (int nz_id = nz_start + lane_id ; nz_id < nnz + lane_id; nz_id += stride) {
        int seg_num = nz_id / SearchSeg;
        if (seg_num & 1) {
          vec = VecsegStarts_odd[seg_num >> 1];
        } else {
          vec = VecsegStarts[seg_num >> 1];
        }
        int p_begin = (vec & 0xffff0000) >> 16;
        int p_end = vec & 0x0000ffff;
        int row = taco_binarySearchBefore(csr_indptr, p_begin, p_end, nz_id);
        //int row_tmp = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);
        row = TACO_MIN(row,M-1);   
        //kassert_eq(row,row_tmp,p_begin,p_end,nz_id);

             
  
      if (nz_id < nnz) {
        k = csr_indices[nz_id];
        v = __guard_load_default_one<float>(csr_data, nz_id);
      } else {
        k = 0;
        v = 0.0f;
      }
  
      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);
  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] = buffer[i] * v;
      }
  
      // reduction
      int row_intv =
          __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
      if (row_intv == 0) {
  // if all non-zeros in this warp belong to the same row, use a simple reduction
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SHFL_DOWN_REDUCE(c[i]);
        }
        if (lane_id == 0) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      } else {
        // if non-zeros belong to different rows, use a parallel-scan primitive
        // thread that holds the start of each segment are responsible for writing
        // results
        bool is_seg_start =
            ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
        float tmpv;
        int tmpr;
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
        }
        if (is_seg_start) {
  // atomic add has no vector-type form.
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            atomicAdd(C_panel + row * ldC + i, c[i]);
          }
        }
      }
    }
    return;
  Ndim_Residue:
    int valid_lane_num = N - col_offset;
  
    for (int nz_id = nz_start + lane_id ; nz_id < nnz + lane_id; nz_id += stride) {
        int seg_num = nz_id / SearchSeg; 
        if (seg_num & 1) {
          vec = VecsegStarts_odd[seg_num >> 1];
        } else {
          vec = VecsegStarts[seg_num >> 1];
        }
        int p_begin = (vec & 0xffff0000) >> 16;
        int p_end = vec & 0x0000ffff;
        int row = taco_binarySearchBefore(csr_indptr, p_begin, p_end, nz_id);
        //int row_tmp = binary_search_segment_number<int>(csr_indptr, M, nnz, nz_id);
        row = TACO_MIN(row,M-1);   
        //kassert_eq(row,row_tmp,p_begin,p_end,nz_id); 
        


      if (nz_id < nnz) {
        k = csr_indices[nz_id];
        v = __guard_load_default_one<float>(csr_data, nz_id);
      } else {
        k = 0;
        v = 0.0f;
      }
  
  #pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          c[i] = B_panel[k * ldB + i] * v;
        }
      }
  
      // reduction
      int row_intv =
          __shfl_sync(FULLMASK, row, (32 - 1)) - __shfl_sync(FULLMASK, row, 0);
      if (row_intv == 0) {
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SHFL_DOWN_REDUCE(c[i]);
        }
        if (lane_id == 0) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_panel + row * ldC + i, c[i]);
            }
          }
        }
      } else {
        bool is_seg_start =
            ((__shfl_up_sync(FULLMASK, row, 1) != row) || (lane_id == 0));
        float tmpv;
        int tmpr;
  #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          SEG_SHFL_SCAN(c[i], tmpv, row, tmpr);
        }
        if (is_seg_start) {
  #pragma unroll
          for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
              atomicAdd(C_panel + row * ldC + i, c[i]);
            }
          }
        }
      }
    }
    return;
}
  void spmm_eb_pr_search(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
    const int M = (int)(C->dimensions[0]);
    const int N = (int)(C->dimensions[1]);
    int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
    const int nnz = A2_pos[M];
    copy_to_device_spmm(C, A, B);  

  // factor of thread coarsening
  int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  // number of parallel warps along M-dimension
  const int segreduce_size_per_warp = 32;
  const int Nnzdim_worker = M; // CEIL(spmatA.nnz, segreduce_size_per_warp);
  // partition large-N and map to blockdim.y to help cache performance
  const int Ndim_threadblock = CEIL(N, 32);
  const int Ndim_warp_per_tb = min(N, 32) / coarsen_factor;

  const int ref_warp_per_tb = RefThreadPerBlock / 32;
  const int Nnzdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

  // total number of warps
  const int gridDimX = CEIL(Nnzdim_worker, Nnzdim_warp_per_tb);
  const int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(Ndim_warp_per_tb * 32, Nnzdim_warp_per_tb, 1);
  const int SearchSeg = 32; // Hyperparameter
  int32_t* segStarts = 0; //
  gpuErrchk(cudaMallocManaged((void**)&segStarts, sizeof(int32_t) * (CEIL(nnz, SearchSeg) + 1))); // element-balance
  gpuErrchk(cudaGetLastError());
  //cudaFuncSetCacheConfig(pr_eb_cache_search_kernel<float4, SearchSeg>, cudaFuncCachePreferL1);
  uint32_t* VecsegStarts = 0;
  const int len_ori = CEIL(nnz, SearchSeg) + 1;
  const int len_vec = CEIL(len_ori,2);
  uint32_t* VecsegStarts_odd = 0;
  const int len_ori_odd = len_ori - 1;
  const int len_vec_odd = CEIL(len_ori_odd,2);
  gpuErrchk(cudaMallocManaged((void**)&VecsegStarts, sizeof(int32_t) * len_vec));
  gpuErrchk(cudaMallocManaged((void**)&VecsegStarts_odd, sizeof(uint32_t) * len_vec_odd));
  gpuErrchk(cudaGetLastError());
      if (coarsen_factor == 4) {
        const int share_size = 1<<8;
        const unsigned int getFirst = 0xffff0000;
        const unsigned int getSecond = 0x0000ffff;
        TIME_GPU(
            segStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, segStarts, (int32_t) 0, M, (int32_t) SearchSeg, (int32_t) SearchSeg, CEIL(nnz, SearchSeg)); // spmm_csr_eb_pr
            //VecsegStarts = taco_vectorizeSearchBeforeBlockLaunch((uint32_t* )segStarts, VecsegStarts, len_ori, len_vec);
            //VecsegStarts_odd = taco_vectorizeSearchBeforeBlockLaunch((uint32_t* )(segStarts+1), VecsegStarts_odd, len_ori_odd, len_vec_odd);
            /*
            for (int i=0;i<len_vec_odd;i++) {
              unsigned int first = (VecsegStarts_odd[i] & getFirst) >> 16;
              unsigned int second = VecsegStarts_odd[i] & getSecond;
              if((i*2+1 < len_ori) && (i*2+2 < len_ori) && first!=segStarts[i*2+1] || second!=segStarts[(i+1)*2]) {
                printf("Wrong! %d, %d, %d \n",first, second, i);
              }
              if((i*2+1 >= len_ori) || (i*2+2 >= len_ori)) {
                printf("%d, %d, %d \n",first, second,i);
              }
            }
            printf("\n");
            for (int i=0;i<len_vec;i++) {
              unsigned int first = (VecsegStarts[i] & getFirst) >> 16;
              unsigned int second = VecsegStarts[i] & getSecond;
              if((i*2 < len_ori) && (i*2+1 < len_ori) && first!=segStarts[i*2] || second!=segStarts[i*2+1]) {
                printf("Wrong! %d, %d, %d \n",first, second, i);
              }
              if((i*2  >= len_ori) || (i*2+1 >= len_ori)) {
                printf("%d, %d, %d \n",first, second, i);
              }
            }
            printf("\n");
            */
            // Can be bounded by max(segStarts[i+1]-segStarts[i])
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
            pr_eb_search_kernel<float4, SearchSeg><<<gridDim, blockDim>>>(segStarts);
            //pr_eb_cache_search_kernel<float4, SearchSeg><<<gridDim, blockDim, share_size*sizeof(int)>>>(segStarts, share_size/Nnzdim_warp_per_tb);
            //pr_eb_vectorization_search_kernel<float4, SearchSeg><<<gridDim, blockDim>>>(VecsegStarts, VecsegStarts_odd);
        );
    } else if (coarsen_factor == 2) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
      csrspmm_parreduce_nnzbalance_kernel<float2><<<gridDim, blockDim>>>();
              );
    } else {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
      csrspmm_parreduce_nnzbalance_kernel<float><<<gridDim, blockDim>>>();
              );
    }
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(segStarts);
  cudaFree(VecsegStarts);
  cudaDeviceSynchronize(); 
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}