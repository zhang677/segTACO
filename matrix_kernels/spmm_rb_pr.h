#include "gpu_library.h"
//#include "gpu_kernels.cuh"
#include "stdio.h"

__global__
void spmm_rb_pr_taco_kernel(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C){
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

void spmm_rb_pr_taco(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
    int C1_dimension = (int)(C->dimensions[0]);
    int C2_dimension = (int)(C->dimensions[1]);
    int A1_dimension = (int)(A->dimensions[0]);
    int B2_dimension = (int)(B->dimensions[1]);
    copy_to_device_spmm(C, A, B); 

    TIME_GPU(
        gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4)); // used with spmm_csr_reduction_scalar
        spmm_rb_pr_taco_kernel<<<(A1_dimension * B2_dimension + 7) / 8, (32 * 8)>>>(A, B, C);
    );
    cudaDeviceSynchronize(); 
    gpuErrchk(cudaGetLastError());
    copy_from_device_spmm(C, A, B);
    free_tensors_spmm();
}

/*
template <int CoarsenFactor>
__global__ void csrspmm_non_transpose_parreduce_rowbalance_kernel() {
    // RFC: Implementing Sparse matrix-vector produce on throughput-oriented
    // processors, SC2009

    int lane_id = (threadIdx.x & (32 - 1));
    int stride = gridDim.x * blockDim.y;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int M = global_A1_dimension;
    const int N = global_B2_dimension;
    const int K = global_B1_dimension;

    int* __restrict__ csr_indptr = global_A2_pos_device;
    int* __restrict__ csr_indices = global_A2_crd_device;
    float* __restrict__ csr_data = global_A_vals_device;
    float* __restrict__ B = global_B_vals_device;
    float* __restrict__ C = global_C_vals_device;
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

    if (col_offset >= N)
        return;
    if (col_offset + CoarsenFactor >= N)
        goto Ndim_Residue;

    for (; row < M; row += stride) {
        // declare accumulators
        float c[CoarsenFactor] = {0};

        int start = csr_indptr[row];
        int end = csr_indptr[row + 1];
        int k;
        float v;

        for (int jj = start + lane_id; jj < end; jj += 32) {
            k = csr_indices[jj];
            v = __guard_load_default_one<float>(csr_data, jj);

        // load B-elements in vector-type
        #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
            c[i] += v * B_panels[i][k];
            }
        }

        #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            // row-wise reduction is a simple merge-tree
            SHFL_DOWN_REDUCE(c[i])
        }

        // store to C in vector-type
        if (lane_id == 0) {
        #pragma unroll
            for (int i = 0; i < CoarsenFactor; i++) {
            C_panels[i][row] = c[i];
            }
        }
    }
    return;

    Ndim_Residue:
    int valid_lane_num = N - col_offset;

    for (; row < M; row += stride) {
    // get row offsets
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor];
    // access_t res = init_zeros<access_t>();

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int jj = start + lane_id; jj < end; jj += 32) {
        k = csr_indices[jj];
        v = __guard_load_default_one<float>(csr_data, jj);

    #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
                buffer[i] = B_panels[i][k];
            }
        }

    #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            c[i] += v * buffer[i];
        }
    }

    #pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
        SHFL_DOWN_REDUCE(c[i])
    }

    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num) {
                C_panels[i][row] = c[i];
            }
        }
    }
    }
}

void spmm_rb_pr(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
    const int M = (int)(C->dimensions[0]);
    const int N = (int)(C->dimensions[1]);
    const int K = (int)(B->dimensions[0]);
    copy_to_device_spmm(C, A, B);
    int coarsen_factor = (N >= 128) ? 4 : (N >= 64) ? 2 : 1;
    int Mdim_worker = M;
    int Ndim_threadblock = CEIL(N, coarsen_factor);
    int ref_warp_per_tb = RefThreadPerBlock / 32;
    int Mdim_warp_per_tb = ref_warp_per_tb;

    int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
    int gridDimY = Ndim_threadblock;
    dim3 gridDim(gridDimX, gridDimY, 1);
    dim3 blockDim(32, Mdim_warp_per_tb, 1);

    if (coarsen_factor == 4) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
            csrspmm_non_transpose_parreduce_rowbalance_kernel<4>
                <<<gridDim, blockDim>>>();
        );
    } else if (coarsen_factor == 2) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
            csrspmm_non_transpose_parreduce_rowbalance_kernel<2>
                <<<gridDim, blockDim>>>();
        );
    } else {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
            csrspmm_non_transpose_parreduce_rowbalance_kernel<1>
                <<<gridDim, blockDim>>>();
        );
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    copy_from_device_spmm(C, A, B);
    free_tensors_spmm(); 
}
*/
 
template <typename access_t>
__global__ void csrspmm_parreduce_rowbalance_kernel() {
  constexpr int CoarsenFactor = sizeof(access_t) / sizeof(float);
    const int M = global_A1_dimension;
    const int N = global_B2_dimension;

    int* __restrict__ csr_indptr = global_A2_pos_device;
    int* __restrict__ csr_indices = global_A2_crd_device;
    float* __restrict__ csr_data = global_A_vals_device;
    float* __restrict__ B = global_B_vals_device;
    float* __restrict__ C = global_C_vals_device; 
  int lane_id = (threadIdx.x & (32 - 1));
  int stride = gridDim.x * blockDim.y;
  int row = blockIdx.x * blockDim.y + threadIdx.y;

  // get the dense column offset
  int col_offset = blockIdx.y * 32 + (threadIdx.x >> 5) * CoarsenFactor;
  const float *B_panel = B + col_offset;
  float *C_panel = C + col_offset;
  int ldB = N;
  int ldC = N;

  if (col_offset >= N)
    return;
  if (col_offset + CoarsenFactor >= N)
    goto Ndim_Residue;

  for (; row < M; row += stride) {
    // declare accumulators
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor];

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      k = csr_indices[jj];
      v = __guard_load_default_one<float>(csr_data, jj);

      // load B-elements in vector-type
      *(access_t *)buffer = *(access_t *)(B_panel + k * ldB);

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      // row-wise reduction is a simple merge-tree
      SHFL_DOWN_REDUCE(c[i])
    }

    // store to C in vector-type
    if (lane_id == 0) {
      *(access_t *)(C_panel + row * ldC) = *(access_t *)c;
    }
  }
  return;

Ndim_Residue:
  int valid_lane_num = N - col_offset;

  for (; row < M; row += stride) {
    // get row offsets
    float c[CoarsenFactor] = {0};
    float buffer[CoarsenFactor];
    // access_t res = init_zeros<access_t>();

    int start = csr_indptr[row];
    int end = csr_indptr[row + 1];
    int k;
    float v;

    for (int jj = start + lane_id; jj < end; jj += 32) {
      k = csr_indices[jj];
      v = __guard_load_default_one<float>(csr_data, jj);

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          buffer[i] = B_panel[k * ldB + i];
        }
      }

#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        c[i] += v * buffer[i];
      }
    }

#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      SHFL_DOWN_REDUCE(c[i])
    }

    if (lane_id == 0) {
#pragma unroll
      for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num) {
          C_panel[row * ldC + i] = c[i];
        }
      }
    }
  }
}

void spmm_rb_pr(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){

    const int M = (int)(C->dimensions[0]);
    const int N = (int)(C->dimensions[1]);
    copy_to_device_spmm(C, A, B); 
  // factor of thread coarsening
  int coarsen_factor = (N % 4 == 0) ? 4 : (N % 2 == 0) ? 2 : 1;
  // number of parallel warps along M-dimension
  int Mdim_worker = M;
  // partition large-N and map to blockdim.y to help cache performance
  int Ndim_threadblock = CEIL(N, 32);
  int Ndim_warp_per_tb = min(N, 32) / coarsen_factor;

  int ref_warp_per_tb = RefThreadPerBlock / 32;
  int Mdim_warp_per_tb = CEIL(ref_warp_per_tb, Ndim_warp_per_tb);

  // total number of warps
  int gridDimX = CEIL(Mdim_worker, Mdim_warp_per_tb);
  int gridDimY = Ndim_threadblock;
  dim3 gridDim(gridDimX, gridDimY, 1);
  dim3 blockDim(Ndim_warp_per_tb * 32, Mdim_warp_per_tb, 1);

    if (coarsen_factor == 4) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
            csrspmm_parreduce_rowbalance_kernel<float4><<<gridDim, blockDim>>>();
        );
    } else if (coarsen_factor == 2) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
      csrspmm_parreduce_rowbalance_kernel<float2><<<gridDim, blockDim>>>();
              );
    } else {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
      csrspmm_parreduce_rowbalance_kernel<float><<<gridDim, blockDim>>>();
              );
    }
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
__global__
void kernel_taco_rb_pr_256(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C){
    int A1_dimension = global_A1_dimension;
    int* __restrict__ A2_pos = global_A2_pos_device;
    int* __restrict__ A2_crd = global_A2_crd_device;
    float* __restrict__ A_vals = global_A_vals_device;
    int B2_dimension = global_B2_dimension;
    float* __restrict__ B_vals = global_B_vals_device;
    int C2_dimension = global_C2_dimension;
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
void spmm_rb_pr_256(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256<<<(A1_dimension * B2_dimension + 7) / 8, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}

template<int c, int g, int group_size>
__global__
void kernel_taco_rb_pr_256_tune(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C){
    int A1_dimension = global_A1_dimension;
    int* __restrict__ A2_pos = global_A2_pos_device;
    int* __restrict__ A2_crd = global_A2_crd_device;
    float* __restrict__ A_vals = global_A_vals_device;
    int B2_dimension = global_B2_dimension;
    float* __restrict__ B_vals = global_B_vals_device;
    int C2_dimension = global_C2_dimension;
    float* __restrict__ C_vals = global_C_vals_device;
    int32_t ko = blockIdx.x;
    int32_t jpos1 = (threadIdx.x % (g)); // tile_size = 8
    int32_t kio = (threadIdx.x / g);
    if (threadIdx.x >= 256) {
        return;
    }
    for (int32_t kii = 0; kii < c; kii++) {
        int32_t ki = kio * c + kii;
        int32_t io = ko * (c * 256 / g) + ki;
        int32_t i = io / B2_dimension;
        if (i >= A1_dimension)
            return;
        int32_t k = io % B2_dimension;
        int32_t kC = i * C2_dimension + k;
        if (k >= B2_dimension)
            return;
        float tjpos1C_val = 0.0;
        for (int32_t jpos0 = A2_pos[i] / g; jpos0 < ((A2_pos[(i + 1)] + g - 1) / g); jpos0++) {
            int32_t jposA = jpos0 * g + jpos1;
            if (jposA < A2_pos[i] || jposA >= A2_pos[(i + 1)])
                continue;
            int32_t j = A2_crd[jposA];
            int32_t kB = j * B2_dimension + k;
            tjpos1C_val = tjpos1C_val + A_vals[jposA] * B_vals[kB];
        }
        //atomicAddWarp<float>(C_vals, kC, tjpos1C_val);
        atomicAddGroup<float,group_size>(C_vals, kC, tjpos1C_val); // group_size = 8
    }
}

void spmm_rb_pr_256_1_4_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<1,4,4><<<(A1_dimension * B2_dimension + 63) / 64, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_2_4_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<2,4,4><<<(A1_dimension * B2_dimension + 127) / 128, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_4_4_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<4,4,4><<<(A1_dimension * B2_dimension + 255) / 256, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_1_8_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<1,8,4><<<(A1_dimension * B2_dimension + 31) / 32, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_2_8_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<2,8,4><<<(A1_dimension * B2_dimension + 63) / 64, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_4_8_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<4,8,4><<<(A1_dimension * B2_dimension + 127) / 128, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_1_8_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<1,8,8><<<(A1_dimension * B2_dimension + 31) / 32, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_2_8_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<2,8,8><<<(A1_dimension * B2_dimension + 63) / 64, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_4_8_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<4,8,8><<<(A1_dimension * B2_dimension + 127) / 128, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_1_16_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<1,16,4><<<(A1_dimension * B2_dimension + 15) / 16, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_2_16_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<2,16,4><<<(A1_dimension * B2_dimension + 31) / 32, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_4_16_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<4,16,4><<<(A1_dimension * B2_dimension + 63) / 64, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_1_16_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<1,16,8><<<(A1_dimension * B2_dimension + 15) / 16, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_2_16_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<2,16,8><<<(A1_dimension * B2_dimension + 31) / 32, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_4_16_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<4,16,8><<<(A1_dimension * B2_dimension + 63) / 64, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_1_16_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<1,16,16><<<(A1_dimension * B2_dimension + 15) / 16, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_2_16_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<2,16,16><<<(A1_dimension * B2_dimension + 31) / 32, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_4_16_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<4,16,16><<<(A1_dimension * B2_dimension + 63) / 64, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_1_32_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<1,32,4><<<(A1_dimension * B2_dimension + 7) / 8, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_2_32_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<2,32,4><<<(A1_dimension * B2_dimension + 15) / 16, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_4_32_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<4,32,4><<<(A1_dimension * B2_dimension + 31) / 32, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_1_32_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<1,32,8><<<(A1_dimension * B2_dimension + 7) / 8, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_2_32_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<2,32,8><<<(A1_dimension * B2_dimension + 15) / 16, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_4_32_8(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<4,32,8><<<(A1_dimension * B2_dimension + 31) / 32, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_1_32_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<1,32,16><<<(A1_dimension * B2_dimension + 7) / 8, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_2_32_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<2,32,16><<<(A1_dimension * B2_dimension + 15) / 16, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_4_32_16(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<4,32,16><<<(A1_dimension * B2_dimension + 31) / 32, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_1_32_32(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<1,32,32><<<(A1_dimension * B2_dimension + 7) / 8, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_2_32_32(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<2,32,32><<<(A1_dimension * B2_dimension + 15) / 16, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_pr_256_4_32_32(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension)*4));
      kernel_taco_rb_pr_256_tune<4,32,32><<<(A1_dimension * B2_dimension + 31) / 32, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
