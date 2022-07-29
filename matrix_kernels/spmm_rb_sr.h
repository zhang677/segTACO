#include "gpu_library.h"
//#include "gpu_kernels.cuh"
#include "stdio.h"

__global__
void spmm_rb_sr_taco_kernel(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (((B2_dimension + 3) / 4)));
  int32_t warp = (threadIdx.x / ((B2_dimension + 3) / 4));
  if (threadIdx.x >= (B2_dimension + 3) / 4 * 16) {
    return;
  }

  for (int32_t warp_row = 0; warp_row < 1; warp_row++) {
    int32_t i1 = warp + warp_row;
    int32_t i = block * 16 + i1;
    if (i >= A1_dimension)
      return;

    for (int32_t thread_col = 0; thread_col < 4; thread_col++) {
      int32_t k = thread * 4 + thread_col;
      if (k >= B2_dimension)
        break;

      int32_t kC = i * C2_dimension + k;
      for (int32_t jA = A2_pos[i]; jA < A2_pos[(i + 1)]; jA++) {
        int32_t j = A2_crd[jA];
        int32_t kB = j * B2_dimension + k;
        C_vals[kC] = C_vals[kC] + A_vals[jA] * B_vals[kB];
      }
    }
  }    
}

void spmm_rb_sr_taco(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

TIME_GPU(
  gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
  spmm_rb_sr_taco_kernel<<<(A1_dimension + 15) / 16, (B2_dimension + 3) / 4 * 16>>>(A, B, C);
);
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();    
}
/*
template <int CoarsenFactor>
__global__ void csrspmm_non_transpose_seqreduce_rowbalance_kernel() {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
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

        for (int p = start; p < end; p++) {
        k = csr_indices[p];
        v = __guard_load_default_one<float>(csr_data, p);

    // load B-elements in vector-type
    #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            c[i] += v * B_panels[i][k];
        }
        }
    #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
        C_panels[i][row] = c[i];
        }
    }
    return;
    Ndim_Residue:
    int valid_lane_num = N - col_offset;

    for (; row < M; row += stride) {
        // declare accumulators
        float c[CoarsenFactor] = {0};

        int start = csr_indptr[row];
        int end = csr_indptr[row + 1];
        int k;
        float v;

        for (int p = start; p < end; p++) {
        k = csr_indices[p];
        v = __guard_load_default_one<float>(csr_data, p);

    // load B-elements in vector-type
    #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
            if (i < valid_lane_num)
            c[i] += v * B_panels[i][k];
        }
        }
    #pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
        if (i < valid_lane_num)
            C_panels[i][row] = c[i];
        }
    }
    return;
}

void spmm_rb_sr(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
    const int M = (int)(C->dimensions[0]);
    const int N = (int)(C->dimensions[1]);
    const int K = (int)(B->dimensions[0]);
    copy_to_device_spmm(C, A, B);
    int coarsen_factor = (N >= 128) ? 4 : (N >= 64) ? 2 : 1;
    int Mdim_worker = M;
    int Ndim_threadblock = CEIL(N, coarsen_factor);
    int Mdim_threadblock = CEIL(Mdim_worker, RefThreadPerBlock);
    dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
    dim3 blockDim(RefThreadPerBlock, 1, 1);    

    if (coarsen_factor == 4) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
            csrspmm_non_transpose_seqreduce_rowbalance_kernel<4>
                <<<gridDim, blockDim>>>();
        );
    } else if (coarsen_factor == 2) {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
            csrspmm_non_transpose_seqreduce_rowbalance_kernel<2>
                <<<gridDim, blockDim>>>();
        );
    } else {
        TIME_GPU(
            gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));
            csrspmm_non_transpose_seqreduce_rowbalance_kernel<1>
                <<<gridDim, blockDim>>>();
        );
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    copy_from_device_spmm(C, A, B);
    free_tensors_spmm();      
}
*/

__global__ void csrspmm_seqreduce_rowbalance_kernel() {
  
    const int nr = global_A1_dimension;
    const int nv = global_B2_dimension;

    int* __restrict__ rowPtr = global_A2_pos_device;
    int* __restrict__ colIdx = global_A2_crd_device;
    float* __restrict__ values = global_A_vals_device;
    float* __restrict__ dnInput = global_B_vals_device;
    float* __restrict__ dnOutput = global_C_vals_device;
  int row_tile = blockDim.y;
  int subwarp_id = threadIdx.y;
  int stride = row_tile * gridDim.x;
  int row = blockIdx.x * row_tile + subwarp_id;
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  dnInput += v_id;
  dnOutput += v_id;

  float res = 0, val;
  int col;
  for (; row < nr; row += stride) {

    int start = __ldg(rowPtr + row);
    int end = __ldg(rowPtr + row + 1);
    for (int p = start; p < end; p++) {
      col = __ldg(colIdx + p);
      val = __guard_load_default_one<float>(values, p);
      res += val * __ldg(dnInput + col * nv);
    }
    dnOutput[row * nv] = res;
  }
}

void spmm_rb_sr(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {

    const int M = (int)(C->dimensions[0]);
    const int N = (int)(C->dimensions[1]);
    copy_to_device_spmm(C, A, B); 
  int Mdim_worker = M;
  int Ndim_worker = N;
  int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
  int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
  int Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  int Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

    TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));    
      csrspmm_seqreduce_rowbalance_kernel<<<gridDim, blockDim>>>();
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
__global__
void kernel_taco_rb_sr_64_4_2(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int A1_dimension = global_A1_dimension;
  int* __restrict__ A2_pos = global_A2_pos_device;
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;
  float* __restrict__ C_vals = global_C_vals_device;
  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (((B2_dimension + 1) / 2)));
  int32_t warp = (threadIdx.x / ((B2_dimension + 1) / 2));
  if (threadIdx.x >= (B2_dimension + 1) / 2 * 16) {
    return;
  }

  for (int32_t warp_row = 0; warp_row < 4; warp_row++) {
    int32_t i1 = warp * 4 + warp_row;
    int32_t i = block * 64 + i1;
    if (i >= A1_dimension)
      return;
    for (int32_t thread_col = 0; thread_col < 2; thread_col++) {
      int32_t k = thread * 2 + thread_col;
      if (k >= B2_dimension)
        return;
      int32_t kC = i * C2_dimension + k;
      for (int32_t jA = A2_pos[i]; jA < A2_pos[(i + 1)]; jA++) {
        int32_t j = A2_crd[jA];
        int32_t kB = j * B2_dimension + k;
        C_vals[kC] = C_vals[kC] + A_vals[jA] * B_vals[kB];
      }
    }
  }
}
void spmm_rb_sr_512_64(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_64_4_2<<<(A1_dimension + 63) / 64, (B2_dimension + 1) / 2 * 16>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}

__global__
void kernel_taco_rb_sr_4_1_1(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int A1_dimension = global_A1_dimension;
  int* __restrict__ A2_pos = global_A2_pos_device;
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;
  float* __restrict__ C_vals = global_C_vals_device;
  int32_t block = blockIdx.x;
  int32_t thread = threadIdx.x % B2_dimension;
  int32_t warp = threadIdx.x / B2_dimension;
  if (threadIdx.x >= B2_dimension * 64) {
    return;
  }

  for (int32_t warp_row = 0; warp_row < 1; warp_row++) {
    int32_t i1 = warp + warp_row;
    int32_t i = block * 64 + i1;
    if(i >= A1_dimension)
      return;
    
    for (int32_t thread_col = 0; thread_col < 1; thread_col++) {
      int32_t k = thread * 1 + thread_col;
      if(k >= B2_dimension)
        break;
      
      int32_t kC = i * C2_dimension + k;
      for (int32_t jA = A2_pos[i]; jA < A2_pos[(i + 1)]; jA++) {
        int32_t j = A2_crd[jA];
        int32_t kB = j * B2_dimension + k;
        C_vals[kC] = C_vals[kC] + A_vals[jA] * B_vals[kB];
      }
      
    }
  }
}
void spmm_rb_sr_256_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_4_1_1<<<(A1_dimension + 63) / 64, B2_dimension * 64>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}

template<int r, int c>
__global__
void kernel_taco_rb_sr_tune(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int A1_dimension = global_A1_dimension;
  int* __restrict__ A2_pos = global_A2_pos_device;
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;
  float* __restrict__ C_vals = global_C_vals_device;
  int32_t block = blockIdx.x;
  int32_t thread = threadIdx.x % (4 / c);
  int32_t warp = threadIdx.x / (4 / c);
  if (threadIdx.x >= 256) {
    return;
  }

  for (int32_t warp_row = 0; warp_row < r; warp_row++) {
    int32_t i1 = warp * r + warp_row;
    int32_t i = block * (64 * r * c) + i1;
    if(i >= A1_dimension)
      return;
    
    for (int32_t thread_col = 0; thread_col < c; thread_col++) {
      int32_t k = thread * c + thread_col;
      if(k >= B2_dimension)
        break;
      
      int32_t kC = i * C2_dimension + k;
      for (int32_t jA = A2_pos[i]; jA < A2_pos[(i + 1)]; jA++) {
        int32_t j = A2_crd[jA];
        int32_t kB = j * B2_dimension + k;
        C_vals[kC] = C_vals[kC] + A_vals[jA] * B_vals[kB];
      }
      
    }
  }
}


void spmm_rb_sr_256_1_1(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_tune<1,1><<<(A1_dimension + 63) / 64, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_sr_256_1_2(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_tune<1,2><<<(A1_dimension + 127) / 128, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_sr_256_1_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_tune<1,4><<<(A1_dimension + 255) / 256, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_sr_256_2_1(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_tune<2,1><<<(A1_dimension + 127) / 128, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_sr_256_2_2(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_tune<2,2><<<(A1_dimension + 255) / 256, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_sr_256_2_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_tune<2,4><<<(A1_dimension + 511) / 512, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_sr_256_4_1(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_tune<4,1><<<(A1_dimension + 255) / 256, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_sr_256_4_2(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_tune<4,2><<<(A1_dimension + 511) / 512, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_rb_sr_256_4_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  TIME_GPU(
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, C2_dimension * C1_dimension * sizeof(float) ));
    kernel_taco_rb_sr_tune<4,4><<<(A1_dimension + 1023) / 1024, 256>>>(A, B, C);
  );
  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}