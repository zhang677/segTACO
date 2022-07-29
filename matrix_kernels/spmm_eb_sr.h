#include "gpu_library.h"
//#include "gpu_kernels.cuh"
#include "stdio.h"

__global__
void spmm_eb_sr_taco_kernel(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = global_A1_dimension;//int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = global_A2_pos_device; //&global_C_vals_host_copy
  int* __restrict__ A2_crd = global_A2_crd_device;
  float* __restrict__ A_vals = global_A_vals_device;
  int B2_dimension = global_B2_dimension;//int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = global_B_vals_device;
  int C2_dimension = global_C2_dimension;//int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = global_C_vals_device;

  
  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) { // 512 threads per block, 16 warps per block
    return;
  }

  for (int32_t dense_val = 0; dense_val < 4; dense_val++) {
    int32_t k = dense_val * 32 + thread;
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block]; 
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 16; // 16 nnz per warp 16=256/16
    int32_t fposA = block * 256 + fpos1; // 256 nnz per block
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA); // find the start row
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 16; nnz++) {
      int32_t fpos1 = warp * 16 + nnz; // 16 nnz per warp, 16 warp per block
      int32_t fposA = block * 256 + fpos1; // A_vals 
      if (fposA >= A2_pos[A1_dimension])
        break;
      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos; 
      }
      int32_t kC = i * C2_dimension + k; // row-major
      int32_t kB = f * B2_dimension + k; // row-major
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[kC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    int32_t kC = i * C2_dimension + k;
    atomicAdd(&C_vals[kC], tnnz_val);
  }
}



void spmm_eb_sr_taco(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  copy_to_device_spmm(C, A, B);
  
  int32_t* i_blockStarts = 0; // start of each segment
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1))); // element-balance
  TIME_GPU(
    i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 512, ((A2_pos[A1_dimension] + 255) / 256)); // spmm_csr_gpu_taco_row
    gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
    spmm_eb_sr_taco_kernel<<<(A2_pos[A1_dimension] + 255) / 256, 32 * 16>>>(A, B, C, i_blockStarts);
  );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
/*
__global__ void csrspmm_seqreduce_nnzbalance_kernel() {

    const int nr = global_A1_dimension;
    const int nv = global_B2_dimension;

    int* __restrict__ rowPtr = global_A2_pos_device;
    int* __restrict__ colIdx = global_A2_crd_device;
    float* __restrict__ values = global_A_vals_device;
    float* __restrict__ dnInput = global_B_vals_device;
    float* __restrict__ dnOutput = global_C_vals_device;

    const int nnz = rowPtr[nr];

  int Nnzdim_thread = blockDim.y * gridDim.x;
  int NE_PER_THREAD = DIV_UP(nnz, Nnzdim_thread);
  int eid = (blockIdx.x * blockDim.y + threadIdx.y) * NE_PER_THREAD;
  int v_id = (blockIdx.y * blockDim.x) + threadIdx.x;
  int col = 0;
  float val = 0.0;

  if (eid < nnz) {
    int row = binary_search_segment_number<int>(rowPtr, nr, nnz, eid);
    int step = __ldg(rowPtr + row + 1) - eid;

    for (int ii = 0; ii < NE_PER_THREAD; ii++) {
      if (eid >= nnz)
        break;
      if (ii < step) {
        col = __ldg(colIdx + eid) * nv;
        val += __guard_load_default_one<float>(values, eid) *
               __ldg(dnInput + col + v_id);

        eid++;
      } else {
        atomicAdd(&dnOutput[row * nv + v_id], val);

        row = binary_search_segment_number<int>(rowPtr, nr, nnz, eid);
        step = __ldg(rowPtr + row + 1) - eid;
        col = __ldg(colIdx + eid) * nv;
        val = __guard_load_default_one<float>(values, eid) *
              __ldg(dnInput + col + v_id);

        eid++;
      }
    }
    atomicAdd(&dnOutput[row * nv + v_id], val);
  }
}

void spmm_eb_sr(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
    const int M = (int)(C->dimensions[0]);
    const int N = (int)(C->dimensions[1]);
    copy_to_device_spmm(C, A, B);    
  int Nnzdim_worker = M * 32;
  int Ndim_worker = N;
  int Ndim_threadblock = CEIL(Ndim_worker, RefThreadPerBlock);
  int Ndim_thread_per_tb = min(Ndim_worker, RefThreadPerBlock);
  int Nnzdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  int Nnzdim_threadblock = CEIL(Nnzdim_worker, Nnzdim_thread_per_tb);

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Nnzdim_thread_per_tb, 1);

    TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4));    
      csrspmm_seqreduce_nnzbalance_kernel<<<gridDim, blockDim>>>();
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
*/
template <int CoarsenFactor, int ThreadNz>
__global__ void csrspmm_rowcaching_nnzbalance_kernel() {
  
  const int N = global_B2_dimension;
  const int M = global_A1_dimension;
  int* __restrict__ csr_indptr = global_A2_pos_device;
  int* __restrict__ csr_indices = global_A2_crd_device;
  float* __restrict__ csr_data = global_A_vals_device;  
  float* __restrict__ B = global_B_vals_device;
  float* __restrict__ C = global_C_vals_device;


  const int nnz = csr_indptr[M];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  extern __shared__ int shared_mem[];
  int *workspace_rowid = &shared_mem[(warp_id << 5)];
  int *workspace_colid = workspace_rowid + blockDim.x;
  float *workspace_data =
      (float *)(workspace_colid +
                blockDim.x); // float and int has the same size

  // get the sparse-value range of this row
  int global_warp_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
  int nz_start = global_warp_id * (ThreadNz * 32);

  // get the dense column offset
  int col_offset = blockIdx.y * 32 * CoarsenFactor;
  const float *B_lanes[CoarsenFactor];
  float *C_lanes[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_lanes[i] = B + col_offset + lane_id + i * 32;
    C_lanes[i] = C + col_offset + lane_id + i * 32;
  }
  int ldB = N;

  // declare accumulators
  float c[CoarsenFactor] = {0.0f};
  int ldC = N;

  int stride = gridDim.x * (blockDim.x >> 5) * ThreadNz * 32;

  if (blockIdx.y == gridDim.y - 1)
    goto Ndim_Residue;

  for (; nz_start < nnz; nz_start += stride) {
  // iterate over the segment of this warp
  for (int tile_base = nz_start; 
    tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {

    int thread_nz_id = tile_base + lane_id;
    if (thread_nz_id < nnz) {
      workspace_colid[lane_id] = csr_indices[thread_nz_id];
      workspace_data[lane_id] =
          __guard_load_default_one<float>(csr_data, thread_nz_id);
    } else {
      workspace_colid[lane_id] = 0;
      workspace_data[lane_id] = 0.0f;
    }
    workspace_rowid[lane_id] =
        binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
    __syncwarp();

    // initialize with first value
    int k = workspace_colid[0];
    float v = workspace_data[0];
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      c[i] = v * B_lanes[i][k * ldB];
    }
    int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
    for (int pp = 1; pp < 32; pp++) {
      next_row = workspace_rowid[pp];
      if (next_row != row_curr) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
        }
        row_curr = next_row;
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          c[i] = v * B_lanes[i][k * ldB];
        }
      } else {
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          c[i] = c[i] + v * B_lanes[i][k * ldB];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
    }
  }
  }
  return;

Ndim_Residue:

  int valid_lane_num = CEIL(N - col_offset - lane_id, 32);
  
  for (; nz_start < nnz; nz_start += stride) {
  // iterate over the segment of this warp
  for (int tile_base = nz_start; 
    tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {

    int thread_nz_id = tile_base + lane_id;
    if (thread_nz_id < nnz) {
      workspace_colid[lane_id] = csr_indices[thread_nz_id];
      workspace_data[lane_id] =
          __guard_load_default_one<float>(csr_data, thread_nz_id);
    } else {
      workspace_colid[lane_id] = 0;
      workspace_data[lane_id] = 0.0f;
    }
    workspace_rowid[lane_id] =
        binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
    __syncwarp();

    // initialize with first value
    int k = workspace_colid[0];
    float v = workspace_data[0];
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        c[i] = v * B_lanes[i][k * ldB];
      }
    }
    int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
    for (int pp = 1; pp < 32; pp++) {
      next_row = workspace_rowid[pp];
      if (next_row != row_curr) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
          }
        }
        row_curr = next_row;
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            c[i] = v * B_lanes[i][k * ldB];
          }
        }
      } else {
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            c[i] = c[i] + v * B_lanes[i][k * ldB];
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
      }
    }
  }
  }
}

void spmm_eb_sr(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  const int M = (int)(C->dimensions[0]);
  const int N = (int)(C->dimensions[1]);
  copy_to_device_spmm(C, A, B); 
  int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2 : 1;
  int Ndim_threadblock = CEIL(N, (32 * coarsen_factor));

  // int thread_nz = (spmatA.nnz > 8000 * 128 * 2) ? 2 : 1;
  int thread_nz = 1;
  int Nnzdim_warp_per_tb = RefThreadPerBlock / 32;
  // int Nnzdim_threadblock = CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 *
  // thread_nz );
  int Nnzdim_threadblock = CEIL(
      M,
      Nnzdim_warp_per_tb *
          thread_nz); // CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 * thread_nz );

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);
  //cudaFuncSetCacheConfig(csrspmm_rowcaching_nnzbalance_kernel<2, 1>, cudaFuncCachePreferEqual);
  size_t smem_size = (2 * sizeof(int) + sizeof(float)) * RefThreadPerBlock;

  // simple heuristic

  if (coarsen_factor == 4) {
    if (thread_nz == 1) {
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<4, 1>
          <<<gridDim, blockDim, smem_size>>>();
      );
    }
    if (thread_nz == 2)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<4, 2>
          <<<gridDim, blockDim, smem_size>>>();
      );
    if (thread_nz == 4)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<4, 4>
          <<<gridDim, blockDim, smem_size>>>();
      );
  } else if (coarsen_factor == 2) {
    if (thread_nz == 1)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<2, 1>
          <<<gridDim, blockDim, smem_size>>>();
      );
    if (thread_nz == 2)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<2, 2>
          <<<gridDim, blockDim, smem_size>>>();
      );
    if (thread_nz == 4)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<2, 4>
          <<<gridDim, blockDim, smem_size>>>();
      );
  } else {
    if (thread_nz == 1)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<1, 1>
          <<<gridDim, blockDim, smem_size>>>();
      );
    if (thread_nz == 2)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<1, 2>
          <<<gridDim, blockDim, smem_size>>>();
      );
    if (thread_nz == 4)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<1, 4>
          <<<gridDim, blockDim, smem_size>>>();
      );
  }
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaDeviceSynchronize(); 
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();  
}

template <int CoarsenFactor, int ThreadNz, int SearchSeg>
__global__ void eb_sr_search_kernel(int* segStarts) {
  
  const int N = global_B2_dimension;
  const int M = global_A1_dimension;
  int* __restrict__ csr_indptr = global_A2_pos_device;
  int* __restrict__ csr_indices = global_A2_crd_device;
  float* __restrict__ csr_data = global_A_vals_device;  
  float* __restrict__ B = global_B_vals_device;
  float* __restrict__ C = global_C_vals_device;


  const int nnz = csr_indptr[M];

  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;

  extern __shared__ int shared_mem[];
  int *workspace_rowid = &shared_mem[(warp_id << 5)];
  int *workspace_colid = workspace_rowid + blockDim.x;
  float *workspace_data =
      (float *)(workspace_colid +
                blockDim.x); // float and int has the same size

  // get the sparse-value range of this row
  int global_warp_id = blockIdx.x * (blockDim.x >> 5) + warp_id;
  int nz_start = global_warp_id * (ThreadNz * 32);

  // get the dense column offset
  int col_offset = blockIdx.y * 32 * CoarsenFactor;
  const float *B_lanes[CoarsenFactor];
  float *C_lanes[CoarsenFactor];
#pragma unroll
  for (int i = 0; i < CoarsenFactor; i++) {
    B_lanes[i] = B + col_offset + lane_id + i * 32;
    C_lanes[i] = C + col_offset + lane_id + i * 32;
  }
  int ldB = N;

  // declare accumulators
  float c[CoarsenFactor] = {0.0f};
  int ldC = N;

  int stride = gridDim.x * (blockDim.x >> 5) * ThreadNz * 32;

  if (blockIdx.y == gridDim.y - 1)
    goto Ndim_Residue;

  for (; nz_start < nnz; nz_start += stride) {
  // iterate over the segment of this warp
  for (int tile_base = nz_start; 
    tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {

    int thread_nz_id = tile_base + lane_id;
    if (thread_nz_id < nnz) {
      workspace_colid[lane_id] = csr_indices[thread_nz_id];
      workspace_data[lane_id] =
          __guard_load_default_one<float>(csr_data, thread_nz_id);
    } else {
      workspace_colid[lane_id] = 0;
      workspace_data[lane_id] = 0.0f;
    }
    int seg_num = thread_nz_id / SearchSeg; 
    int p_begin = segStarts[seg_num];
    int p_end = segStarts[seg_num+1];
    workspace_rowid[lane_id] = taco_binarySearchBefore(csr_indptr, p_begin, p_end, thread_nz_id);    
    //workspace_rowid[lane_id] =
    //    binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
    __syncwarp();

    // initialize with first value
    int k = workspace_colid[0];
    float v = workspace_data[0];
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      c[i] = v * B_lanes[i][k * ldB];
    }
    int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
    for (int pp = 1; pp < 32; pp++) {
      next_row = workspace_rowid[pp];
      if (next_row != row_curr) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
        }
        row_curr = next_row;
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          c[i] = v * B_lanes[i][k * ldB];
        }
      } else {
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          c[i] = c[i] + v * B_lanes[i][k * ldB];
        }
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
    }
  }
  }
  return;

Ndim_Residue:

  int valid_lane_num = CEIL(N - col_offset - lane_id, 32);
  
  for (; nz_start < nnz; nz_start += stride) {
  // iterate over the segment of this warp
  for (int tile_base = nz_start; 
    tile_base < min(nz_start + ThreadNz * 32, nnz); tile_base += 32) {

    int thread_nz_id = tile_base + lane_id;
    if (thread_nz_id < nnz) {
      workspace_colid[lane_id] = csr_indices[thread_nz_id];
      workspace_data[lane_id] =
          __guard_load_default_one<float>(csr_data, thread_nz_id);
    } else {
      workspace_colid[lane_id] = 0;
      workspace_data[lane_id] = 0.0f;
    }
    int seg_num = thread_nz_id / SearchSeg; 
    int p_begin = segStarts[seg_num];
    int p_end = segStarts[seg_num+1];
    workspace_rowid[lane_id] = taco_binarySearchBefore(csr_indptr, p_begin, p_end, thread_nz_id);
    //workspace_rowid[lane_id] =
    //    binary_search_segment_number<int>(csr_indptr, M, nnz, thread_nz_id);
    __syncwarp();

    // initialize with first value
    int k = workspace_colid[0];
    float v = workspace_data[0];
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        c[i] = v * B_lanes[i][k * ldB];
      }
    }
    int row_curr = workspace_rowid[0], next_row;

// scan
#pragma unroll
    for (int pp = 1; pp < 32; pp++) {
      next_row = workspace_rowid[pp];
      if (next_row != row_curr) {
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
          }
        }
        row_curr = next_row;
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            c[i] = v * B_lanes[i][k * ldB];
          }
        }
      } else {
        k = workspace_colid[pp];
        v = workspace_data[pp];
#pragma unroll
        for (int i = 0; i < CoarsenFactor; i++) {
          if (i < valid_lane_num) {
            c[i] = c[i] + v * B_lanes[i][k * ldB];
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < CoarsenFactor; i++) {
      if (i < valid_lane_num) {
        atomicAdd(C_lanes[i] + row_curr * ldC, c[i]);
      }
    }
  }
  }
}

void spmm_eb_sr_search(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  const int M = (int)(C->dimensions[0]);
  const int N = (int)(C->dimensions[1]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  const int nnz = A2_pos[M];
  copy_to_device_spmm(C, A, B); 
  int coarsen_factor = (N >= 512) ? 4 : (N >= 128) ? 2 : 1;
  int Ndim_threadblock = CEIL(N, (32 * coarsen_factor));

  // int thread_nz = (spmatA.nnz > 8000 * 128 * 2) ? 2 : 1;
  int thread_nz = 1;
  int Nnzdim_warp_per_tb = RefThreadPerBlock / 32;
  // int Nnzdim_threadblock = CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 *
  // thread_nz );
  int Nnzdim_threadblock = CEIL(
      M,
      Nnzdim_warp_per_tb *
          thread_nz); // CEIL(spmatA.nnz, Nnzdim_warp_per_tb * 32 * thread_nz );

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(RefThreadPerBlock, 1, 1);

  const int SearchSeg = 32; // Hyperparameter
  int32_t* segStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&segStarts, sizeof(int32_t) * (CEIL(nnz, SearchSeg) + 1))); // element-balance
  gpuErrchk(cudaGetLastError());
  //cudaFuncSetCacheConfig(eb_sr_search_kernel<4, 1, SearchSeg>, cudaFuncCachePreferEqual);
  size_t smem_size = (2 * sizeof(int) + sizeof(float)) * RefThreadPerBlock;

  // simple heuristic

  if (coarsen_factor == 4) {
    if (thread_nz == 1) {
      TIME_GPU(
      segStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, segStarts, (int32_t) 0, M, (int32_t) SearchSeg, (int32_t) SearchSeg, CEIL(nnz, SearchSeg));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      eb_sr_search_kernel<4, 1, SearchSeg>
          <<<gridDim, blockDim, smem_size>>>(segStarts);
      );
    }
    if (thread_nz == 2)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<4, 2>
          <<<gridDim, blockDim, smem_size>>>();
      );
    if (thread_nz == 4)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<4, 4>
          <<<gridDim, blockDim, smem_size>>>();
      );
  } else if (coarsen_factor == 2) {
    if (thread_nz == 1)
      TIME_GPU(
      segStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, segStarts, (int32_t) 0, M, (int32_t) SearchSeg, (int32_t) SearchSeg, CEIL(nnz, SearchSeg));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      eb_sr_search_kernel<4, 1, SearchSeg>
          <<<gridDim, blockDim, smem_size>>>(segStarts);
      );
    if (thread_nz == 2)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<2, 2>
          <<<gridDim, blockDim, smem_size>>>();
      );
    if (thread_nz == 4)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<2, 4>
          <<<gridDim, blockDim, smem_size>>>();
      );
  } else {
    if (thread_nz == 1)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<1, 1>
          <<<gridDim, blockDim, smem_size>>>();
      );
    if (thread_nz == 2)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<1, 2>
          <<<gridDim, blockDim, smem_size>>>();
      );
    if (thread_nz == 4)
      TIME_GPU(
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) M * N) * 4)); 
      csrspmm_rowcaching_nnzbalance_kernel<1, 4>
          <<<gridDim, blockDim, smem_size>>>();
      );
  }
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaDeviceSynchronize(); 
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();  
}


__global__
void kernel_taco_eb_sr_256_16_2(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
    int A1_dimension = global_A1_dimension;
    int* __restrict__ A2_pos = global_A2_pos_device;
    int* __restrict__ A2_crd = global_A2_crd_device;
    float* __restrict__ A_vals = global_A_vals_device;
    int B2_dimension = global_B2_dimension;
    float* __restrict__ B_vals = global_B_vals_device;
    int C2_dimension = global_C2_dimension;
    float* __restrict__ C_vals = global_C_vals_device;
    int32_t block = blockIdx.x;
    int32_t thread = (threadIdx.x % (32));
    int32_t warp = (threadIdx.x / 32);
    if (threadIdx.x >= 512) {
        return;
    }
    for (int32_t dense_val = 0; dense_val < 2; dense_val++) {
        int32_t k = dense_val * 32 + thread;
        float tnnz_val = 0.0;
        int32_t pA2_begin = i_blockStarts[block]; 
        int32_t pA2_end = i_blockStarts[(block + 1)];
        int32_t fpos1 = warp * 16; 
        int32_t fposA = block * 256 + fpos1; 
        int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA); 
        int32_t i = i_pos;
        for (int32_t nnz = 0; nnz < 16; nnz++) {
        int32_t fpos1 = warp * 16 + nnz; 
        int32_t fposA = block * 256 + fpos1; 
        if (fposA >= A2_pos[A1_dimension])
            break;
        int32_t f = A2_crd[fposA];
        while (fposA == A2_pos[(i_pos + 1)]) {
            i_pos = i_pos + 1;
            i = i_pos; 
        }
        int32_t kC = i * C2_dimension + k; 
        int32_t kB = f * B2_dimension + k; 
        tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB]; 
        if (fposA + 1 == A2_pos[(i_pos + 1)]) { 
            atomicAdd(&C_vals[kC], tnnz_val);
            tnnz_val = 0.0;
        }
    }
    int32_t kC = i * C2_dimension + k;
    atomicAdd(&C_vals[kC], tnnz_val);
  }
}
void spmm_eb_sr_512_64(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1)));
  TIME_GPU(
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 512, ((A2_pos[A1_dimension] + 255) / 256));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
      kernel_taco_eb_sr_256_16_2<<<(A2_pos[A1_dimension] + 255) / 256, 512>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}

__global__
void kernel_taco_eb_sr_1024_16_1(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
    int A1_dimension = global_A1_dimension;
    int* __restrict__ A2_pos = global_A2_pos_device;
    int* __restrict__ A2_crd = global_A2_crd_device;
    float* __restrict__ A_vals = global_A_vals_device;
    int B2_dimension = global_B2_dimension;
    float* __restrict__ B_vals = global_B_vals_device;
    int C2_dimension = global_C2_dimension;
    float* __restrict__ C_vals = global_C_vals_device;
    int32_t block = blockIdx.x;
    int32_t thread = (threadIdx.x % (4));
    int32_t warp = (threadIdx.x / 4); // 256 = 1024 / 16 * 4
    if (threadIdx.x >= 256) {
        return;
    }
    for (int32_t dense_val = 0; dense_val < 1; dense_val++) {
      int32_t k = dense_val * 4 + thread;
      if (k >= B2_dimension)
        break;
      float tnnz_val = 0.0;
      int32_t pA2_begin = i_blockStarts[block]; 
      int32_t pA2_end = i_blockStarts[(block + 1)];
      int32_t fpos1 = warp * 16;
      int32_t fposA = block * 1024 + fpos1;
      int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA); 
      int32_t i = i_pos;
      for (int32_t nnz = 0; nnz < 16; nnz++) {
        int32_t fpos1 = warp * 16 + nnz;
        int32_t fposA = block * 1024 + fpos1;
        if (fposA >= A2_pos[A1_dimension]) 
          break;
        int32_t f = A2_crd[fposA];
        while (fposA == A2_pos[(i_pos + 1)]) {
            i_pos = i_pos + 1;
            i = i_pos; 
        }
        int32_t kC = i * C2_dimension + k; 
        int32_t kB = f * B2_dimension + k; 
        tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB]; 
        if (fposA + 1 == A2_pos[(i_pos + 1)]) { 
            atomicAdd(&C_vals[kC], tnnz_val);
            tnnz_val = 0.0;
        }
      }
    int32_t kC = i * C2_dimension + k;
    atomicAdd(&C_vals[kC], tnnz_val);
    }
}

void spmm_eb_sr_256_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 1023) / 1024 + 1)));
  TIME_GPU(
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 1024, (int32_t) 256, ((A2_pos[A1_dimension] + 1023) / 1024));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
      kernel_taco_eb_sr_1024_16_1<<<(A2_pos[A1_dimension] + 1023) / 1024, 256>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}

template<int g, int c>
__global__
void kernel_taco_eb_sr_tune(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
    int A1_dimension = global_A1_dimension;
    int* __restrict__ A2_pos = global_A2_pos_device;
    int* __restrict__ A2_crd = global_A2_crd_device;
    float* __restrict__ A_vals = global_A_vals_device;
    int B2_dimension = global_B2_dimension;
    float* __restrict__ B_vals = global_B_vals_device;
    int C2_dimension = global_C2_dimension;
    float* __restrict__ C_vals = global_C_vals_device;
    int32_t block = blockIdx.x;
    int32_t thread = (threadIdx.x % (4 / c));
    int32_t warp = (threadIdx.x / (4 / c)); // 256 = 1024 / 16 * 4
    if (threadIdx.x >= 256) {
        return;
    }
    for (int32_t dense_val = 0; dense_val < c; dense_val++) {
      int32_t k = dense_val * 4 / c + thread;
      if (k >= B2_dimension)
        break;
      float tnnz_val = 0.0;
      int32_t pA2_begin = i_blockStarts[block]; 
      int32_t pA2_end = i_blockStarts[(block + 1)];
      int32_t fpos1 = warp * g;
      int32_t fposA = block * (64 * g * c) + fpos1;
      int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA); 
      int32_t i = i_pos;
      for (int32_t nnz = 0; nnz < g; nnz++) {
        int32_t fpos1 = warp * g + nnz;
        int32_t fposA = block * (64 * g * c) + fpos1;
        if (fposA >= A2_pos[A1_dimension]) 
          break;
        int32_t f = A2_crd[fposA];
        while (fposA == A2_pos[(i_pos + 1)]) {
            i_pos = i_pos + 1;
            i = i_pos; 
        }
        int32_t kC = i * C2_dimension + k; 
        int32_t kB = f * B2_dimension + k; 
        tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB]; 
        if (fposA + 1 == A2_pos[(i_pos + 1)]) { 
            atomicAdd(&C_vals[kC], tnnz_val);
            tnnz_val = 0.0;
        }
      }
    int32_t kC = i * C2_dimension + k;
    atomicAdd(&C_vals[kC], tnnz_val);
    }
}


void spmm_eb_sr_256_4_1(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
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
      kernel_taco_eb_sr_tune<4,1><<<(A2_pos[A1_dimension] + 255) / 256, 256>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_sr_256_4_2(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 511) / 512 + 1)));
  TIME_GPU(
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 512, (int32_t) 256, ((A2_pos[A1_dimension] + 511) / 512));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
      kernel_taco_eb_sr_tune<4,2><<<(A2_pos[A1_dimension] + 511) / 512, 256>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_sr_256_4_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 1023) / 1024 + 1)));
  TIME_GPU(
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 1024, (int32_t) 256, ((A2_pos[A1_dimension] + 1023) / 1024));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
      kernel_taco_eb_sr_tune<4,4><<<(A2_pos[A1_dimension] + 1023) / 1024, 256>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_sr_256_8_1(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 511) / 512 + 1)));
  TIME_GPU(
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 512, (int32_t) 256, ((A2_pos[A1_dimension] + 511) / 512));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
      kernel_taco_eb_sr_tune<8,1><<<(A2_pos[A1_dimension] + 511) / 512, 256>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_sr_256_8_2(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 1023) / 1024 + 1)));
  TIME_GPU(
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 1024, (int32_t) 256, ((A2_pos[A1_dimension] + 1023) / 1024));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
      kernel_taco_eb_sr_tune<8,2><<<(A2_pos[A1_dimension] + 1023) / 1024, 256>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_sr_256_8_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 2047) / 2048 + 1)));
  TIME_GPU(
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 2048, (int32_t) 256, ((A2_pos[A1_dimension] + 2047) / 2048));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
      kernel_taco_eb_sr_tune<8,4><<<(A2_pos[A1_dimension] + 2047) / 2048, 256>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_sr_256_16_1(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 1023) / 1024 + 1)));
  TIME_GPU(
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 1024, (int32_t) 256, ((A2_pos[A1_dimension] + 1023) / 1024));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
      kernel_taco_eb_sr_tune<16,1><<<(A2_pos[A1_dimension] + 1023) / 1024, 256>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_sr_256_16_2(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 2047) / 2048 + 1)));
  TIME_GPU(
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 2048, (int32_t) 256, ((A2_pos[A1_dimension] + 2047) / 2048));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
      kernel_taco_eb_sr_tune<16,2><<<(A2_pos[A1_dimension] + 2047) / 2048, 256>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}
void spmm_eb_sr_256_16_4(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  copy_to_device_spmm(C, A, B);

  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 4095) / 4096 + 1)));
  TIME_GPU(
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(global_A2_pos_host_copy, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 4096, (int32_t) 256, ((A2_pos[A1_dimension] + 4095) / 4096));
      gpuErrchk(cudaMemset(global_C_vals_host_copy, 0, ((size_t) C2_dimension * C1_dimension) * 4));
      kernel_taco_eb_sr_tune<16,4><<<(A2_pos[A1_dimension] + 4095) / 4096, 256>>>(A, B, C, i_blockStarts);
    );
  cudaDeviceSynchronize(); 
  gpuErrchk(cudaGetLastError());
  cudaFree(i_blockStarts);
  copy_from_device_spmm(C, A, B);
  free_tensors_spmm();
}