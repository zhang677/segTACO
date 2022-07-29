
#include "gpu_library.h"
//#include "gpu_kernels.cuh"
//#include "cusparse.h"
#include "cusparse_v2.h"
#include "cublas_v2.h"
/*
void spmm_csr_gpu_cusparse_cu100(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
    
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descra=0;

    int nr = (int)(C->dimensions[0]); //nr
    int sc = (int)(C->dimensions[1]); //sc
    int nc = (int)(B->dimensions[0]); //nc, col-major
    int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
    int A1_dimension = (int)(A->dimensions[0]); //nr
    long nnz = A2_pos[A1_dimension]; //nnz

    copy_to_device_spmm(C, A, B);

    float* __restrict__ cy_out = global_C_vals_host_copy;
    float* __restrict__ cy_in = global_B_vals_host_copy;
    int* __restrict__ csr_v = global_A2_pos_host_copy;
    int* __restrict__ csr_e = global_A2_crd_host_copy;
    float* __restrict__ csr_ev = global_A_vals_host_copy;

    // initialize cusparse library 
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // create and setup matrix descriptor 
    CUSPARSE_CHECK(cusparseCreateMatDescr(&descra));

    cusparseSetMatType(descra,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descra,CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    const float alpha=1.0f, beta=0.0f;

TIME_GPU(
    CUSPARSE_CHECK(cusparseDcsrmm2(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,  nr, sc, nc, nnz,
                           &alpha, descra, csr_ev, csr_v, csr_e, cy_in, sc, &beta, cy_out, nr));
);

    copy_from_device_spmm(C, A, B);
    free_tensors_spmm();
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descra));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    
}*/

void spmm_csr_gpu_cusparse(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
    cusparseHandle_t handle=0;
    cusparseSpMatDescr_t matA=0;
    cusparseDnMatDescr_t matB=0;
    cusparseDnMatDescr_t matC=0;


    size_t space=0;
    float *workspace;
    const float alpha=1.0f;
    const float beta=0.0f;
    cusparseSpMMAlg_t alg=CUSPARSE_SPMM_CSR_ALG2;
    
    int nr = (int)(C->dimensions[0]); //nr
    int sc = (int)(C->dimensions[1]); //sc
    int nc = (int)(B->dimensions[0]); //nc,row-major
    int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
    int A1_dimension = (int)(A->dimensions[0]); //nr
    long nnz = A2_pos[A1_dimension]; //nnz

//    gpuErrchk((cudaMalloc(&Ct)),sizeof(float)*nr*sc));
//   cublasCreate(&(cublas_handle));
//    cublasSetPointerMode(cublas_handle,CUBLAS_POINTER_MODE_HOST);

    CUSPARSE_CHECK(cusparseCreate(&handle));
    copy_to_device_spmm(C, A, B);

    
    CUSPARSE_CHECK(cusparseCreateCsr(&matA,
        nr,nc,nnz,global_A2_pos_host_copy,global_A2_crd_host_copy,
        global_A_vals_host_copy,CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB,
        nc,
        sc,
        sc,
        global_B_vals_host_copy,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW
    ));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC,
        nr,
        sc,
        sc,
        global_C_vals_host_copy,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW
    ));

    
    CUSPARSE_CHECK( cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,//CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        alg,
        &space
    ));

    gpuErrchk( cudaMalloc(&workspace,space));
TIME_GPU(
    CUSPARSE_CHECK(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,//CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        alg,
        workspace
    ));
    
);   
    
    
    cublasHandle_t cublas_handle = 0;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetPointerMode(cublas_handle,CUBLAS_POINTER_MODE_HOST));
    float *Ct;
    const int ldb_Ct = sc;  // leading demension
    int ldb_C = nr;
    gpuErrchk(cudaMalloc((void**)&Ct, sizeof(float)*ldb_Ct*nr));
    CUBLAS_CHECK(cublasSgeam(
        cublas_handle, 
        CUBLAS_OP_T, 
        CUBLAS_OP_T, 
        nr, sc, 
        &alpha, 
        global_C_vals_host_copy, ldb_Ct, 
        &beta, 
        global_C_vals_host_copy, ldb_Ct, 
        Ct, ldb_C
    ));


    //copy_from_device_spmm(C, A, B);
    float*  C_vals = (float*)(C->vals);
    gpuErrchk(cudaMemcpy(C_vals, Ct, sizeof(float)*ldb_Ct*nr, cudaMemcpyDeviceToHost));
    //C->vals = (uint8_t*)C_vals;
    C->vals = C_vals;
    gpuErrchk( cudaFree(workspace) );
    free_tensors_spmm();
    gpuErrchk( cudaFree(Ct) );
    CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));


}
void spmm_csr_cusparse_row(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B){
    cusparseHandle_t handle=0;
    cusparseSpMatDescr_t matA=0;
    cusparseDnMatDescr_t matB=0;
    cusparseDnMatDescr_t matC=0;


    size_t space=0;
    float *workspace;
    const float alpha=1.0f;
    const float beta=0.0f;
    cusparseSpMMAlg_t alg=CUSPARSE_SPMM_CSR_ALG2;
    
    int nr = (int)(C->dimensions[0]); //nr
    int sc = (int)(C->dimensions[1]); //sc
    int nc = (int)(B->dimensions[0]); //nc,row-major
    int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
    int A1_dimension = (int)(A->dimensions[0]); //nr
    long nnz = A2_pos[A1_dimension]; //nnz


    CUSPARSE_CHECK(cusparseCreate(&handle));
    copy_to_device_spmm(C, A, B);

    
    CUSPARSE_CHECK(cusparseCreateCsr(&matA,
        nr,nc,nnz,global_A2_pos_host_copy,global_A2_crd_host_copy,
        global_A_vals_host_copy,CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matB,
        nc,
        sc,
        sc,
        global_B_vals_host_copy,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW
    ));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC,
        nr,
        sc,
        sc,
        global_C_vals_host_copy,
        CUDA_R_32F,
        CUSPARSE_ORDER_ROW
    ));

    
    CUSPARSE_CHECK( cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,//CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        alg,
        &space
    ));

    gpuErrchk( cudaMalloc(&workspace,space));
TIME_GPU(
    CUSPARSE_CHECK(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,//CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_32F,
        alg,
        workspace
    ));
    
);   
    cudaDeviceSynchronize(); 
    gpuErrchk(cudaGetLastError());  
    gpuErrchk( cudaFree(workspace) );
    copy_from_device_spmm(C, A, B);
    free_tensors_spmm();
    CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    
}