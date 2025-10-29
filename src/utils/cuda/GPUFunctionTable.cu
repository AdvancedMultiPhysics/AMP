#ifndef included_AMP_GPUFunctionTable_cu
#define included_AMP_GPUFunctionTable_cu

#define included_AMP_AMP_GPUFunctionTable_HPP_

#include <cuda.h>
#include <cuda_runtime.h>

#include "AMP/utils/cuda/GPUFunctionTable.hpp"


namespace AMP{

//Kernels
template <class TYPE, typename LAMBDA>
__global__ void transform(LAMBDA &fun, TYPE* d_x, TYPE* d_y, size_t n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= blockDim.x*gridDim.x){
        d_y[i] = fun(d_x[i]);
    }
}

template <class TYPE, typename LAMBDA>
__global__ void transform(LAMBDA &fun, TYPE* d_x, TYPE* d_y, TYPE* d_z, size_t n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= blockDim.x*gridDim.x){
        d_z[i] = fun( d_x[i], d_y[i]);
    }
}


template <class TYPE>
__global__ void transformReLUKernel(const TYPE* d_a, TYPE* d_b, size_t n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= blockDim.x*gridDim.x){
        d_b[i] = fmax(d_a[i], static_cast<TYPE>(0));
    }

}

template <class TYPE>
__global__ void transformAbsKernel(const TYPE* d_a, TYPE* d_b, size_t n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= blockDim.x*gridDim.x){
        d_b[i] = fabs(d_a[i]);
    }
}

template <class TYPE>
__global__ void transformTanhKernel(const TYPE* d_a, TYPE* d_b, size_t n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= blockDim.x*gridDim.x){
        d_b[i] = tanh(d_a[i]);
    }
}

template <class TYPE>
__global__ void transformHardTanhKernel(const TYPE* d_a, TYPE* d_b, size_t n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= blockDim.x*gridDim.x){
        d_b[i] = fmax( -1.0, fmin(1.0, d_a[i]));
    }
}

template <class TYPE>
__global__ void transformSigmoidKernel(const TYPE* d_a, TYPE* d_b, size_t n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= blockDim.x*gridDim.x){
        d_b[i] = 1.0 / (1.0 + exp(-d_a[i]));
    }
}

template <class TYPE>
__global__ void transformSoftPlusKernel(const TYPE* d_a, TYPE* d_b, size_t n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i+= blockDim.x*gridDim.x){
        d_b[i] = log1p(exp(d_a[i]));
    }
}

template <class TYPE, unsigned int blockSize>
__global__ void sumKernel(const TYPE* g_idata,TYPE* g_odata, size_t n)
{

    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;

    while(i < n){
        sdata[tid] += g_idata[i];
        if((i + blockSize) < n){
            sdata[tid] += g_idata[i+blockSize];
        }
        i+= gridSize;
    }
    __syncthreads();

    if(blockSize >= 512){ if(tid < 256) { sdata[tid] += sdata[tid + 256];} __syncthreads();}
    if(blockSize >= 256){ if(tid < 128) { sdata[tid] += sdata[tid + 128];} __syncthreads();}
    if(blockSize >= 128){ if(tid < 64) { sdata[tid] += sdata[tid + 64];} __syncthreads();}

    if(blockSize >= 64){ if(tid < 32){ sdata[tid] += sdata[tid + 32];} }
    __syncthreads();
    if(blockSize >= 32){ if(tid < 16){ sdata[tid] += sdata[tid + 16];} }
    __syncthreads();
    if(blockSize >= 16){ if(tid < 8) { sdata[tid] += sdata[tid + 8]; } }
    __syncthreads();
    if(blockSize >= 8){  if(tid < 4) { sdata[tid] += sdata[tid + 4]; } }
    __syncthreads();
    if(blockSize >= 4){  if(tid < 2) { sdata[tid] += sdata[tid + 2]; } }
    __syncthreads();
    if(blockSize >= 2){  if(tid < 1) { sdata[tid] += sdata[tid + 1]; } }
    __syncthreads();

    if(tid == 0) g_odata[blockIdx.x] = sdata[0];
}

template <class TYPE, unsigned int blockSize>
__global__ void equalsKernel(const TYPE* d_a,const TYPE* d_b, bool* g_odata, size_t n, TYPE tol)
{

    extern __shared__ bool esdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockSize*2 + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    esdata[tid] = true;

    while(i < n){
        esdata[tid] = esdata[tid] && (fabs(d_a[i] - d_b[i]) < tol);
        if((i + blockSize) < n){
            esdata[tid] = esdata[tid] && (fabs(d_a[i+blockSize] - d_b[i+blockSize]) < tol);
        }
        i+= gridSize;
    }
    __syncthreads();

    if(blockSize >= 512){ if(tid < 256) { esdata[tid] = esdata[tid] && esdata[tid + 256];} __syncthreads();}
    if(blockSize >= 256){ if(tid < 128) { esdata[tid] = esdata[tid] && esdata[tid + 128];} __syncthreads();}
    if(blockSize >= 128){ if(tid < 64) { esdata[tid] =  esdata[tid] && esdata[tid + 64];} __syncthreads();}

    if(blockSize >= 64){ if(tid < 32){ esdata[tid] = esdata[tid] && esdata[tid + 32];} }
    __syncthreads();
    if(blockSize >= 32){ if(tid < 16){ esdata[tid] = esdata[tid] && esdata[tid + 16];} }
    __syncthreads();
    if(blockSize >= 16){ if(tid < 8) { esdata[tid] = esdata[tid] && esdata[tid + 8]; } }
    __syncthreads();
    if(blockSize >= 8){  if(tid < 4) { esdata[tid] = esdata[tid] && esdata[tid + 4]; } }
    __syncthreads();
    if(blockSize >= 4){  if(tid < 2) { esdata[tid] = esdata[tid] && esdata[tid + 2]; } }
    __syncthreads();
    if(blockSize >= 2){  if(tid < 1) { esdata[tid] = esdata[tid] && esdata[tid + 1]; } }
    __syncthreads();

    if(tid == 0) g_odata[blockIdx.x] = esdata[0];
}

//Wrappers
template <class TYPE>
void transformReLUW(const TYPE* d_a, TYPE* d_b, size_t n)
{   
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims(n,BlockDim,GridDim); 
    transformReLUKernel<TYPE><<<GridDim,BlockDim>>>(d_a, d_b, n);
    cudaDeviceSynchronize();

}

template <class TYPE>
void transformAbsW(const TYPE* d_a, TYPE* d_b, size_t n)
{   
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims(n,BlockDim,GridDim); 
    transformAbsKernel<TYPE><<<GridDim,BlockDim>>>(d_a, d_b, n);
    cudaDeviceSynchronize();

}

template <class TYPE>
void transformTanhW(const TYPE* d_a, TYPE* d_b, size_t n)
{   
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims(n,BlockDim,GridDim); 
    transformTanhKernel<TYPE><<<GridDim,BlockDim>>>(d_a, d_b, n);
    cudaDeviceSynchronize();

}

template <class TYPE>
void transformHardTanhW(const TYPE* d_a, TYPE* d_b, size_t n)
{   
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims(n,BlockDim,GridDim); 
    transformHardTanhKernel<TYPE><<<GridDim,BlockDim>>>( d_a, d_b, n);
    cudaDeviceSynchronize();

}

template <class TYPE>
void transformSigmoidW(const TYPE* d_a, TYPE* d_b, size_t n)
{   
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims(n,BlockDim,GridDim); 
    transformSigmoidKernel<TYPE><<<GridDim,BlockDim>>>(d_a, d_b, n);
    cudaDeviceSynchronize();

}

template <class TYPE>
void transformSoftPlusW(const TYPE* d_a, TYPE* d_b, size_t n)
{   
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims(n,BlockDim,GridDim); 
    transformSoftPlusKernel<TYPE><<<GridDim,BlockDim>>>(d_a, d_b, n);
    cudaDeviceSynchronize();

}

template <class TYPE>
TYPE sumW(const TYPE* d_a, size_t n)
{
    unsigned int threads = 128;
    int blocks = 64;
    dim3 gridSize(blocks,1,1);
    dim3 blockSize(threads,1,1);
    int smemsize = (threads <= 32) ? 2 * threads * sizeof(TYPE) : threads * sizeof(TYPE);
    TYPE* d_odata, *h_odata;
    size_t bytes = sizeof(TYPE)*blocks;
    h_odata = (TYPE*)malloc(bytes);
    cudaMalloc((void**)&d_odata, bytes); 
    sumKernel<TYPE, 128><<<gridSize,blockSize,smemsize>>>(d_a,d_odata,n);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost);
    TYPE sum = (TYPE) 0;
    for(int i = 0; i < blocks; i++){
        sum += h_odata[i];
    }
    free(h_odata);
    cudaFree(d_odata);
    return sum;
}


template <class TYPE>
bool equalsW(const TYPE* d_a, const TYPE* d_b, TYPE tol, size_t n)
{
    bool eq = true;
    unsigned int threads = 128;
    int blocks = 64;
    dim3 gridSize(blocks,1,1);
    dim3 blockSize(threads,1,1);
    int smemsize = (threads <= 32) ? 2 * threads * sizeof(bool) : threads * sizeof(bool);
    bool* d_odata, *h_odata;
    size_t bytes = sizeof(bool)*blocks;
    h_odata = (bool*)malloc(bytes);
    cudaMalloc((void**)&d_odata, bytes); 
    equalsKernel<TYPE, 128><<<gridSize,blockSize,smemsize>>>(d_a,d_b,d_odata,n,tol);
    cudaDeviceSynchronize();
    cudaMemcpy(h_odata, d_odata, bytes, cudaMemcpyDeviceToHost);
    for(int i = 0; i < blocks; i++){
        eq = eq && h_odata[i];
    }
    free(h_odata);
    cudaFree(d_odata);
    return eq;

} 

//Explicit Instantiation of the wrappers
template double sumW<double>(const double* d_a, size_t n);
template float sumW<float>(const float* d_a, size_t n);
template bool equalsW<double>(const double* d_a, const double* d_b, double tol, size_t n);
template bool equalsW<float>(const float* d_a, const float* d_b, float tol, size_t n);

template void transformReLUW(const double* d_a, double* d_b, size_t n);
template void transformAbsW(const double* d_a, double* d_b, size_t n);
template void transformTanhW(const double* d_a, double* d_b, size_t n);
template void transformHardTanhW(const double* d_a, double* d_b, size_t n);
template void transformSigmoidW(const double* d_a, double* d_b, size_t n);
template void transformSoftPlusW(const double* d_a, double* d_b, size_t n);

}


/********************************************************
 *  Explicit instantiations of GPUFunctionTable          *
 ********************************************************/
template class AMP::GPUFunctionTable<float>;
template class AMP::GPUFunctionTable<double>;

#endif
