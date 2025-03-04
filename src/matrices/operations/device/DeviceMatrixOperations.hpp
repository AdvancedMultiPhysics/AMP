#include "AMP/matrices/operations/device/DeviceMatrixOperations.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/device/device.h"

namespace AMP {
namespace LinearAlgebra {

// sparce matrix vector multiplication
template<typename L, typename S>
__global__ void mult_kernel(
    const L *row_starts, const L *cols_loc, const S *coeffs, const unsigned N, const S *x, S *y )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x ) {
        int start = row_starts[i];
        int end   = row_starts[i + 1];
        for ( int j = start; j < end; j++ )
            y[i] += coeffs[j] * x[cols_loc[j]];
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::mult(
    const L *row_starts, const L *cols_loc, const S *coeffs, const size_t N, const S *in, S *out )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    mult_kernel<<<GridDim, BlockDim>>>( row_starts, cols_loc, coeffs, N, in, out );
    deviceSynchronize();
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::mult( const L *row_starts,
                                            const L *cols_loc,
                                            const S *coeffs,
                                            const size_t N,
                                            const S *in_h,
                                            const size_t Ng,
                                            S *out )
{
    S *in;
    deviceMalloc( &in, Ng * sizeof( S ) );
    deviceMemcpy( in, in_h, Ng * sizeof( S ), deviceMemcpyHostToDevice );

    const auto in_d = in;

    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    mult_kernel<<<GridDim, BlockDim>>>( row_starts, cols_loc, coeffs, N, in_d, out );
    deviceSynchronize();
    deviceFree( in );
}

// set to scalar
template<typename S>
__global__ void setScalar_kernel( const size_t N, S *x, const S alpha )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x ) {
        x[i] = alpha;
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::setScalar( const size_t N, S *x, const S alpha )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    setScalar_kernel<<<GridDim, BlockDim>>>( N, x, alpha );
    deviceSynchronize();
}

// scale
template<typename S>
__global__ void scale_kernel( const size_t N, S *x, const S alpha )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x ) {
        x[i] *= alpha;
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::scale( const size_t N, S *x, const S alpha )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    scale_kernel<<<GridDim, BlockDim>>>( N, x, alpha );
    deviceSynchronize();
}

// axpy
template<typename S>
__global__ void axpy_kernel( const size_t N, const S alpha, S *x, S *y )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x ) {
        y[i] += alpha * x[i];
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::axpy( const size_t N, const S alpha, S *x, S *y )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    axpy_kernel<<<GridDim, BlockDim>>>( N, alpha, x, y );
    deviceSynchronize();
}

// copy
template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::copy( const size_t N, const S *x, S *y )
{
    deviceMemcpy( y, x, N * sizeof( S ), deviceMemcpyDeviceToDevice );
}

// extract diagonal
template<typename L, typename S>
__global__ static void
extractDiagonal_kernel( const L *row_starts, const S *coeffs, const size_t N, S *diag )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x ) {
        diag[i] = coeffs[row_starts[i]];
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::extractDiagonal( const L *row_starts,
                                                       const S *coeffs,
                                                       const size_t N,
                                                       S *diag )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    extractDiagonal_kernel<<<GridDim, BlockDim>>>( row_starts, coeffs, N, diag );
    deviceSynchronize();
}

// set diagonal
template<typename L, typename S>
__global__ static void
setDiagonal_kernel( const L *row_starts, S *coeffs, const size_t N, const S *diag )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x ) {
        coeffs[row_starts[i]] = diag[i];
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::setDiagonal( const L *row_starts,
                                                   S *coeffs,
                                                   const size_t N,
                                                   const S *diag )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    setDiagonal_kernel<<<GridDim, BlockDim>>>( row_starts, coeffs, N, diag );
    deviceSynchronize();
}

// set identity
template<typename L, typename S>
__global__ static void setIdentity_kernel( const L *row_starts, S *coeffs, const size_t N )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x ) {
        coeffs[row_starts[i]] = 1.0;
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::setIdentity( const L *row_starts, S *coeffs, const size_t N )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    setIdentity_kernel<<<GridDim, BlockDim>>>( row_starts, coeffs, N );
    deviceSynchronize();
}

// Linf norms
template<typename L, typename S>
__global__ static void
LinfNorm_kernel( const size_t N, const S *x, const L *row_starts, S *row_sums )
{
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x ) {
        const auto start = row_starts[i];
        const auto end   = row_starts[i + 1];

        for ( auto j = start; j < end; j++ ) {
            row_sums[i] += abs( x[j] );
        }
    }
}

// LinfNorm for diagonal only matrix
template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::LinfNorm( const size_t N,
                                                const S *x,
                                                const L *row_starts,
                                                S *row_sums )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    LinfNorm_kernel<<<GridDim, BlockDim>>>( N, x, row_starts, row_sums );
    deviceSynchronize();
}

} // namespace LinearAlgebra
} // namespace AMP
