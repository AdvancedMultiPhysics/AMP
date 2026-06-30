#include "AMP/matrices/operations/device/DeviceMatrixOperations.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/device/Device.h"

namespace AMP {
namespace LinearAlgebra {

// sparce matrix vector multiplication
template<typename L, typename S>
__global__ void mult_kernel( const L *__restrict__ row_starts,
                             const L *__restrict__ cols_loc,
                             const S *__restrict__ coeffs,
                             const unsigned N,
                             const S *__restrict__ x,
                             S *__restrict__ y )
{
    for ( unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
          i += blockDim.x * gridDim.x ) {
        L start = row_starts[i];
        L end   = row_starts[i + 1];
        S sum   = 0.0;
        for ( L j = start; j < end; j++ )
            sum += coeffs[j] * x[cols_loc[j]];
        y[i] += sum;
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::mult( const L *row_starts,
                                            const L *cols_loc,
                                            const S *coeffs,
                                            const size_t N,
                                            const S *in,
                                            S *out,
                                            AMP::Utilities::ComputeStream stream )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, mult_kernel<L, S>, BlockDim, GridDim );
    mult_kernel<<<GridDim, BlockDim, 0, stream>>>( row_starts, cols_loc, coeffs, N, in, out );
    deviceStreamSynchronize( stream );
}

// scale
template<typename S>
__global__ void scale_kernel( const size_t N, S *__restrict__ x, const S alpha )
{
    for ( unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
          i += blockDim.x * gridDim.x ) {
        x[i] *= alpha;
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::scale( const size_t N,
                                             S *x,
                                             const S alpha,
                                             AMP::Utilities::ComputeStream stream )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, scale_kernel<S>, BlockDim, GridDim );
    scale_kernel<<<GridDim, BlockDim, 0, stream>>>( N, x, alpha );
    deviceStreamSynchronize( stream );
}

// axpy
template<typename S>
__global__ void axpy_kernel( const size_t N, const S alpha, S *__restrict__ x, S *__restrict__ y )
{
    for ( unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
          i += blockDim.x * gridDim.x ) {
        y[i] += alpha * x[i];
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::axpy(
    const size_t N, const S alpha, S *x, S *y, AMP::Utilities::ComputeStream stream )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, axpy_kernel<S>, BlockDim, GridDim );
    axpy_kernel<<<GridDim, BlockDim, 0, stream>>>( N, alpha, x, y );
    deviceStreamSynchronize( stream );
}

// copy
template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::copy( const size_t N,
                                            const S *__restrict__ x,
                                            S *__restrict__ y,
                                            AMP::Utilities::ComputeStream stream )
{
    deviceMemcpyAsync( y, x, N * sizeof( S ), deviceMemcpyDeviceToDevice, stream );
    deviceStreamSynchronize( stream );
}

// extract diagonal
template<typename L, typename S>
__global__ static void extractDiagonal_kernel( const L *row_starts,
                                               const S *__restrict__ coeffs,
                                               const size_t N,
                                               S *__restrict__ diag )
{
    for ( unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
          i += blockDim.x * gridDim.x ) {
        diag[i] = coeffs[row_starts[i]];
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::extractDiagonal( const L *row_starts,
                                                       const S *coeffs,
                                                       const size_t N,
                                                       S *diag,
                                                       AMP::Utilities::ComputeStream stream )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, extractDiagonal_kernel<L, S>, BlockDim, GridDim );
    extractDiagonal_kernel<<<GridDim, BlockDim, 0, stream>>>( row_starts, coeffs, N, diag );
    deviceStreamSynchronize( stream );
}

// set diagonal
template<typename L, typename S>
__global__ static void setDiagonal_kernel( const L *__restrict__ row_starts,
                                           S *__restrict__ coeffs,
                                           const size_t N,
                                           const S *__restrict__ diag )
{
    for ( unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
          i += blockDim.x * gridDim.x ) {
        coeffs[row_starts[i]] = diag[i];
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::setDiagonal( const L *row_starts,
                                                   S *coeffs,
                                                   const size_t N,
                                                   const S *diag,
                                                   AMP::Utilities::ComputeStream stream )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, setDiagonal_kernel<L, S>, BlockDim, GridDim );
    setDiagonal_kernel<<<GridDim, BlockDim, 0, stream>>>( row_starts, coeffs, N, diag );
    deviceStreamSynchronize( stream );
}

// set identity
template<typename L, typename S>
__global__ static void
setIdentity_kernel( const L *__restrict__ row_starts, S *__restrict__ coeffs, const size_t N )
{
    for ( unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
          i += blockDim.x * gridDim.x ) {
        coeffs[row_starts[i]] = 1.0;
    }
}

template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::setIdentity( const L *row_starts,
                                                   S *coeffs,
                                                   const size_t N,
                                                   AMP::Utilities::ComputeStream stream )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, setIdentity_kernel<L, S>, BlockDim, GridDim );
    setIdentity_kernel<<<GridDim, BlockDim, 0, stream>>>( row_starts, coeffs, N );
    deviceStreamSynchronize( stream );
}

// Linf norms
template<typename L, typename S>
__global__ static void LinfNorm_kernel( const size_t N,
                                        const S *__restrict__ x,
                                        const L *__restrict__ row_starts,
                                        S *__restrict__ row_sums )
{
    for ( unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
          i += blockDim.x * gridDim.x ) {
        const auto start = row_starts[i];
        const auto end   = row_starts[i + 1];
        S sum            = 0.0;
        for ( auto j = start; j < end; j++ ) {
            sum += abs( x[j] );
        }
        row_sums[i] += sum;
    }
}

// LinfNorm for diagonal only matrix
template<typename G, typename L, typename S>
void DeviceMatrixOperations<G, L, S>::LinfNorm( const size_t N,
                                                const S *x,
                                                const L *row_starts,
                                                S *row_sums,
                                                AMP::Utilities::ComputeStream stream )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, LinfNorm_kernel<L, S>, BlockDim, GridDim );
    LinfNorm_kernel<<<GridDim, BlockDim, 0, stream>>>( N, x, row_starts, row_sums );
    deviceStreamSynchronize( stream );
}

} // namespace LinearAlgebra
} // namespace AMP
