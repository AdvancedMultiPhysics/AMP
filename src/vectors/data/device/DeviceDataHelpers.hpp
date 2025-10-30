#ifndef included_AMP_DeviceDataHelpers_hpp
#define included_AMP_DeviceDataHelpers_hpp

#include "AMP/IO/PIO.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/device/Device.h"
#include "AMP/vectors/data/device/DeviceDataHelpers.h"

#include <string>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/logical.h>
#include <thrust/scatter.h>
#include <thrust/transform.h>

namespace AMP {
namespace LinearAlgebra {

template<typename TYPE>
void print( const std::string &title, const size_t &N, const TYPE *vals )
{
    AMP::pout << title << ", number of values " << N << std::endl;
    for ( size_t i = 0; i < N; ++i )
        AMP::pout << "vals[" << i << "] " << vals[i] << std::endl;
}

template<typename STYPE, typename DTYPE>
__global__ void
set_vals_kernel( const size_t N, const size_t *indices, const STYPE *src, DTYPE *dst )
{
    if constexpr ( std::is_same_v<STYPE, DTYPE> ) {
        for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x )
            dst[indices[i]] = src[i];
    } else {
        for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x )
            dst[indices[i]] = static_cast<DTYPE>( src[i] );
    }
}

template<typename STYPE, typename DTYPE>
bool DeviceDataHelpers<STYPE, DTYPE>::containsIndex( const size_t N,
                                                     const size_t *indices,
                                                     const size_t i )
{
    thrust::device_ptr<const size_t> ndx_ptr = thrust::device_pointer_cast( indices );

    auto it = thrust::find( ndx_ptr, ndx_ptr + N, i );
    return ( it != ( ndx_ptr + N ) );
}

template<typename STYPE, typename DTYPE>
bool DeviceDataHelpers<STYPE, DTYPE>::allGhostIndices( const size_t N,
                                                       const size_t *indices,
                                                       const size_t start,
                                                       const size_t end )
{
    thrust::device_ptr<const size_t> ndx_ptr = thrust::device_pointer_cast( indices );

    auto out_of_range = [start, end] __host__ __device__( const size_t &x ) {
        return x < start || x >= end;
    };
    return thrust::all_of( ndx_ptr, ndx_ptr + N, out_of_range );
}

template<typename STYPE, typename DTYPE>
void DeviceDataHelpers<STYPE, DTYPE>::setValuesByIndex( const size_t N,
                                                        const size_t *indices,
                                                        const STYPE *src,
                                                        DTYPE *dst )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    set_vals_kernel<<<GridDim, BlockDim>>>( N, indices, src, dst );
    // deviceSynchronize();
}

template<typename STYPE, typename DTYPE>
__global__ void
add_vals_kernel( const size_t N, const size_t *indices, const STYPE *src, DTYPE *dst )
{
#if 1
    if constexpr ( std::is_same_v<STYPE, DTYPE> ) {
        for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x )
            atomicAdd( &dst[indices[i]], src[i] );
    } else {
        for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x )
            atomicAdd( &dst[indices[i]], static_cast<DTYPE>( src[i] ) );
    }
#else
    if constexpr ( std::is_same_v<STYPE, DTYPE> ) {
        for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x )
            dst[indices[i]] += src[i];
    } else {
        for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x )
            dst[indices[i]] += static_cast<DTYPE>( src[i] );
    }
#endif
}

template<typename STYPE, typename DTYPE>
void DeviceDataHelpers<STYPE, DTYPE>::addValuesByIndex( const size_t N,
                                                        const size_t *indices,
                                                        const STYPE *src,
                                                        DTYPE *dst )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    add_vals_kernel<<<GridDim, BlockDim>>>( N, indices, src, dst );
    // deviceSynchronize();
}

template<typename STYPE, typename DTYPE>
__global__ void
get_vals_kernel( const size_t N, const size_t *indices, STYPE *const src, DTYPE *dst )
{
    if constexpr ( std::is_same_v<STYPE, DTYPE> ) {
        for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x )
            dst[i] = src[indices[i]];
    } else {
        for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x )
            dst[i] = static_cast<DTYPE>( src[indices[i]] );
    }
}

template<typename STYPE, typename DTYPE>
void DeviceDataHelpers<STYPE, DTYPE>::getValuesByIndex( const size_t N,
                                                        const size_t *indices,
                                                        const STYPE *src,
                                                        DTYPE *dst )
{
    dim3 BlockDim;
    dim3 GridDim;
    setKernelDims( N, BlockDim, GridDim );
    get_vals_kernel<<<GridDim, BlockDim>>>( N, indices, src, dst );
    // deviceSynchronize();
}

template<typename STYPE, typename DTYPE>
void DeviceDataHelpers<STYPE, DTYPE>::setGhostValuesByGlobalID( const size_t gsize,
                                                                const size_t *globalids,
                                                                const size_t N,
                                                                const size_t *ndx,
                                                                const STYPE *src,
                                                                const size_t dst_size,
                                                                DTYPE *dst )
{
    AMP_INSIST( AMP::Utilities::getMemoryType( globalids ) >= AMP::Utilities::MemoryType::managed,
                "globalids not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( ndx ) >= AMP::Utilities::MemoryType::managed,
                "ndx not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( src ) >= AMP::Utilities::MemoryType::managed,
                "src not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( dst ) >= AMP::Utilities::MemoryType::managed,
                "dst not on device" );
    thrust::device_ptr<const size_t> gid_ptr = thrust::device_pointer_cast( globalids );
    thrust::device_ptr<const size_t> ndx_ptr = thrust::device_pointer_cast( ndx );
    thrust::device_ptr<const STYPE> src_ptr  = thrust::device_pointer_cast( src );
    thrust::device_ptr<DTYPE> dst_ptr        = thrust::device_pointer_cast( dst );

    thrust::device_vector<size_t> pos( N );

    // Perform vectorized lower_bound
    thrust::lower_bound(
        thrust::device, gid_ptr, gid_ptr + gsize, ndx_ptr, ndx_ptr + N, pos.begin() );
    thrust::scatter( thrust::device, src_ptr, src_ptr + N, pos.begin(), dst_ptr );
}


template<typename STYPE, typename DTYPE>
void DeviceDataHelpers<STYPE, DTYPE>::addGhostValuesByGlobalID( const size_t gsize,
                                                                const size_t *globalids,
                                                                const size_t N,
                                                                const size_t *ndx,
                                                                const STYPE *src,
                                                                const size_t dst_size,
                                                                DTYPE *dst )
{
    AMP_INSIST( AMP::Utilities::getMemoryType( globalids ) >= AMP::Utilities::MemoryType::managed,
                "globalids not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( ndx ) >= AMP::Utilities::MemoryType::managed,
                "ndx not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( src ) >= AMP::Utilities::MemoryType::managed,
                "src not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( dst ) >= AMP::Utilities::MemoryType::managed,
                "dst not on device" );

    thrust::device_ptr<const size_t> gid_ptr = thrust::device_pointer_cast( globalids );
    thrust::device_ptr<const size_t> ndx_ptr = thrust::device_pointer_cast( ndx );
    thrust::device_ptr<const STYPE> src_ptr  = thrust::device_pointer_cast( src );
    thrust::device_ptr<DTYPE> dst_ptr        = thrust::device_pointer_cast( dst );

    thrust::device_vector<size_t> pos( N );

    // Perform vectorized lower_bound to find positions in destination
    thrust::lower_bound(
        thrust::device, gid_ptr, gid_ptr + gsize, ndx_ptr, ndx_ptr + N, pos.begin() );
    // construct the [begin, end) for the map
    auto begin_map = thrust::make_permutation_iterator( dst_ptr, pos.begin() );
    auto end_map   = thrust::make_permutation_iterator( dst_ptr, pos.end() );

    // add the src vector to the mapped locations using transform with a binary op
    thrust::transform(
        thrust::device, begin_map, end_map, src_ptr, begin_map, thrust::plus<DTYPE>() );
}

template<typename TYPE>
struct pair_plus_op {
    pair_plus_op() = default;
    __host__ __device__ TYPE operator()( const thrust::tuple<TYPE, TYPE> &a )
    {
        return thrust::get<0>( a ) + thrust::get<1>( a );
    }
};

template<typename STYPE, typename DTYPE>
void DeviceDataHelpers<STYPE, DTYPE>::getGhostValuesByGlobalID( const size_t gsize,
                                                                const size_t *globalids,
                                                                const size_t N,
                                                                const size_t *ndx,
                                                                const size_t src_size,
                                                                const STYPE *src1,
                                                                const STYPE *src2,
                                                                DTYPE *dst )
{

    AMP_INSIST( AMP::Utilities::getMemoryType( globalids ) >= AMP::Utilities::MemoryType::managed,
                "globalids not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( ndx ) >= AMP::Utilities::MemoryType::managed,
                "ndx not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( src1 ) >= AMP::Utilities::MemoryType::managed,
                "src1 not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( src2 ) >= AMP::Utilities::MemoryType::managed,
                "src2 not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( dst ) >= AMP::Utilities::MemoryType::managed,
                "dst not on device" );

    // print<size_t>( "Global IDs", gsize, globalids );
    // print<size_t>( "Ghost IDs to map to local", N, ndx );
    // print<STYPE>( "Ghost set buffer values", src_size, src1 );
    // print<STYPE>( "Ghost add buffer values", src_size, src2 );

    thrust::device_ptr<const size_t> gid_ptr = thrust::device_pointer_cast( globalids );
    thrust::device_ptr<const size_t> ndx_ptr = thrust::device_pointer_cast( ndx );
    thrust::device_ptr<const STYPE> src1_ptr = thrust::device_pointer_cast( src1 );
    thrust::device_ptr<const STYPE> src2_ptr = thrust::device_pointer_cast( src2 );
    thrust::device_ptr<DTYPE> dst_ptr        = thrust::device_pointer_cast( dst );

    thrust::device_vector<size_t> pos( N );

    // Perform vectorized lower_bound to find positions in src
    thrust::lower_bound(
        thrust::device, gid_ptr, gid_ptr + gsize, ndx_ptr, ndx_ptr + N, pos.begin() );

    //    print<size_t>( "Positions to map to local", N, thrust::raw_pointer_cast( pos.data() ) );
    // AMP::pout << "Contents of device_vector pos:" << std::endl;
    // thrust::copy( pos.begin(), pos.end(), std::ostream_iterator<float>( AMP::pout, " " ) );
    // AMP::pout << std::endl; // Add a newline after printing

    auto map_data_1_begin = thrust::make_permutation_iterator( src1_ptr, pos.begin() );
    auto map_data_1_end   = thrust::make_permutation_iterator( src1_ptr, pos.end() );

    auto map_data_2_begin = thrust::make_permutation_iterator( src2_ptr, pos.begin() );
    auto map_data_2_end   = thrust::make_permutation_iterator( src2_ptr, pos.end() );

    auto zip_begin =
        thrust::make_zip_iterator( thrust::make_tuple( map_data_1_begin, map_data_2_begin ) );
    auto zip_end =
        thrust::make_zip_iterator( thrust::make_tuple( map_data_1_end, map_data_2_end ) );
    thrust::transform( thrust::device, zip_begin, zip_end, dst_ptr, pair_plus_op<DTYPE>() );
}

template<typename STYPE, typename DTYPE>
void DeviceDataHelpers<STYPE, DTYPE>::getGhostAddValuesByGlobalID( const size_t gsize,
                                                                   const size_t *globalids,
                                                                   const size_t N,
                                                                   const size_t *ndx,
                                                                   const size_t src_size,
                                                                   const STYPE *src,
                                                                   DTYPE *dst )
{
    AMP_INSIST( AMP::Utilities::getMemoryType( globalids ) >= AMP::Utilities::MemoryType::managed,
                "globalids not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( ndx ) >= AMP::Utilities::MemoryType::managed,
                "ndx not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( src ) >= AMP::Utilities::MemoryType::managed,
                "src not on device" );
    AMP_INSIST( AMP::Utilities::getMemoryType( dst ) >= AMP::Utilities::MemoryType::managed,
                "dst not on device" );
    thrust::device_ptr<const size_t> gid_ptr = thrust::device_pointer_cast( globalids );
    thrust::device_ptr<const size_t> ndx_ptr = thrust::device_pointer_cast( ndx );
    thrust::device_ptr<const STYPE> src_ptr  = thrust::device_pointer_cast( src );
    thrust::device_ptr<DTYPE> dst_ptr        = thrust::device_pointer_cast( dst );

    thrust::device_vector<size_t> pos( N );

    // Perform vectorized lower_bound to find positions in src
    thrust::lower_bound(
        thrust::device, gid_ptr, gid_ptr + gsize, ndx_ptr, ndx_ptr + N, pos.begin() );

    // construct the [begin, end) for the src map
    auto begin_map = thrust::make_permutation_iterator( src_ptr, pos.begin() );
    auto end_map   = thrust::make_permutation_iterator( src_ptr, pos.end() );

    thrust::gather( thrust::device, pos.begin(), pos.end(), src_ptr, dst_ptr );
}

} // namespace LinearAlgebra
} // namespace AMP

#endif
