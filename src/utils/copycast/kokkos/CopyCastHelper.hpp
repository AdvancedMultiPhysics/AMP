#ifndef included_AMP_CopyCast_kokkos_HPP_
#define included_AMP_CopyCast_kokkos_HPP_

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/memory.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

#include <memory>


namespace AMP::Utilities {
#ifdef KOKKOS_ENABLE_OPENMP
typedef Kokkos::OpenMP HostExecSpace;
#else
typedef Kokkos::Serial HostExecSpace;
#endif
typedef Kokkos::RangePolicy<HostExecSpace> host_range_policy;
typedef Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> dev_range_policy;

/*!
 * Helper function to copy and cast (single<->double precision) values between two arrays
 * @param[in]    len      Length of above vectors
 * @param[in]    vec_in   The incoming vector to get the values from
 * @param[inout] vec_out  The outgoing vector to with the up/down-casted values from vec_in
 *                        It is assumed that vec_out is properly allocated
 */
template<typename T1, typename T2>
struct copyCast_<T1, T2, AMP::Utilities::Backend::Kokkos, AMP::HostAllocator<void>> {
    void static apply( size_t len, const T1 *vec_in, T2 *vec_out )
    {
#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
        int err = 0;
        Kokkos::parallel_reduce(
            "Copy cast",
            host_range_policy( 0, len ),
            KOKKOS_LAMBDA( const int &i, int &lerr ) {
                if ( std::abs( vec_in[i] ) > std::numeric_limits<T2>::max() )
                    lerr = 1;
            },
            Kokkos::Max<int>( err ) );
        AMP_ASSERT( err < 1 );
#endif
        Kokkos::parallel_for(
            "Copy cast", host_range_policy( 0, len ), KOKKOS_LAMBDA( const int &i ) {
                vec_out[i] = static_cast<T2>( vec_in[i] );
            } );
    }
};

#ifdef AMP_USE_DEVICE

template<typename T1, typename T2>
struct copyCast_<T1, T2, AMP::Utilities::Backend::Kokkos, AMP::ManagedAllocator<void>> {
    void static apply( size_t len, const T1 *vec_in, T2 *vec_out )
    {
    #if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
        int err = 0;
        Kokkos::parallel_reduce(
            "Copy cast",
            dev_range_policy( 0, len ),
            KOKKOS_LAMBDA( const int &i, int &lerr ) {
                if ( std::abs( vec_in[i] ) > std::numeric_limits<T2>::max() )
                    lerr = 1;
            },
            Kokkos::Max<int>( err ) );
        AMP_ASSERT( err < 1 );
    #endif
        Kokkos::parallel_for(
            "Copy cast", dev_range_policy( 0, len ), KOKKOS_LAMBDA( const int &i ) {
                vec_out[i] = static_cast<T2>( vec_in[i] );
            } );
        Kokkos::fence();
    }
};

template<typename T1, typename T2>
struct copyCast_<T1, T2, AMP::Utilities::Backend::Kokkos, AMP::DeviceAllocator<void>> {
    void static apply( size_t len, const T1 *vec_in, T2 *vec_out )
    {
    #if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
        int err = 0;
        Kokkos::parallel_reduce(
            "Copy cast",
            dev_range_policy( 0, len ),
            KOKKOS_LAMBDA( const int &i, int &lerr ) {
                if ( std::abs( vec_in[i] ) > std::numeric_limits<T2>::max() )
                    lerr = 1;
            },
            Kokkos::Max<int>( err ) );
        AMP_ASSERT( err < 1 );
    #endif
        Kokkos::parallel_for(
            "Copy cast", dev_range_policy( 0, len ), KOKKOS_LAMBDA( const int &i ) {
                vec_out[i] = static_cast<T2>( vec_in[i] );
            } );
        // Might not be needed for device memory
        Kokkos::fence();
    }
};
#endif

} // namespace AMP::Utilities


#endif
