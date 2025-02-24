#ifndef included_AMP_CopyCast_HPP_
#define included_AMP_CopyCast_HPP_

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/memory.h"
#include "CopyCastHelper.h"


#if defined( USE_DEVICE )
    #include "AMP/utils/copycast/device/CopyCastHelper.hpp"
#endif
#if defined( AMP_USE_KOKKOS ) || defined( AMP_USE_TRILINOS_KOKKOS )
    #include "AMP/utils/copycast/kokkos/CopyCastHelper.hpp"
#endif
#if defined( USE_OPENMP )
    #include "AMP/utils/copycast/openmp/CopyCastHelper.hpp"
#endif
#include "AMP/utils/copycast/serial/CopyCastHelper.hpp"


#include <iostream>
#include <memory>

namespace AMP::Utilities {

/*!
 * Helper function to copy and cast (single<->double precision) values between two arrays
 * @param[in]    len      Length of above vectors
 * @param[in]    vec_in   The incoming vector to get the values from
 * @param[inout] vec_out  The outgoing vector to with the up/down-casted values from vec_in
 *                        It is assumed that vec_out is properly allocated
 */
template<typename T1, typename T2, typename Backend, typename MemoryType>
void copyCast( size_t len, const T1 *vec_in, T2 *vec_out )
{
    AMP_DEBUG_ASSERT( getMemoryType( vec_in ) == getMemoryType( vec_out ) );

    copyCast_<T1, T2, Backend, MemoryType>::apply( len, vec_in, vec_out );
}

/*!
 * Helper function to copy and cast (single<->double precision) values between two arrays
 * @param[in]    len      Length of above vectors
 * @param[in]    vec_in   The incoming vector to get the values from
 * @param[inout] vec_out  The outgoing vector to with the up/down-casted values from vec_in
 *                        It is assumed that vec_out is properly allocated
 */
template<typename T1, typename T2, typename Backend>
void copyCast( size_t len, const T1 *vec_in, T2 *vec_out )
{
    AMP_DEBUG_ASSERT( getMemoryType( vec_in ) == getMemoryType( vec_out ) );
    if ( ( getMemoryType( vec_in ) == AMP::Utilities::MemoryType::host ) ||
         ( getMemoryType( vec_in ) == AMP::Utilities::MemoryType::unregistered ) ) {
        copyCast_<T1, T2, Backend, AMP::HostAllocator<void>>::apply( len, vec_in, vec_out );
    }
#ifdef USE_DEVICE
    else if ( getMemoryType( vec_in ) == AMP::Utilities::MemoryType::managed ) {
        copyCast_<T1, T2, Backend, AMP::ManagedAllocator<void>>::apply( len, vec_in, vec_out );
    } else if ( getMemoryType( vec_in ) == AMP::Utilities::MemoryType::device ) {
        copyCast_<T1, T2, Backend, AMP::DeviceAllocator<void>>::apply( len, vec_in, vec_out );
    }
#endif
    else {
        AMP_ERROR( "Memory Type not supported" );
    }
}

} // namespace AMP::Utilities

#endif
