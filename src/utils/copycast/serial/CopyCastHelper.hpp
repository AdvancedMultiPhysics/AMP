#ifndef included_AMP_CopyCast_serial_HPP_
#define included_AMP_CopyCast_serial_HPP_

#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"

#include <limits>
#include <memory>


namespace AMP::Utilities {

/*!
 * Helper function to copy and cast (single<->double precision) values between two arrays
 * @param[in]    len      Length of above vectors
 * @param[in]    vec_in   The incoming vector to get the values from
 * @param[inout] vec_out  The outgoing vector to with the up/down-casted values from vec_in
 *                        It is assumed that vec_out is properly allocated
 */
template<typename T1, typename T2>
struct copyCast_<T1, T2, AMP::Utilities::Backend::Serial, AMP::HostAllocator<void>> {
    static void apply( const size_t len, const T1 *vec_in, T2 *vec_out )
    {
        for ( size_t i = 0; i < len; i++ ) {
            if constexpr ( std::numeric_limits<T1>::max() > std::numeric_limits<T2>::max() ) {
                if constexpr ( std::is_signed_v<T1> )
                    AMP_ASSERT( std::abs( vec_in[i] ) <= std::numeric_limits<T2>::max() );
                else
                    AMP_ASSERT( vec_in[i] <= std::numeric_limits<T2>::max() );
            }
            vec_out[i] = static_cast<T2>( vec_in[i] );
        }
    }
};

} // namespace AMP::Utilities

#endif
