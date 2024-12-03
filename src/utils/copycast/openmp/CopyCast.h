#ifndef included_AMP_CopyCast_openmp_H_
#define included_AMP_CopyCast_openmp_H_

#include "AMP/utils/memory.h"

namespace AMP::Utilities {

/*!
 * Helper function to copy and cast (single<->double precision) values between two arrays
 * @param[in]    vec_in   The incoming vector to get the values from
 * @param[inout] vec_out  The outgoing vector to with the up/down-casted values from vec_in
 *                        It is assumed that vec_out is properly allocated
 * @param[in]    len      Length of above vectors
 */
template<typename T1, typename T2, AMP::Utilities::MemoryType MEMTYPE>
struct copyCast_ {
    void static apply( size_t, const T1 *, T2 * ) { AMP_ERROR( "Not implemented" ); }
};

} // namespace AMP::Utilities

// #include "CopyCast.hpp"

#endif
