#ifndef included_DeviceMatrixOperationsHelpers_H_
#define included_DeviceMatrixOperationsHelpers_H_

#include "AMP/utils/device/Device.h"

#include <cstddef>

namespace AMP {
namespace LinearAlgebra {


template<typename G, typename L, typename S>
struct DeviceMatrixOperations {
    static void mult( const L *row_starts,
                      const L *cols_loc,
                      const S *coeffs,
                      const size_t N,
                      const S *in,
                      S *out,
                      AMP::Utilities::ComputeStream stream );

    static void
    scale( const size_t N, S *coeffs, const S alpha, AMP::Utilities::ComputeStream stream );

    static void
    axpy( const size_t N, const S alpha, S *x, S *y, AMP::Utilities::ComputeStream stream );

    static void copy( const size_t N, const S *x, S *y, AMP::Utilities::ComputeStream stream );

    static void setDiagonal( const L *row_starts,
                             S *coeffs,
                             const size_t N,
                             const S *diag,
                             AMP::Utilities::ComputeStream stream );

    static void extractDiagonal( const L *row_starts,
                                 const S *coeffs,
                                 const size_t N,
                                 S *diag,
                                 AMP::Utilities::ComputeStream stream );

    static void setIdentity( const L *row_starts,
                             S *coeffs,
                             const size_t N,
                             AMP::Utilities::ComputeStream stream );

    static void LinfNorm( const size_t N,
                          const S *x,
                          const L *row_starts,
                          S *row_sums,
                          AMP::Utilities::ComputeStream stream );
};

} // namespace LinearAlgebra
} // namespace AMP
#endif
