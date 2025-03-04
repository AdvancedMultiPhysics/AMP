#ifndef included_DeviceMatrixOperationsHelpers_H_
#define included_DeviceMatrixOperationsHelpers_H_

#include <cstddef>

namespace AMP {
namespace LinearAlgebra {


template<typename G, typename L, typename S>
struct DeviceMatrixOperations {
    static void mult( const L *row_starts,
                      const L *cols_loc,
                      const S *coeffs,
                      const size_t N,
                      const S *in_h,
                      const size_t Ng,
                      S *out );
    static void mult( const L *row_starts,
                      const L *cols_loc,
                      const S *coeffs,
                      const size_t N,
                      const S *in_h,
                      S *out );

    static void setScalar( const size_t N, S *coeffs, const S alpha );
    static void scale( const size_t N, S *coeffs, const S alpha );
    static void axpy( const size_t N, const S alpha, S *x, S *y );
    static void copy( const size_t N, const S *x, S *y );
    static void setDiagonal( const L *row_starts, S *coeffs, const size_t N, const S *diag );
    static void extractDiagonal( const L *row_starts, const S *coeffs, const size_t N, S *diag );
    static void setIdentity( const L *row_starts, S *coeffs, const size_t N );
    static void LinfNorm( const size_t N, const S *x, const L *row_starts, S *row_sums );
};

} // namespace LinearAlgebra
} // namespace AMP
#endif
