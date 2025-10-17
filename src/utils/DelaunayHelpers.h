#ifndef included_AMP_DelaunayHelpers
#define included_AMP_DelaunayHelpers

#include <array>
#include <stdexcept>
#include <stdlib.h>

#include "AMP/utils/Array.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/extended_int.h"


namespace AMP::DelaunayHelpers {


// clang-format off
template<int NDIM, class TYPE> struct getETYPE;
template<> struct getETYPE<1,int> { typedef int ETYPE; };
template<> struct getETYPE<2,int> { typedef int64_t ETYPE; };
#ifdef __SIZEOF_INT128__
DISABLE_WARNINGS
template<> struct getETYPE<3,int> { typedef __int128 ETYPE; };
template<> struct getETYPE<4,int> { typedef __int128 ETYPE; };
ENABLE_WARNINGS
#else
template<> struct getETYPE<3,int> { typedef AMP::extended::int128_t ETYPE; };
template<> struct getETYPE<4,int> { typedef AMP::extended::int128_t ETYPE; };
#endif
template<> struct getETYPE<1,double> { typedef long double ETYPE; };
template<> struct getETYPE<2,double> { typedef long double ETYPE; };
template<> struct getETYPE<3,double> { typedef long double ETYPE; };
template<> struct getETYPE<4,double> { typedef long double ETYPE; };
// clang-format on


/********************************************************************
 * Create triangles neighbors from the triangles                     *
 ********************************************************************/
template<size_t NG>
std::vector<std::array<int, NG + 1>>
create_tri_neighbors( const std::vector<std::array<int, NG + 1>> &tri );


/********************************************************************
 * Sort the triangles by index                                       *
 ********************************************************************/
template<size_t NG>
std::vector<int> sortTri( std::vector<std::array<int, NG + 1>> &tri );
template<size_t NG>
std::vector<std::array<int, NG + 1>> uniqueTri( const std::vector<std::array<int, NG + 1>> &tri );


/********************************************************************
 * Compute the determinant of a matrix                               *
 * Note: The matrix is stored in column-major order                  *
 * Note: For the determinant to be exact, we must support a signed   *
 *   range of +- N^D, where N is the largest input value and D is    *
 *   the size of the matrix.                                         *
 ********************************************************************/
template<class TYPE, std::size_t NDIM>
inline TYPE det( const TYPE *M )
{
    if constexpr ( NDIM == 1 ) {
        return M[0];
    } else if constexpr ( NDIM == 2 ) {
        return M[0] * M[3] - M[1] * M[2];
    } else if constexpr ( NDIM == 3 ) {
        TYPE det( 0 );
        det += M[0] * ( M[4] * M[8] - M[7] * M[5] );
        det -= M[3] * ( M[1] * M[8] - M[7] * M[2] );
        det += M[6] * ( M[1] * M[5] - M[4] * M[2] );
        return det;
    } else if constexpr ( NDIM == 4 ) {
        TYPE tmp[6];
        tmp[0] = M[2] * M[7] - M[6] * M[3];
        tmp[1] = M[2] * M[11] - M[10] * M[3];
        tmp[2] = M[2] * M[15] - M[14] * M[3];
        tmp[3] = M[6] * M[11] - M[10] * M[7];
        tmp[4] = M[6] * M[15] - M[14] * M[7];
        tmp[5] = M[10] * M[15] - M[14] * M[11];
        TYPE det( 0 );
        det += M[0] * ( M[5] * tmp[5] - M[9] * tmp[4] + M[13] * tmp[3] );
        det -= M[4] * ( M[1] * tmp[5] - M[9] * tmp[2] + M[13] * tmp[1] );
        det += M[8] * ( M[1] * tmp[4] - M[5] * tmp[2] + M[13] * tmp[0] );
        det -= M[12] * ( M[1] * tmp[3] - M[5] * tmp[1] + M[9] * tmp[0] );
        return det;
    } else {
        throw std::logic_error( "Not programmed" );
    }
}
template<std::size_t NDIM>
inline typename getETYPE<NDIM, int>::ETYPE det2( const int *M );
template<>
inline int det2<1>( const int *M )
{
    return M[0];
}
template<>
inline int64_t det2<2>( const int *M )
{
    return int64_t( M[0] ) * int64_t( M[3] ) - int64_t( M[1] ) * int64_t( M[2] );
}
template<>
inline typename getETYPE<3, int>::ETYPE det2<3>( const int *M )
{
    using int64  = int64_t;
    using int128 = typename getETYPE<3, int>::ETYPE;
    int128 t1    = int128( int64( M[4] ) * int64( M[8] ) - int64( M[7] ) * int64( M[5] ) );
    int128 t2    = int128( int64( M[1] ) * int64( M[8] ) - int64( M[7] ) * int64( M[2] ) );
    int128 t3    = int128( int64( M[1] ) * int64( M[5] ) - int64( M[4] ) * int64( M[2] ) );
    auto det     = int128( M[0] ) * t1 + int128( -M[3] ) * t2 + int128( M[6] ) * t3;
    return det;
}
template<std::size_t NDIM>
inline long double det2( const double *M )
{
    long double M2[NDIM * NDIM];
    for ( size_t i = 0; i < NDIM * NDIM; i++ )
        M2[i] = M[i];
    return det<long double, NDIM>( M2 );
}


/********************************************************************
 * Solve a small dense system                                        *
 * Note: The matrix is stored in column-major order                  *
 * Note: The solve functions do not normalize the inverse:           *
 *      x = det * M^-1 * b                                           *
 *   Instead they return the normalization constant det so the user  *
 *   can perform the normalization if necessary.  This helps to      *
 *   preserve accuracy.                                              *
 ********************************************************************/
template<class TYPE, std::size_t NDIM>
inline void solve( const TYPE *M, const TYPE *b, TYPE *x, TYPE &det_M )
{
    if constexpr ( NDIM == 1 ) {
        det_M = M[0];
        x[0]  = b[0];
    } else if constexpr ( NDIM == 2 ) {
        det_M = M[0] * M[3] - M[1] * M[2];
        x[0]  = M[3] * b[0] - M[2] * b[1];
        x[1]  = M[0] * b[1] - M[1] * b[0];
    } else if constexpr ( NDIM == 3 ) {
        TYPE inv[9];
        inv[0] = M[4] * M[8] - M[7] * M[5];
        inv[1] = M[7] * M[2] - M[1] * M[8];
        inv[2] = M[1] * M[5] - M[4] * M[2];
        inv[3] = M[6] * M[5] - M[3] * M[8];
        inv[4] = M[0] * M[8] - M[6] * M[2];
        inv[5] = M[3] * M[2] - M[0] * M[5];
        inv[6] = M[3] * M[7] - M[6] * M[4];
        inv[7] = M[6] * M[1] - M[0] * M[7];
        inv[8] = M[0] * M[4] - M[3] * M[1];
        det_M  = M[0] * inv[0] + M[3] * inv[1] + M[6] * inv[2];
        x[0]   = inv[0] * b[0] + inv[3] * b[1] + inv[6] * b[2];
        x[1]   = inv[1] * b[0] + inv[4] * b[1] + inv[7] * b[2];
        x[2]   = inv[2] * b[0] + inv[5] * b[1] + inv[8] * b[2];
    } else if constexpr ( NDIM == 4 ) {
        TYPE inv[16];
        TYPE m00 = M[0], m01 = M[4], m02 = M[8], m03 = M[12];
        TYPE m10 = M[1], m11 = M[5], m12 = M[9], m13 = M[13];
        TYPE m20 = M[2], m21 = M[6], m22 = M[10], m23 = M[14];
        TYPE m30 = M[3], m31 = M[7], m32 = M[11], m33 = M[15];
        det_M  = det<TYPE, NDIM>( M );
        inv[0] = m12 * m23 * m31 - m13 * m22 * m31 + m13 * m21 * m32 - m11 * m23 * m32 -
                 m12 * m21 * m33 + m11 * m22 * m33;
        inv[1] = m03 * m22 * m31 - m02 * m23 * m31 - m03 * m21 * m32 + m01 * m23 * m32 +
                 m02 * m21 * m33 - m01 * m22 * m33;
        inv[2] = m02 * m13 * m31 - m03 * m12 * m31 + m03 * m11 * m32 - m01 * m13 * m32 -
                 m02 * m11 * m33 + m01 * m12 * m33;
        inv[3] = m03 * m12 * m21 - m02 * m13 * m21 - m03 * m11 * m22 + m01 * m13 * m22 +
                 m02 * m11 * m23 - m01 * m12 * m23;
        inv[4] = m13 * m22 * m30 - m12 * m23 * m30 - m13 * m20 * m32 + m10 * m23 * m32 +
                 m12 * m20 * m33 - m10 * m22 * m33;
        inv[5] = m02 * m23 * m30 - m03 * m22 * m30 + m03 * m20 * m32 - m00 * m23 * m32 -
                 m02 * m20 * m33 + m00 * m22 * m33;
        inv[6] = m03 * m12 * m30 - m02 * m13 * m30 - m03 * m10 * m32 + m00 * m13 * m32 +
                 m02 * m10 * m33 - m00 * m12 * m33;
        inv[7] = m02 * m13 * m20 - m03 * m12 * m20 + m03 * m10 * m22 - m00 * m13 * m22 -
                 m02 * m10 * m23 + m00 * m12 * m23;
        inv[8] = m11 * m23 * m30 - m13 * m21 * m30 + m13 * m20 * m31 - m10 * m23 * m31 -
                 m11 * m20 * m33 + m10 * m21 * m33;
        inv[9] = m03 * m21 * m30 - m01 * m23 * m30 - m03 * m20 * m31 + m00 * m23 * m31 +
                 m01 * m20 * m33 - m00 * m21 * m33;
        inv[10] = m01 * m13 * m30 - m03 * m11 * m30 + m03 * m10 * m31 - m00 * m13 * m31 -
                  m01 * m10 * m33 + m00 * m11 * m33;
        inv[11] = m03 * m11 * m20 - m01 * m13 * m20 - m03 * m10 * m21 + m00 * m13 * m21 +
                  m01 * m10 * m23 - m00 * m11 * m23;
        inv[12] = m12 * m21 * m30 - m11 * m22 * m30 - m12 * m20 * m31 + m10 * m22 * m31 +
                  m11 * m20 * m32 - m10 * m21 * m32;
        inv[13] = m01 * m22 * m30 - m02 * m21 * m30 + m02 * m20 * m31 - m00 * m22 * m31 -
                  m01 * m20 * m32 + m00 * m21 * m32;
        inv[14] = m02 * m11 * m30 - m01 * m12 * m30 - m02 * m10 * m31 + m00 * m12 * m31 +
                  m01 * m10 * m32 - m00 * m11 * m32;
        inv[15] = m01 * m12 * m20 - m02 * m11 * m20 + m02 * m10 * m21 - m00 * m12 * m21 -
                  m01 * m10 * m22 + m00 * m11 * m22;
        x[0] = ( inv[0] * b[0] + inv[1] * b[1] + inv[2] * b[2] + inv[3] * b[3] );
        x[1] = ( inv[4] * b[0] + inv[5] * b[1] + inv[6] * b[2] + inv[7] * b[3] );
        x[2] = ( inv[8] * b[0] + inv[9] * b[1] + inv[10] * b[2] + inv[11] * b[3] );
        x[3] = ( inv[12] * b[0] + inv[13] * b[1] + inv[14] * b[2] + inv[15] * b[3] );
    }
}


/********************************************************************
 * Solve a small dense system                                        *
 * Note: The matrix is stored in column-major order                  *
 ********************************************************************/
void solve( int NDIM, const double *M, const double *b, double *x );


/****************************************************************
 * Function to calculate the inverse of a matrix                 *
 ****************************************************************/
void inverse( int NDIM, const double *M, double *M_inv );


/****************************************************************
 * Function to compute the Barycentric coordinates               *
 * Note: we use exact math until we perform the normalization    *
 *    The exact math component requires N^D precision            *
 ****************************************************************/
template<int NDIM, class TYPE>
std::array<double, NDIM + 1> computeBarycentric( const std::array<TYPE, NDIM> *x,
                                                 const std::array<TYPE, NDIM> &xi )
{
    // Compute the barycentric coordinates T*L=r-r0
    // http://en.wikipedia.org/wiki/Barycentric_coordinate_system_(mathematics)
    using ETYPE = typename getETYPE<NDIM, TYPE>::ETYPE;
    ETYPE T[NDIM * NDIM];
    for ( int i = 0; i < NDIM; i++ ) {
        for ( int j = 0; j < NDIM; j++ )
            T[j + i * NDIM] = ETYPE( x[i][j] - x[NDIM][j] );
    }
    ETYPE r[NDIM];
    for ( int i = 0; i < NDIM; i++ )
        r[i] = ETYPE( xi[i] - x[NDIM][i] );
    ETYPE L2[NDIM + 1], det( 0 );
    DelaunayHelpers::solve<ETYPE, NDIM>( T, r, L2, det );
    L2[NDIM] = det;
    for ( int i = 0; i < NDIM; i++ )
        L2[NDIM] -= L2[i];
    // Perform the normalization (will require inexact math)
    double scale                   = 1.0 / static_cast<double>( det );
    std::array<double, NDIM + 1> L = { 0 };
    for ( int i = 0; i < NDIM + 1; i++ )
        L[i] = static_cast<double>( L2[i] ) * scale;
    return L;
}


/********************************************************************
 * This function calculates the volume of a N-dimensional simplex    *
 *         1  |  x1-x4   x2-x4   x3-x4  |                            *
 *    V = --  |  y1-y4   y2-y4   y3-y4  |   (3D)                     *
 *        n!  |  z1-z4   z2-z4   z3-z4  |                            *
 * Note: the sign of the volume depends on the order of the points.  *
 *   It will be positive for points stored in a clockwise mannor     *
 * This version is optimized for performance.                        *
 * Note:  If the volume is zero, then the simplex is invalid         *
 *   Eg. a line in 2D or a plane in 3D.                              *
 * Note: exact math requires N^D precision                           *
 ********************************************************************/
static constexpr double inv_factorial( int N )
{
    double x = 1;
    for ( int i = 2; i <= N; i++ )
        x *= i;
    return 1.0 / x;
}
template<int NDIM, class TYPE>
double calcVolume( const std::array<TYPE, NDIM> *x )
{
    if constexpr ( NDIM == 1 )
        return static_cast<double>( x[1][0] - x[0][0] );
    TYPE M[NDIM * NDIM];
    for ( int d = 0; d < NDIM; d++ ) {
        auto tmp = x[NDIM][d];
        for ( int j = 0; j < NDIM; j++ )
            M[d + j * NDIM] = x[j][d] - tmp;
    }
    constexpr double C = inv_factorial( NDIM );
    return C * static_cast<double>( DelaunayHelpers::det2<NDIM>( M ) );
}


/********************************************************************
 * Helper interfaces to convert arrays                               *
 ********************************************************************/
template<class TYPE, std::size_t NDIM>
AMP::Array<TYPE> convert( const std::vector<std::array<TYPE, NDIM>> &x )
{
    AMP::Array<TYPE> y( NDIM, x.size() );
    for ( size_t i = 0; i < x.size(); i++ ) {
        for ( size_t d = 0; d < NDIM; d++ )
            y( d, i ) = x[i][d];
    }
    return y;
}
template<class TYPE, std::size_t NDIM>
std::vector<std::array<TYPE, NDIM>> convert( const AMP::Array<TYPE> &x )
{
    AMP_ASSERT( x.size( 0 ) == NDIM );
    std::vector<std::array<TYPE, NDIM>> y( x.size( 1 ) );
    for ( size_t i = 0; i < y.size(); i++ ) {
        for ( size_t d = 0; d < NDIM; d++ )
            y[i][d] = x( d, i );
    }
    return y;
}


} // namespace AMP::DelaunayHelpers

#endif
