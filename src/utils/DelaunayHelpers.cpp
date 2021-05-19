#include "AMP/utils/DelaunayHelpers.h"

#include "LapackWrappers.h"


namespace AMP::DelaunayHelpers {


/********************************************************************
 * Solve dense linear system                                         *
 * Note: The matrix is stored in column-major order                  *
 ********************************************************************/
void solve( int N, const double *M, const double *b, double *x )
{
    if ( N == 1 ) {
        // 1x1 matrix is trivial
        x[0] = b[0] / M[0];
    } else if ( N == 2 ) {
        // 2x2 matrix has a simple inverse
        double inv_det = 1.0 / ( M[0] * M[3] - M[1] * M[2] );
        x[0]           = ( M[3] * b[0] - M[2] * b[1] ) * inv_det;
        x[1]           = ( M[0] * b[1] - M[1] * b[0] ) * inv_det;
    } else if ( N == 3 ) {
        // 3x3 matrix
        double det = 0;
        solve<double, 3>( M, b, x, det );
        double inv_det = 1.0 / det;
        x[0] *= inv_det;
        x[1] *= inv_det;
        x[2] *= inv_det;
    } else if ( N == 4 ) {
        // 3x3 matrix
        double det = 0;
        solve<double, 4>( M, b, x, det );
        double inv_det = 1.0 / det;
        x[0] *= inv_det;
        x[1] *= inv_det;
        x[2] *= inv_det;
    } else {
        // Call Lapack to compute the inverse
        int error;
        int *IPIV;
        double *M2;
        double tmp1[64]; // Use the stack for small matricies (n<=8)
        int tmp2[8];     // Use the stack for small matricies (n<=8)
        if ( N <= 20 ) {
            M2   = tmp1;
            IPIV = tmp2;
        } else {
            M2   = new double[N * N];
            IPIV = new int[N];
        }
        for ( int i = 0; i < N * N; i++ )
            M2[i] = M[i];
        for ( int i = 0; i < N; i++ )
            x[i] = b[i];
        Lapack<double>::gesv( N, 1, M2, N, IPIV, x, N, error );
        if ( M2 != tmp1 ) {
            delete[] M2;
            delete[] IPIV;
        }
    }
}


/****************************************************************
 * Function to calculate the inverse of a matrix                 *
 ****************************************************************/
void inverse( const int N, const double *M, double *M_inv )
{
    if ( N == 1 ) {
        // 1x1 matrix is trivial
        M_inv[0] = 1.0 / M[0];
    } else if ( N == 2 ) {
        // 2x2 matrix has a simple inverse
        double inv_det = 1.0 / ( M[0] * M[3] - M[1] * M[2] );
        M_inv[0]       = M[3] * inv_det;
        M_inv[1]       = -M[1] * inv_det;
        M_inv[2]       = -M[2] * inv_det;
        M_inv[3]       = M[0] * inv_det;
    } else if ( N == 3 ) {
        // 3x3 matrix
        M_inv[0]       = M[4] * M[8] - M[7] * M[5];
        M_inv[1]       = M[7] * M[2] - M[1] * M[8];
        M_inv[2]       = M[1] * M[5] - M[4] * M[2];
        M_inv[3]       = M[6] * M[5] - M[3] * M[8];
        M_inv[4]       = M[0] * M[8] - M[6] * M[2];
        M_inv[5]       = M[3] * M[2] - M[0] * M[5];
        M_inv[6]       = M[3] * M[7] - M[6] * M[4];
        M_inv[7]       = M[6] * M[1] - M[0] * M[7];
        M_inv[8]       = M[0] * M[4] - M[3] * M[1];
        double inv_det = 1.0 / ( M[0] * M_inv[0] + M[3] * M_inv[1] + M[6] * M_inv[2] );
        for ( int i = 0; i < 9; i++ )
            M_inv[i] *= inv_det;
    } else if ( N == 4 ) {
        double m00 = M[0], m01 = M[4], m02 = M[8], m03 = M[12];
        double m10 = M[1], m11 = M[5], m12 = M[9], m13 = M[13];
        double m20 = M[2], m21 = M[6], m22 = M[10], m23 = M[14];
        double m30 = M[3], m31 = M[7], m32 = M[11], m33 = M[15];
        M_inv[0] = m12 * m23 * m31 - m13 * m22 * m31 + m13 * m21 * m32 - m11 * m23 * m32 -
                   m12 * m21 * m33 + m11 * m22 * m33;
        M_inv[1] = m03 * m22 * m31 - m02 * m23 * m31 - m03 * m21 * m32 + m01 * m23 * m32 +
                   m02 * m21 * m33 - m01 * m22 * m33;
        M_inv[2] = m02 * m13 * m31 - m03 * m12 * m31 + m03 * m11 * m32 - m01 * m13 * m32 -
                   m02 * m11 * m33 + m01 * m12 * m33;
        M_inv[3] = m03 * m12 * m21 - m02 * m13 * m21 - m03 * m11 * m22 + m01 * m13 * m22 +
                   m02 * m11 * m23 - m01 * m12 * m23;
        M_inv[4] = m13 * m22 * m30 - m12 * m23 * m30 - m13 * m20 * m32 + m10 * m23 * m32 +
                   m12 * m20 * m33 - m10 * m22 * m33;
        M_inv[5] = m02 * m23 * m30 - m03 * m22 * m30 + m03 * m20 * m32 - m00 * m23 * m32 -
                   m02 * m20 * m33 + m00 * m22 * m33;
        M_inv[6] = m03 * m12 * m30 - m02 * m13 * m30 - m03 * m10 * m32 + m00 * m13 * m32 +
                   m02 * m10 * m33 - m00 * m12 * m33;
        M_inv[7] = m02 * m13 * m20 - m03 * m12 * m20 + m03 * m10 * m22 - m00 * m13 * m22 -
                   m02 * m10 * m23 + m00 * m12 * m23;
        M_inv[8] = m11 * m23 * m30 - m13 * m21 * m30 + m13 * m20 * m31 - m10 * m23 * m31 -
                   m11 * m20 * m33 + m10 * m21 * m33;
        M_inv[9] = m03 * m21 * m30 - m01 * m23 * m30 - m03 * m20 * m31 + m00 * m23 * m31 +
                   m01 * m20 * m33 - m00 * m21 * m33;
        M_inv[10] = m01 * m13 * m30 - m03 * m11 * m30 + m03 * m10 * m31 - m00 * m13 * m31 -
                    m01 * m10 * m33 + m00 * m11 * m33;
        M_inv[11] = m03 * m11 * m20 - m01 * m13 * m20 - m03 * m10 * m21 + m00 * m13 * m21 +
                    m01 * m10 * m23 - m00 * m11 * m23;
        M_inv[12] = m12 * m21 * m30 - m11 * m22 * m30 - m12 * m20 * m31 + m10 * m22 * m31 +
                    m11 * m20 * m32 - m10 * m21 * m32;
        M_inv[13] = m01 * m22 * m30 - m02 * m21 * m30 + m02 * m20 * m31 - m00 * m22 * m31 -
                    m01 * m20 * m32 + m00 * m21 * m32;
        M_inv[14] = m02 * m11 * m30 - m01 * m12 * m30 - m02 * m10 * m31 + m00 * m12 * m31 +
                    m01 * m10 * m32 - m00 * m11 * m32;
        M_inv[15] = m01 * m12 * m20 - m02 * m11 * m20 + m02 * m10 * m21 - m00 * m12 * m21 -
                    m01 * m10 * m22 + m00 * m11 * m22;
        double inv_det = 1.0 / det<double, 4>( M );
        for ( int i = 0; i < 16; i++ )
            M_inv[i] *= inv_det;
    } else {
        // Call Lapack to compute the inverse
        int error;
        int LWORK;
        int *IPIV;
        double *WORK;
        double tmp1[64 * 8]; // Use the stack for small matricies (N<=8)
        int tmp2[8];         // Use the stack for small matricies (N<=8)
        if ( N <= 8 ) {
            LWORK = 64 * 8;
            WORK  = tmp1;
            IPIV  = tmp2;
        } else {
            LWORK = 64 * N;
            WORK  = new double[LWORK];
            IPIV  = new int[N];
        }
        for ( int i = 0; i < N * N; i++ )
            M_inv[i] = M[i];
        Lapack<double>::getrf( N, N, M_inv, N, IPIV, error );
        Lapack<double>::getri( N, M_inv, N, IPIV, WORK, LWORK, error );
        if ( WORK != tmp1 ) {
            delete[] IPIV;
            delete[] WORK;
        }
    }
}


} // namespace AMP::DelaunayHelpers
