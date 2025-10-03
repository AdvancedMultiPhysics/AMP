#include "AMP/utils/DelaunayHelpers.h"
#include "AMP/utils/Utilities.h"

#ifdef AMP_USE_LAPACK_WRAPPERS
    #include "LapackWrappers.h"
#endif

#include "ProfilerApp.h"

#include <limits>


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
#ifdef AMP_USE_LAPACK_WRAPPERS
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
#else
        AMP_ERROR( "Lapack required" );
#endif
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
#ifdef AMP_USE_LAPACK_WRAPPERS
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
#else
        AMP_ERROR( "Lapack required" );
#endif
    }
}


/****************************************************************
 * Find the first n intersections in multiple lists              *
 * This function assumes the lists are in sorted order           *
 ****************************************************************/
static int intersect_sorted(
    const int N_lists, const int size[], int *list[], const int N_max, int *intersection )
{
    if ( N_max <= 0 )
        return ~( (unsigned int) 0 );
    int N_int = 0;
    std::vector<int> index( N_lists );
    for ( int i = 0; i < N_lists; i++ )
        index[i] = 0;
    int current_val = list[0][0];
    bool finished   = false;
    while ( true ) {
        int min_val          = std::numeric_limits<int>::max();
        bool in_intersection = true;
        for ( int i = 0; i < N_lists; i++ ) {
            if ( index[i] >= size[i] ) {
                finished = true;
                break;
            }
            while ( list[i][index[i]] < current_val ) {
                index[i]++;
                if ( index[i] >= size[i] ) {
                    finished = true;
                    break;
                }
            }
            if ( list[i][index[i]] == current_val ) {
                index[i]++;
            } else {
                in_intersection = false;
            }
            if ( index[i] < size[i] ) {
                if ( list[i][index[i]] < min_val )
                    min_val = list[i][index[i]];
            }
        }
        if ( finished )
            break;
        if ( in_intersection ) {
            intersection[N_int] = current_val;
            N_int++;
            if ( N_int >= N_max )
                break;
        }
        current_val = min_val;
    }
    return N_int;
}


/****************************************************************
 * Create triangles neighbors from the triangles                 *
 ****************************************************************/
template<size_t N>
static constexpr std::array<int, N + 1> createNullTri()
{
    std::array<int, N + 1> null = { -1 };
    for ( size_t i = 0; i < null.size(); i++ )
        null[i] = -1;
    return null;
}
template<>
std::vector<std::array<int, 2>>
create_tri_neighbors<1>( const std::vector<std::array<int, 2>> &tri )
{
    std::vector<std::array<int, 2>> tri_nab( tri.size(), createNullTri<1>() );
    if ( tri.size() == 1 )
        return tri_nab;
    for ( size_t i = 0; i < tri.size(); i++ ) {
        tri_nab[i][0] = i + 1;
        tri_nab[i][1] = i - 1;
    }
    tri_nab[0][1]              = -1;
    tri_nab[tri.size() - 1][0] = -1;
    return tri_nab;
}
template<size_t NG>
std::vector<std::array<int, NG + 1>>
create_tri_neighbors( const std::vector<std::array<int, NG + 1>> &tri )
{
    PROFILE( "create_tri_neighbors", 1 );
    // Allocate memory
    std::vector<std::array<int, NG + 1>> tri_nab( tri.size(), createNullTri<NG>() );
    if ( tri.size() == 1 )
        return tri_nab;
    // Get the number of vertices
    size_t N_vertex = 0;
    for ( const auto &t : tri ) {
        for ( size_t i = 0; i < NG + 1; i++ )
            N_vertex = std::max<size_t>( N_vertex, t[i] + 1 );
    }
    // Count the number of triangles connected to each vertex
    std::vector<int> N_tri_nab( N_vertex, 0 );
    for ( size_t i = 0; i < tri.size(); i++ ) {
        for ( size_t d = 0; d < NG + 1; d++ )
            N_tri_nab[tri[i][d]]++;
    }
    // For each node, get a list of the triangles that connect to that node
    auto tri_list = new int *[N_vertex]; // List of triangles connected each node (N)
    tri_list[0]   = new int[( NG + 1 ) * tri.size()];
    for ( size_t i = 1; i < N_vertex; i++ )
        tri_list[i] = &tri_list[i - 1][N_tri_nab[i - 1]];
    for ( size_t i = 0; i < ( NG + 1 ) * tri.size(); i++ )
        tri_list[0][i] = -1;
    // Create a sorted list of all triangles that have each node as a vertex
    for ( size_t i = 0; i < N_vertex; i++ )
        N_tri_nab[i] = 0;
    for ( size_t i = 0; i < tri.size(); i++ ) {
        for ( size_t j = 0; j <= NG; j++ ) {
            int k                     = tri[i][j];
            tri_list[k][N_tri_nab[k]] = i;
            N_tri_nab[k]++;
        }
    }
    for ( size_t i = 0; i < N_vertex; i++ )
        AMP::Utilities::quicksort( N_tri_nab[i], tri_list[i] );
    int N_tri_max = 0;
    for ( size_t i = 0; i < N_vertex; i++ ) {
        if ( N_tri_nab[i] > N_tri_max )
            N_tri_max = N_tri_nab[i];
    }
    // Note, if a triangle is a neighbor, it will share all but the current node
    int size[NG];
    for ( int i = 0; i < (int) tri.size(); i++ ) {
        // Loop through the different faces of the triangle
        for ( size_t j = 0; j <= NG; j++ ) {
            int *list[NG] = { nullptr };
            int k1        = 0;
            for ( size_t k2 = 0; k2 <= NG; k2++ ) {
                if ( k2 == j )
                    continue;
                int k    = tri[i][k2];
                list[k1] = tri_list[k];
                size[k1] = N_tri_nab[k];
                k1++;
            }
            // Find the intersection of all triangle lists except the current node
            int intersection[5] = { -1, -1, -1, -1, -1 };
            int N_int           = intersect_sorted( NG, size, list, 5, intersection );
            int m               = 0;
            if ( N_int == 0 || N_int > 2 ) {
                // We cannot have less than 1 triangle or more than 2 triangles sharing NDIM nodes
                AMP_ERROR( "Error in create_tri_neighbors detected" );
            } else if ( intersection[0] == i ) {
                m = intersection[1];
            } else if ( intersection[1] == i ) {
                m = intersection[0];
            } else {
                // One of the triangles must be the current triangle
                AMP_ERROR( "Error in create_tri_neighbors detected" );
            }
            tri_nab[i][j] = m;
        }
    }
    // Check tri_nab
    for ( int i = 0; i < (int) tri.size(); i++ ) {
        for ( size_t d = 0; d <= NG; d++ ) {
            if ( tri_nab[i][d] < -1 || tri_nab[i][d] >= (int) tri.size() || tri_nab[i][d] == i )
                AMP_ERROR( "Internal error" );
        }
    }
    delete[] tri_list[0];
    delete[] tri_list;
    return tri_nab;
}
template std::vector<std::array<int, 2>>
create_tri_neighbors<1>( const std::vector<std::array<int, 2>> & );
template std::vector<std::array<int, 3>>
create_tri_neighbors<2>( const std::vector<std::array<int, 3>> & );
template std::vector<std::array<int, 4>>
create_tri_neighbors<3>( const std::vector<std::array<int, 4>> & );


} // namespace AMP::DelaunayHelpers
