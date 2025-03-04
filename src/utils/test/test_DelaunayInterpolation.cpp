#include "AMP/IO/PIO.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/DelaunayInterpolation.h"
#include "AMP/utils/DelaunayTessellation.h"
#include "AMP/utils/NearestPairSearch.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/typeid.h"

#include "test_DelaunayInterpolation.h"

#include "ProfilerApp.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>


#define printp printf


constexpr int NDIM_MAX = 3; // The maximum number of dimensions supported (currently 3)
constexpr int NTRI_MAX = NDIM_MAX + 1;
constexpr int NVTX_MAX = NTRI_MAX * NDIM_MAX;


// Helper function to check if two numbers are approximately equal
inline bool approx_equal( double x, double y, double tol = 1e-8 )
{
    return 2.0 * fabs( x - y ) <= tol * fabs( x + y );
}


// Helper function to write points to a file
template<class TYPE>
void writePoints( const char *filename, const AMP::Array<TYPE> &x );
template<>
void writePoints<double>( const char *filename, const AMP::Array<double> &x )
{
    int ndim = x.size( 0 );
    int N    = x.size( 1 );
    FILE *fid;
    fid = fopen( filename, "wb" );
    fprintf( fid, "%i points in %iD in double precision\n", N, ndim );
    fwrite( x.data(), sizeof( double ), x.length(), fid );
    fclose( fid );
    printp( "  Failed points written to %s\n", filename );
}
template<>
void writePoints<int>( const char *filename, const AMP::Array<int> &x )
{
    int ndim = x.size( 0 );
    int N    = x.size( 1 );
    FILE *fid;
    fid = fopen( filename, "wb" );
    fprintf( fid, "%i points in %iD in int precision\n", N, ndim );
    fwrite( x.data(), sizeof( int ), x.length(), fid );
    fclose( fid );
    printp( "  Failed points written to %s\n", filename );
}


// Helper function to read points from a file
AMP::Array<double> readPoints( const char *filename )
{
    auto fid = fopen( filename, "rb" );
    AMP_INSIST( fid, "Input file not found" );
    char tline[200];
    char *tmp1 = fgets( tline, 200, fid );
    AMP_ASSERT( tmp1 == tline );
    std::string header( tline );
    size_t i = header.find( " points in " );
    AMP_ASSERT( i != std::string::npos );
    std::string tmp = header.substr( 0, i );
    int N           = atoi( tmp.c_str() );
    i               = header.find( "D in " );
    AMP_ASSERT( i != std::string::npos );
    tmp   = header.substr( i - 2, 2 );
    int D = atoi( tmp.c_str() );
    AMP_ASSERT( N > 0 && D > 0 );
    AMP::Array<double> x( D, N );
    x.fill( 0.0 );
    size_t tmp2 = fread( x.data(), sizeof( double ), N * D, fid );
    AMP_ASSERT( tmp2 == (size_t) N * D );
    fclose( fid );
    return x;
}


// Test for the nearest pair search
template<class TYPE>
void testPointSearch( AMP::UnitTest &ut, const AMP::Array<TYPE> &x )
{
    PROFILE( "testPointSearch" );
    if ( x.empty() )
        return;
    // Test the nearest point search
    int ndim     = x.size( 0 );
    size_t N     = x.size( 1 );
    double dist1 = 1e100;
    double dist2 = 0.0;
    std::pair<int, int> index1( -1, -1 );
    std::pair<int, int> index2;
    {
        PROFILE( "Test point search 1" );
        for ( size_t i = 0; i < N; i++ ) {
            for ( size_t j = i + 1; j < N; j++ ) {
                double dist2 = 0.0;
                for ( int d = 0; d < ndim; d++ ) {
                    auto tmp = static_cast<double>( x( d, i ) - x( d, j ) );
                    dist2 += tmp * tmp;
                }
                if ( dist2 < dist1 ) {
                    dist1  = dist2;
                    index1 = std::pair<int, int>( (int) i, (int) j );
                }
            }
        }
    }
    {
        PROFILE( "Test point search 2" );
        if ( ndim == 1 )
            index2 = AMP::find_min_dist<1, TYPE>( N, x.data() );
        else if ( ndim == 2 )
            index2 = AMP::find_min_dist<2, TYPE>( N, x.data() );
        else if ( ndim == 3 )
            index2 = AMP::find_min_dist<3, TYPE>( N, x.data() );
        else if ( ndim == 4 )
            index2 = AMP::find_min_dist<4, TYPE>( N, x.data() );
        else if ( ndim == 5 )
            index2 = AMP::find_min_dist<5, TYPE>( N, x.data() );
        for ( int d = 0; d < ndim; d++ ) {
            auto tmp = static_cast<double>( x( d, index2.first ) - x( d, index2.second ) );
            dist2 += tmp * tmp;
        }
    }
    auto msg = AMP::Utilities::stringf(
        "Test point search (%i,%i,%s)", ndim, (int) N, AMP::getTypeID<TYPE>().name );
    if ( approx_equal( dist1, dist2, 1e-12 ) ) {
        ut.passes( msg );
    } else {
        auto msg2 = AMP::Utilities::stringf( "%s:  (%i,%i,%e) - (%i,%i,%e)",
                                             msg.data(),
                                             index1.first,
                                             index1.second,
                                             dist1,
                                             index2.first,
                                             index2.second,
                                             dist2 );
        ut.failure( msg2 );
        writePoints( "failed_points", x );
    }
}


// Function to create and test the construction of the tessellation
template<class TYPE>
std::shared_ptr<AMP::DelaunayInterpolation<TYPE>>
createAndTestDelaunayInterpolation( AMP::UnitTest &ut, const AMP::Array<TYPE> &x )
{
    PROFILE( "createAndTestDelaunayInterpolation" );
    if ( x.empty() )
        return nullptr;
    int ndim = x.size( 0 );
    size_t N = x.size( 1 );
    auto msg = AMP::Utilities::stringf( "(%i,%i,%s)", ndim, (int) N, AMP::getTypeID<TYPE>().name );

    // Create the tessellation
    auto data = std::make_shared<AMP::DelaunayInterpolation<TYPE>>();
    data->create_tessellation( x );
    size_t N_tri = data->get_N_tri();
    if ( N_tri > 0 ) {
        ut.passes( "Created tessellation " + msg );
    } else {
        ut.failure( "Created tessellation " + msg );
        return nullptr;
    }

    // Check the triangle neighbors
    auto tri_nab = data->get_tri_nab();
    bool pass    = true;
    for ( size_t i = 0; i < N_tri; i++ ) {
        for ( int d = 0; d < ndim + 1; d++ ) {
            if ( tri_nab( d, i ) == (int) i || tri_nab( d, i ) < -1 ||
                 tri_nab( d, i ) >= (int) N_tri )
                pass = false;
        }
    }
    if ( !pass ) {
        ut.failure( "Triangle neighbors are invalid " + msg );
        return nullptr;
    }

    // Copy the tessellation
    {
        PROFILE( "Copy tessellation", 1 );
        auto [x2, tri] = data->copy_tessellation();
        int N2         = x2.size( 1 );
        int N_tri2     = tri.size( 1 );
        bool error     = static_cast<size_t>( N2 ) != N || static_cast<size_t>( N_tri2 ) != N_tri;
        if ( error == 0 ) {
            for ( size_t i = 0; i < x.length(); i++ ) {
                if ( x( i ) != x2( i ) )
                    error = true;
            }
            for ( size_t i = 0; i < tri.length(); i++ ) {
                if ( tri( i ) < 0 || tri( i ) >= N2 )
                    error = true;
            }
        }
        AMP::DelaunayInterpolation<TYPE> data2;
        data2.create_tessellation( x, tri );
        if ( data2.get_N_tri() != N_tri )
            error = true;
        if ( !error )
            ut.passes( "Copy of tessellation " + msg );
        else
            ut.failure( "Copy of tessellation " + msg );
    }

    // Check the behavior of get_circumsphere and test_in_circumsphere
    {
        PROFILE( "Check circumsphere", 1 );
        pass                = true;
        double R            = 0;
        auto tri            = data->get_tri();
        TYPE x1[NVTX_MAX]   = { 0 };
        double c[NDIM_MAX]  = { 0 };
        double xi[NDIM_MAX] = { 0 };
        double x2[NVTX_MAX] = { 0 };
        for ( size_t i = 0; i < N_tri; i++ ) {
            for ( int d1 = 0; d1 < ndim + 1; d1++ ) {
                int k = tri( d1, i );
                for ( int d2 = 0; d2 < ndim; d2++ ) {
                    x1[d2 + d1 * ndim] = x( d2, k );
                    x2[d2 + d1 * ndim] = x( d2, k );
                }
            }
            AMP::DelaunayTessellation::get_circumsphere( ndim, x1, R, c );
            if ( R < 0 ) {
                pass = false;
            }
            for ( int j = 0; j < ndim + 1; j++ ) {
                double dist = 0.0;
                for ( int d = 0; d < ndim; d++ )
                    dist += ( x2[d + j * ndim] - c[d] ) * ( x2[d + j * ndim] - c[d] );
                if ( fabs( std::sqrt( dist ) - R ) / R > 1e-7 )
                    pass = false;
            }
            for ( int j = 0; j < ndim; j++ ) {
                for ( int d = 0; d < ndim; d++ )
                    xi[d] = c[d];
                xi[j]     = c[j] - 10.0 * R;
                int test1 = AMP::DelaunayTessellation::test_in_circumsphere( ndim, x2, xi, 0 );
                xi[j]     = c[j] - 0.1 * R;
                int test2 = AMP::DelaunayTessellation::test_in_circumsphere( ndim, x2, xi, 0 );
                xi[j]     = c[j] + 10.0 * R;
                int test3 = AMP::DelaunayTessellation::test_in_circumsphere( ndim, x2, xi, 0 );
                xi[j]     = c[j] + 0.1 * R;
                int test4 = AMP::DelaunayTessellation::test_in_circumsphere( ndim, x2, xi, 0 );
                if ( test1 != -1 || test2 != 1 || test3 != -1 || test4 != 1 ) {
                    pass = false;
                }
            }
        }
        if ( pass )
            ut.passes( "get_circumsphere and test_in_circumsphere " + msg );
        else
            ut.expected_failure( "get_circumsphere and test_in_circumsphere " + msg );
    }

    // Check that the volume of each triangle is positive and > 0
    {
        PROFILE( "Check volume", 1 );
        auto tri       = data->get_tri();
        double vol_min = 1e100;
        double vol_max = 0.0;
        bool neg_vol   = false;
        for ( size_t i = 0; i < N_tri; i++ ) {
            int tri2[NTRI_MAX]  = { 0 };
            double x2[NVTX_MAX] = { 0 };
            for ( int d1 = 0; d1 < ndim + 1; d1++ ) {
                tri2[d1] = tri( d1, i );
                for ( int d2 = 0; d2 < ndim; d2++ )
                    x2[d2 + d1 * ndim] = x( d2, tri2[d1] );
            }
            double vol = AMP::DelaunayTessellation::calc_volume( ndim, x2 );
            if ( vol < 0 ) {
                neg_vol = true;
                vol     = -vol;
            }
            vol_min = std::min( vol_min, vol );
            vol_max = std::max( vol_max, vol );
        }
        if ( !neg_vol && vol_min > 0.0 && vol_min >= 1e-10 * vol_max )
            ut.passes( "Tessellation volume is valid " + msg );
        else
            ut.failure( "Tessellation volume is invalid " + msg );
    }

    // Perform a rigorous check of the tessellation by checking that no point in the
    // tessellation lies withing the circumsphere of any triangle
    // Note: This is an N^2 test and will only be run for "reasonable" problem sizes
    if ( N <= 5000 ) {
        PROFILE( "Check tessellation", 1 );
        auto tri               = data->get_tri();
        bool pass_circumsphere = true;
        for ( size_t i = 0; i < N_tri; i++ ) {
            int tri2[NTRI_MAX] = { 0 };
            TYPE x2[NVTX_MAX]  = { 0 };
            TYPE xi[NDIM_MAX]  = { 0 };
            for ( int d1 = 0; d1 < ndim + 1; d1++ ) {
                tri2[d1] = tri( d1, i );
                for ( int d2 = 0; d2 < ndim; d2++ )
                    x2[d2 + d1 * ndim] = x( d2, tri2[d1] );
            }
            for ( size_t j = 0; j < N; j++ ) {
                for ( int d = 0; d < ndim; d++ )
                    xi[d] = x( d, j );
                int test = AMP::DelaunayTessellation::test_in_circumsphere( ndim, x2, xi, 1e-8 );
                if ( test == 1 ) {
                    pass_circumsphere = false;
                    break;
                }
            }
        }
        if ( pass_circumsphere )
            ut.passes( "Tessellation is valid " + msg );
        else
            ut.failure( "Tessellation is invalid " + msg );
    }

    return data;
}


// Create f and g for a given problem
template<class TYPE>
std::string initialize_problem( int p,
                                const AMP::Array<TYPE> &x0,
                                const double xrange[3],
                                AMP::Array<double> &f,
                                AMP::Array<double> &g,
                                double &tol_grad,
                                double &tol_linear,
                                double &tol_cubic,
                                double &tol_cubic_grad,
                                double &tol_cubic_extrap1,
                                double &tol_cubic_extrap2 )
{
    PROFILE( "initialize_problem" );
    int ndim = x0.size( 0 );
    size_t N = x0.size( 1 );
    std::string problem;
    tol_grad          = 1e-8;  // Tolerance for gradient calculation
    tol_linear        = 1e-8;  // Tolerance for linear interpolation
    tol_cubic         = 1e-10; // Tolerance for cubic interpolation (interior)
    tol_cubic_grad    = 1e-8;  // Tolerance for cubic interpolation (interior)
    tol_cubic_extrap1 = 1e-10; // Tolerance for linear extrapolation with cubic interpolation
    tol_cubic_extrap2 = 1e-10; // Tolerance for quadratic extrapolation with cubic interpolation
    // Set f and g for each problem
    double grad_max[NDIM_MAX] = { 0 };
    if ( p == 0 ) {
        // Constant profile
        problem = "f(x) = 2.5";
        f.fill( 2.5 );
        g.fill( 0.0 );
    } else if ( p == 1 ) {
        // Linear profile
        problem               = std::string( "f(x) = x-y+2*z" );
        const double df_dx[3] = { 1.0, -1.0, 2.0 };
        for ( size_t i = 0; i < N; i++ ) {
            f( i ) = 0.0;
            for ( int d = 0; d < ndim; d++ ) {
                f( i ) += df_dx[d] * x0( d, i ) / xrange[d];
                g( d, i ) = df_dx[d] / xrange[d];
            }
        }
        for ( int d = 0; d < ndim; d++ )
            grad_max[d] = df_dx[d] / xrange[d];
    } else if ( p == 2 ) {
        // Quadratic profile
        problem           = "f(x) = 1+x-y+2*z+0.1*x^2+0.2*y^2-0.3*z^2";
        double dx2        = 1.0 / std::pow( 1.0 * N, 1.0 / ndim );
        tol_grad          = 2.0 * dx2;
        tol_linear        = std::min( 4 * std::pow( 3.0, ndim ) * dx2, 0.5 );
        tol_cubic         = 1e-8;
        tol_cubic_grad    = 1e-6;
        tol_cubic_extrap1 = tol_linear;
        tol_cubic_extrap2 = 1e-8;
        for ( size_t i = 0; i < N; i++ ) {
            double x2[3] = { 0, 0, 0 }, g2[3];
            for ( int d = 0; d < ndim; d++ )
                x2[d] = x0( d, i );
            double x = x2[0] / xrange[0];
            double y = x2[1] / xrange[1];
            double z = x2[2] / xrange[2];
            f( i )   = 1 + x - y + 2 * z + 0.1 * x * x + 0.2 * y * y - 0.3 * z * z;
            g2[0]    = 1 + 0.2 * x;
            g2[1]    = -1 + 0.4 * y;
            g2[2]    = 2 - 0.6 * z;
            for ( int d = 0; d < ndim; d++ )
                g( d, i ) = g2[d] / xrange[d];
        }
        const double grad_max0[3] = { 1.2, 1, 2 };
        for ( int d = 0; d < ndim; d++ )
            grad_max[d] = grad_max0[d] / xrange[d];
        if ( N < 10 ) {
            tol_grad          = 1e100;
            tol_linear        = 1e100;
            tol_cubic         = 1e100;
            tol_cubic_grad    = 1e100;
            tol_cubic_extrap1 = 1e100;
            tol_cubic_extrap2 = 1e100;
        }
    } else {
        throw std::logic_error( "Unknown problem" );
    }
    tol_grad *= std::sqrt( grad_max[0] * grad_max[0] + grad_max[1] * grad_max[1] +
                           grad_max[2] * grad_max[2] );
    tol_cubic_grad *= std::sqrt( grad_max[0] * grad_max[0] + grad_max[1] * grad_max[1] +
                                 grad_max[2] * grad_max[2] );
    tol_grad       = std::max( tol_grad, 1e-12 );
    tol_cubic_grad = std::max( tol_cubic_grad, 1e-12 );
    return problem;
}


template<class TYPE>
std::vector<double> convert_to_double( const std::vector<TYPE> &x0 );
template<>
std::vector<double> convert_to_double<double>( const std::vector<double> &x0 )
{
    return x0;
}
template<>
std::vector<double> convert_to_double<int>( const std::vector<int> &x0 )
{
    std::vector<double> x( x0.size(), 0.0 );
    for ( size_t i = 0; i < x0.size(); i++ )
        x[i] = static_cast<double>( x0[i] ) / static_cast<double>( R_INT );
    return x;
}
template<class TYPE>
void testInterpolation( AMP::UnitTest &ut, const AMP::Array<TYPE> &x, bool check_extrap = true )
{
    PROFILE( "testInterpolation" );
    check_extrap = false; // Extrapolation is having major issues (disable for now)

    int ndim = x.size( 0 );
    size_t N = x.size( 1 );
    if ( N == 0 )
        return;

    // Check the points
    if ( N < 10000 )
        testPointSearch( ut, x );

    // Create the tessellation
    auto data = createAndTestDelaunayInterpolation<TYPE>( ut, x );
    if ( data == nullptr )
        return;
    size_t N_tri = data->get_N_tri();

    // Create a grid for interpolation at the center of each triangle
    size_t N1  = N_tri;
    auto order = new size_t[N1];
    for ( size_t i = 0; i < N1; i++ )
        order[i] = i;
    std::shuffle( order, &order[N1], std::mt19937( std::random_device()() ) );
    AMP::Array<double> x1( ndim, N1 );
    auto triangles = data->get_tri();
    for ( size_t i = 0; i < N1; i++ ) {
        for ( int d = 0; d < ndim; d++ ) {
            x1( d, i ) = 0.0;
            for ( int j = 0; j <= ndim; j++ ) {
                int k = triangles( j, order[i] );
                x1( d, i ) += x( d, k );
            }
            x1( d, i ) /= ndim + 1;
        }
    }

    // Create a second grid for interpolation
    double dx[3]     = { 0, 0, 0 };
    double xrange[3] = { 1, 1, 1 };
    double tol       = std::min( 1e-2, 1.0 / std::pow( 1.0 * N, 1.0 / ndim ) );
    for ( int d = 0; d < ndim; d++ ) {
        double x_min = 1e100;
        double x_max = -1e100;
        for ( size_t i = 0; i < N; i++ ) {
            x_min = std::min<double>( x_min, x( d, i ) );
            x_max = std::max<double>( x_max, x( d, i ) );
        }
        dx[d]     = tol * ( x_max - x_min );
        xrange[d] = std::max( fabs( x_max ), fabs( x_min ) );
    }
    int Nn    = 3;
    size_t N2 = Nn * N;
    AMP::Array<TYPE> x2( ndim, N2 );
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dist( -1, 1 );
    for ( size_t i = 0; i < N; i++ ) {
        for ( int j = 0; j < Nn; j++ ) {
            for ( int d = 0; d < ndim; d++ )
                x2( d, j + i * Nn ) = x( d, i ) + 0.1 * dx[d] * dist( gen );
        }
    }

    // Test the nearest point search
    auto nearest      = data->find_nearest( x );
    bool pass_nearest = true;
    for ( size_t i = 0; i < N; i++ ) {
        if ( nearest( i ) != i )
            pass_nearest = false;
    }
    if ( pass_nearest )
        ut.passes( "Found nearest point" );
    else
        ut.failure( "Found nearest point" );

    // Test find_tri
    auto index1        = data->find_tri( x1, false );
    auto index2        = data->find_tri( x2, true );
    bool pass_find_tri = true;
    for ( size_t i = 0; i < N1; i++ ) {
        if ( index1( i ) != (int) order[i] )
            pass_find_tri = false;
    }
    delete[] order;
    if ( pass_find_tri )
        ut.passes( "Found triangle centroids" );
    else
        ut.failure( "Found triangle centroids" );

    for ( int p = 0; p < 3; p++ ) {
        if ( N < 10 )
            continue;

        // Evaluate f and the gradient at each point
        std::string problem;
        AMP::Array<double> f( N );         // The function at the vertices
        AMP::Array<double> g( ndim, N );   // The gradient at the vertices
        AMP::Array<double> f1( N1 );       // The function at points within the triangles
        AMP::Array<double> g1( ndim, N1 ); // The gradient at points within the triangles
        AMP::Array<double> f2( N2 );       // The function at points "near" the vertices
        AMP::Array<double> g2( ndim, N2 ); // The gradient at points "near" the vertices
        double tol_grad, tol_linear, tol_cubic, tol_c_grad, tol_ce1, tol_ce2;
        initialize_problem(
            p, x1, xrange, f1, g1, tol_grad, tol_linear, tol_cubic, tol_c_grad, tol_ce1, tol_ce2 );
        initialize_problem(
            p, x2, xrange, f2, g2, tol_grad, tol_linear, tol_cubic, tol_c_grad, tol_ce1, tol_ce2 );
        problem = initialize_problem(
            p, x, xrange, f, g, tol_grad, tol_linear, tol_cubic, tol_c_grad, tol_ce1, tol_ce2 );
        auto msg = AMP::Utilities::stringf( "%s (%i,%i,%s)",
                                            problem.c_str(),
                                            (int) x.size( 0 ),
                                            (int) x.size( 1 ),
                                            AMP::getTypeID<TYPE>().name );

        // Test the gradient
        AMP::Array<double> grad( ndim, N );
        [[maybe_unused]] double error[4] = { 0 };
        for ( int method = 1; method <= 4; method++ ) {
            if ( method == 2 || N < 20 )
                continue;
            data->calc_node_gradient( f.data(), method, grad.data() );
            double err = 0.0;
            for ( size_t i = 0; i < N; i++ ) {
                for ( int d = 0; d < ndim; d++ )
                    err += ( grad( d, i ) - g( d, i ) ) * ( grad( d, i ) - g( d, i ) );
            }
            err               = std::sqrt( err / N );
            error[method - 1] = err;
            auto message      = AMP::Utilities::stringf( "calc_gradient (%i,%s) (%i,%i) (%e)",
                                                    method,
                                                    problem.c_str(),
                                                    (int) x.size( 0 ),
                                                    (int) x.size( 1 ),
                                                    err );
            if ( err <= tol_grad )
                ut.passes( message );
            else
                ut.failure( message );
        }

        // Test nearest-neighbor interpolation
        auto fi                  = data->interp_nearest( f, x, nearest );
        bool pass_interp_nearest = true;
        for ( size_t i = 0; i < N; i++ ) {
            if ( !approx_equal( f( i ), fi( i ) ) )
                pass_interp_nearest = false;
        }
        if ( pass_interp_nearest )
            ut.passes( "nearest-neighbor interpolation: " + msg );
        else
            ut.failure( "nearest-neighbor interpolation: " + msg );

        // Allocate variables for interpolation
        AMP::Array<double> fi1, fi2, fi3, fi4, gi1, gi2, gi3, gi4;
        double f1_norm, f2_norm, f3_norm, f4_norm;
        double g1_norm, g2_norm, g3_norm, g4_norm;
        double f1_err, f2_err, f3_err, f4_err;
        double g1_err, g2_err, g3_err, g4_err;
        auto f_mask         = new bool[N2];
        auto g_mask         = new bool[N2 * ndim];
        bool *f_extrap_mask = nullptr, *g_extrap_mask = nullptr;
        if ( !check_extrap || N < 50 ) {
            f_extrap_mask = f_mask;
            g_extrap_mask = g_mask;
        }

        // Test the linear interpolation
        std::tie( fi1, std::ignore ) = data->interp_linear( f, x1, index1, false );
        std::tie( fi2, std::ignore ) = data->interp_linear( f, x2, index2, false );
        std::tie( fi3, std::ignore ) = data->interp_linear( f, x2, index2, true );
        for ( unsigned int i = 0; i < N2; i++ ) {
            f_mask[i] = fi2( i ) == fi2( i );
            for ( int d = 0; d < ndim; d++ )
                g_mask[d + i * ndim] = f_mask[i];
        }
        f1_norm = L2norm( f1 );
        f2_norm = L2norm( f2, f_mask );
        f3_norm = L2norm( f2, f_extrap_mask );
        f1_err  = L2errNorm( f1, fi1 );
        f2_err  = L2errNorm( f2, fi2, f_mask );
        f3_err  = L2errNorm( f2, fi3, f_extrap_mask );
        if ( f1_err <= tol_linear * f1_norm && f2_err <= tol_linear * f2_norm &&
             f3_err <= tol_linear * f3_norm ) {
            ut.passes( "linear interpolation: " + msg );
        } else {
            printf( "Failed linear interpolation: %s (%e)\n", msg.data(), tol_linear );
            printf( "   %e %e\n", f1_err, f1_norm );
            printf( "   %e %e\n", f2_err, f2_norm );
            printf( "   %e %e\n", f3_err, f3_norm );
            ut.failure( "linear interpolation: " + msg );
        }
        // Test the cubic interpolation (using the exact gradient)
        std::tie( fi1, gi1 )    = data->interp_cubic( f, g, x1, index1, false );
        std::tie( fi2, gi2 )    = data->interp_cubic( f, g, x2, index2, 0 );
        std::tie( fi3, gi3 )    = data->interp_cubic( f, g, x2, index2, 1 );
        std::tie( fi4, gi4 )    = data->interp_cubic( f, g, x2, index2, 2 );
        f1_norm                 = L2norm( f1 );
        f2_norm                 = L2norm( f2, f_mask );
        f3_norm                 = L2norm( f2, f_extrap_mask );
        f4_norm                 = L2norm( f2, f_extrap_mask );
        g1_norm                 = std::max( L2norm( g1 ), 1.0 );
        g2_norm                 = std::max( L2norm( g2, g_mask ), 1.0 );
        g3_norm                 = std::max( L2norm( g2 ), 1.0 );
        g4_norm                 = std::max( L2norm( g2 ), 1.0 );
        f1_err                  = L2errNorm( f1, fi1 );
        f2_err                  = L2errNorm( f2, fi2, f_mask );
        f3_err                  = L2errNorm( f2, fi3, f_extrap_mask );
        f4_err                  = L2errNorm( f2, fi4, f_extrap_mask );
        g1_err                  = L2errNorm( g1, gi1 );
        g2_err                  = L2errNorm( g2, gi2, g_mask );
        g3_err                  = L2errNorm( g2, gi3, g_extrap_mask );
        g4_err                  = L2errNorm( g2, gi4, g_extrap_mask );
        bool pass_interp_cubic1 = f1_err <= tol_cubic * f1_norm && g1_err <= tol_c_grad * g1_norm;
        bool pass_interp_cubic2 = f2_err <= tol_cubic * f2_norm && g2_err <= tol_c_grad * g2_norm;
        // The error in the gradient (and points) is large for points outside the domain
        bool pass_interp_cubic3 =
            f3_err <= tol_linear * f3_norm && g3_norm <= 1e100 && g3_err < 1e100;
        bool pass_interp_cubic4 = f4_err <= tol_cubic * f4_norm && g4_err <= tol_c_grad * g4_norm;
        if ( pass_interp_cubic1 && pass_interp_cubic2 && pass_interp_cubic3 &&
             pass_interp_cubic4 ) {
            ut.passes( "cubic interpolation: " + msg );
        } else {
            printf( "Failed cubic interpolation: %s (%e,%e)\n", msg.data(), tol_cubic, tol_c_grad );
            printf( "   %e %e %e %e\n", f1_err, f1_norm, g1_err, g1_norm );
            printf( "   %e %e %e %e\n", f2_err, f2_norm, g2_err, g2_norm );
            printf( "   %e %e %e %e\n", f3_err, f3_norm, g3_err, g3_norm );
            printf( "   %e %e %e %e\n", f4_err, f4_norm, g4_err, g4_norm );
            ut.failure( "cubic interpolation: " + msg );
        }

        // Free some memory
        delete[] f_mask;
        delete[] g_mask;
    }
}
template<class TYPE>
void testInterpolationRandom( AMP::UnitTest &ut, int d, int N )
{
    // Generate random points
    auto x = createRandomPoints<TYPE>( d, N );
    // Verify small number of points are not collinear
    if ( d == 2 && N <= 10 ) {
        int i = 0;
        while ( AMP::DelaunayTessellation::collinear( x ) && i < 5 ) {
            x = createRandomPoints<TYPE>( d, N );
            i++;
        }
    }
    // Run the tests
    testInterpolation( ut, x );
}


// This function creates a set of points within box1 that are not contained within box2
// Each box is specified as [xmin,xmax,ymin,ymax,zmin,zmax] and box2 may be NULL
// The grid spacing is given by dx
template<class TYPE>
AMP::Array<TYPE>
createBoxPoints( int ndim, const int *box1, const int *dx, const int *box2 = nullptr )
{
    int N[5] = { 0, 0, 0, 0, 0 };
    int Nt   = 1;
    for ( int i = 0; i < ndim; i++ ) {
        N[i] = static_cast<int>( floor( ( box1[2 * i + 1] - box1[2 * i + 0] ) / dx[i] ) ) + 1;
        Nt *= N[i];
    }
    std::vector<TYPE> points;
    // Create all points in the box1 and not within box2
    points.reserve( ndim * Nt );
    if ( ndim == 1 ) {
        for ( int i = 0; i < N[0]; i++ ) {
            int x = box1[0] + i * dx[0];
            if ( box2 != nullptr ) {
                if ( x >= box2[0] && x <= box2[1] )
                    continue;
            }
            points.push_back( x );
        }
    } else if ( ndim == 2 ) {
        for ( int i = 0; i < N[0]; i++ ) {
            int x = box1[0] + i * dx[0];
            for ( int j = 0; j < N[1]; j++ ) {
                int y = box1[2] + j * dx[1];
                if ( box2 != nullptr ) {
                    if ( x >= box2[0] && x <= box2[1] && y >= box2[3] && y <= box2[4] )
                        continue;
                }
                points.push_back( x );
                points.push_back( y );
            }
        }
    } else if ( ndim == 3 ) {
        for ( int i = 0; i < N[0]; i++ ) {
            int x = box1[0] + i * dx[0];
            for ( int j = 0; j < N[1]; j++ ) {
                int y = box1[2] + j * dx[1];
                for ( int k = 0; k < N[2]; k++ ) {
                    int z = box1[4] + k * dx[2];
                    if ( box2 != nullptr ) {
                        if ( x >= box2[0] && x <= box2[1] && y >= box2[2] && y <= box2[3] &&
                             z >= box2[4] && z <= box2[5] )
                            continue;
                    }
                    points.push_back( x );
                    points.push_back( y );
                    points.push_back( z );
                }
            }
        }
    }
    AMP::Array<TYPE> p( ndim, points.size() / ndim );
    for ( size_t i = 0; i < points.size(); i++ )
        p( i ) = points[i];
    return p;
}


template<class TYPE>
AMP::Array<TYPE> createProblem( int problem )
{
    PROFILE( "createProblem" );
    AMP::Array<TYPE> points;
    if ( problem == 1 ) {
        // This is a simple problem of a uniform mesh
        int box[6] = { 0, 4, 0, 4, 0, 4 };
        int dx[3]  = { 1, 1, 1 };
        points     = createBoxPoints<TYPE>( 3, box, dx );
    } else if ( problem == 2 ) {
        // This set was taken from a failed hierarchy that caused an infinite loop
        int box1[6] = { -3, 65, 61, 129, 29, 65 };
        int box2[6] = { 0, 62, 60, 126, 28, 62 };
        int box3[6] = { 0, 62, 62, 126, 30, 62 };
        int dx1[3]  = { 4, 4, 4 };
        int dx3[3]  = { 2, 2, 2 };
        auto set1   = createBoxPoints<TYPE>( 3, box1, dx1, box2 );
        auto set2   = createBoxPoints<TYPE>( 3, box3, dx3 );
        points      = AMP::Array<TYPE>::cat( { set1, set2 }, 1 );
    } else if ( problem == 3 ) {
        // This is reduced version of problem 2
        int box1[6] = { -3, 57, -1, 51, -1, 11 };
        int box2[6] = { 0, 55, -1, 51, -1, 11 };
        int box3[6] = { 0, 52, 0, 38, 0, 12 };
        int box4[6] = { 0, 50, 0, 38, 2, 12 };
        int dx1[3]  = { 4, 4, 4 };
        int dx3[3]  = { 2, 2, 4 };
        auto set1   = createBoxPoints<TYPE>( 3, box1, dx1, box2 );
        auto set2   = createBoxPoints<TYPE>( 3, box3, dx3, box4 );
        points      = AMP::Array<TYPE>::cat( { set1, set2 }, 1 );
    } else if ( problem == 4 ) {
        // Invalid triangle neighbor
        int box1[6] = { -3, 15, -1, 43, -1, 11 };
        int box2[6] = { 0, 15, -1, 43, -1, 11 };
        int box3[6] = { 0, 12, 36, 44, 0, 12 };
        int box4[6] = { 0, 10, 36, 44, 2, 12 };
        int dx1[3]  = { 4, 4, 4 };
        int dx3[3]  = { 2, 2, 4 };
        auto set1   = createBoxPoints<TYPE>( 3, box1, dx1, box2 );
        auto set2   = createBoxPoints<TYPE>( 3, box3, dx3, box4 );
        points      = AMP::Array<TYPE>::cat( { set1, set2 }, 1 );
    } else if ( problem == 5 ) {
        // This is a problem that resulted in no valid flips
        int box1[6] = { -3, 55, -1, 55, -1, 15 };
        int box2[6] = { 0, 52, -2, 52, -2, 12 };
        int box3[6] = { 0, 52, 0, 52, 0, 12 };
        int dx1[3]  = { 4, 4, 4 };
        int dx3[3]  = { 2, 2, 2 };
        auto set1   = createBoxPoints<TYPE>( 3, box1, dx1, box2 );
        auto set2   = createBoxPoints<TYPE>( 3, box3, dx3 );
        points      = AMP::Array<TYPE>::cat( { set1, set2 }, 1 );
    } else if ( problem == 6 ) {
        // A set of points that caused the nearest-neighbor point search to fail (01/18/13)
        if ( typeid( TYPE ) == typeid( double ) ) {
            const double const_points[30] = {
                0.228134639925963, 0.107455922361451, 0.865652664047092, 0.184470611307978,
                0.838401675869785, 0.668757150119921, 0.369538598042889, 0.123727772818859,
                0.640994658126969, 0.499916995230264, 0.448223924105178, 0.071055349040041,
                0.234558133130565, 0.459801521022238, 0.439658494159647, 0.053472364370121,
                0.256549138105335, 0.295241277565778, 0.375079637861114, 0.623130377070684,
                0.385421525726560, 0.965203651821110, 0.545386155470929, 0.355259056526340,
                0.034658433895993, 0.917076353750111, 0.761392645215632, 0.350240130632096,
                0.309851148106666, 0.654482311175926
            };
            points.resize( 3, 10 );
            for ( int i = 0; i < 30; i++ )
                points( i ) = const_points[i];
        }
    }
    return points;
}


// Test the frequency with which random points are collinear
// Note the number of collinear sets is primarilly determined by TOL_COLLINEAR
void testCollinear( AMP::UnitTest &ut )
{
    PROFILE( "testCollinear" );
    int N = 100000;
    int C = 0;
    for ( int i = 0; i < N; i++ ) {
        auto x    = createRandomPoints<double>( 2, 3 );
        bool test = AMP::DelaunayTessellation::collinear( x );
        if ( test )
            C++;
    }
    printf( "   %i of %i sets of points were collinear\n", C, N );
    if ( C <= 0.04 * N )
        ut.passes( "expected number of sets are collinear" );
    else
        ut.failure( "excessive number of sets are collinear" );
}


// Get the convergence
template<class TYPE>
void testConvergence( AMP::UnitTest &ut, int ndim )
{
    // Create two grids with different resolution
    int box[6] = { -1000, 1000, -1000, 1000, -1000, 1000 };
    int dx1[6] = { 100, 100, 100 };
    int dx2[6] = { 50, 50, 50 };
    auto set1  = createBoxPoints<TYPE>( 3, box, dx1 );
    auto set2  = createBoxPoints<TYPE>( 3, box, dx2 );
    // Create the tessellation
    [[maybe_unused]] auto data1 = createAndTestDelaunayInterpolation<TYPE>( ut, ndim, set1 );
    [[maybe_unused]] auto data2 = createAndTestDelaunayInterpolation<TYPE>( ut, ndim, set2 );
    // Check the convergence for different interpolation method
    for ( int method = 0; method < 3; method++ ) {}
}


// Main
int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    PROFILE_ENABLE( 5 ); // 0: code, 1: tests, 3: basic timers, 5: all timers

    // Run some basic tests on the point search
    printp( "Running point search tests\n" );
    for ( int d = 1; d <= 5; d++ ) {
        testPointSearch( ut, createRandomPoints<double>( d, 10 ) );
        testPointSearch( ut, createRandomPoints<double>( d, 20 ) );
        testPointSearch( ut, createRandomPoints<double>( d, 50 ) );
        testPointSearch( ut, createRandomPoints<double>( d, 3000 ) );
        testPointSearch( ut, createRandomPoints<int>( d, 100 ) );
    }

    // Test collinear
    printp( "Running collinear test\n" );
    testCollinear( ut );

    // Test DelaunayInterpolation with random points in 1D, 2D and 3D
    for ( int d = 1; d <= NDIM_MAX; d++ ) {
        printp( "Running interpolation tests with random points %iD (double)\n", d );
        testInterpolationRandom<double>( ut, d, d + 1 ); // minimum # of points
        testInterpolationRandom<double>( ut, d, 10 );    // small # of points
        testInterpolationRandom<double>( ut, d, 1000 );  // medium # of points
        testInterpolationRandom<double>( ut, d, 10000 ); // large # of points
        printp( "Running interpolation tests with random points %iD (int)\n", d );
        testInterpolationRandom<int>( ut, d, d + 1 ); // minimum # of points
        testInterpolationRandom<int>( ut, d, 10 );    // small # of points
        testInterpolationRandom<int>( ut, d, 1000 );  // medium # of points
        // testInterpolationRandom<int>( ut, d, 10000 ); // large # of points
    }

    // Run some predefined tests
    for ( int i = 1; i <= 6; i++ ) {
        printp( "Running predefined test %i\n", i );
        auto tmp = AMP::Utilities::stringf( "Problem %i", i );
        PROFILE2( tmp );
        testInterpolation( ut, createProblem<int>( i ) );
        testInterpolation( ut, createProblem<double>( i ) );
    }

    // Run any input file problems
    for ( int i = 1; i < argc; i++ ) {
        printp( "Running input: %s\n", argv[i] );
        auto data = readPoints( argv[i] );
        int ndim  = data.size( 0 );
        if ( ndim <= NDIM_MAX )
            testInterpolation<double>( ut, data, false );
        else
            testPointSearch( ut, data );
    }

    PROFILE_SAVE( "test_DelaunayInterpolation" );
    ut.report();
    auto N_errors = static_cast<int>( ut.NumFailGlobal() );
    AMP::AMPManager::shutdown();
    return N_errors;
}
