#ifndef included_AMP_FunctionTable_hpp
#define included_AMP_FunctionTable_hpp

#include "AMP/utils/FunctionTable.h"
#include "AMP/utils/TypeTraits.h"
#include "AMP/utils/UtilityMacros.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>

namespace AMP {


/********************************************************
 *  Random number initialization                         *
 ********************************************************/
template<class TYPE>
void FunctionTable<TYPE>::rand( size_t N, TYPE *x )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    if constexpr ( std::is_same_v<TYPE, bool> ) {
        std::uniform_int_distribution<unsigned short> dis( 0, 1 );
        for ( size_t i = 0; i < N; i++ )
            x[i] = dis( gen ) != 0;
    } else if constexpr ( std::is_integral_v<TYPE> ) {
        if ( std::is_signed_v<TYPE> ) {
            auto min = static_cast<int64_t>( std::numeric_limits<TYPE>::min() );
            auto max = static_cast<int64_t>( std::numeric_limits<TYPE>::max() );
            std::uniform_int_distribution<int64_t> dis( min, max );
            for ( size_t i = 0; i < N; i++ )
                x[i] = static_cast<TYPE>( dis( gen ) );
        } else {
            auto min = static_cast<uint64_t>( std::numeric_limits<TYPE>::min() );
            auto max = static_cast<uint64_t>( std::numeric_limits<TYPE>::max() );
            std::uniform_int_distribution<uint64_t> dis( min, max );
            for ( size_t i = 0; i < N; i++ )
                x[i] = static_cast<TYPE>( dis( gen ) );
        }
    } else if constexpr ( std::is_floating_point_v<TYPE> ) {
        std::uniform_real_distribution<TYPE> dis;
        for ( size_t i = 0; i < N; i++ )
            x[i] = dis( gen );
    } else if constexpr ( std::is_same_v<TYPE, std::complex<float>> ) {
        std::uniform_real_distribution<float> dis;
        for ( size_t i = 0; i < N; i++ )
            x[i] = std::complex<float>( dis( gen ), dis( gen ) );
    } else if constexpr ( std::is_same_v<TYPE, std::complex<double>> ) {
        std::uniform_real_distribution<double> dis;
        for ( size_t i = 0; i < N; i++ )
            x[i] = std::complex<double>( dis( gen ), dis( gen ) );
    } else {
        AMP_ERROR( "rand not implemented" );
    }
}


/********************************************************
 *  Reduction                                            *
 ********************************************************/
template<class TYPE>
template<typename LAMBDA>
TYPE FunctionTable<TYPE>::reduce( LAMBDA &op, size_t N, const TYPE *x, TYPE y )
{
    for ( size_t i = 0; i < N; i++ )
        y = op( x[i], y );
    return y;
}
template<class TYPE>
template<typename LAMBDA>
TYPE FunctionTable<TYPE>::reduce( LAMBDA &op, size_t N, const TYPE *x, const TYPE *y, TYPE z )
{
    for ( size_t i = 0; i < N; i++ )
        z = op( x[i], y[i], z );
    return z;
}
template<class TYPE>
TYPE FunctionTable<TYPE>::min( size_t N, const TYPE *x )
{
    if constexpr ( std::is_arithmetic_v<TYPE> && !std::is_same_v<TYPE, bool> ) {
        if ( N == 0 )
            return 0;
        TYPE y = x[0];
        for ( size_t i = 0; i < N; i++ )
            y = x[i] < y ? x[i] : y;
        return y;
    } else {
        AMP_ERROR( "min not implemented" );
    }
}
template<class TYPE>
TYPE FunctionTable<TYPE>::max( size_t N, const TYPE *x )
{
    if constexpr ( std::is_arithmetic_v<TYPE> && !std::is_same_v<TYPE, bool> ) {
        if ( N == 0 )
            return 0;
        TYPE y = x[0];
        for ( size_t i = 0; i < N; i++ )
            y = x[i] > y ? x[i] : y;
        return y;
    } else {
        AMP_ERROR( "max not implemented" );
    }
}
template<class TYPE>
TYPE FunctionTable<TYPE>::sum( size_t N, const TYPE *x )
{
    if constexpr ( std::is_arithmetic_v<TYPE> && !std::is_same_v<TYPE, bool> ) {
        if ( N == 0 )
            return 0;
        TYPE y = x[0];
        for ( size_t i = 1; i < N; i++ )
            y += x[i];
        return y;
    } else {
        AMP_ERROR( "sum not implemented" );
    }
}


/********************************************************
 *  Unary transformation                                 *
 ********************************************************/
template<class TYPE>
template<typename LAMBDA>
void FunctionTable<TYPE>::transform( LAMBDA &fun, size_t N, const TYPE *x, TYPE *y )
{
    for ( size_t i = 0; i < N; i++ )
        y[i] = fun( x[i] );
}
template<class TYPE>
template<typename LAMBDA>
void FunctionTable<TYPE>::transform( LAMBDA &fun, size_t N, const TYPE *x, const TYPE *y, TYPE *z )
{
    for ( size_t i = 0; i < N; i++ )
        z[i] = fun( x[i], y[i] );
}


/********************************************************
 *  axpy                                                 *
 ********************************************************/
template<class TYPE>
void call_axpy( size_t N, const TYPE alpha, const TYPE *x, TYPE *y );
template<>
void call_axpy<float>( size_t N, const float alpha, const float *x, float *y );
template<>
void call_axpy<double>( size_t N, const double alpha, const double *x, double *y );
template<class TYPE>
void call_axpy( size_t N, const TYPE alpha, const TYPE *x, TYPE *y )
{
    for ( size_t i = 0; i < N; i++ )
        y[i] += alpha * x[i];
}
template<class TYPE>
void FunctionTable<TYPE>::axpy( TYPE alpha, size_t N, const TYPE *x, TYPE *y )
{
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        call_axpy( N, alpha, x, y );
    } else {
        AMP_ERROR( "axpy not implemented" );
    }
}
template<class TYPE>
void FunctionTable<TYPE>::axpby( TYPE alpha, size_t N, const TYPE *x, TYPE beta, TYPE *y )
{
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        if ( beta == 1 ) {
            call_axpy( N, alpha, x, y );
        } else {
            for ( size_t i = 0; i < N; i++ )
                y[i] = alpha * x[i] + beta * y[i];
        }
    } else {
        AMP_ERROR( "axpby not implemented" );
    }
}
template<class TYPE>
void FunctionTable<TYPE>::px( size_t N, TYPE x, TYPE *y )
{
    for ( size_t i = 0; i < N; i++ )
        y[i] += x;
}
template<class TYPE>
void FunctionTable<TYPE>::px( size_t N, const TYPE *x, TYPE *y )
{
    for ( size_t i = 0; i < N; i++ )
        y[i] += x[i];
}
template<class TYPE>
void FunctionTable<TYPE>::mx( size_t N, TYPE x, TYPE *y )
{
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        for ( size_t i = 0; i < N; i++ )
            y[i] -= x;
    } else {
        AMP_ERROR( "mx not implemented" );
    }
}
template<class TYPE>
void FunctionTable<TYPE>::mx( size_t N, const TYPE *x, TYPE *y )
{
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        for ( size_t i = 0; i < N; i++ )
            y[i] -= x[i];
    } else {
        AMP_ERROR( "mx not implemented" );
    }
}


/********************************************************
 *  Multiply two arrays                                  *
 ********************************************************/
template<class TYPE>
void call_gemv( size_t M, size_t N, TYPE alpha, TYPE beta, const TYPE *A, const TYPE *x, TYPE *y );
template<>
void call_gemv<double>(
    size_t M, size_t N, double alpha, double beta, const double *A, const double *x, double *y );
template<>
void call_gemv<float>(
    size_t M, size_t N, float alpha, float beta, const float *A, const float *x, float *y );
template<class TYPE>
void call_gemv( size_t M, size_t N, TYPE alpha, TYPE beta, const TYPE *A, const TYPE *x, TYPE *y )
{
    if constexpr ( std::is_same_v<TYPE, bool> ) {
        AMP_ERROR( "gemv not implemented for bool" );
    } else if constexpr ( std::is_arithmetic_v<TYPE> ) {
        for ( size_t i = 0; i < M; i++ )
            y[i] = beta * y[i];
        for ( size_t j = 0; j < N; j++ ) {
            for ( size_t i = 0; i < M; i++ )
                y[i] += alpha * A[i + j * M] * x[j];
        }
    } else {
        AMP_ERROR( "gemv not implemented" );
    }
}
template<class TYPE>
void call_gemm(
    size_t M, size_t N, size_t K, TYPE alpha, TYPE beta, const TYPE *A, const TYPE *B, TYPE *C );
template<>
void call_gemm<double>( size_t M,
                        size_t N,
                        size_t K,
                        double alpha,
                        double beta,
                        const double *A,
                        const double *B,
                        double *C );
template<>
void call_gemm<float>( size_t M,
                       size_t N,
                       size_t K,
                       float alpha,
                       float beta,
                       const float *A,
                       const float *B,
                       float *C );
template<class TYPE>
void call_gemm(
    size_t M, size_t N, size_t K, TYPE alpha, TYPE beta, const TYPE *A, const TYPE *B, TYPE *C )
{
    if constexpr ( std::is_same_v<TYPE, bool> ) {
        AMP_ERROR( "gemv not implemented for bool" );
    } else if constexpr ( std::is_arithmetic_v<TYPE> ) {
        for ( size_t i = 0; i < K * M; i++ )
            C[i] = beta * C[i];
        for ( size_t k = 0; k < K; k++ ) {
            for ( size_t j = 0; j < N; j++ ) {
                for ( size_t i = 0; i < M; i++ )
                    C[i + k * M] += alpha * A[i + j * M] * B[j + k * N];
            }
        }
    } else {
        AMP_ERROR( "gemv not implemented" );
    }
}
template<class TYPE>
ArraySize FunctionTable<TYPE>::multiplySize( const ArraySize &sa, const ArraySize &sb )
{
    if ( sa[1] != sb[0] )
        AMP_ERROR( "Inner dimensions must match" );
    if ( sa.ndim() == 2 && sb.ndim() == 1 ) {
        return { sa[0] };
    } else if ( sa.ndim() <= 2 && sb.ndim() <= 2 ) {
        return { sa[0], sb[1] };
    } else {
        AMP_ERROR( "Not finished yet" );
    }
}
template<class TYPE>
void FunctionTable<TYPE>::gemm( TYPE alpha,
                                const ArraySize &sa,
                                const TYPE *a,
                                const ArraySize &sb,
                                const TYPE *b,
                                TYPE beta,
                                const ArraySize &sc,
                                TYPE *c )
{
    AMP_ASSERT( sc == multiplySize( sa, sb ) );
    if ( sa.ndim() == 2 && sb.ndim() == 1 ) {
        call_gemv<TYPE>( sa[0], sa[1], alpha, beta, a, b, c );
    } else if ( sa.ndim() <= 2 && sb.ndim() <= 2 ) {
        call_gemm<TYPE>( sa[0], sa[1], sb[1], alpha, beta, a, b, c );
    } else {
        AMP_ERROR( "Not finished yet" );
    }
}
template<class TYPE>
void FunctionTable<TYPE>::multiply( const ArraySize &sa,
                                    const TYPE *a,
                                    const ArraySize &sb,
                                    const TYPE *b,
                                    const ArraySize &sc,
                                    TYPE *c )
{
    if constexpr ( !std::is_arithmetic_v<TYPE> ) {
        AMP_ERROR( "Not finished yet" );
    } else {
        AMP_ASSERT( sc == multiplySize( sa, sb ) );
        if ( sa.ndim() == 2 && sb.ndim() == 1 ) {
            call_gemv<TYPE>( sa[0], sa[1], 1, 0, a, b, c );
        } else if ( sa.ndim() <= 2 && sb.ndim() <= 2 ) {
            call_gemm<TYPE>( sa[0], sa[1], sb[1], 1, 0, a, b, c );
        } else {
            AMP_ERROR( "Not finished yet" );
        }
    }
}


/********************************************************
 *  Check if two arrays are equal                        *
 ********************************************************/
template<class TYPE>
bool FunctionTable<TYPE>::equals( size_t N, const TYPE *a, const TYPE *b, TYPE tol )
{
    bool pass = true;
    if constexpr ( std::is_same_v<TYPE, bool> || std::is_integral_v<TYPE> ) {
        for ( size_t i = 0; i < N; i++ )
            pass = pass && a[i] == b[i];
    } else if constexpr ( std::is_arithmetic_v<TYPE> ) {
        for ( size_t i = 0; i < N; i++ )
            pass = pass && ( std::abs( a[i] - b[i] ) < tol );
    } else {
        for ( size_t i = 0; i < N; i++ )
            pass = pass && a[i] == b[i];
    }
    return pass;
}


/********************************************************
 *  Specialized Functions                                *
 ********************************************************/
template<class TYPE>
void FunctionTable<TYPE>::transformReLU( size_t N, const TYPE *A, TYPE *B )
{
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        const auto &fun = []( const TYPE &a ) { return std::max( a, static_cast<TYPE>( 0 ) ); };
        transform( fun, N, A, B );
    } else {
        AMP_ERROR( "min not implemented" );
    }
}

template<class TYPE>
void FunctionTable<TYPE>::transformAbs( size_t N, const TYPE *A, TYPE *B )
{
    if constexpr ( std::is_signed_v<TYPE> ) {
        const auto &fun = []( const TYPE &a ) { return std::abs( a ); };
        transform( fun, N, A, B );
    } else {
        AMP_ERROR( "min not implemented" );
    }
}
template<class TYPE>
void FunctionTable<TYPE>::transformTanh( size_t N, const TYPE *A, TYPE *B )
{
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        const auto &fun = []( const TYPE &a ) { return std::tanh( a ); };
        transform( fun, N, A, B );
    } else {
        AMP_ERROR( "min not implemented" );
    }
}

template<class TYPE>
void FunctionTable<TYPE>::transformHardTanh( size_t N, const TYPE *A, TYPE *B )
{
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        const auto &fun = []( const TYPE &a ) {
            return std::max<TYPE>( -static_cast<TYPE>( 1.0 ),
                                   std::min( static_cast<TYPE>( 1.0 ), a ) );
        };
        transform( fun, N, A, B );
    } else {
        AMP_ERROR( "min not implemented" );
    }
}

template<class TYPE>
void FunctionTable<TYPE>::transformSigmoid( size_t N, const TYPE *A, TYPE *B )
{
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        const auto &fun = []( const TYPE &a ) { return 1.0 / ( 1.0 + std::exp( -a ) ); };
        transform( fun, N, A, B );
    } else {
        AMP_ERROR( "min not implemented" );
    }
}

template<class TYPE>
void FunctionTable<TYPE>::transformSoftPlus( size_t N, const TYPE *A, TYPE *B )
{
    if constexpr ( std::is_arithmetic_v<TYPE> ) {
        const auto &fun = []( const TYPE &a ) { return std::log1p( std::exp( a ) ); };
        transform( fun, N, A, B );
    } else {
        AMP_ERROR( "min not implemented" );
    }
}


} // namespace AMP

#endif
