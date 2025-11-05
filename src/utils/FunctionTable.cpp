#include "AMP/utils/FunctionTable.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Array.h"

#include <complex>
#include <functional>

#ifdef AMP_USE_LAPACK_WRAPPERS
    #include "LapackWrappers.h"
#else
template<class TYPE>
class Lapack
{
public:
    static void axpy( size_t, TYPE, const TYPE *, size_t, TYPE *, size_t )
    {
        AMP_ERROR( "Lapack required" );
    }
    static void gemv( char,
                      size_t,
                      size_t,
                      TYPE,
                      const TYPE *,
                      size_t,
                      const TYPE *,
                      size_t,
                      TYPE,
                      TYPE *,
                      size_t )
    {
        AMP_ERROR( "Lapack required" );
    }
    static void gemm( char,
                      char,
                      size_t,
                      size_t,
                      size_t,
                      TYPE,
                      const TYPE *,
                      size_t,
                      const TYPE *,
                      size_t,
                      TYPE,
                      TYPE *,
                      size_t )
    {
        AMP_ERROR( "Lapack required" );
    }
};
#endif


namespace AMP {


/********************************************************
 *  axpy                                                 *
 ********************************************************/
template<>
void call_axpy<float>( size_t N, const float alpha, const float *x, float *y )
{
    Lapack<float>::axpy( N, alpha, x, 1, y, 1 );
}
template<>
void call_axpy<double>( size_t N, const double alpha, const double *x, double *y )
{
    Lapack<double>::axpy( N, alpha, x, 1, y, 1 );
}


/********************************************************
 *  Multiply two arrays                                  *
 ********************************************************/
template<>
void call_gemv<double>(
    size_t M, size_t N, double alpha, double beta, const double *A, const double *x, double *y )
{
    Lapack<double>::gemv( 'N', M, N, alpha, A, M, x, 1, beta, y, 1 );
}
template<>
void call_gemv<float>(
    size_t M, size_t N, float alpha, float beta, const float *A, const float *x, float *y )
{
    Lapack<float>::gemv( 'N', M, N, alpha, A, M, x, 1, beta, y, 1 );
}
template<>
void call_gemm<double>( size_t M,
                        size_t N,
                        size_t K,
                        double alpha,
                        double beta,
                        const double *A,
                        const double *B,
                        double *C )
{
    Lapack<double>::gemm( 'N', 'N', M, K, N, alpha, A, M, B, N, beta, C, M );
}
template<>
void call_gemm<float>( size_t M,
                       size_t N,
                       size_t K,
                       float alpha,
                       float beta,
                       const float *A,
                       const float *B,
                       float *C )
{
    Lapack<float>::gemm( 'N', 'N', M, K, N, alpha, A, M, B, N, beta, C, M );
}


} // namespace AMP


static_assert( std::is_same_v<typename AMP::FunctionTable<double>::value_type, double> );
static_assert(
    std::is_same_v<typename AMP::FunctionTable<double>::cloneTo<float>::value_type, float> );


/********************************************************
 *  Explicit instantiations of FunctionTable             *
 ********************************************************/
template<class T>
using FUN1 = std::function<T( const T & )>;
template<class T>
using FUN2 = std::function<T( const T &, const T & )>;
#define INSTANTIATE( T )                                                                          \
    template class AMP::FunctionTable<T>;                                                         \
    template void AMP::FunctionTable<T>::transform<FUN1<T>>( FUN1<T> &, size_t, const T *, T * ); \
    template void AMP::FunctionTable<T>::transform<FUN2<T>>(                                      \
        FUN2<T> &, size_t, const T *, const T *, T * )
INSTANTIATE( bool );
INSTANTIATE( char );
INSTANTIATE( uint8_t );
INSTANTIATE( uint16_t );
INSTANTIATE( uint32_t );
INSTANTIATE( uint64_t );
INSTANTIATE( int8_t );
INSTANTIATE( int16_t );
INSTANTIATE( int32_t );
INSTANTIATE( int64_t );
INSTANTIATE( long long );
INSTANTIATE( float );
INSTANTIATE( double );
INSTANTIATE( long double );
template class AMP::FunctionTable<std::complex<float>>;
template class AMP::FunctionTable<std::complex<double>>;
template class AMP::FunctionTable<std::string>;
