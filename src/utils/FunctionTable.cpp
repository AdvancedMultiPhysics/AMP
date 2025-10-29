#include "AMP/utils/FunctionTable.hpp"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/Array.h"


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


/********************************************************
 *  Explicit instantiations of FunctionTable             *
 ********************************************************/
template class AMP::FunctionTable<float>;
template class AMP::FunctionTable<double>;
