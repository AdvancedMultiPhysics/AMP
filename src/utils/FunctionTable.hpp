#ifndef included_FunctionTable_hpp
#define included_FunctionTable_hpp

#include "utils/FunctionTable.h"
#include "utils/Utilities.h"
#include "utils/LapackWrappers.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <random>


namespace AMP {


/********************************************************
*  Random number initialization                         *
********************************************************/
template <class TYPE, class FUN>
void FunctionTable::rand( Array<TYPE, FUN> &x )
{
    FunctionTable::rand<TYPE>( x.length(), x.data() );
}
template <>
inline void FunctionTable::rand<double>( size_t N, double *x )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<> dis( 0, 1 );
    for ( size_t i = 0; i < N; i++ )
        x[i]       = dis( gen );
}
template <>
inline void FunctionTable::rand<float>( size_t N, float *x )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<> dis( 0, 1 );
    for ( size_t i = 0; i < N; i++ )
        x[i]       = dis( gen );
}
template <>
inline void FunctionTable::rand<int>( size_t N, int *x )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_int_distribution<> dis;
    for ( size_t i = 0; i < N; i++ )
        x[i]       = dis( gen );
}


/********************************************************
*  Reduction                                            *
********************************************************/
template <class TYPE, class FUN, typename LAMBDA>
inline TYPE FunctionTable::reduce( LAMBDA &op, const Array<TYPE, FUN> &A, const TYPE& initialValue )
{
    if ( A.length() == 0 )
        return TYPE();
    const TYPE *x = A.data();
    TYPE y        = initialValue;  
    for ( size_t i = 0; i < A.length(); i++ ) 
        y = op( x[i], y );
    return y;
}
template <class TYPE, class FUN, typename LAMBDA>
inline TYPE FunctionTable::reduce( LAMBDA &op, const Array<TYPE, FUN> &A, const Array<TYPE, FUN> &B, const TYPE& initialValue )
{
    ARRAY_ASSERT(A.length()==B.length());
    if ( A.length() == 0 )
        return TYPE();
    const TYPE *x = A.data();
    const TYPE *y = B.data();
    TYPE z        = initialValue;  
    for ( size_t i = 0; i < A.length(); i++ )
        z = op( x[i], y[i], z );
    return z;
}


/********************************************************
*  Unary transformation                                 *
********************************************************/
template <class TYPE, class FUN, typename LAMBDA>
inline void FunctionTable::transform( LAMBDA &fun, const Array<TYPE, FUN> &x, Array<TYPE, FUN> &y )
{
    y.resize( x.size() );
    for ( size_t i = 0; i < x.length(); i++ )
        y( i )     = fun( x( i ) );
}
template <class TYPE, class FUN, typename LAMBDA>
inline void FunctionTable::transform( LAMBDA &fun,
                                      const Array<TYPE, FUN> &x,
                                      const Array<TYPE, FUN> &y,
                                      Array<TYPE, FUN> &z )
{
    ARRAY_INSIST( x.sizeMatch( y ), "Sizes of x and y do not match" );
    z.resize( x.size() );
    for ( size_t i = 0; i < x.length(); i++ )
        z( i )     = fun( x( i ), y( i ) );
}


/********************************************************
*  Multiply two arrays                                  *
********************************************************/
template <class TYPE, class FUN>
void FunctionTable::multiply( const Array<TYPE, FUN> &a,
                              const Array<TYPE, FUN> &b,
                              Array<TYPE, FUN> &c )
{
    if ( a.d_ndim <= 2 && b.d_ndim <= 2 ) {
        ARRAY_INSIST( a.size( 1 ) == b.size( 0 ), "Inner dimensions must match" );
        c.resize( a.size( 0 ), b.size( 1 ) );
        c.fill( 0 );
        for ( size_t k = 0; k < b.size( 1 ); k++ ) {
            for ( size_t j = 0; j < a.size( 1 ); j++ ) {
                for ( size_t i = 0; i < a.size( 0 ); i++ ) {
                    c( i, k ) += a( i, j ) * b( j, k );
                }
            }
        }
    } else {
        ARRAY_ERROR( "Not finished yet" );
    }
}

template <class TYPE, class FUN>
bool FunctionTable::equals( const Array<TYPE, FUN> &a, const Array<TYPE, FUN> &b, TYPE tol )
{
    bool pass = true;
    ARRAY_INSIST( a.sizeMatch( b ), "Sizes of x and y do not match" );
    for ( size_t i = 0; i < a.length(); i++ )
        pass  = pass && (std::abs(a(i)-b(i))<tol);
    return pass;
}

/********************************************************
*  Specialized Functions                                *
********************************************************/
template <class TYPE, class FUN, class ALLOC>
void FunctionTable::transformReLU(const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B)
{
    const auto &fun = [](const TYPE &a) { return std::max(a, static_cast<TYPE>( 0 ) );};
    transform(fun,A,B); 
}

template <class TYPE, class FUN, class ALLOC>
void FunctionTable::transformAbs(const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B)
{
    B.resize(A.size());
    const auto &fun = [](const TYPE &a) { return std::abs(a) ;};
    transform(fun,A,B); 
}
template <class TYPE, class FUN, class ALLOC>
void FunctionTable::transformTanh(const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B)
{
    B.resize(A.size());
    const auto &fun = [](const TYPE &a) { return tanh(a);};
    transform(fun,A,B); 
}

template <class TYPE, class FUN, class ALLOC>
void FunctionTable::transformHardTanh(const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B)
{
    B.resize(A.size());
    const auto &fun = [](const TYPE &a) { return std::max(-static_cast<TYPE>(1.0),std::min(static_cast<TYPE>(1.0), a));  };
    transform(fun,A,B); 
}

template <class TYPE, class FUN, class ALLOC>
void FunctionTable::transformSigmoid(const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B)
{
    B.resize(A.size());
    const auto &fun = [](const TYPE &a) { return 1.0/(1.0 + exp( -a));};
    transform(fun,A,B); 
}

template <class TYPE, class FUN, class ALLOC>
void FunctionTable::transformSoftPlus(const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B)
{
    B.resize(A.size());
    const auto &fun = [](const TYPE &a) { return log1p(exp(a));};
    transform(fun,A,B); 
}

template <class TYPE, class FUN, class ALLOC>
TYPE FunctionTable::sum(const Array<TYPE, FUN, ALLOC> &A)
{
    const auto &fun = []( const TYPE &a, const TYPE &b ) { return a + b; };
    return reduce( fun, A, (TYPE) 0 );
}

template <class TYPE>
inline void FunctionTable::gemmWrapper(char TRANSA, char TRANSB, int M, int N, int K, TYPE alpha, const TYPE* A, int LDA, const TYPE* B, int LDB, TYPE beta, TYPE* C, int LDC)
{

    AMP::Lapack<TYPE>::gemm(TRANSA,TRANSB,M,N,K,alpha,A,LDA,B,LDB,beta,C,LDC);

}



} // namespace AMP

#endif