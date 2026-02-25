#ifndef included_AMP_GPUFunctionTable_HPP_
#define included_AMP_GPUFunctionTable_HPP_

#include "hipblas/hipblas.h"
#include "hiprand/hiprand.h"


#include "AMP/utils/Array.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/hip/GPUFunctionTable.h"


namespace AMP {

// Kernel Wrappers
template<class TYPE>
void transformReLUW( const TYPE *d_a, TYPE *d_b, size_t n );

template<class TYPE>
void transformAbsW( const TYPE *d_a, TYPE *d_b, size_t n );

template<class TYPE>
void transformTanhW( const TYPE *d_a, TYPE *d_b, size_t n );

template<class TYPE>
void transformHardTanhW( const TYPE *d_a, TYPE *d_b, size_t n );

template<class TYPE>
void transformSigmoidW( const TYPE *d_a, TYPE *d_b, size_t n );

template<class TYPE>
void transformSoftPlusW( const TYPE *d_a, TYPE *d_b, size_t n );

template<class TYPE>
TYPE sumW( const TYPE *d_a, size_t n );

template<class TYPE>
bool equalsW( const TYPE *d_a, const TYPE *d_b, TYPE tol, size_t n );

// Rand functions
template<class TYPE>
void GPUFunctionTable<TYPE>::rand( size_t N, TYPE *x )
{
    if constexpr ( std::is_same_v<TYPE, int> ) {
        hiprandGenerator_t gen;
        hiprandCreateGenerator( &gen, HIPRAND_RNG_PSEUDO_DEFAULT );
        hiprandSetPseudoRandomGeneratorSeed( gen, time( NULL ) );
        hiprandGenerate( gen, (unsigned int *) x, N );
        hiprandDestroyGenerator( gen );
    } else if constexpr ( std::is_same_v<TYPE, float> ) {
        hiprandGenerator_t gen;
        hiprandCreateGenerator( &gen, HIPRAND_RNG_PSEUDO_DEFAULT );
        hiprandSetPseudoRandomGeneratorSeed( gen, time( NULL ) );
        hiprandGenerateUniform( gen, x, N );
        hiprandDestroyGenerator( gen );
    } else if constexpr ( std::is_same_v<TYPE, double> ) {
        hiprandGenerator_t gen;
        hiprandCreateGenerator( &gen, HIPRAND_RNG_PSEUDO_DEFAULT );
        hiprandSetPseudoRandomGeneratorSeed( gen, time( NULL ) );
        hiprandGenerateUniformDouble( gen, x, N );
        hiprandDestroyGenerator( gen );
    } else {
        AMP_ERROR( "Not finished" );
    }
}


// Specialized transform functions - temporary solution
template<class TYPE>
void GPUFunctionTable<TYPE>::transformReLU( size_t N, const TYPE *A, TYPE *B )
{
    transformReLUW<TYPE>( A, B, N );
}

template<class TYPE>
void GPUFunctionTable<TYPE>::transformAbs( size_t N, const TYPE *A, TYPE *B )
{
    transformAbsW<TYPE>( A, B, N );
}
template<class TYPE>
void GPUFunctionTable<TYPE>::transformTanh( size_t N, const TYPE *A, TYPE *B )
{
    transformTanhW<TYPE>( A, B, N );
}

template<class TYPE>
void GPUFunctionTable<TYPE>::transformHardTanh( size_t N, const TYPE *A, TYPE *B )
{
    transformHardTanhW<TYPE>( A, B, N );
}

template<class TYPE>
void GPUFunctionTable<TYPE>::transformSigmoid( size_t N, const TYPE *A, TYPE *B )
{
    transformSigmoidW<TYPE>( A, B, N );
}

template<class TYPE>
void GPUFunctionTable<TYPE>::transformSoftPlus( size_t N, const TYPE *A, TYPE *B )
{
    transformSoftPlusW<TYPE>( A, B, N );
}

// Specialized reductions
template<class TYPE>
TYPE GPUFunctionTable<TYPE>::sum( size_t N, const TYPE *A )
{
    if ( N == 0 )
        return TYPE( 0 );
    return sumW<TYPE>( A, N );
}
template<class TYPE>
TYPE GPUFunctionTable<TYPE>::min( size_t N, const TYPE *A )
{
    AMP_ERROR( "Not finished" );
}
template<class TYPE>
TYPE GPUFunctionTable<TYPE>::max( size_t N, const TYPE *A )
{
    AMP_ERROR( "Not finished" );
}

template<class TYPE>
bool GPUFunctionTable<TYPE>::equals( size_t N, const TYPE *A, const TYPE *B, TYPE tol )
{
    return equalsW( A, B, tol, N );
}


/* Functions not yet implemented */
template<class TYPE>
void GPUFunctionTable<TYPE>::multiply(
    const ArraySize &, const TYPE *, const ArraySize &, const TYPE *, const ArraySize &, TYPE * )
{
    AMP_ERROR( "not implemented" );
}
template<class TYPE>
void GPUFunctionTable<TYPE>::scale( size_t N, TYPE x, TYPE *y )
{
    AMP_ERROR( "not implemented" );
}
template<class TYPE>
void GPUFunctionTable<TYPE>::px( size_t N, TYPE x, TYPE *y )
{
    AMP_ERROR( "not implemented" );
}
template<class TYPE>
void GPUFunctionTable<TYPE>::px( size_t N, const TYPE *x, TYPE *y )
{
    AMP_ERROR( "not implemented" );
}
template<class TYPE>
void GPUFunctionTable<TYPE>::mx( size_t N, TYPE x, TYPE *y )
{
    AMP_ERROR( "not implemented" );
}
template<class TYPE>
void GPUFunctionTable<TYPE>::mx( size_t N, const TYPE *x, TYPE *y )
{
    AMP_ERROR( "not implemented" );
}
template<class TYPE>
void GPUFunctionTable<TYPE>::axpy( TYPE, size_t, const TYPE *, TYPE * )
{
    AMP_ERROR( "not implemented" );
}
template<class TYPE>
void GPUFunctionTable<TYPE>::axpby( TYPE, size_t, const TYPE *, TYPE, TYPE * )
{
    AMP_ERROR( "not implemented" );
}


} // namespace AMP
#endif
