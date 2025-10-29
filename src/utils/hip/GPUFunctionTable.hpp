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
template<class TYPE, class FUN, class ALLOC>
void GPUFunctionTable::rand( Array<TYPE, FUN, ALLOC> &x )
{
    rand<TYPE>( x.length(), x.data() );
}


// Specialized transform functions - temporary solution
template<class TYPE, class FUN, class ALLOC>
void GPUFunctionTable::transformReLU( const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B )
{
    B.resize( A.size() );
    transformReLUW<TYPE>( A.data(), B.data(), A.length() );
}

template<class TYPE, class FUN, class ALLOC>
void GPUFunctionTable::transformAbs( const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B )
{
    B.resize( A.size() );
    transformAbsW<TYPE>( A.data(), B.data(), A.length() );
}
template<class TYPE, class FUN, class ALLOC>
void GPUFunctionTable::transformTanh( const Array<TYPE, FUN, ALLOC> &A, Array<TYPE, FUN, ALLOC> &B )
{
    B.resize( A.size() );
    transformTanhW<TYPE>( A.data(), B.data(), A.length() );
}

template<class TYPE, class FUN, class ALLOC>
void GPUFunctionTable::transformHardTanh( const Array<TYPE, FUN, ALLOC> &A,
                                          Array<TYPE, FUN, ALLOC> &B )
{
    B.resize( A.size() );
    transformHardTanhW<TYPE>( A.data(), B.data(), A.length() );
}

template<class TYPE, class FUN, class ALLOC>
void GPUFunctionTable::transformSigmoid( const Array<TYPE, FUN, ALLOC> &A,
                                         Array<TYPE, FUN, ALLOC> &B )
{
    B.resize( A.size() );
    transformSigmoidW<TYPE>( A.data(), B.data(), A.length() );
}

template<class TYPE, class FUN, class ALLOC>
void GPUFunctionTable::transformSoftPlus( const Array<TYPE, FUN, ALLOC> &A,
                                          Array<TYPE, FUN, ALLOC> &B )
{
    B.resize( A.size() );
    transformSoftPlusW<TYPE>( A.data(), B.data(), A.length() );
}

// Specialized reductions
template<class TYPE, class FUN, class ALLOC>
TYPE GPUFunctionTable::sum( const Array<TYPE, FUN, ALLOC> &A )
{
    if ( A.length() == 0 ) {
        return TYPE();
    }
    return sumW<TYPE>( A.data(), A.length() );
}

template<class TYPE, class FUN, class ALLOC>
bool GPUFunctionTable::equals( const Array<TYPE, FUN, ALLOC> &A,
                               const Array<TYPE, FUN, ALLOC> &B,
                               TYPE tol )
{
    bool eq = true;
    AMP_INSIST( A.size() == B.size(), "Sizes of A and B do not match" );
    eq = equalsW( A.data(), B.data(), tol, A.length() );

    return eq;
}


/* Functions not yet implemented */

template<class TYPE, class FUN, class ALLOC, typename LAMBDA>
void GPUFunctionTable::transform( LAMBDA &,
                                  const Array<TYPE, FUN, ALLOC> &,
                                  Array<TYPE, FUN, ALLOC> & )
{
    AMP_ERROR( "Not implemented for GPU" );
}

template<class TYPE, class FUN, class ALLOC, typename LAMBDA>
void GPUFunctionTable::transform( LAMBDA &,
                                  const Array<TYPE, FUN, ALLOC> &,
                                  const Array<TYPE, FUN, ALLOC> &,
                                  Array<TYPE, FUN, ALLOC> & )
{
    AMP_ERROR( "Not implemented for GPU" );
}

template<class TYPE, class FUN, class ALLOC, typename LAMBDA>
TYPE GPUFunctionTable::reduce( LAMBDA &, const Array<TYPE, FUN, ALLOC> &, const TYPE & )
{
    AMP_ERROR( "Not implemented for GPU" );
    return 0;
}

template<class TYPE, class FUN, class ALLOC, typename LAMBDA>
TYPE GPUFunctionTable::reduce( LAMBDA &,
                               const Array<TYPE, FUN, ALLOC> &,
                               const Array<TYPE, FUN, ALLOC> &,
                               const TYPE & )
{
    AMP_ERROR( "Not implemented for GPU" );
    return 0;
}

template<class TYPE, class FUN, class ALLOC>
void GPUFunctionTable::multiply( const Array<TYPE, FUN, ALLOC> &,
                                 const Array<TYPE, FUN, ALLOC> &,
                                 Array<TYPE, FUN, ALLOC> & )
{
    AMP_ERROR( "not implemented" );
}
} // namespace AMP
#endif
