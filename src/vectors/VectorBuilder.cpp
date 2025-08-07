#include "AMP/vectors/VectorBuilder.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/utils/memory.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.hpp"


/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
INSTANTIATE_ARRAY_VECTOR( float );
INSTANTIATE_ARRAY_VECTOR( double );
INSTANTIATE_SIMPLE_VECTOR( float,
                           AMP::LinearAlgebra::VectorOperationsDefault<float>,
                           AMP::LinearAlgebra::VectorDataDefault<double> );
INSTANTIATE_SIMPLE_VECTOR( double,
                           AMP::LinearAlgebra::VectorOperationsDefault<double>,
                           AMP::LinearAlgebra::VectorDataDefault<double> );

#ifdef AMP_USE_DEVICE
using float_op    = AMP::LinearAlgebra::VectorOperationsDevice<float>;
using double_op   = AMP::LinearAlgebra::VectorOperationsDevice<double>;
using float_data  = AMP::LinearAlgebra::VectorDataDefault<float, AMP::ManagedAllocator<void>>;
using double_data = AMP::LinearAlgebra::VectorDataDefault<double, AMP::ManagedAllocator<void>>;
INSTANTIATE_SIMPLE_VECTOR( float, float_op, float_data );
INSTANTIATE_SIMPLE_VECTOR( double, double_op, double_data );
#endif
