#include "AMP/vectors/operations/OpenMP/VectorOperationsOpenMP.h"
#include "AMP/vectors/VectorBuilder.hpp"
#include "AMP/vectors/operations/OpenMP/VectorOperationsOpenMP.hpp"


/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
template class AMP::LinearAlgebra::VectorOperationsOpenMP<double>;
template class AMP::LinearAlgebra::VectorOperationsOpenMP<float>;
template class AMP::LinearAlgebra::VectorOperationsOpenMP<int>;
INSTANTIATE_SIMPLE_VECTOR( int,
                           AMP::LinearAlgebra::VectorOperationsOpenMP<int>,
                           AMP::LinearAlgebra::VectorDataDefault<int> );
INSTANTIATE_SIMPLE_VECTOR( float,
                           AMP::LinearAlgebra::VectorOperationsOpenMP<float>,
                           AMP::LinearAlgebra::VectorDataDefault<double> );
INSTANTIATE_SIMPLE_VECTOR( double,
                           AMP::LinearAlgebra::VectorOperationsOpenMP<double>,
                           AMP::LinearAlgebra::VectorDataDefault<double> );
