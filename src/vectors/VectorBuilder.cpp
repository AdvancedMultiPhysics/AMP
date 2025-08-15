#include "AMP/vectors/VectorBuilder.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/typeid.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.hpp"

namespace AMP::LinearAlgebra {

std::shared_ptr<Vector> createVector( std::shared_ptr<const Vector> vector,
                                      AMP::Utilities::MemoryType memType )
{
    if ( !vector )
        return nullptr;
    // Check if we are dealing with a multiVector
    auto multiVector = std::dynamic_pointer_cast<const MultiVector>( vector );
    if ( multiVector ) {
        std::vector<std::shared_ptr<Vector>> vecs;
        for ( auto vec : *multiVector )
            vecs.push_back( createVector( vec, memType ) );
        auto multiVector = MultiVector::create( vector->getVariable(), vector->getComm() );
        multiVector->addVector( vecs );
        return multiVector;
    }
    // Create a vector that mimics the original vector
    auto type = vector->getVectorData()->getType( 0 );
    if ( type == getTypeID<double>() ) {
        return createVector<double>(
            vector->getDOFManager(), vector->getVariable(), false, memType );
    } else if ( type == AMP::getTypeID<float>() ) {
        return createVector<float>(
            vector->getDOFManager(), vector->getVariable(), false, memType );
    } else {
        AMP_ERROR( "Currently only float and double supported" );
    }
    return nullptr;
}
} // namespace AMP::LinearAlgebra

/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
INSTANTIATE_ARRAY_VECTOR( float );
INSTANTIATE_ARRAY_VECTOR( double );

INSTANTIATE_CREATE_VECTOR( float );
INSTANTIATE_CREATE_VECTOR( double );

INSTANTIATE_SIMPLE_VECTOR( float,
                           AMP::LinearAlgebra::VectorOperationsDefault<float>,
                           AMP::LinearAlgebra::VectorDataDefault<double> );
INSTANTIATE_SIMPLE_VECTOR( double,
                           AMP::LinearAlgebra::VectorOperationsDefault<double>,
                           AMP::LinearAlgebra::VectorDataDefault<double> );

#ifdef AMP_USE_DEVICE
using float_op           = AMP::LinearAlgebra::VectorOperationsDevice<float>;
using double_op          = AMP::LinearAlgebra::VectorOperationsDevice<double>;
using float_managed_data = AMP::LinearAlgebra::VectorDataDevice<float, AMP::ManagedAllocator<void>>;
using double_managed_data =
    AMP::LinearAlgebra::VectorDataDevice<double, AMP::ManagedAllocator<void>>;
INSTANTIATE_SIMPLE_VECTOR( float, float_op, float_managed_data );
INSTANTIATE_SIMPLE_VECTOR( double, double_op, double_managed_data );
#endif
