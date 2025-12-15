#include "AMP/vectors/petsc/PetscVector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/data/ManagedVectorData.h"

#include "petsc/private/petscimpl.h"


namespace AMP::LinearAlgebra {


/****************************************************************
 * view                                                          *
 ****************************************************************/
std::shared_ptr<const PetscVector> PetscVector::constView( Vector::const_shared_ptr inVector )
{
    return inVector ? view( std::const_pointer_cast<Vector>( inVector ) ) : nullptr;
}
std::shared_ptr<PetscVector> PetscVector::view( Vector::shared_ptr inVector )
{
    // Check if we have an existing view
    if ( std::dynamic_pointer_cast<PetscVector>( inVector ) )
        return std::dynamic_pointer_cast<PetscVector>( inVector );
    if ( inVector->hasView<PetscVector>() )
        return inVector->getView<PetscVector>();
    // Check if we are dealing with a managed vector
    auto managedData = std::dynamic_pointer_cast<ManagedVectorData>( inVector->getVectorData() );
    if ( managedData ) {
        auto retVal = view( managedData->getVectorEngine() );
        retVal->getManagedVec()->setVariable( inVector->getVariable() );
        return retVal;
    }
    // Create the view
    std::shared_ptr<PetscVector> ptr( new PetscVector( inVector ) );
    inVector->registerView( ptr );
    return ptr;
}

PetscVector::PetscVector() = default;
PetscVector::PetscVector( std::shared_ptr<Vector> vec )
    : d_Vec( PETSC::getVec( vec ) ), d_vector( vec )
{
}
PetscVector::~PetscVector() { PETSC::vecDestroy( &d_Vec ); }


} // namespace AMP::LinearAlgebra
