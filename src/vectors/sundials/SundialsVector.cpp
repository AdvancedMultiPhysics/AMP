#include "AMP/vectors/sundials/SundialsVector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/data/ManagedVectorData.h"
#include "AMP/vectors/sundials/ManagedSundialsVector.h"


namespace AMP::LinearAlgebra {


/****************************************************************
 * view                                                          *
 ****************************************************************/
std::shared_ptr<const SundialsVector> SundialsVector::constView( Vector::const_shared_ptr inVector )
{
    return view( std::const_pointer_cast<Vector>( inVector ) );
}
std::shared_ptr<SundialsVector> SundialsVector::view( Vector::shared_ptr inVector )
{
    // Check if we have an existing view
    if ( std::dynamic_pointer_cast<SundialsVector>( inVector ) )
        return std::dynamic_pointer_cast<SundialsVector>( inVector );
    if ( inVector->hasView<SundialsVector>() )
        return inVector->getView<SundialsVector>();
    // Check if we are dealing with a managed vector
    auto managedData = std::dynamic_pointer_cast<ManagedVectorData>( inVector->getVectorData() );
    if ( managedData ) {
        auto retVal = view( managedData->getVectorEngine() );
        retVal->getManagedVec()->setVariable( inVector->getVariable() );
        return retVal;
    }
    // Check if we are dealing with a multivector
    if ( std::dynamic_pointer_cast<MultiVector>( inVector ) ) {
        auto retVal = std::make_shared<ManagedSundialsVector>( inVector );
        retVal->setVariable( inVector->getVariable() );
        retVal->getVectorData()->setUpdateStatusPtr(
            inVector->getVectorData()->getUpdateStatusPtr() );
        inVector->registerView( retVal );
        return retVal;
    }
    // Create a multivector to wrap the given vector and create a view
    auto retVal = view( MultiVector::view( inVector, inVector->getComm() ) );
    retVal->getManagedVec()->setVariable( inVector->getVariable() );
    inVector->registerView( retVal );
    return retVal;
}

SundialsVector::SundialsVector() = default;


N_Vector &SundialsVector::getNVector() { return d_n_vector; }

const N_Vector &SundialsVector::getNVector() const { return d_n_vector; }


} // namespace AMP::LinearAlgebra


/********************************************************
 * Get the AMP vector from the PETSc Vec or Mat          *
 ********************************************************/
std::shared_ptr<AMP::LinearAlgebra::Vector> getAMP( N_Vector t )
{
    auto ptr = static_cast<AMP::LinearAlgebra::ManagedSundialsVector *>( t->content );
    AMP_ASSERT( ptr );
    std::shared_ptr<AMP::LinearAlgebra::Vector> vec( ptr, []( auto ) {} );
    return vec;
}
