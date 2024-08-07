#include "AMP/vectors/trilinos/epetra/EpetraVector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/ManagedVectorData.h"
#include "AMP/vectors/operations/ManagedVectorOperations.h"
#include "AMP/vectors/trilinos/epetra/EpetraHelpers.h"
#include "AMP/vectors/trilinos/epetra/EpetraVectorData.h"


namespace AMP::LinearAlgebra {


/********************************************************
 * Constructors / De-constructors                        *
 ********************************************************/
EpetraVector::~EpetraVector() = default;


/********************************************************
 * View                                                  *
 ********************************************************/
std::shared_ptr<EpetraVector> EpetraVector::view( Vector::shared_ptr inVector )
{
    AMP_INSIST( inVector->numberOfDataBlocks() == 1,
                "Epetra does not support more than 1 data block" );
    // Check if we have an existing view
    if ( std::dynamic_pointer_cast<EpetraVector>( inVector ) )
        return std::dynamic_pointer_cast<EpetraVector>( inVector );
    if ( std::dynamic_pointer_cast<MultiVector>( inVector ) ) {
        auto multivec = std::dynamic_pointer_cast<MultiVector>( inVector );
        if ( multivec->getNumberOfSubvectors() == 1 ) {
            return view( multivec->getVector( 0 ) );
        } else {
            AMP_ERROR( "View of multi-block MultiVector is not supported yet" );
        }
    }
    // Check if we are dealing with a managed vector
    auto managedData = std::dynamic_pointer_cast<ManagedVectorData>( inVector->getVectorData() );
    if ( managedData ) {
        auto root = managedData->getVectorEngine();
        return view( root );
    }
    // Create the view
    std::shared_ptr<EpetraVector> ptr( new EpetraVector( inVector ) );
    inVector->registerView( ptr );
    return ptr;
}
std::shared_ptr<const EpetraVector> EpetraVector::constView( Vector::const_shared_ptr inVector )
{
    return view( std::const_pointer_cast<Vector>( inVector ) );
}


EpetraVector::EpetraVector( std::shared_ptr<Vector> vec )
    : d_epetra( getEpetra( vec ) ), d_AMP( vec )
{
}


} // namespace AMP::LinearAlgebra
