#include "AMP/vectors/trilinos/thyra/ManagedThyraVector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/SimpleVector.h"
#include "AMP/vectors/trilinos/thyra/ThyraVectorWrapper.h"


namespace AMP {
namespace LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
ManagedThyraVector::ManagedThyraVector( VectorParameters::shared_ptr params )
    : ManagedVector( params )
{
    Vector::shared_ptr vec = std::dynamic_pointer_cast<Vector>( d_Engine );
    d_thyraVec             = Teuchos::RCP<Thyra::VectorBase<double>>(
        new ThyraVectorWrapper( std::vector<Vector::shared_ptr>( 1, vec ) ) );
}
ManagedThyraVector::ManagedThyraVector( Vector::shared_ptr alias ) : ManagedVector( alias )
{
    d_thyraVec = Teuchos::RCP<Thyra::VectorBase<double>>(
        new ThyraVectorWrapper( std::vector<Vector::shared_ptr>( 1, alias ) ) );
}
ManagedVector *ManagedThyraVector::getNewRawPtr() const
{
    return new ManagedThyraVector( std::dynamic_pointer_cast<VectorParameters>( d_pParameters ) );
}


/****************************************************************
 * Destructor                                                    *
 ****************************************************************/
ManagedThyraVector::~ManagedThyraVector() = default;


/****************************************************************
 * Return the vector type                                        *
 ****************************************************************/
std::string ManagedThyraVector::ManagedThyraVector::type() const
{
    std::string retVal = "Managed Thyra Vector";
    retVal += ManagedVector::type();
    return retVal;
}


/****************************************************************
 * Clone the vector                                              *
 ****************************************************************/
Vector::shared_ptr ManagedThyraVector::cloneVector( const Variable::shared_ptr var ) const
{
    std::shared_ptr<ManagedThyraVectorParameters> p( new ManagedThyraVectorParameters() );
    auto vec = std::dynamic_pointer_cast<Vector>( d_Engine );
    if ( vec ) {
        auto vec2   = vec->cloneVector( "ManagedThyraVectorClone" );
        p->d_Buffer = std::dynamic_pointer_cast<VectorData>( vec2 );
        p->d_Engine = std::dynamic_pointer_cast<VectorOperations>( vec2 );
    } else {
        AMP_ERROR( "ManagedThyraVector::rawClone() should not have reached here!" );
    }
    p->d_CommList             = getCommunicationList();
    p->d_DOFManager           = getDOFManager();
    Vector::shared_ptr retVal = Vector::shared_ptr( new ManagedThyraVector( p ) );
    retVal->setVariable( var );
    return retVal;
}


/****************************************************************
 * Copy the vector                                               *
 ****************************************************************/
void ManagedThyraVector::copyVector( Vector::const_shared_ptr vec )
{
    Vector::shared_ptr engineVec = std::dynamic_pointer_cast<Vector>( d_Engine );
    engineVec->copyVector( vec );
}
} // namespace LinearAlgebra
} // namespace AMP
