#include "AMP/vectors/trilinos/thyra/ManagedThyraVector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/data/ManagedVectorData.h"
#include "AMP/vectors/operations/ManagedVectorOperations.h"
#include "AMP/vectors/trilinos/thyra/ThyraVectorWrapper.h"


namespace AMP::LinearAlgebra {


static inline auto getVectorEngine( std::shared_ptr<VectorData> data )
{
    auto managed = std::dynamic_pointer_cast<ManagedVectorData>( data );
    AMP_ASSERT( managed );
    return managed->getVectorEngine();
}
static inline auto getVectorEngine( std::shared_ptr<const VectorData> data )
{
    auto managed = std::dynamic_pointer_cast<const ManagedVectorData>( data );
    AMP_ASSERT( managed );
    return managed->getVectorEngine();
}


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
ManagedThyraVector::ManagedThyraVector( Vector::shared_ptr vec ) : Vector()
{
    AMP_ASSERT( !std::dynamic_pointer_cast<ManagedVectorData>( vec->getVectorData() ) );
    d_VectorOps  = std::make_shared<ManagedVectorOperations>();
    d_VectorData = std::make_shared<ManagedVectorData>( vec );
    d_DOFManager = vec->getDOFManager();
    setVariable( vec->getVariable() );
    d_thyraVec = Teuchos::RCP<Thyra::VectorBase<double>>(
        new ThyraVectorWrapper( std::vector<Vector::shared_ptr>( 1, vec ) ) );
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
    return "Managed Thyra Vector" + d_VectorData->VectorDataName();
}


/****************************************************************
 * Clone the vector                                              *
 ****************************************************************/
std::unique_ptr<Vector> ManagedThyraVector::rawClone() const
{
    auto vec    = getVectorEngine( getVectorData() );
    auto vec2   = vec->clone( "ManagedThyraVectorClone" );
    auto engine = std::dynamic_pointer_cast<Vector>( vec2 );
    auto retVal = std::make_unique<ManagedThyraVector>( engine );
    retVal->setVariable( d_Variable );
    return retVal;
}
void ManagedThyraVector::swapVectors( Vector &other )
{
    d_VectorData->swapData( *other.getVectorData() );
}


/****************************************************************
 * Copy the vector                                               *
 ****************************************************************/
void ManagedThyraVector::copyVector( Vector::const_shared_ptr vec )
{
    AMP_ASSERT( vec );
    if ( vec.get() == this )
        return;

    auto engineVec = getVectorEngine( getVectorData() );
    engineVec->copyVector( vec );
}


} // namespace AMP::LinearAlgebra
