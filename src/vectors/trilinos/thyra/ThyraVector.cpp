#include "AMP/vectors/trilinos/thyra/ThyraVector.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/data/ManagedVectorData.h"
#include "AMP/vectors/operations/ManagedVectorOperations.h"
#include "AMP/vectors/trilinos/thyra/ManagedThyraVector.h"
#include "AMP/vectors/trilinos/thyra/ThyraVectorWrapper.h"


namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
ThyraVector::ThyraVector() { d_thyraVec.reset(); }


/****************************************************************
 * Destructors                                                   *
 ****************************************************************/
ThyraVector::~ThyraVector() { d_thyraVec.reset(); }


/****************************************************************
 * view                                                          *
 ****************************************************************/
std::shared_ptr<const ThyraVector> ThyraVector::constView( Vector::const_shared_ptr inVector )
{
    return view( std::const_pointer_cast<Vector>( inVector ) );
}
std::shared_ptr<ThyraVector> ThyraVector::view( Vector::shared_ptr inVector )
{
    // Check if we have an exisiting view
    auto retVal = std::dynamic_pointer_cast<ThyraVector>( inVector );
    if ( retVal )
        return retVal;
    if ( inVector->hasView<ManagedThyraVector>() )
        return inVector->getView<ManagedThyraVector>();
    // Check if we are dealing with a managed vector
    auto managedData = std::dynamic_pointer_cast<ManagedVectorData>( inVector->getVectorData() );
    if ( managedData ) {
        auto retVal = view( managedData->getVectorEngine() );
        retVal->getManagedVec()->setVariable( inVector->getVariable() );
        return retVal;
    }
    // Create a new view
    if ( std::dynamic_pointer_cast<MultiVector>( inVector ) ) {
        auto newVector = std::make_shared<ManagedThyraVector>( inVector );
        newVector->setVariable( inVector->getVariable() );
        newVector->getVectorData()->setUpdateStatusPtr(
            inVector->getVectorData()->getUpdateStatusPtr() );
        retVal = newVector;
        inVector->registerView( retVal );
    } else {
        retVal = view( MultiVector::view( inVector, inVector->getComm() ) );
        inVector->registerView( retVal );
    }
    return retVal;
}


/****************************************************************
 * Return the thyra vector                                       *
 ****************************************************************/
Teuchos::RCP<Thyra::VectorBase<double>> ThyraVector::getVec() { return d_thyraVec; }
Teuchos::RCP<const Thyra::VectorBase<double>> ThyraVector::getVec() const { return d_thyraVec; }


/****************************************************************
 * Return the views to the AMP vectors                           *
 ****************************************************************/
template<class T>
static void nullDeleter( T * )
{
}
AMP::LinearAlgebra::Vector::shared_ptr ThyraVector::view( Thyra::VectorBase<double> *vec )
{
    AMP::LinearAlgebra::Vector::shared_ptr vec_out;
    if ( !vec ) {
        // Null vec, do nothing
    } else if ( dynamic_cast<AMP::LinearAlgebra::ThyraVectorWrapper *>( vec ) ) {
        auto *tmp = dynamic_cast<AMP::LinearAlgebra::ThyraVectorWrapper *>( vec );
        if ( tmp->numVecs() == 0 ) {
            vec_out.reset();
        } else if ( tmp->numVecs() == 1 ) {
            vec_out = tmp->getVec( 0 );
        } else {
            std::vector<std::shared_ptr<AMP::LinearAlgebra::Variable>> vars;
            for ( size_t i = 0; i < tmp->d_vecs.size(); i++ ) {
                auto name = AMP::Utilities::stringf( "col-%i\n", (int) tmp->d_cols[i] );
                vars.push_back( std::make_shared<AMP::LinearAlgebra::Variable>( name ) );
            }
            auto multiVar =
                std::make_shared<AMP::LinearAlgebra::MultiVariable>( "ThyraMultiVec", vars );
            vec_out = AMP::LinearAlgebra::MultiVector::create(
                multiVar, tmp->d_vecs[0]->getComm(), tmp->d_vecs );
            // Currently our multivectors can't be easily subsetted to create the original vectors
            AMP_ERROR( "Not ready for ThyraMultiVectors yet" );
        }
    } else {
        AMP_ERROR( "Not finished" );
    }
    return vec_out;
}
AMP::LinearAlgebra::Vector::const_shared_ptr
ThyraVector::constView( const Thyra::VectorBase<double> *vec )
{
    AMP::LinearAlgebra::Vector::const_shared_ptr vec_out;
    if ( !vec ) {
        // Null vec, do nothing
    } else if ( dynamic_cast<const AMP::LinearAlgebra::ThyraVectorWrapper *>( vec ) ) {
        const auto *tmp = dynamic_cast<const AMP::LinearAlgebra::ThyraVectorWrapper *>( vec );
        if ( tmp->numVecs() == 0 ) {
            vec_out.reset();
        } else if ( tmp->numVecs() == 1 ) {
            vec_out = tmp->getVec( 0 );
        } else {
            std::vector<std::shared_ptr<AMP::LinearAlgebra::Variable>> vars;
            for ( size_t i = 0; i < tmp->d_vecs.size(); i++ ) {
                auto name = AMP::Utilities::stringf( "col-%i\n", (int) tmp->d_cols[i] );
                vars.push_back( std::make_shared<AMP::LinearAlgebra::Variable>( name ) );
            }
            auto multiVar =
                std::make_shared<AMP::LinearAlgebra::MultiVariable>( "ThyraMultiVec", vars );
            vec_out = AMP::LinearAlgebra::MultiVector::create(
                multiVar, tmp->d_vecs[0]->getComm(), tmp->d_vecs );
            // Currently our multivectors can't be easily subsetted to create the original vectors
            AMP_ERROR( "Not ready for ThyraMultiVectors yet" );
        }
    } else {
        AMP_ERROR( "Not finished" );
    }
    return vec_out;
}


} // namespace AMP::LinearAlgebra
