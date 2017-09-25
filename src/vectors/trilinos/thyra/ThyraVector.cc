#include "vectors/trilinos/thyra/ThyraVector.h"
#include "vectors/MultiVariable.h"
#include "vectors/MultiVector.h"
#include "vectors/SimpleVector.h"
#include "vectors/trilinos/thyra/ManagedThyraVector.h"
#include "vectors/trilinos/thyra/ThyraVectorWrapper.h"


namespace AMP {
namespace LinearAlgebra {


/************************************************************************
* Constructors                                                          *
************************************************************************/
ThyraVector::ThyraVector() { d_thyraVec.reset(); }


/************************************************************************
* Destructors                                                           *
************************************************************************/
ThyraVector::~ThyraVector() { d_thyraVec.reset(); }


/****************************************************************
* constView                                                     *
****************************************************************/
Vector::const_shared_ptr ThyraVector::constView( Vector::const_shared_ptr inVector )
{
    // Check if we have an exisiting view
    if ( AMP::dynamic_pointer_cast<const ThyraVector>( inVector ) != nullptr )
        return inVector;
    if ( inVector->hasView<ManagedThyraVector>() )
        return inVector->getView<ManagedThyraVector>();
    // Create a new view
    Vector::shared_ptr retVal;
    if ( dynamic_pointer_cast<const ManagedVector>( inVector ) ) {
        Vector::shared_ptr inVector2 = AMP::const_pointer_cast<Vector>( inVector );
        retVal                       = Vector::shared_ptr( new ManagedThyraVector( inVector2 ) );
        retVal->setVariable( inVector->getVariable() );
        inVector->registerView( retVal );
    } else if ( dynamic_pointer_cast<const VectorEngine>( inVector ) ) {
        Vector::shared_ptr inVector2            = AMP::const_pointer_cast<Vector>( inVector );
        ManagedThyraVectorParameters *newParams = new ManagedThyraVectorParameters;
        newParams->d_Engine      = AMP::dynamic_pointer_cast<VectorEngine>( inVector2 );
        newParams->d_CloneEngine = false;
        AMP_INSIST( inVector->getCommunicationList().get() != NULL,
                    "All vectors must have a communication list" );
        newParams->d_CommList = inVector->getCommunicationList();
        AMP_INSIST( inVector->getDOFManager().get() != NULL,
                    "All vectors must have a DOFManager list" );
        newParams->d_DOFManager = inVector->getDOFManager();
        ManagedThyraVector *t = new ManagedThyraVector( VectorParameters::shared_ptr( newParams ) );
        retVal                = Vector::shared_ptr( t );
        inVector->registerView( retVal );
    } else {
        Vector::shared_ptr inVector2 = AMP::const_pointer_cast<Vector>( inVector );
        retVal                       = view( MultiVector::view( inVector2, inVector->getComm() ) );
        inVector->registerView( retVal );
    }
    return retVal;
}


/****************************************************************
* View                                                          *
****************************************************************/
Vector::shared_ptr ThyraVector::view( Vector::shared_ptr inVector )
{
    // Check if we have an exisiting view
    if ( AMP::dynamic_pointer_cast<ThyraVector>( inVector ) != NULL )
        return inVector;
    if ( inVector->hasView<ManagedThyraVector>() )
        return inVector->getView<ManagedThyraVector>();
    // Create a new view
    Vector::shared_ptr retVal;
    if ( dynamic_pointer_cast<ManagedVector>( inVector ) ) {
        retVal = Vector::shared_ptr( new ManagedThyraVector( inVector ) );
        inVector->registerView( retVal );
    } else if ( dynamic_pointer_cast<VectorEngine>( inVector ) ) {
        ManagedThyraVectorParameters *newParams = new ManagedThyraVectorParameters;
        newParams->d_Engine      = AMP::dynamic_pointer_cast<VectorEngine>( inVector );
        newParams->d_CloneEngine = false;
        AMP_INSIST( inVector->getCommunicationList().get() != NULL,
                    "All vectors must have a communication list" );
        newParams->d_CommList = inVector->getCommunicationList();
        AMP_INSIST( inVector->getDOFManager().get() != NULL,
                    "All vectors must have a DOFManager list" );
        newParams->d_DOFManager = inVector->getDOFManager();
        ManagedThyraVector *newVector =
            new ManagedThyraVector( VectorParameters::shared_ptr( newParams ) );
        newVector->setVariable( inVector->getVariable() );
        newVector->setUpdateStatusPtr( inVector->getUpdateStatusPtr() );
        retVal = Vector::shared_ptr( newVector );
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
template <class T>
static void nullDeleter( T * ){};
AMP::LinearAlgebra::Vector::shared_ptr ThyraVector::view( Thyra::VectorBase<double> *vec )
{
    AMP::LinearAlgebra::Vector::shared_ptr vec_out;
    if ( vec == NULL ) {
        // Null vec, do nothing
    } else if ( dynamic_cast<AMP::LinearAlgebra::ThyraVectorWrapper *>( vec ) ) {
        AMP::LinearAlgebra::ThyraVectorWrapper *tmp =
            dynamic_cast<AMP::LinearAlgebra::ThyraVectorWrapper *>( vec );
        if ( tmp->numVecs() == 0 ) {
            vec_out.reset();
        } else if ( tmp->numVecs() == 1 ) {
            vec_out = tmp->getVec( 0 );
        } else {
            std::vector<AMP::LinearAlgebra::Variable::shared_ptr> vars;
            for ( size_t i = 0; i < tmp->d_vecs.size(); i++ ) {
                char name[100];
                sprintf( name, "col-%i\n", (int) tmp->d_cols[i] );
                vars.push_back( AMP::LinearAlgebra::Variable::shared_ptr(
                    new AMP::LinearAlgebra::Variable( name ) ) );
            }
            AMP::LinearAlgebra::Variable::shared_ptr multiVar(
                new AMP::LinearAlgebra::MultiVariable( "ThyraMultiVec", vars ) );
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
    if ( vec == NULL ) {
        // Null vec, do nothing
    } else if ( dynamic_cast<const AMP::LinearAlgebra::ThyraVectorWrapper *>( vec ) ) {
        const AMP::LinearAlgebra::ThyraVectorWrapper *tmp =
            dynamic_cast<const AMP::LinearAlgebra::ThyraVectorWrapper *>( vec );
        if ( tmp->numVecs() == 0 ) {
            vec_out.reset();
        } else if ( tmp->numVecs() == 1 ) {
            vec_out = tmp->getVec( 0 );
        } else {
            std::vector<AMP::LinearAlgebra::Variable::shared_ptr> vars;
            for ( size_t i = 0; i < tmp->d_vecs.size(); i++ ) {
                char name[100];
                sprintf( name, "col-%i\n", (int) tmp->d_cols[i] );
                vars.push_back( AMP::LinearAlgebra::Variable::shared_ptr(
                    new AMP::LinearAlgebra::Variable( name ) ) );
            }
            AMP::LinearAlgebra::Variable::shared_ptr multiVar(
                new AMP::LinearAlgebra::MultiVariable( "ThyraMultiVec", vars ) );
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


} // LinearAlgebra namespace
} // AMP namespace
