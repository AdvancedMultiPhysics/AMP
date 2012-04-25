#include "vectors/sundials/ManagedSundialsVector.h"


namespace AMP {
namespace LinearAlgebra {

const Vector::shared_ptr  SundialsVector::constView ( const Vector::shared_ptr inVector )
{
    Vector::shared_ptr  retVal;

    if ( inVector->isA<SundialsVector> () )
        return inVector;
    if ( inVector->hasView<SundialsVector> () )
        return inVector->getView<SundialsVector>();

    if ( inVector->isA<ManagedVector> () ) {
        retVal = Vector::shared_ptr ( new ManagedSundialsVector ( inVector ) );
        inVector->registerView ( retVal );
    } else if ( inVector->isA<VectorEngine> () ) {
        ManagedSundialsVectorParameters *new_params = new ManagedSundialsVectorParameters;
        new_params->d_Engine = boost::dynamic_pointer_cast<VectorEngine> ( inVector );
        new_params->d_CloneEngine = false;
        if ( inVector->getCommunicationList().get()!=NULL )
            new_params->d_CommList = inVector->getCommunicationList();
        else
            new_params->d_CommList = CommunicationList::createEmpty ( inVector->getLocalSize(), inVector->getComm() );
        if ( inVector->getDOFManager().get()!=NULL )
            new_params->d_DOFManager = inVector->getDOFManager();
        else
            new_params->d_DOFManager = AMP::Discretization::DOFManager::shared_ptr( new AMP::Discretization::DOFManager( inVector->getLocalSize(), inVector->getComm() ) );
        ManagedSundialsVector *t = new ManagedSundialsVector ( VectorParameters::shared_ptr ( new_params ) );
        t->setVariable ( inVector->getVariable() );
        t->setUpdateStatusPtr ( inVector->getUpdateStatusPtr () );
        retVal = Vector::shared_ptr ( t );
        inVector->registerView ( retVal );
    } else {
        AMP_ERROR( "Cannot create view!" );
    }

    return retVal;
}


Vector::shared_ptr  SundialsVector::view ( Vector::shared_ptr inVector )
{
    Vector::shared_ptr  retVal;

    if ( inVector->isA<SundialsVector> () )
        return inVector;
    if ( inVector->hasView<SundialsVector> () )
        return inVector->getView<SundialsVector>();

    if ( inVector->isA<ManagedVector> () ) {
        retVal = Vector::shared_ptr ( new ManagedSundialsVector ( inVector ) );
        inVector->registerView ( retVal );
    } else if ( inVector->isA<VectorEngine> () ) {
        ManagedSundialsVectorParameters *new_params = new ManagedSundialsVectorParameters;
        new_params->d_Engine = boost::dynamic_pointer_cast<VectorEngine> ( inVector );
        new_params->d_CloneEngine = false;
        if ( inVector->getCommunicationList().get()!=NULL )
            new_params->d_CommList = inVector->getCommunicationList();
        else
            new_params->d_CommList = CommunicationList::createEmpty ( inVector->getLocalSize(), inVector->getComm() );
        if ( inVector->getDOFManager().get()!=NULL )
            new_params->d_DOFManager = inVector->getDOFManager();
        else
            new_params->d_DOFManager = AMP::Discretization::DOFManager::shared_ptr( new AMP::Discretization::DOFManager( inVector->getLocalSize(), inVector->getComm() ) );
        ManagedSundialsVector *t = new ManagedSundialsVector ( VectorParameters::shared_ptr ( new_params ) );
        t->setVariable ( inVector->getVariable() );
        t->setUpdateStatusPtr ( inVector->getUpdateStatusPtr () );
        retVal = Vector::shared_ptr ( t );
        inVector->registerView ( retVal );
    } else {
        AMP_ERROR( "Cannot create view!" );
    }

    return retVal;
}


}
}

