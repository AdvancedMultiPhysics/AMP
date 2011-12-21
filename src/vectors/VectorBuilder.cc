#ifdef USE_AMP_DISCRETIZATION

#include "VectorBuilder.h"
#include "vectors/petsc/ManagedPetscVector.h"
#include "vectors/trilinos/EpetraVectorEngine.h"
#include "discretization/MultiDOF_Manager.h"

#include <iostream>


namespace AMP {
namespace LinearAlgebra {


/********************************************************
* Vector builder                                        *
********************************************************/
AMP::LinearAlgebra::Vector::shared_ptr  createVector( 
    AMP::Discretization::DOFManager::shared_ptr DOFs, 
    AMP::LinearAlgebra::Variable::shared_ptr variable,
    bool split )
{
    // Check if we are dealing with a multiDOFManager
    boost::shared_ptr<AMP::Discretization::multiDOFManager> multiDOF;
    if ( split )
        multiDOF = boost::dynamic_pointer_cast<AMP::Discretization::multiDOFManager>(DOFs);
    if (  multiDOF.get() != NULL ) {
        // We are dealing with a multiDOFManager and want to split the vector based on the DOF managers
        std::vector<AMP::Discretization::DOFManager::shared_ptr> subDOFs = multiDOF->getDOFManagers();
        // Get the vectors for each DOF manager
        std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vectors(subDOFs.size());
        for (size_t i=0; i<subDOFs.size(); i++)
            vectors[i] = createVector( subDOFs[i], variable, split );
        // Create the multivector
        AMP_MPI comm = DOFs->getComm();
        AMP_ASSERT( !comm.isNull() );
        comm.barrier();
        boost::shared_ptr<AMP::LinearAlgebra::MultiVector> multiVector = AMP::LinearAlgebra::MultiVector::create( variable, comm );
        multiVector->addVector(vectors);
        return multiVector;
    } else {
        // We are ready to create a single vector
        // Create the communication list
        AMP_MPI comm = DOFs->getComm();
        AMP_ASSERT( !comm.isNull() );
        comm.barrier();
        AMP::LinearAlgebra::CommunicationList::shared_ptr comm_list;
        std::vector<size_t> remote_DOFs = DOFs->getRemoteDOFs();
        bool ghosts = comm.maxReduce<char>(remote_DOFs.size()>0)==1;
        if ( !ghosts ) {
            // No need for a communication list
            comm_list = AMP::LinearAlgebra::CommunicationList::createEmpty( DOFs->numLocalDOF(), DOFs->getComm() );
        } else {
            // Construct the communication list
            AMP::LinearAlgebra::CommunicationListParameters::shared_ptr params( new AMP::LinearAlgebra::CommunicationListParameters );
            params->d_comm = comm;
            params->d_localsize = DOFs->numLocalDOF();
            params->d_remote_DOFs = remote_DOFs;
            comm_list = AMP::LinearAlgebra::CommunicationList::shared_ptr( new AMP::LinearAlgebra::CommunicationList(params) );
        }
        comm.barrier();
        // Create the vector parameters
        boost::shared_ptr<AMP::LinearAlgebra::ManagedPetscVectorParameters> mvparams(
            new AMP::LinearAlgebra::ManagedPetscVectorParameters() );
        boost::shared_ptr<AMP::LinearAlgebra::EpetraVectorEngineParameters> eveparams(
            new AMP::LinearAlgebra::EpetraVectorEngineParameters( DOFs->numLocalDOF(), DOFs->numGlobalDOF(), DOFs->getComm() ) );
        comm.barrier();
        AMP::LinearAlgebra::VectorEngine::BufferPtr t_buffer ( new AMP::LinearAlgebra::VectorEngine::Buffer( DOFs->numLocalDOF() ) );
        AMP::LinearAlgebra::VectorEngine::shared_ptr epetra_engine( new AMP::LinearAlgebra::EpetraVectorEngine( eveparams, t_buffer ) );
        mvparams->d_Engine = epetra_engine;
        mvparams->d_Buffer = t_buffer;
        mvparams->d_CommList = comm_list;
        mvparams->d_DOFManager = DOFs;
        // Create the vector
        comm.barrier();
        AMP::LinearAlgebra::Vector::shared_ptr vector = AMP::LinearAlgebra::Vector::shared_ptr( new AMP::LinearAlgebra::ManagedPetscVector(mvparams) );
        vector->setVariable(variable);
        comm.barrier();
        return vector;
    } 
    return AMP::LinearAlgebra::Vector::shared_ptr();
}


}
}

#endif

