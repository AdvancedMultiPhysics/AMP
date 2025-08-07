#ifndef included_AMP_VectorBuider_hpp
#define included_AMP_VectorBuider_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/memory.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/ArrayVectorData.h"

#ifdef AMP_USE_DEVICE
    #include "AMP/utils/device/GPUFunctionTable.h"
    #include "AMP/vectors/operations/device/VectorOperationsDevice.h"
#endif

#include "math.h"

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

/********************************************************
 * Vector builder                                        *
 ********************************************************/
template<typename TYPE, typename OPS, typename DATA>
Vector::shared_ptr createVector( std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
                                 std::shared_ptr<Variable> variable,
                                 bool split )
{
    PROFILE( "createVector" );

    if ( !DOFs )
        return Vector::shared_ptr();
    AMP_ASSERT( variable );
    // Check if we are dealing with a multiDOFManager
    std::shared_ptr<AMP::Discretization::multiDOFManager> multiDOF;
    if ( split )
        multiDOF = std::dynamic_pointer_cast<AMP::Discretization::multiDOFManager>( DOFs );
    // Check if we are dealing with a multiVariable
    auto multiVariable = std::dynamic_pointer_cast<MultiVariable>( variable );
    if ( multiVariable ) {
        // We are dealing with a MultiVariable, first check that there are no duplicate or null
        // variables
        for ( size_t i = 0; i < multiVariable->numVariables(); i++ ) {
            auto var1 = multiVariable->getVariable( i );
            AMP_INSIST( var1,
                        "Error using a MultiVariable in createVector, NULL variables detected" );
            for ( size_t j = 0; j < i; j++ ) {
                auto var2 = multiVariable->getVariable( j );
                AMP_INSIST(
                    ( *var1 ) != ( *var2 ),
                    "Error using a MultiVariable in createVector, duplicate variables detected" );
            }
        }
        // Check that all processors have the same number of variables
        size_t N_var  = multiVariable->numVariables();
        size_t N_var0 = DOFs->getComm().bcast( N_var, 0 );
        AMP_INSIST(
            N_var == N_var0,
            "The multivariable has a different number of varaibles on different processors" );
        // Create the Vector for each variable, then combine
        std::vector<Vector::shared_ptr> vectors;
        for ( auto var : *multiVariable )
            vectors.push_back( createVector( DOFs, var, split ) );
        // Create the multivector
        AMP_MPI comm = DOFs->getComm();
        AMP_ASSERT( !comm.isNull() );
        comm.barrier();
        auto multiVector = MultiVector::create( variable, comm );
        multiVector->addVector( vectors );
        return multiVector;
    } else if ( multiDOF ) {
        // We are dealing with a multiDOFManager and want to split the vector based on the DOF
        // managers
        auto subDOFs = multiDOF->getDOFManagers();
        // Get the vectors for each DOF manager
        std::vector<Vector::shared_ptr> vectors( subDOFs.size() );
        for ( size_t i = 0; i < subDOFs.size(); i++ )
            vectors[i] = createVector( subDOFs[i], variable, split );
        // Create the multivector
        AMP_MPI comm = DOFs->getComm();
        AMP_ASSERT( !comm.isNull() );
        comm.barrier();
        auto multiVector = MultiVector::create( variable, comm );
        multiVector->addVector( vectors );
        return multiVector;
    } else {
        // We are ready to create a single vector
        // Create the communication list
        AMP_MPI comm = DOFs->getComm();
        AMP_ASSERT( !comm.isNull() );
        comm.barrier();
        std::shared_ptr<CommunicationList> comm_list;
        auto remote_DOFs = DOFs->getRemoteDOFs();
        bool ghosts      = comm.anyReduce( !remote_DOFs.empty() );
        if ( !ghosts ) {
            // No need for a communication list
            comm_list = std::make_shared<CommunicationList>( DOFs->numLocalDOF(), DOFs->getComm() );
        } else {
            // Construct the communication list
            auto params           = std::make_shared<CommunicationListParameters>();
            params->d_comm        = comm;
            params->d_localsize   = DOFs->numLocalDOF();
            params->d_remote_DOFs = remote_DOFs;
            comm_list             = std::make_shared<CommunicationList>( params );
        }
        comm.barrier();
        // Create the vector
        auto vector = createSimpleVector<TYPE, OPS, DATA>( variable, DOFs, comm_list );
        return vector;
    }
}

template<typename TYPE>
std::shared_ptr<Vector> createVector( std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
                                      std::shared_ptr<Variable> variable,
                                      bool split,
                                      AMP::Utilities::MemoryType memType )
{
    PROFILE( "createVector" );

    if ( memType <= AMP::Utilities::MemoryType::host ) {
        return createVector<TYPE>( DOFs, variable, split );
    } else if ( memType == AMP::Utilities::MemoryType::managed ) {
#ifdef AMP_USE_DEVICE
        return createVector<TYPE,
                            VectorOperationsDevice<TYPE>,
                            VectorDataDefault<TYPE, AMP::ManagedAllocator<void>>>(
            DOFs, variable, split );
#else
        AMP_ERROR( "Creating Vector in managed memory requires HIP or CUDA support" );
#endif
    } else if ( memType == AMP::Utilities::MemoryType::device ) {
#ifdef AMP_USE_DEVICE
        return createVector<TYPE,
                            VectorOperationsDevice<TYPE>,
                            VectorDataDefault<TYPE, AMP::DeviceAllocator<void>>>(
            DOFs, variable, split );
#else
        AMP_ERROR( "Creating Vector in device memory requires HIP or CUDA support" );
#endif
    } else {
        AMP_ERROR( "Unknown memory space in createVector" );
    }
}
template<typename TYPE>
std::shared_ptr<Vector> createVector( std::shared_ptr<Vector> vector,
                                      AMP::Utilities::MemoryType memType )
{
    if ( !vector )
        return nullptr;
    // Check if we are dealing with a multiVector
    auto multiVector = std::dynamic_pointer_cast<MultiVector>( vector );
    if ( multiVector ) {
        std::vector<std::shared_ptr<Vector>> vecs;
        for ( auto vec : *multiVector )
            vecs.push_back( createVector<TYPE>( vec, memType ) );
        auto multiVector = MultiVector::create( vector->getVariable(), vector->getComm() );
        multiVector->addVector( vecs );
        return multiVector;
    }
    // Create a vector that mimics the original vector
    return createVector<TYPE>( vector->getDOFManager(), vector->getVariable(), false, memType );
}


/****************************************************************
 * SimpleVector                                                  *
 ****************************************************************/
template<typename TYPE, typename OPS, typename DATA>
Vector::shared_ptr createSimpleVector( size_t localSize, const std::string &name )
{
    AMP_MPI comm( AMP_COMM_SELF );
    auto var = std::make_shared<Variable>( name );
    return createSimpleVector<TYPE, OPS, DATA>( localSize, var, comm );
}

template<typename TYPE, typename OPS, typename DATA>
Vector::shared_ptr createSimpleVector( size_t localSize, std::shared_ptr<Variable> var )
{
    AMP_MPI comm( AMP_COMM_SELF );
    return createSimpleVector<TYPE, OPS, DATA>( localSize, var, comm );
}
template<typename TYPE, typename OPS, typename DATA>
Vector::shared_ptr
createSimpleVector( size_t localSize, std::shared_ptr<Variable> var, AMP_MPI comm )
{
    auto DOFs = std::make_shared<AMP::Discretization::DOFManager>( localSize, comm );
    auto ops  = std::make_shared<OPS>();
    auto data =
        std::make_shared<DATA>( DOFs->beginDOF(), DOFs->numLocalDOF(), DOFs->numGlobalDOF() );
    data->setCommunicationList( std::make_shared<CommunicationList>( localSize, comm ) );
    // set the state to be unchanged since
    // setCommunicationList sets it to LOCAL_CHANGED
    data->setUpdateStatus( UpdateState::UNCHANGED );
    return std::make_shared<Vector>( data, ops, var, DOFs );
}
template<typename TYPE, typename OPS, typename DATA>
Vector::shared_ptr createSimpleVector( std::shared_ptr<Variable> var,
                                       std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
                                       std::shared_ptr<CommunicationList> commlist )
{
    auto ops = std::make_shared<OPS>();
    auto data =
        std::make_shared<DATA>( DOFs->beginDOF(), DOFs->numLocalDOF(), DOFs->numGlobalDOF() );
    data->setCommunicationList( commlist );
    // set the state to be unchanged since
    // setCommunicationList sets it to LOCAL_CHANGED
    data->setUpdateStatus( UpdateState::UNCHANGED );
    return std::make_shared<Vector>( data, ops, var, DOFs );
}


/****************************************************************
 * ArrayVector                                                  *
 ****************************************************************/
template<typename T, typename FUN, typename Allocator>
Vector::shared_ptr createArrayVector( const ArraySize &localSize, const std::string &name )
{
    auto var = std::make_shared<Variable>( name );
    return createArrayVector<T, FUN, Allocator>( localSize, { 0, 0, 0 }, AMP_COMM_SELF, var );
}
template<typename T, typename FUN, typename Allocator>
Vector::shared_ptr createArrayVector( const ArraySize &localSize, std::shared_ptr<Variable> var )
{
    return createArrayVector<T, FUN, Allocator>( localSize, { 0, 0, 0 }, AMP_COMM_SELF, var );
}
template<typename T, typename FUN, typename Allocator>
Vector::shared_ptr createArrayVector( const ArraySize &localSize,
                                      const ArraySize &blockIndex,
                                      const AMP_MPI &comm,
                                      std::shared_ptr<Variable> var )
{
    size_t N  = localSize.length();
    auto ops  = std::make_shared<VectorOperationsDefault<T>>();
    auto data = ArrayVectorData<T, FUN, Allocator>::create( localSize, blockIndex, comm );
    auto DOFs = std::make_shared<AMP::Discretization::DOFManager>( N, comm );
    return std::make_shared<Vector>( data, ops, var, DOFs );
}


/****************************************************************
 * Vector view based on ArrayVector                             *
 ****************************************************************/
template<typename T>
Vector::shared_ptr createVectorAdaptor( const std::string &name,
                                        std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
                                        T *data )
{
    auto var              = std::make_shared<Variable>( name );
    auto params           = std::make_shared<CommunicationListParameters>();
    params->d_comm        = DOFs->getComm();
    params->d_localsize   = DOFs->numLocalDOF();
    params->d_remote_DOFs = DOFs->getRemoteDOFs();
    auto commList         = std::make_shared<CommunicationList>( params );

    auto memType = AMP::Utilities::getMemoryType( data );

    std::shared_ptr<VectorData> vecData;
    std::shared_ptr<VectorOperations> vecOps;
    if ( memType <= AMP::Utilities::MemoryType::host ) {
        vecOps  = std::make_shared<VectorOperationsDefault<T>>();
        vecData = ArrayVectorData<T>::create( DOFs->numLocalDOF(), commList, data );
    } else if ( memType == AMP::Utilities::MemoryType::managed ) {
#ifdef AMP_USE_DEVICE
        vecOps  = std::make_shared<VectorOperationsDevice<T>>();
        vecData = ArrayVectorData<T, AMP::GPUFunctionTable, AMP::ManagedAllocator<void>>::create(
            DOFs->numLocalDOF(), commList, data );
#endif
    } else if ( memType == AMP::Utilities::MemoryType::device ) {
#ifdef AMP_USE_DEVICE
        vecOps  = std::make_shared<VectorOperationsDevice<T>>();
        vecData = ArrayVectorData<T, AMP::GPUFunctionTable, AMP::DeviceAllocator<void>>::create(
            DOFs->numLocalDOF(), commList, data );
#endif
    } else {
        AMP_ERROR( "Unknown memory location specified for data" );
    }

    auto vec = std::make_shared<Vector>( vecData, vecOps, var, DOFs );
    vec->makeConsistent( ScatterType::CONSISTENT_SET );
    return vec;
}


} // namespace AMP::LinearAlgebra


/********************************************************
 *  Macros to instantiate vectors                        *
 ********************************************************/
#define INSTANTIATE_SIMPLE_VECTOR( TYPE, OP, DATA )                                          \
    template std::shared_ptr<AMP::LinearAlgebra::Vector>                                     \
    AMP::LinearAlgebra::createVector<TYPE, OP, DATA>(                                        \
        std::shared_ptr<AMP::Discretization::DOFManager>, std::shared_ptr<Variable>, bool ); \
    template std::shared_ptr<AMP::LinearAlgebra::Vector>                                     \
    AMP::LinearAlgebra::createSimpleVector<TYPE, OP, DATA>( size_t, const std::string & );   \
    template std::shared_ptr<AMP::LinearAlgebra::Vector>                                     \
        AMP::LinearAlgebra::createSimpleVector<TYPE, OP, DATA>( size_t,                      \
                                                                std::shared_ptr<Variable> ); \
    template std::shared_ptr<AMP::LinearAlgebra::Vector>                                     \
        AMP::LinearAlgebra::createSimpleVector<TYPE, OP, DATA>(                              \
            size_t, std::shared_ptr<Variable>, AMP_MPI );                                    \
    template std::shared_ptr<AMP::LinearAlgebra::Vector>                                     \
    AMP::LinearAlgebra::createSimpleVector<TYPE, OP, DATA>(                                  \
        std::shared_ptr<Variable>,                                                           \
        std::shared_ptr<AMP::Discretization::DOFManager>,                                    \
        std::shared_ptr<CommunicationList> )
#define INSTANTIATE_ARRAY_VECTOR( TYPE )                                                         \
    template std::shared_ptr<AMP::LinearAlgebra::Vector>                                         \
    AMP::LinearAlgebra::createArrayVector<TYPE>( const ArraySize &, const std::string & );       \
    template std::shared_ptr<AMP::LinearAlgebra::Vector>                                         \
    AMP::LinearAlgebra::createArrayVector<TYPE>( const ArraySize &, std::shared_ptr<Variable> ); \
    template std::shared_ptr<AMP::LinearAlgebra::Vector>                                         \
    AMP::LinearAlgebra::createArrayVector<TYPE>(                                                 \
        const ArraySize &, const ArraySize &, const AMP_MPI &, std::shared_ptr<Variable> );      \
    template std::shared_ptr<AMP::LinearAlgebra::Vector>                                         \
    AMP::LinearAlgebra::createVectorAdaptor<TYPE>(                                               \
        const std::string &, std::shared_ptr<AMP::Discretization::DOFManager>, TYPE * );         \
    template std::shared_ptr<AMP::LinearAlgebra::Vector> AMP::LinearAlgebra::createVector<TYPE>( \
        std::shared_ptr<AMP::Discretization::DOFManager> DOFs,                                   \
        std::shared_ptr<AMP::LinearAlgebra::Variable> variable,                                  \
        bool split,                                                                              \
        AMP::Utilities::MemoryType memType )


#endif
