#include "AMP/vectors/petsc/NativePetscVectorData.h"
#include "AMP/vectors/petsc/NativePetscVectorOperations.h"
#include "AMP/vectors/petsc/PetscHelpers.h"

#include "petsc.h"
#include "petsc/private/vecimpl.h"
#include "petscvec.h"

namespace AMP::LinearAlgebra {

NativePetscVectorData::NativePetscVectorData( Vec v, bool deleteable, AMP_MPI comm )
    : GhostDataHelper<PetscScalar>()
{
    // Set the vector
    d_petscVec   = v;
    d_pArray     = nullptr;
    d_pArrayRead = nullptr;
    d_bDeleteMe  = deleteable;
    // Get the correct communicator if it is not set
    MPI_Comm comm2 = comm.getCommunicator(); // Get a MPI_comm object from AMP_MPI to pass to PETSc
    PetscObjectGetComm( reinterpret_cast<PetscObject>( v ), &comm2 );
    if ( comm2 != (MPI_Comm) comm.getCommunicator() )
        comm = AMP_MPI( comm2 );
    // Get the local size
    PetscInt lsize;
    VecGetLocalSize( d_petscVec, &lsize );
    // Create the communication list
    auto params         = std::make_shared<CommunicationListParameters>();
    params->d_comm      = comm;
    params->d_localsize = lsize;
    setCommunicationList( std::make_shared<CommunicationList>( params ) );
    // set the state to be unchanged since setCommunicationList sets
    // it to LOCAL_CHANGED
    setUpdateStatus( UpdateState::UNCHANGED );
    // Cache vector sizes
    d_localStart = d_CommList->getStartGID();
    d_localSize  = lsize;
    d_globalSize = d_CommList->getTotalSize();
}


NativePetscVectorData::~NativePetscVectorData()
{
    resetArray();
    if ( d_bDeleteMe ) {
        PETSC::vecDestroy( &d_petscVec );
    }
}

void NativePetscVectorData::makeConsistent( [[maybe_unused]] ScatterType t )
{
    // resetArray ensures that any grabbed raw data is returned
    resetArray();
    // assemble inserts/adds any values that have changed
    assemble();
    setUpdateStatus( UpdateState::UNCHANGED );
}

void NativePetscVectorData::makeConsistent()
{
    resetArray();
    assemble();
    setUpdateStatus( UpdateState::UNCHANGED );
}

void NativePetscVectorData::putRawData( const void *in, const typeID &id )
{
    resetArray();
    constexpr auto type = getTypeID<PetscScalar>();
    AMP_ASSERT( id == type );
    auto data = reinterpret_cast<const PetscScalar *>( in );
    PetscInt a, b;
    VecGetOwnershipRange( d_petscVec, &a, &b );
    AMP_ASSERT( b - a == static_cast<PetscInt>( getLocalSize() ) );
    std::vector<PetscInt> offs( b - a );
    for ( size_t j = 0; j != offs.size(); j++ )
        offs[j] = a + j;
    VecSetValues( d_petscVec, offs.size(), offs.data(), data, INSERT_VALUES );
}

void NativePetscVectorData::getRawData( void *out, const typeID &id ) const
{
    resetArray(); // return possibly outstanding raw data block
    constexpr auto type = getTypeID<PetscScalar>();
    AMP_ASSERT( id == type );
    auto data = reinterpret_cast<PetscScalar *>( out );
    std::copy( getRawDataBlock<PetscScalar>( 0 ),
               getRawDataBlock<PetscScalar>( 0 ) + getLocalSize(),
               data );
    resetArray(); // return block just requested
}

void NativePetscVectorData::swapData( VectorData &other )
{
    resetArray();
    auto otherData = dynamic_cast<NativePetscVectorData *>( &other );
    otherData->resetArray();
    VecSwap( d_petscVec, otherData->getVec() );
}

std::shared_ptr<VectorData> NativePetscVectorData::cloneData( const std::string & ) const
{
    resetArray();
    Vec new_petscVec;
    VecDuplicate( d_petscVec, &new_petscVec );
    return std::make_shared<NativePetscVectorData>( new_petscVec, true, getComm() );
}

size_t NativePetscVectorData::numberOfDataBlocks() const { return 1; }

size_t NativePetscVectorData::sizeOfDataBlock( size_t i ) const
{
    if ( i != 0 ) {
        return 0;
    }
    return getLocalSize();
}

void NativePetscVectorData::resetArray()
{
    if ( d_pArray ) {
        VecRestoreArray( d_petscVec, &d_pArray );
        d_pArray = nullptr;
    }
    if ( d_pArrayRead ) {
        VecRestoreArrayRead( d_petscVec, &d_pArrayRead );
        d_pArrayRead = nullptr;
    }
}

void NativePetscVectorData::resetArray() const
{
    if ( d_pArray ) {
        VecRestoreArray( d_petscVec, &d_pArray );
        d_pArray = nullptr;
    }
    if ( d_pArrayRead ) {
        VecRestoreArrayRead( d_petscVec, &d_pArrayRead );
        d_pArrayRead = nullptr;
    }
}

void NativePetscVectorData::setValuesByLocalID( size_t N,
                                                const size_t *indices,
                                                const void *vals,
                                                const typeID &id )
{
    resetArray();
    AMP_ASSERT( id == getTypeID<PetscScalar>() );
    auto data = reinterpret_cast<const PetscScalar *>( vals );
    std::vector<PetscInt> idx( N );
    for ( size_t i = 0; i < N; i++ ) {
        idx[i] = indices[i] + d_localStart;
    }
    VecSetValues( d_petscVec, N, idx.data(), data, INSERT_VALUES );
}


void NativePetscVectorData::addValuesByLocalID( size_t N,
                                                const size_t *indices,
                                                const void *vals,
                                                const typeID &id )
{
    resetArray();
    AMP_ASSERT( id == getTypeID<PetscScalar>() );
    auto data = reinterpret_cast<const PetscScalar *>( vals );
    std::vector<PetscInt> idx( N );
    for ( size_t i = 0; i < N; i++ ) {
        idx[i] = indices[i] + d_localStart;
    }
    VecSetValues( d_petscVec, N, idx.data(), data, ::ADD_VALUES );
}

void NativePetscVectorData::getValuesByLocalID( size_t N,
                                                const size_t *indices,
                                                void *vals,
                                                const typeID &id ) const
{
    resetArray();
    AMP_ASSERT( id == getTypeID<PetscScalar>() );
    auto data = reinterpret_cast<PetscScalar *>( vals );
    std::vector<PetscInt> idx( N );
    for ( size_t i = 0; i < N; i++ ) {
        idx[i] = indices[i] + d_localStart;
    }
    VecGetValues( d_petscVec, N, idx.data(), data );
}

void *NativePetscVectorData::getRawDataBlockAsVoid( size_t i )
{
    if ( i > 0 ) {
        return nullptr;
    }
    if ( d_pArray == nullptr ) {
        VecGetArray( d_petscVec, &d_pArray );
    }
    return static_cast<void *>( d_pArray );
}

const void *NativePetscVectorData::getRawDataBlockAsVoid( size_t i ) const
{
    if ( i > 0 ) {
        return nullptr;
    }
    if ( d_pArrayRead == nullptr ) {
        VecGetArrayRead( d_petscVec, &d_pArrayRead );
    }
    return static_cast<const void *>( d_pArrayRead );
}

void NativePetscVectorData::assemble()
{
    VecAssemblyBegin( d_petscVec );
    VecAssemblyEnd( d_petscVec );
}


} // namespace AMP::LinearAlgebra
