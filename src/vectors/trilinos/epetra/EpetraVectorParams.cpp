#include "AMP/vectors/trilinos/epetra/EpetraVectorData.h"

DISABLE_WARNINGS
#ifdef AMP_USE_MPI
    #include <Epetra_MpiComm.h>
    #include <utility>
#else
    #include <Epetra_SerialComm.h>
#endif
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {


/********************************************************
 * EpetraVectorEngineParameters constructors             *
 ********************************************************/
EpetraVectorEngineParameters::EpetraVectorEngineParameters(
    size_t N_local, const AMP_MPI &comm, std::shared_ptr<CommunicationList> commList )
    : d_CommList( commList ), d_comm( comm )
{
    d_global = d_comm.sumReduce( N_local );
    d_comm.sumScan( &N_local, &d_end, 1 );
    d_begin = d_end - N_local;
}
EpetraVectorEngineParameters::~EpetraVectorEngineParameters() = default;


/********************************************************
 * Function to return (and create) the Epetra_Map        *
 ********************************************************/
Epetra_Map &EpetraVectorEngineParameters::getEpetraMap()
{
    if ( d_emap )
        return *d_emap;
// Create the epetra map
#ifdef AMP_USE_MPI
    Epetra_MpiComm comm = d_comm.getCommunicator();
#else
    Epetra_SerialComm comm;
#endif
    AMP_INSIST( d_global < 0x80000000,
                "Epetra does not support vectors with global size greater than 2^31" );
    size_t local_size = d_end - d_begin;
    d_emap            = std::make_shared<Epetra_Map>( (int) d_global, (int) local_size, 0, comm );
    // Check the map to make sure it is correct
    AMP_ASSERT( local_size == (size_t) d_emap->NumMyPoints() );
    AMP_ASSERT( d_global == (size_t) d_emap->NumGlobalPoints() );
    AMP_ASSERT( d_begin == (size_t) d_emap->MinMyGID() );
    AMP_ASSERT( d_end - 1 == (size_t) d_emap->MaxMyGID() );
    AMP_ASSERT( 0 == (size_t) d_emap->MinAllGID() );
    AMP_ASSERT( d_global - 1 == (size_t) d_emap->MaxAllGID() );
    AMP_ASSERT( 0 == (size_t) d_emap->MinLID() );
    AMP_ASSERT( d_emap->LinearMap() );
    if ( local_size == 0 )
        AMP_ASSERT( 0 == (size_t) d_emap->MaxLID() );
    else
        AMP_ASSERT( local_size - 1 == (size_t) d_emap->MaxLID() );
    return *d_emap;
}


} // namespace AMP::LinearAlgebra
