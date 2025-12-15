#include "AMP/matrices/operations/MatrixOperations.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/matrices/operations/MatrixOperationsFactory.h"


namespace AMP::LinearAlgebra {

/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
// this should change to be unique across processes
MatrixOperations::MatrixOperations() : d_hash( reinterpret_cast<uint64_t>( this ) ) {}

/****************************************************************
 * Get an id                                                     *
 ****************************************************************/
uint64_t MatrixOperations::getID() const { return d_hash; }

void MatrixOperations::copyCast( const MatrixData &, MatrixData & )
{
    AMP_ERROR( "NOT IMPLEMENTED" );
}

/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void MatrixOperations::registerChildObjects( AMP::IO::RestartManager * ) const {}
void MatrixOperations::writeRestart( int64_t ) const {}


} // namespace AMP::LinearAlgebra

/********************************************************
 *  Restart operations                                   *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixOperations>::DataStoreType(
    std::shared_ptr<const AMP::LinearAlgebra::MatrixOperations> data, RestartManager *manager )
    : d_data( data )
{
    d_hash = data->getID();
    d_data->registerChildObjects( manager );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixOperations>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid = createGroup( fid, name );
    writeHDF5( gid, "ClassType", d_data->type() );
    d_data->writeRestart( gid );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::LinearAlgebra::MatrixOperations>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixOperations>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    hid_t gid = openGroup( fid, name );
    auto ops  = AMP::LinearAlgebra::MatrixOperationsFactory::create( gid, manager );
    closeGroup( gid );
    return ops;
}
