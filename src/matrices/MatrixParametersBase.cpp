#include "AMP/matrices/MatrixParametersBase.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/matrices/MatrixParametersFactory.h"
#include "AMP/vectors/Variable.h"

namespace AMP::LinearAlgebra {

void MatrixParametersBase::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    if ( d_VariableLeft ) {
        auto id = manager->registerObject( d_VariableLeft );
        AMP_ASSERT( id == d_VariableLeft->getID() );
    }
    if ( d_VariableRight ) {
        auto id = manager->registerObject( d_VariableRight );
        AMP_ASSERT( id == d_VariableRight->getID() );
    }
}

void MatrixParametersBase::writeRestart( int64_t fid ) const
{
    AMP::IO::writeHDF5( fid, "commHash", d_comm.hash() );

    uint64_t leftVarID = d_VariableLeft ? d_VariableLeft->getID() : 0;
    AMP::IO::writeHDF5( fid, "leftVarID", leftVarID );

    uint64_t rightVarID = d_VariableRight ? d_VariableRight->getID() : 0;
    AMP::IO::writeHDF5( fid, "rightVarID", rightVarID );

    AMP::IO::writeHDF5( fid, "backend", static_cast<signed char>( d_backend ) );
}

MatrixParametersBase::MatrixParametersBase( int64_t fid, AMP::IO::RestartManager *manager )
{
    uint64_t commHash, varID;
    AMP::IO::readHDF5( fid, "commHash", commHash );
    auto comm = manager->getComm( commHash );

    std::shared_ptr<AMP::LinearAlgebra::Variable> varLeft;
    AMP::IO::readHDF5( fid, "leftVarID", varID );
    if ( varID ) {
        varLeft = manager->getData<AMP::LinearAlgebra::Variable>( varID );
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> varRight;
    AMP::IO::readHDF5( fid, "rightVarID", varID );
    if ( varID ) {
        varRight = manager->getData<AMP::LinearAlgebra::Variable>( varID );
    }

    signed char backend;
    AMP::IO::readHDF5( fid, "backend", backend );
    auto backend_e = static_cast<AMP::Utilities::Backend>( backend );
    // This is not quite right as yet!!
    auto params = std::make_shared<AMP::LinearAlgebra::MatrixParametersBase>(
        comm, varLeft, varRight, backend_e );
}

} // namespace AMP::LinearAlgebra

/********************************************************
 *  Restart operations                                   *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixParametersBase>::DataStoreType(
    std::shared_ptr<const AMP::LinearAlgebra::MatrixParametersBase> data, RestartManager *manager )
    : d_data( data )
{
    d_hash = data->getID();
    d_data->registerChildObjects( manager );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixParametersBase>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid = createGroup( fid, name );
    AMP::IO::writeHDF5( gid, "type", d_data->type() );
    d_data->writeRestart( gid );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::LinearAlgebra::MatrixParametersBase>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixParametersBase>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    hid_t gid   = openGroup( fid, name );
    auto params = AMP::LinearAlgebra::MatrixParametersFactory::create( gid, manager );
    closeGroup( gid );
    return params;
}
