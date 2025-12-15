#include "AMP/matrices/data/MatrixData.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/data/MatrixDataFactory.h"
#include "AMP/utils/AMPManager.h"

namespace AMP::LinearAlgebra {


/********************************************************
 * Constructors/Destructor                               *
 ********************************************************/
MatrixData::MatrixData() : d_hash( reinterpret_cast<uint64_t>( this ) )
{
    AMPManager::incrementResource( "MatrixData" );
}

MatrixData::MatrixData( std::shared_ptr<MatrixParametersBase> params )
    : d_pParameters( params ), d_hash( reinterpret_cast<uint64_t>( this ) )
{
    AMPManager::incrementResource( "MatrixData" );
}
MatrixData::~MatrixData() { AMPManager::decrementResource( "MatrixData" ); }


/********************************************************
 * Get the number of rows/columns in the matrix          *
 ********************************************************/
size_t MatrixData::numLocalRows() const
{
    auto DOF = getLeftDOFManager();
    return DOF->numLocalDOF();
}
size_t MatrixData::numGlobalRows() const
{
    auto DOF = getLeftDOFManager();
    return DOF->numGlobalDOF();
}
size_t MatrixData::numLocalColumns() const
{
    auto DOF = getRightDOFManager();
    return DOF->numLocalDOF();
}
size_t MatrixData::numGlobalColumns() const
{
    auto DOF = getRightDOFManager();
    return DOF->numGlobalDOF();
}


/********************************************************
 * Get iterators                                         *
 ********************************************************/
size_t MatrixData::beginRow() const
{
    auto DOF = getLeftDOFManager();
    return DOF->beginDOF();
}

size_t MatrixData::endRow() const
{
    auto DOF = getLeftDOFManager();
    return DOF->endDOF();
}

size_t MatrixData::beginCol() const
{
    auto DOF = getRightDOFManager();
    return DOF->beginDOF();
}
size_t MatrixData::endCol() const
{
    auto DOF = getRightDOFManager();
    return DOF->endDOF();
}

/****************************************************************
 * Get an id                                                     *
 ****************************************************************/
uint64_t MatrixData::getID() const { return d_hash; }

/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void MatrixData::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    auto id = manager->registerObject( d_pParameters );
    AMP_ASSERT( id == d_pParameters->getID() );
}

void MatrixData::writeRestart( int64_t fid ) const
{
    uint64_t paramsID = d_pParameters->getID();
    IO::writeHDF5( fid, "paramsID", paramsID );
}

MatrixData::MatrixData( int64_t fid, AMP::IO::RestartManager *manager )
{
    uint64_t paramsID;
    IO::readHDF5( fid, "paramsID", paramsID );
    if ( paramsID )
        d_pParameters = manager->getData<MatrixParametersBase>( paramsID );
}

} // namespace AMP::LinearAlgebra

/********************************************************
 *  Restart operations                                   *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixData>::DataStoreType(
    std::shared_ptr<const AMP::LinearAlgebra::MatrixData> data, RestartManager *manager )
    : d_data( data )
{
    d_hash = data->getID();
    d_data->registerChildObjects( manager );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixData>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid = createGroup( fid, name );
    writeHDF5( gid, "ClassType", d_data->type() );
    d_data->writeRestart( gid );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::LinearAlgebra::MatrixData>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::MatrixData>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    hid_t gid       = openGroup( fid, name );
    auto matrixData = AMP::LinearAlgebra::MatrixDataFactory::create( gid, manager );
    closeGroup( gid );
    return matrixData;
}
