#include "AMP/mesh/SAMRAI/SAMRAILevelAdaptor.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/mesh/SAMRAI/SAMRAIPatchAdaptor.h"
#include "AMP/utils/UtilityMacros.h"

#include "SAMRAI/geom/CartesianGridGeometry.h"
#include "SAMRAI/hier/PatchLevel.h"


namespace AMP::Mesh {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
SAMRAILevelParameters::SAMRAILevelParameters( const std::shared_ptr<AMP::Database> db )
    : AMP::Mesh::MeshParameters( db )
{
}
SAMRAILevelParameters::SAMRAILevelParameters( const std::shared_ptr<AMP::Database> db,
                                              std::shared_ptr<SAMRAI::hier::PatchLevel> level )
    : AMP::Mesh::MeshParameters( db ), d_level( level )
{
    AMP_ASSERT( d_level );
    AMP::AMP_MPI mpi_comm( d_level->getBoxLevel()->getMPI() );
    setComm( mpi_comm );
}


/****************************************************************
 * SAMRAILevelAdaptor                                            *
 ****************************************************************/
SAMRAILevelAdaptor::SAMRAILevelAdaptor( std::shared_ptr<SAMRAI::hier::PatchLevel> level )
    : d_samrai_level( level )
{
    AMP_ASSERT( level );
    d_comm = AMP::AMP_MPI( level->getBoxLevel()->getMPI() );
    setPatchAdaptors();
}
SAMRAILevelAdaptor::SAMRAILevelAdaptor( const std::shared_ptr<SAMRAILevelParameters> &params )
    : SAMRLevel( params ), d_samrai_level( params->d_level )
{
    setPatchAdaptors();
}
void SAMRAILevelAdaptor::reset() { setPatchAdaptors(); }
bool SAMRAILevelAdaptor::inHierarchy() { return d_samrai_level->inHierarchy(); }
unsigned short SAMRAILevelAdaptor::getDim() { return d_samrai_level->getDim().getValue(); }
int SAMRAILevelAdaptor::getLevelNumber() { return d_samrai_level->getLevelNumber(); }
void SAMRAILevelAdaptor::setPatchAdaptors()
{
    AMP_ASSERT( d_samrai_level );
    d_patches.clear();
    for ( const auto &patch : *d_samrai_level )
        d_patches.emplace_back( std::make_shared<SAMRAIPatchAdaptor>( patch ) );
}

std::vector<int> SAMRAILevelAdaptor::getRatioToLevelZero()
{
    auto ratio = d_samrai_level->getRatioToLevelZero();
    std::vector<int> rv;
    for ( auto i = 0u; i < getDim(); ++i )
        rv.push_back( ratio[i] );
    return rv;
}

std::vector<int> SAMRAILevelAdaptor::getRatioToCoarserLevel()
{
    auto ratio = d_samrai_level->getRatioToCoarserLevel();
    std::vector<int> rv;
    for ( auto i = 0u; i < getDim(); ++i )
        rv.push_back( ratio[i] );
    return rv;
}

std::shared_ptr<SAMRLevel> SAMRAILevelAdaptor::constructCoarsenedLevel()
{
    // for now restrict to levels in the hierarchy and coarsen
    // by the ratio to the next coarser level. In future this could change
    AMP_ASSERT( inHierarchy() );
    const auto &coarseningRatio = d_samrai_level->getRatioToCoarserLevel();
    auto coarsened_level = std::make_shared<SAMRAI::hier::PatchLevel>( d_samrai_level->getDim() );
    AMP_ASSERT( coarsened_level != nullptr );
    coarsened_level->setCoarsenedPatchLevel( d_samrai_level, coarseningRatio );
    return std::make_shared<SAMRAILevelAdaptor>( coarsened_level );
}

std::vector<int> SAMRAILevelAdaptor::getPeriodicShift()
{
    auto grid_geometry = std::dynamic_pointer_cast<SAMRAI::geom::CartesianGridGeometry>(
        d_samrai_level->getGridGeometry() );
    AMP_ASSERT( grid_geometry );
    auto shift = grid_geometry->getPeriodicShift( d_samrai_level->getRatioToLevelZero() );

    std::vector<int> shift2( 3 );
    for ( int i = 0; i < getDim(); ++i )
        shift2[i] = shift( i );

    return shift2;
}

std::shared_ptr<SAMRAI::hier::PatchLevel>
SAMRAILevelAdaptor::getSAMRAILevel( std::shared_ptr<AMP::Mesh::SAMRLevel> samr_level )
{
    auto level_adaptor = std::dynamic_pointer_cast<AMP::Mesh::SAMRAILevelAdaptor>( samr_level );
    AMP_ASSERT( level_adaptor );
    auto level =
        std::dynamic_pointer_cast<SAMRAI::hier::PatchLevel>( level_adaptor->getSAMRAILevel() );
    AMP_ASSERT( level );
    return level;
}

int SAMRAILevelAdaptor::getLocalNumberOfPatches()
{
    return d_samrai_level->getLocalNumberOfPatches();
}
unsigned long SAMRAILevelAdaptor::getGlobalNumberOfCells()
{
    return d_samrai_level->getGlobalNumberOfCells();
}


#if 0
/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void SAMRAILevelAdaptor::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    manager->registerObject( d_samrai_level );
}
void SAMRAILevelAdaptor::writeRestart( int64_t fid ) const
{
    uint64_t levelID = reinterpret_cast<uint64_t>( d_samrai_level.get() );
    AMP::writeHDF5( fid, "levelID", levelID );
}
SAMRAILevelAdaptor::SAMRAILevelAdaptor( int64_t fid, AMP::IO::RestartManager *manager )
{
    uint64_t levelID;
    AMP::readHDF5( fid, "levelID", levelID );
    d_samrai_level = manager->getData<SAMRAI::hier::PatchLevel>( levelID );
    AMP_ASSERT( d_samrai_level );
    setPatchAdaptors();
}
#endif

} // namespace AMP::Mesh
