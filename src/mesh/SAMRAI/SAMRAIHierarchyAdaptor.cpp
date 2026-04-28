#include "AMP/mesh/SAMRAI/SAMRAIHierarchyAdaptor.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/mesh/SAMRAI/SAMRAILevelAdaptor.h"
#include "AMP/utils/UtilityMacros.h"

#include "SAMRAI/geom/CartesianGridGeometry.h"


namespace AMP::Mesh {


SAMRAIHierarchyAdaptor::SAMRAIHierarchyAdaptor(
    std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy )
    : d_samrai_hierarchy( hierarchy )
{
    AMP_ASSERT( hierarchy );
    d_comm = AMP::AMP_MPI( hierarchy->getMPI() );
    setLevelAdaptors();
}

SAMRAIHierarchyAdaptor::SAMRAIHierarchyAdaptor( std::shared_ptr<AMP::Mesh::MeshParameters> )
{
    AMP_ERROR( "Not finished" );
}

SAMRAIHierarchyAdaptor::SAMRAIHierarchyAdaptor(
    const std::shared_ptr<SAMRAIHierarchyParameters> &params )
    : SAMRHierarchy( params ), d_samrai_hierarchy( params->d_hierarchy )
{
    setLevelAdaptors();
}

void SAMRAIHierarchyAdaptor::reset( void ) { setLevelAdaptors(); }

void SAMRAIHierarchyAdaptor::setLevelAdaptors( void )
{
    AMP_ASSERT( d_samrai_hierarchy );
    d_levels.clear();
    const int nLevels = d_samrai_hierarchy->getNumberOfLevels();
    d_levels.resize( nLevels );
    for ( int ln = 0; ln < nLevels; ++ln ) {
        d_levels[ln] =
            std::make_shared<SAMRAILevelAdaptor>( d_samrai_hierarchy->getPatchLevel( ln ) );
        AMP_ASSERT( d_levels[ln] );
        d_levels[ln]->reset();
    }
}

std::vector<int> SAMRAIHierarchyAdaptor::getPeriodicShift( void )
{
    auto grid_geometry = std::dynamic_pointer_cast<SAMRAI::geom::CartesianGridGeometry>(
        d_samrai_hierarchy->getGridGeometry() );
    SAMRAI::hier::IntVector shiftVec = grid_geometry->getPeriodicShift(
        SAMRAI::hier::IntVector::getOne( d_samrai_hierarchy->getDim() ) );

    auto dim = getDim();
    std::vector<int> shifts( dim, 0 );
    for ( auto i = 0u; i < dim; ++i )
        shifts[i] = shiftVec[i];
    return shifts;
}

std::shared_ptr<SAMRAI::hier::PatchHierarchy> SAMRAIHierarchyAdaptor::getSAMRAIHierarchy(
    std::shared_ptr<AMP::Mesh::SAMRHierarchy> samr_hierarchy )

{
    auto hierarchy_adaptor =
        std::dynamic_pointer_cast<AMP::Mesh::SAMRAIHierarchyAdaptor>( samr_hierarchy );
    AMP_ASSERT( hierarchy_adaptor );
    auto hierarchy = std::dynamic_pointer_cast<SAMRAI::hier::PatchHierarchy>(
        hierarchy_adaptor->getSAMRAIHierarchy() );
    AMP_ASSERT( hierarchy );
    return hierarchy;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void SAMRAIHierarchyAdaptor::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    manager->registerObject( d_samrai_hierarchy );
}
void SAMRAIHierarchyAdaptor::writeRestart( int64_t fid ) const
{
    Mesh::writeRestart( fid );
    uint64_t hierID = reinterpret_cast<uint64_t>( d_samrai_hierarchy.get() );
    AMP::IO::writeHDF5( fid, "hierarchyID", hierID );
}
SAMRAIHierarchyAdaptor::SAMRAIHierarchyAdaptor( int64_t fid, AMP::IO::RestartManager *manager )
    : SAMRHierarchy( fid, manager )
{
    uint64_t hierID;
    AMP::IO::readHDF5( fid, "hierarchyID", hierID );
    d_samrai_hierarchy = manager->getData<SAMRAI::hier::PatchHierarchy>( hierID );
    AMP_ASSERT( d_samrai_hierarchy );
    setLevelAdaptors();
}


} // namespace AMP::Mesh
