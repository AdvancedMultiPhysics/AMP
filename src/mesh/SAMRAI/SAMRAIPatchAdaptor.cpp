#include "AMP/mesh/SAMRAI/SAMRAIPatchAdaptor.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/utils/UtilityMacros.h"


namespace AMP::Mesh {

std::shared_ptr<SAMRAI::hier::Patch>
SAMRAIPatchAdaptor::getSAMRAIPatch( std::shared_ptr<SAMRPatch> samr_patch )
{
    auto patch_adaptor = std::dynamic_pointer_cast<SAMRAIPatchAdaptor>( samr_patch );
    AMP_ASSERT( patch_adaptor );
    auto patch = std::dynamic_pointer_cast<SAMRAI::hier::Patch>( patch_adaptor->getSAMRAIPatch() );
    AMP_ASSERT( patch );
    return patch;
}


#if 0
/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void SAMRAIPatchAdaptor::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    manager->registerObject( d_samrai_patch );
}
void SAMRAIPatchAdaptor::writeRestart( int64_t fid ) const
{
    uint64_t patchID = reinterpret_cast<uint64_t>( d_samrai_patch.get() );
    AMP::writeHDF5( fid, "patchID", patchID );
}
SAMRAIPatchAdaptor::SAMRAIPatchAdaptor( int64_t fid, AMP::IO::RestartManager *manager )
{
    uint64_t patchID;
    AMP::readHDF5( fid, "patchID", patchID );
    d_samrai_patch = manager->getData<SAMRAI::hier::Patch>( patchID );
    AMP_ASSERT( d_samrai_patch );
}
#endif


} // namespace AMP::Mesh
