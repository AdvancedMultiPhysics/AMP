#ifndef included_AMP_SAMRAIPatchAdaptor
#define included_AMP_SAMRAIPatchAdaptor

#include <memory>

#include "SAMRAI/hier/Patch.h"

#include "AMP/mesh/SAMRAI/SAMRPatch.h"

namespace AMP::IO {
class RestartManager;
}

namespace AMP::Mesh {

class SAMRAIPatchAdaptor : public SAMRPatch
{

public:
    explicit SAMRAIPatchAdaptor( std::shared_ptr<SAMRAI::hier::Patch> patch )
        : d_samrai_patch( patch )
    {
    }

    virtual ~SAMRAIPatchAdaptor() = default;
    bool inHierarchy() override { return d_samrai_patch->inHierarchy(); }
    unsigned short getDim() override { return d_samrai_patch->getDim().getValue(); }
    std::shared_ptr<SAMRAI::hier::Patch> getSAMRAIPatch( void ) { return d_samrai_patch; }

    static std::shared_ptr<SAMRAI::hier::Patch>
    getSAMRAIPatch( std::shared_ptr<AMP::Mesh::SAMRPatch> samr_patch );

    SAMRAIPatchAdaptor()                                         = default;
    SAMRAIPatchAdaptor( const SAMRAIPatchAdaptor & )             = delete;
    SAMRAIPatchAdaptor( SAMRAIPatchAdaptor && )                  = delete;
    SAMRAIPatchAdaptor &operator=( const SAMRAIPatchAdaptor & )  = delete;
    SAMRAIPatchAdaptor &operator=( const SAMRAIPatchAdaptor && ) = delete;

    //    SAMRAIPatchAdaptor( int64_t fid, AMP::IO::RestartManager *manager );
    //    void registerChildObjects( AMP::IO::RestartManager *manager ) const;
    //    void writeRestart( int64_t fid ) const;

private:
    std::shared_ptr<SAMRAI::hier::Patch> d_samrai_patch = nullptr;
};

} // namespace AMP::Mesh

#endif
