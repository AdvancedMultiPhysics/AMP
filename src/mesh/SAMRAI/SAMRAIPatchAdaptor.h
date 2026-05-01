#ifndef included_AMP_SAMRAIPatchAdaptor
#define included_AMP_SAMRAIPatchAdaptor

#include "AMP/mesh/SAMRAI/SAMRPatch.h"

#include <memory>


namespace SAMRAI::hier {
class Patch;
}


namespace AMP::IO {
class RestartManager;
}

namespace AMP::Mesh {

class SAMRAIPatchAdaptor : public SAMRPatch
{

public:
    SAMRAIPatchAdaptor()                                         = default;
    SAMRAIPatchAdaptor( const SAMRAIPatchAdaptor & )             = delete;
    SAMRAIPatchAdaptor( SAMRAIPatchAdaptor && )                  = delete;
    SAMRAIPatchAdaptor &operator=( const SAMRAIPatchAdaptor & )  = delete;
    SAMRAIPatchAdaptor &operator=( const SAMRAIPatchAdaptor && ) = delete;
    explicit SAMRAIPatchAdaptor( std::shared_ptr<SAMRAI::hier::Patch> patch );
    virtual ~SAMRAIPatchAdaptor() = default;

    bool inHierarchy() override;
    unsigned short getDim() override;
    std::shared_ptr<SAMRAI::hier::Patch> getSAMRAIPatch();
    static std::shared_ptr<SAMRAI::hier::Patch>
    getSAMRAIPatch( std::shared_ptr<AMP::Mesh::SAMRPatch> samr_patch );

    //    SAMRAIPatchAdaptor( int64_t fid, AMP::IO::RestartManager *manager );
    //    void registerChildObjects( AMP::IO::RestartManager *manager ) const;
    //    void writeRestart( int64_t fid ) const;

private:
    std::shared_ptr<SAMRAI::hier::Patch> d_samrai_patch = nullptr;
};

} // namespace AMP::Mesh

#endif
