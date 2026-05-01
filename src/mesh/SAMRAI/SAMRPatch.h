#ifndef included_AMP_SAMRPatch
#define included_AMP_SAMRPatch

namespace AMP::Mesh {

class SAMRPatch
{

public:
    virtual ~SAMRPatch()            = default;
    virtual bool inHierarchy()      = 0;
    virtual unsigned short getDim() = 0;

    SAMRPatch()                    = default;
    SAMRPatch( const SAMRPatch & ) = delete;
    SAMRPatch( SAMRPatch && )      = delete;
    SAMRPatch &operator=( const SAMRPatch & ) = delete;
    SAMRPatch &operator=( const SAMRPatch && ) = delete;
};

} // namespace AMP::Mesh

#endif
