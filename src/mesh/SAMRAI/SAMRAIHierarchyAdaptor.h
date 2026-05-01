#ifndef included_AMP_SAMRAIHierarchyParameters
#define included_AMP_SAMRAIHierarchyParameters

#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/SAMRAI/SAMRAILevelAdaptor.h"
#include "AMP/mesh/SAMRAI/SAMRHierarchy.h"

#include <memory>


namespace SAMRAI::hier {
class PatchHierarchy;
}


namespace AMP::Mesh {

class SAMRLevel;

class SAMRAIHierarchyParameters : public AMP::Mesh::MeshParameters
{
public:
    explicit SAMRAIHierarchyParameters( const std::shared_ptr<AMP::Database> db );

    SAMRAIHierarchyParameters( const std::shared_ptr<AMP::Database> db,
                               std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy );

    virtual ~SAMRAIHierarchyParameters() {}

    SAMRAIHierarchyParameters() = delete;

    std::shared_ptr<SAMRAI::hier::PatchHierarchy> d_hierarchy = nullptr;
};

class SAMRAIHierarchyAdaptor : public SAMRHierarchy
{
public:
    explicit SAMRAIHierarchyAdaptor( std::shared_ptr<SAMRAI::hier::PatchHierarchy> );
    explicit SAMRAIHierarchyAdaptor( std::shared_ptr<AMP::Mesh::MeshParameters> );
    explicit SAMRAIHierarchyAdaptor( const std::shared_ptr<SAMRAIHierarchyParameters> &params );
    virtual ~SAMRAIHierarchyAdaptor() = default;
    unsigned short getDim() override;
    std::shared_ptr<SAMRLevel> getPatchLevel( const int ln ) override;
    std::shared_ptr<SAMRAI::hier::PatchLevel> getSAMRAIPatchLevel( const int ln ) override;
    int getFinestLevelNumber() override;
    int getNumberOfLevels() override;
    std::vector<int> getPeriodicShift() override;
    void reset() override;

    std::shared_ptr<SAMRAI::hier::PatchHierarchy> getSAMRAIHierarchy()
    {
        return d_samrai_hierarchy;
    }

    static std::shared_ptr<SAMRAI::hier::PatchHierarchy>
    getSAMRAIHierarchy( std::shared_ptr<AMP::Mesh::SAMRHierarchy> samr_hierarchy );

    SAMRAIHierarchyAdaptor()                                 = default;
    SAMRAIHierarchyAdaptor( const SAMRAIHierarchyAdaptor & ) = delete;
    SAMRAIHierarchyAdaptor( SAMRAIHierarchyAdaptor && )      = delete;

    SAMRAIHierarchyAdaptor &operator=( const SAMRAIHierarchyAdaptor & ) = delete;

    SAMRAIHierarchyAdaptor &operator=( const SAMRAIHierarchyAdaptor && ) = delete;

    std::string meshClass() const override { return "SAMRAIHierarchyAdaptor"; }

public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t fid ) const override;
    SAMRAIHierarchyAdaptor( int64_t fid, AMP::IO::RestartManager *manager );

protected:
    void setLevelAdaptors();

private:
    std::shared_ptr<SAMRAI::hier::PatchHierarchy> d_samrai_hierarchy = nullptr;
};

} // namespace AMP::Mesh

#endif
