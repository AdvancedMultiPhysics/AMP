#ifndef included_AMP_SAMRAILevelParameters
#define included_AMP_SAMRAILevelParameters

#include <memory>
#include <vector>

#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/SAMRAI/SAMRLevel.h"


namespace SAMRAI::hier {
class PatchLevel;
}


namespace AMP::IO {
class RestartManager;
}

namespace AMP::Mesh {

class SAMRAILevelParameters : public AMP::Mesh::MeshParameters
{
public:
    explicit SAMRAILevelParameters( const std::shared_ptr<AMP::Database> db );

    SAMRAILevelParameters( const std::shared_ptr<AMP::Database> db,
                           std::shared_ptr<SAMRAI::hier::PatchLevel> level );

    virtual ~SAMRAILevelParameters() {}

    SAMRAILevelParameters() = delete;

    std::shared_ptr<SAMRAI::hier::PatchLevel> d_level = nullptr;
};

class SAMRAILevelAdaptor : public SAMRLevel
{

public:
    explicit SAMRAILevelAdaptor( std::shared_ptr<SAMRAI::hier::PatchLevel> );
    explicit SAMRAILevelAdaptor( const std::shared_ptr<SAMRAILevelParameters> & );
    //    SAMRAILevelAdaptor( int64_t fid, AMP::IO::RestartManager *manager );
    virtual ~SAMRAILevelAdaptor() = default;

    bool inHierarchy() override;
    unsigned short getDim() override;
    int getLevelNumber() override;
    int getLocalNumberOfPatches() override;
    unsigned long getGlobalNumberOfCells() override;
    std::shared_ptr<SAMRPatch> getPatch( const int pn ) override { return d_patches[pn]; };
    std::vector<int> getRatioToLevelZero() override;
    std::vector<int> getRatioToCoarserLevel() override;
    std::shared_ptr<SAMRLevel> constructCoarsenedLevel() override;
    std::vector<int> getPeriodicShift() override;
    void reset() override;

    static std::shared_ptr<SAMRAI::hier::PatchLevel>
    getSAMRAILevel( std::shared_ptr<AMP::Mesh::SAMRLevel> samr_level );

    std::shared_ptr<SAMRAI::hier::PatchLevel> getSAMRAILevel() { return d_samrai_level; }

    SAMRAILevelAdaptor()                             = default;
    SAMRAILevelAdaptor( const SAMRAILevelAdaptor & ) = delete;
    SAMRAILevelAdaptor( SAMRAILevelAdaptor && )      = delete;

    SAMRAILevelAdaptor &operator=( const SAMRAILevelAdaptor & ) = delete;

    SAMRAILevelAdaptor &operator=( const SAMRAILevelAdaptor && ) = delete;

    //    void registerChildObjects( AMP::IO::RestartManager *manager ) const;
    //    void writeRestart( int64_t fid ) const;

protected:
    void setPatchAdaptors();

private:
    std::shared_ptr<SAMRAI::hier::PatchLevel> d_samrai_level = nullptr;
};

} // namespace AMP::Mesh

#endif
