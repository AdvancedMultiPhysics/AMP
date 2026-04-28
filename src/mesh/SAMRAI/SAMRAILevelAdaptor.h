#ifndef included_AMP_SAMRAILevelParameters
#define included_AMP_SAMRAILevelParameters

#include <memory>
#include <vector>

#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/SAMRAI/SAMRLevel.h"

#include "SAMRAI/hier/PatchLevel.h"


namespace AMP::IO {
class RestartManager;
}

namespace AMP::Mesh {

class SAMRAILevelParameters : public AMP::Mesh::MeshParameters
{
public:
    explicit SAMRAILevelParameters( const std::shared_ptr<AMP::Database> db )
        : AMP::Mesh::MeshParameters( db )
    {
    }

    SAMRAILevelParameters( const std::shared_ptr<AMP::Database> db,
                           std::shared_ptr<SAMRAI::hier::PatchLevel> level )
        : AMP::Mesh::MeshParameters( db ), d_level( level )
    {
        AMP_ASSERT( d_level );
        AMP::AMP_MPI mpi_comm( d_level->getBoxLevel()->getMPI() );
        setComm( mpi_comm );
    }

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

    virtual ~SAMRAILevelAdaptor() { d_patches.clear(); };
    bool inHierarchy() override { return d_samrai_level->inHierarchy(); }
    unsigned short getDim() override { return d_samrai_level->getDim().getValue(); }
    int getLevelNumber( void ) override { return d_samrai_level->getLevelNumber(); }
    int getLocalNumberOfPatches( void ) override
    {
        return d_samrai_level->getLocalNumberOfPatches();
    }
    unsigned long getGlobalNumberOfCells( void ) override
    {
        return d_samrai_level->getGlobalNumberOfCells();
    }
    std::shared_ptr<SAMRPatch> getPatch( const int pn ) override { return d_patches[pn]; };
    std::vector<int> getRatioToLevelZero( void ) override;
    std::vector<int> getRatioToCoarserLevel( void ) override;
    std::shared_ptr<SAMRLevel> constructCoarsenedLevel( void ) override;
    std::vector<int> getPeriodicShift( void ) override;
    void reset( void ) override;

    static std::shared_ptr<SAMRAI::hier::PatchLevel>
    getSAMRAILevel( std::shared_ptr<AMP::Mesh::SAMRLevel> samr_level );

    std::shared_ptr<SAMRAI::hier::PatchLevel> getSAMRAILevel( void ) { return d_samrai_level; }

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
