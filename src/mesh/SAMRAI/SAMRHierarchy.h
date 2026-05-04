#ifndef included_AMP_SAMRHierarchy
#define included_AMP_SAMRHierarchy

#include <ctime>
#include <memory>
#include <vector>

#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/SAMRAI/SAMRLevel.h"
#include "AMP/utils/Utilities.h"


namespace SAMRAI::hier {
class PatchLevel;
}


namespace AMP::Mesh {


class SAMRHierarchy : public AMP::Mesh::Mesh
{
public:
    SAMRHierarchy( const std::shared_ptr<AMP::Mesh::MeshParameters> &params )
        : AMP::Mesh::Mesh( params )
    {
    }
    virtual ~SAMRHierarchy() {}
    virtual unsigned short getDim()                                                       = 0;
    virtual std::shared_ptr<SAMRLevel> getPatchLevel( const int ln )                      = 0;
    virtual std::shared_ptr<SAMRAI::hier::PatchLevel> getSAMRAIPatchLevel( const int ln ) = 0;
    virtual int getFinestLevelNumber( void )                                              = 0;
    virtual int getNumberOfLevels( void )                                                 = 0;
    virtual std::vector<int> getPeriodicShift( void )                                     = 0;
    virtual void reset( void )                                                            = 0;

    SAMRHierarchy()                        = default;
    SAMRHierarchy( const SAMRHierarchy & ) = delete;
    SAMRHierarchy( SAMRHierarchy && )      = delete;
    SAMRHierarchy &operator=( const SAMRHierarchy & ) = delete;
    SAMRHierarchy &operator=( const SAMRHierarchy && ) = delete;

    // functions derived from AMP::Mesh
    using AMP::Mesh::Mesh::clone;
    using AMP::Mesh::Mesh::displaceMesh;
    using AMP::Mesh::Mesh::isMeshMovable;
    using AMP::Mesh::Mesh::positionHash;
    std::unique_ptr<AMP::Mesh::Mesh> clone() const override { return nullptr; }
    bool operator==( const Mesh & ) const override
    {
        AMP_ERROR( "Not implemented" );
        return 0;
    }
    AMP::Mesh::Mesh::Movable isMeshMovable() const override
    {
        return AMP::Mesh::Mesh::Movable::Fixed;
    }
    std::string meshClass() const override { return "SAMRHierarchy"; }
    uint64_t positionHash() const override { return 0u; }
    void displaceMesh( const std::vector<double> & ) override {}
    void displaceMesh( std::shared_ptr<const AMP::LinearAlgebra::Vector> ) override {}
    void writeRestart( int64_t ) const override {}


protected:
    SAMRHierarchy( int64_t fid, AMP::IO::RestartManager *manager ) : Mesh( fid, manager ) {}

    using LevelVector = std::vector<std::shared_ptr<SAMRLevel>>;
    LevelVector d_levels;
};

} // namespace AMP::Mesh

#endif
