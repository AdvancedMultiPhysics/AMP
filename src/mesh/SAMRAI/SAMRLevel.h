#ifndef included_AMP_SAMRLevel
#define included_AMP_SAMRLevel

#include <memory>
#include <vector>

#include "AMP/mesh/Mesh.h"

namespace AMP::Mesh {

class SAMRPatch;

class SAMRLevel : public AMP::Mesh::Mesh
{

public:
    SAMRLevel( const std::shared_ptr<AMP::Mesh::MeshParameters> &params )
        : AMP::Mesh::Mesh( params )
    {
    }
    virtual ~SAMRLevel() {}
    virtual bool inHierarchy()                                         = 0;
    virtual unsigned short getDim()                                    = 0;
    virtual int getLevelNumber( void )                                 = 0;
    virtual int getLocalNumberOfPatches( void )                        = 0;
    virtual unsigned long getGlobalNumberOfCells( void )               = 0;
    virtual std::shared_ptr<SAMRPatch> getPatch( const int pn )        = 0;
    virtual std::vector<int> getRatioToLevelZero( void )               = 0;
    virtual std::vector<int> getRatioToCoarserLevel( void )            = 0;
    virtual std::shared_ptr<SAMRLevel> constructCoarsenedLevel( void ) = 0;
    virtual std::vector<int> getPeriodicShift( void )                  = 0;
    virtual void reset( void )                                         = 0;

    SAMRLevel()                                = default;
    SAMRLevel( const SAMRLevel & )             = delete;
    SAMRLevel( SAMRLevel && )                  = delete;
    SAMRLevel &operator=( const SAMRLevel & )  = delete;
    SAMRLevel &operator=( const SAMRLevel && ) = delete;

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
    std::string meshClass() const override { return "SAMRLevel"; }
    uint64_t positionHash() const override { return 0u; }
    void displaceMesh( const std::vector<double> & ) override {}
    void displaceMesh( std::shared_ptr<const AMP::LinearAlgebra::Vector> ) override {}
    void writeRestart( int64_t ) const override {}

protected:
    using patch_vector = std::vector<std::shared_ptr<SAMRPatch>>;
    patch_vector d_patches;

public:
    using iterator       = patch_vector::iterator;
    using const_iterator = patch_vector::const_iterator;
    iterator begin() { return d_patches.begin(); }
    iterator end() { return d_patches.end(); }
};

} // namespace AMP::Mesh

#endif
