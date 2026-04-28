#ifndef included_AMP_SAMRAIHierarchyParameters
#define included_AMP_SAMRAIHierarchyParameters

#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/SAMRAI/SAMRAILevelAdaptor.h"
#include "AMP/mesh/SAMRAI/SAMRHierarchy.h"

#include "SAMRAI/hier/PatchHierarchy.h"

#include <memory>


namespace AMP::Mesh {

class SAMRLevel;

class SAMRAIHierarchyParameters : public AMP::Mesh::MeshParameters
{
public:
    explicit SAMRAIHierarchyParameters( const std::shared_ptr<AMP::Database> db )
        : AMP::Mesh::MeshParameters( db )
    {
    }

    SAMRAIHierarchyParameters( const std::shared_ptr<AMP::Database> db,
                               std::shared_ptr<SAMRAI::hier::PatchHierarchy> hierarchy )
        : AMP::Mesh::MeshParameters( db ), d_hierarchy( hierarchy )
    {
        AMP_ASSERT( d_hierarchy );
        AMP::AMP_MPI mpi_comm( d_hierarchy->getMPI() );
        setComm( mpi_comm );
    }

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
    virtual ~SAMRAIHierarchyAdaptor() { d_levels.clear(); }
    unsigned short getDim() override { return d_samrai_hierarchy->getDim().getValue(); }
    std::shared_ptr<SAMRLevel> getPatchLevel( const int ln ) override { return d_levels[ln]; }
    std::shared_ptr<SAMRAI::hier::PatchLevel> getSAMRAIPatchLevel( const int ln ) override
    {
        const auto &src_level_adaptor =
            std::dynamic_pointer_cast<AMP::Mesh::SAMRAILevelAdaptor>( d_levels[ln] );
        return src_level_adaptor->getSAMRAILevel();
    }
    int getFinestLevelNumber( void ) override { return d_samrai_hierarchy->getFinestLevelNumber(); }
    int getNumberOfLevels( void ) override { return d_samrai_hierarchy->getNumberOfLevels(); }
    std::vector<int> getPeriodicShift( void ) override;
    void reset( void ) override;

    std::shared_ptr<SAMRAI::hier::PatchHierarchy> getSAMRAIHierarchy( void )
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
