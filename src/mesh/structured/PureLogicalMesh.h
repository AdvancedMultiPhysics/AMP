#ifndef included_AMP_PureLogicalMesh
#define included_AMP_PureLogicalMesh

#include "AMP/geometry/LogicalGeometry.h"
#include "AMP/mesh/structured/BoxMesh.h"

#include <array>
#include <vector>


namespace AMP::Mesh {


/**
 * \class PureLogicalMesh
 * \brief A derived version of BoxMesh for a given geometry
 * \details A concrete implementation of BoxMesh for a object which has a
 *     well defined geometry than can be used to define the mapping between
 *     physical and logical coordinates
 */
class PureLogicalMesh final : public AMP::Mesh::BoxMesh
{
public:
    //! Default constructor
    explicit PureLogicalMesh( std::shared_ptr<const MeshParameters> );

    //! Copy constructor
    explicit PureLogicalMesh( const PureLogicalMesh & );

    //! Assignment operator
    PureLogicalMesh &operator=( const PureLogicalMesh & ) = delete;

    //! Return a string with the mesh class name
    std::string meshClass() const override;

    //! Check if two meshes are equal
    bool operator==( const Mesh &mesh ) const override;

public: // Functions derived from BoxMesh
    Mesh::Movable isMeshMovable() const override;
    uint64_t positionHash() const override;
    void displaceMesh( const std::vector<double> &x ) override;
    void displaceMesh( std::shared_ptr<const AMP::LinearAlgebra::Vector> ) override;
    AMP::Geometry::Point physicalToLogical( const AMP::Geometry::Point &x ) const override;
    void coord( const MeshElementIndex &index, double *pos ) const override;
    std::array<AMP::Array<double>, 3> localCoord() const override;
    std::array<AMP::Array<double>, 3> globalCoord() const override;
    std::unique_ptr<Mesh> clone() const override;

public: // Restart functions
    void writeRestart( int64_t ) const override;
    PureLogicalMesh( int64_t, AMP::IO::RestartManager * );

protected: // Functions derived from BoxMesh
    virtual void createBoundingBox() override;
};


} // namespace AMP::Mesh


#endif
