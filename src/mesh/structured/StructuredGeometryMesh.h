#ifndef included_AMP_StructuredGeometryMesh
#define included_AMP_StructuredGeometryMesh

#include "AMP/geometry/LogicalGeometry.h"
#include "AMP/mesh/structured/BoxMesh.h"

#include <array>
#include <vector>


namespace AMP::Mesh {


/**
 * \class StructuredGeometryMesh
 * \brief A derived version of BoxMesh for a given geometry
 * \details A concrete implementation of BoxMesh for a object which has a
 *     well defined geometry than can be used to define the mapping between
 *     physical and logical coordinates
 */
class StructuredGeometryMesh final : public AMP::Mesh::BoxMesh
{
public:
    //! Default constructor
    explicit StructuredGeometryMesh( std::shared_ptr<const MeshParameters> );

    //! Create from a geometry
    explicit StructuredGeometryMesh( std::shared_ptr<AMP::Geometry::LogicalGeometry>,
                                     const ArraySize &size,
                                     const AMP::AMP_MPI &comm );

    //! Copy constructor
    explicit StructuredGeometryMesh( const StructuredGeometryMesh & );

    //! Assignment operator
    StructuredGeometryMesh &operator=( const StructuredGeometryMesh & ) = delete;

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
    StructuredGeometryMesh( int64_t, AMP::IO::RestartManager * );

private:
    uint32_t d_pos_hash = 0;
    std::shared_ptr<AMP::Geometry::LogicalGeometry> d_geometry2;
};


} // namespace AMP::Mesh


#endif
