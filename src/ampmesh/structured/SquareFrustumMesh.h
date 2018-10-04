#ifndef included_AMP_SquareFrustumBoxMesh
#define included_AMP_SquareFrustumBoxMesh

#include "AMP/ampmesh/structured/StructuredGeometryMesh.h"

#include <array>


namespace AMP {
namespace Mesh {


/**
 * \class SquareFrustumMesh
 * \brief A derived version of BoxMesh for a square frustrum
 * \details A concrete implementation of BoxMesh for a square frustrum
 */
class SquareFrustumMesh : public AMP::Mesh::StructuredGeometryMesh
{
public:
    //! Default constructor
    explicit SquareFrustumMesh( MeshParameters::shared_ptr params );

    /**
     * \brief   Estimate the number of elements in the mesh
     * \details  This function will estimate the number of elements in the mesh.
     *   This is used so that we can properly balance the meshes across multiple processors.
     *   Ideally this should be both an accurate estimate and very fast.  It should not require
     *   any communication and should not have to actually load a mesh.
     * \param params Parameters for constructing a mesh from an input database
     */
    static std::vector<size_t> estimateLogicalMeshSize( const MeshParameters::shared_ptr &params );

    virtual AMP::shared_ptr<Mesh> clone() const override;

private:
    SquareFrustumMesh(); // Private empty constructor
};


} // namespace Mesh
} // namespace AMP


#endif
