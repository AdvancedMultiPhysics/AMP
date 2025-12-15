#ifndef included_AMP_boxMeshDOFManager
#define included_AMP_boxMeshDOFManager

#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/structured/BoxMesh.h"


namespace AMP::Discretization {


/**
 * \class boxMeshDOFManager
 * \brief A derived class to create a simple DOF manager on a box mesh
 * \details  This derived class implements a simpleDOFManager for creating Vectors
 *    over a mesh on boxMesh.
 */
class boxMeshDOFManager final : public simpleDOFManager
{
public:
    /**
     * \brief Create a new DOF manager object
     * \details  This is the standard constructor for creating a new DOF manager object.
     * \param mesh          Mesh over which we want to construct the DOF map (must be a boxMesh)
     * \param type          The geometric entity type for the DOF map
     * \param gcw           The desired ghost width
     * \param DOFsPerElement The desired number of DOFs per element
     * multiDOFManager
     */
    boxMeshDOFManager( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                       AMP::Mesh::GeomType type,
                       int gcw,
                       int DOFsPerElement );


    //! Destructor
    virtual ~boxMeshDOFManager() = default;

    //! Get the underlying box mesh
    auto getBoxMesh() const { return d_boxMesh; }

    //! Get the array size for the local variables
    ArraySize getArraySize() const;


public: // Advanced interfaces
    //! Get the row DOFs given a mesh element
    size_t getRowDOFs( const AMP::Mesh::MeshElementID &id,
                       size_t *dofs,
                       size_t N_alloc,
                       bool sort = true ) const override;
    using DOFManager::getRowDOFs;

    // Append DOFs to the list
    size_t appendDOFs( const AMP::Mesh::MeshElementID &id,
                       size_t *dofs,
                       size_t index,
                       size_t capacity ) const override;

private:
    // Private constructor
    boxMeshDOFManager() = delete;

    // Convert a MeshElementID to a DOF index
    size_t convert( const AMP::Mesh::MeshElementID & ) const;

    // Convert a MeshElementIndex to a DOF index
    size_t convert( const AMP::Mesh::BoxMesh::MeshElementIndex &, int ) const;

    // Convert a DOF index to a MeshElementID
    AMP::Mesh::MeshElementID convert( size_t ) const;

private: // Data
    int d_gcw;
    int d_rank;
    std::shared_ptr<const AMP::Mesh::BoxMesh> d_boxMesh;
    std::vector<std::array<AMP::Mesh::BoxMesh::MeshElementIndex, 3>> d_ifirst;
    std::vector<std::array<std::array<int, 3>, 3>> d_boxSize;
    std::vector<size_t> d_start;
};

} // namespace AMP::Discretization

#endif
