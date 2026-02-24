#ifndef included_AMP_SubsetMesh
#define included_AMP_SubsetMesh

#include "AMP/mesh/Mesh.h"

#include <memory>

#include <map>

namespace AMP::Mesh {


/**
 * \class SubsetMesh
 * \brief A class used to handle a subset mesh
 * \details  This class provides routines for using subset meshes.
 */
class SubsetMesh : public Mesh
{
public:
    /**
     * \brief    Create a subset mesh
     * \details  This function will create the desired subset mesh.
     *           If the underlying mesh is a multimesh, it will return
     *           a multimesh of the individual subset meshes
     * \param mesh      Mesh to subset
     * \param iterator  Iterator containing the desired mesh elements
     * \param isGlobal  Do we want the communicator to span the provided mesh
     */
    static std::shared_ptr<Mesh> create( std::shared_ptr<const Mesh> mesh,
                                         const AMP::Mesh::MeshIterator &iterator,
                                         bool isGlobal );


    //! Deconstructor
    virtual ~SubsetMesh();


    /**
     * \brief    Subset a mesh given a MeshID
     * \details  This function will return the mesh with the given meshID.
     *    Note: for multimeshes, this will return the mesh with the given id.
     *    For a single mesh this will return a pointer to itself if the meshID
     *    matches the meshID of the mesh, and a null pointer otherwise.
     * \param meshID  MeshID of the desired mesh
     */
    std::shared_ptr<Mesh> Subset( MeshID meshID ) const override;


    /**
     * \brief    Subset a mesh given a mesh name
     * \details  This function will return the mesh with the given name.
     *    For a single mesh this will return a pointer to itself if the mesh name
     *    matches the name of the mesh, and a null pointer otherwise.
     *    Note: The mesh name is not guaranteed to be unique.  If there are multiple
     *    meshes with the same name, all meshes with the given name will be returned
     *    within a new multimesh.
     *    It is strongly recommended to use the meshID when possible.
     * \param name  Name of the desired mesh
     */
    std::shared_ptr<Mesh> Subset( std::string name ) const override;


    //! Return a string with the mesh class name
    std::string meshClass() const override;


    //! Function to copy the mesh (allows use to properly copy the derived class)
    std::unique_ptr<Mesh> clone() const override;


    /* Return the number of local element of the given type
     * \param type   Geometric type
     */
    size_t numLocalElements( const GeomType type ) const override;


    /* Return the global number of elements of the given type
     * Note: depending on the mesh this routine may require global communication across the mesh.
     * \param type   Geometric type
     */
    size_t numGlobalElements( const GeomType type ) const override;


    /* Return the number of ghost elements of the given type on the current processor
     * \param type   Geometric type
     */
    size_t numGhostElements( const GeomType type, const int gcw ) const override;


    /**
     * \brief    Return an MeshIterator over the given geometric objects
     * \details  Return an MeshIterator over the given geometric objects
     * \param type   Geometric type to iterate over
     * \param gcw    Desired ghost cell width
     */
    MeshIterator getIterator( const GeomType type, const int gcw = 0 ) const override;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the surface
     * \details  Return an MeshIterator over the given geometric objects on the surface
     * \param type   Geometric type to iterate over
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator getSurfaceIterator( const GeomType type,
                                             const int gcw = 0 ) const override;


    /**
     * \brief    Return the list of all boundary ID sets in the mesh
     * \details  Return the list of all boundary ID sets in the mesh
     * Note: depending on the mesh this routine may require global communication across the mesh.
     */
    std::vector<int> getBoundaryIDs() const override;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the given boundary ID
     * set
     * \details  Return an MeshIterator over the given geometric objects on the given boundary ID
     * set
     * \param type   Geometric type to iterate over
     * \param id     Boundary id for the elements (example: sideset id)
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator
    getBoundaryIDIterator( const GeomType type, const int id, const int gcw = 0 ) const override;


    /**
     * \brief    Return the list of all boundary ID sets in the mesh
     * \details  Return the list of all boundary ID sets in the mesh
     * Note: depending on the mesh this routine may require global communication across the mesh.
     */
    std::vector<int> getBlockIDs() const override;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the given block ID set
     * \details  Return an MeshIterator over the given geometric objects on the given block ID set
     * \param type   Geometric type to iterate over
     * \param id     Block id for the elements (example: block id in cubit, subdomain in libmesh)
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator
    getBlockIDIterator( const GeomType type, const int id, const int gcw = 0 ) const override;


    /**
     * \brief    Check if an element is in the mesh
     * \details  This function queries the mesh to determine if the given element is a member of the
     * mesh
     * \param id    Mesh element id we are querying.
     */
    bool isMember( const MeshElementID &id ) const override;


    /**
     * \brief    Check if elements are in the mesh
     * \details  This function queries the mesh to determine if each of the given elements
     *           is a member of the mesh and returns an iterator over those elements
     * \param it    Mesh element id we are querying.
     */
    MeshIterator isMember( const MeshIterator &it ) const override;


    /**
     * \brief    Return a mesh element given it's id.
     * \details  This function queries the mesh to get an element given the mesh id.
     *    This function is only required to return an element if the id is local.
     *    Ideally, this should be done in O(1) time, but the implementation is up to
     *    the underlying mesh.  The base class provides a basic implementation, but
     *    uses mesh iterators and requires O(N) time on the number of elements in the mesh.
     * \param id    Mesh element id we are requesting.
     */
    MeshElementPtr getElement( const MeshElementID &id ) const override;


    /**
     * \brief    Return the parent elements of the given mesh element
     * \details  This function queries the mesh to get an element given the mesh id,
     *    then returns the parent elements that have the element as a child
     * \param elem  Mesh element of interest
     * \param type  Element type of the parents requested
     */
    ElementListPtr getElementParents( const MeshElement &elem, const GeomType type ) const override;


    //! Is the current mesh a base mesh
    inline bool isBaseMesh() const override { return false; }


    //! Check if two meshes are equal
    bool operator==( const Mesh &mesh ) const override;


    /**
     *  Get the meshIDs of all meshes that compose the current mesh (including its self)
     *  Note: This function may require global communication depending on the implementation
     */
    std::vector<MeshID> getAllMeshIDs() const override;


    /**
     *  Get the meshIDs of all the basic meshes that compose the current mesh (excluding multimeshes
     * and subset meshes)
     *  Note: This function may require global communication depending on the implementation
     */
    std::vector<MeshID> getBaseMeshIDs() const override;


    /**
     *  Get the meshIDs of all meshes that compose the current mesh (including its self)
     *  on the current processor.
     */
    std::vector<MeshID> getLocalMeshIDs() const override;


    /**
     *  Get the meshIDs of all the basic meshes that compose the current mesh
     *  (excluding multimeshes and subset meshes) on the current processor.
     */
    std::vector<MeshID> getLocalBaseMeshIDs() const override;


    /**
     *  Check if the mesh contains the given mesh element
     */
    bool containsElement( const MeshElementID &id ) const override;


    /**
     * \brief    Is the mesh movable
     * \details  This function will check if the mesh can be displaced.
     *    It will return 0 if the mesh cannont be moved, 1 if it can be displaced,
     *    and 2 if the individual nodes can be moved.
     * @return  The if
     */
    Mesh::Movable isMeshMovable() const override;


    /**
     * \brief    Identify if the position has moved
     * \details  This function will return a hash that can be used to
     *    identify if the mesh has been moved.  Any time that displaceMesh
     *    is called, the hash value should change.  There is no requirement
     *    that dispacing a mesh and returning it back to the original position
     *    will return the original hash.
     * @return   hash value with current position id
     */
    uint64_t positionHash() const override;


    /**
     * \brief    Displace the entire mesh
     * \details  This function will displace the entire mesh by a scalar value.
     *   This function is a blocking call for the mesh communicator, and requires
     *   the same value on all processors.  The displacement vector should be the
     *   size of the physical dimension.
     * \param x  Displacement vector
     */
    void displaceMesh( const std::vector<double> &x ) override;


    /**
     * \brief    Displace the entire mesh
     * \details  This function will displace the entire mesh by displacing
     *   each node by the values provided in the vector.  This function is
     *   a blocking call for the mesh communicator
     * \param x  Displacement vector.  Must have N DOFs per node where N
     *           is the physical dimension of the mesh.
     */
    void displaceMesh( std::shared_ptr<const AMP::LinearAlgebra::Vector> x ) override;


    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    void writeRestart( int64_t fid ) const override;


    // Needed to prevent problems with virtual functions
    using Mesh::Subset;


protected:
    //! Default constructor
    SubsetMesh( std::shared_ptr<const Mesh> mesh,
                const AMP::Mesh::MeshIterator &iterator,
                bool isGlobal );


protected:
    using ElementList    = std::vector<std::unique_ptr<MeshElement>>;
    using ElementListPtr = std::shared_ptr<ElementList>;

    // Data to store the id sets
    struct map_id_struct {
        int id;
        GeomType type;
        int gcw;
        inline bool operator==( const map_id_struct &rhs ) const
        {
            return id == rhs.id && type == rhs.type && gcw == rhs.gcw;
        }
        inline bool operator!=( const map_id_struct &rhs ) const
        {
            return id != rhs.id && type != rhs.type && gcw != rhs.gcw;
        }
        inline bool operator>=( const map_id_struct &rhs ) const
        {
            if ( id != rhs.id ) {
                return id > rhs.id;
            }
            if ( type != rhs.type ) {
                return type > rhs.type;
            }
            return gcw >= rhs.gcw;
        }
        inline bool operator>( const map_id_struct &rhs ) const
        {
            if ( id != rhs.id ) {
                return id > rhs.id;
            }
            if ( type != rhs.type ) {
                return type > rhs.type;
            }
            return gcw > rhs.gcw;
        }
        inline bool operator<( const map_id_struct &rhs ) const { return !operator>=( rhs ); }
        inline bool operator<=( const map_id_struct &rhs ) const { return !operator>( rhs ); }
    };


protected:                                               // Member data
    std::shared_ptr<const Mesh> d_parentMesh;            // Parent mesh for the subset
    MeshID d_parentMeshID;                               // Parent mesh id
    std::vector<size_t> N_global;                        // Number of global elements
    std::vector<std::vector<ElementListPtr>> d_elements; // Elements in the subset [type][gcw][elem]
    std::vector<std::vector<ElementListPtr>> d_surface; // Elements on the surface [type][gcw][elem]
    std::vector<int> d_boundaryIdSets;
    std::map<map_id_struct, ElementListPtr> d_boundarySets;
    std::vector<int> d_blockIdSets;
    std::map<map_id_struct, ElementListPtr> d_blockSets;
};

} // namespace AMP::Mesh

#endif
