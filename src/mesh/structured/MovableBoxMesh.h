#ifndef included_AMP_MovableBoxMesh
#define included_AMP_MovableBoxMesh

#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/utils/Array.h"

#include <array>
#include <vector>


namespace AMP::Mesh {


/**
 * \class MovableBoxMesh
 * \brief A general BoxMesh
 * \details A concrete implementation of BoxMesh in which the
 *    coordinates of the nodes may be moved.  The base class
 *    BoxMesh is an arbitrary logically rectangular mesh class.
 *    To allow for derived implementations which can leverage specific
 *    properties, a derived class is required.
 */
class MovableBoxMesh : public AMP::Mesh::BoxMesh
{
public:
    //! Construct a movable box mesh from any existing box mesh
    explicit MovableBoxMesh( const AMP::Mesh::BoxMesh &mesh );


    //! Return a string with the mesh class name
    std::string meshClass() const override;


    /**
     * \brief    Is the mesh movable
     * \details  This function will check if the mesh can be displaced.
     *    It will return 0 if the mesh cannont be moved, 1 if it can be displaced,
     *    and 2 if the individual nodes can be moved.
     * @return  The if
     */
    Mesh::Movable isMeshMovable() const override;


    //! Check if two meshes are equal
    bool operator==( const Mesh &mesh ) const override;


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

    //! Virtual function to copy the mesh (allows us to properly copy the derived class)
    std::unique_ptr<Mesh> clone() const override;

    /**
     * \brief    Return a mesh element's coordinates given it's id.
     * \details  This function queries the mesh to get an element's coordinates given the mesh id.
     *    Ideally, this should be done in O(1) time, but the implementation is up to
     *    the underlying mesh.
     * \param[in] index     Mesh element index we are requesting.
     * \param[out] pos      Mesh element coordinates
     */
    void coord( const MeshElementIndex &index, double *pos ) const override;


public: // BoxMesh specific functionality
    /**
     * \brief    Return the logical coordinates
     * \details  This function queries the mesh to get the logical coordinates in [0,1]
     *     from the physical coordinates.  Not all meshes support this functionallity.
     * \param[in] x         Physical coordinates
     * @return              Returns the logical coordinates
     */
    AMP::Geometry::Point physicalToLogical( const AMP::Geometry::Point &x ) const override;


public: // Restart functions
    void writeRestart( int64_t ) const override;
    MovableBoxMesh( int64_t, AMP::IO::RestartManager * );


private:
    MovableBoxMesh(); // Private empty constructor

    // Index indicating number of times the position has changed
    uint64_t d_pos_hash;

    // The coordinates of the nodes
    MeshElementIndex d_first;
    MeshElementIndex d_last;
    AMP::Array<double> d_coord;
};


} // namespace AMP::Mesh


#endif
