#ifndef included_AMP_TriangleMeshElement
#define included_AMP_TriangleMeshElement


#include "AMP/mesh/MeshElement.h"
#include <memory>
#include <vector>


namespace AMP::Mesh {


template<uint8_t NG, uint8_t NP>
class TriangleMesh;
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
class TriangleMeshIterator;


/**
 * \class TriangleMeshElement
 * \brief A derived class used to define a mesh element
 * \details  This class provides routines for accessing and using a mesh element.
 * A mesh element can be thought of as the smallest unit of a mesh.  It is of a type
 * of GeomType.  This class is derived to store a TriangleMesh element.
 */
template<uint8_t NG, uint8_t NP, uint8_t TYPE>
class TriangleMeshElement final : public MeshElement
{
public:
    //! Empty constructor for a MeshElement
    TriangleMeshElement();

    //! Copy constructor
    TriangleMeshElement( const TriangleMeshElement & );

    //! Assignment operator
    TriangleMeshElement &operator=( const TriangleMeshElement & );

    //! Move operator
    TriangleMeshElement( TriangleMeshElement && );

    //! Move assignment operator
    TriangleMeshElement &operator=( TriangleMeshElement && );

    //! De-constructor for a MeshElement
    virtual ~TriangleMeshElement() = default;

    //! Return the unique global ID of the element
    MeshElementID globalID() const override { return d_globalID; }

    //! Return the element class
    std::string elementClass() const override;

    //! Return the elements composing the current element
    virtual void getElements( const GeomType type,
                              std::vector<MeshElement> &elements ) const override;

    //! Return the IDs of the elements composing the current element
    virtual void getElementsID( const GeomType type,
                                std::vector<MeshElementID> &ID ) const override;

    //! Return the elements neighboring the current element
    void getNeighbors( std::vector<std::unique_ptr<MeshElement>> &neighbors ) const override;

    //! Return the volume of the current element (does not apply to vertices)
    double volume() const override;

    //! Return the normal to the current element (does not apply to all elements)
    MeshPoint<double> norm() const override;

    //! Return the coordinates of all vertices composing the element
    MeshPoint<double> coord() const override;

    /**
     * \brief     Return the centroid of the element
     * \details   This function returns the centroid of the element.  The
     *   centroid is defined as the average of the coordinates of the vertices.
     *   The centroid of a vertex is the vertex and will return the same result as coord().
     */
    MeshPoint<double> centroid() const override;

    /**
     * \brief     Return true if the element contains the point
     * \details   This function checks if the given point is inside or
     *   within TOL of the given element.  If the current element is a vertex,
     *   this function checks if the point is with TOL of the vertex.
     * \param pos   The coordinates of the point to check.
     * \param TOL   The tolerance to use for the computation.
     */
    bool containsPoint( const MeshPoint<double> &pos, double TOL = 1e-12 ) const override;

    /**
     * \brief    Calculate the nearest point on the element
     * \details  This function computes nearest point on/in the element to the given point
     * \param[in] pos   Current position of the point
     */
    MeshPoint<double> nearest( const MeshPoint<double> &pos ) const override;

    /**
     * \brief    Calculate the distance to the element given a ray
     * \details  This function computes the distance to the element given a ray.
     *     If the ray will never intersect the element, this distance is inf.
     * \param[in] pos   Current position of ray
     * \param[in] dir   Direction of ray (should be normalized for most uses)
     * @return          Returns the distance to the element surface
     */
    virtual double distance( const MeshPoint<double> &pos,
                             const MeshPoint<double> &dir ) const override;

    //! Check if the element is on the surface
    bool isOnSurface() const override;

    /**
     * \brief     Check if the current element is on the given boundary
     * \details   Check if the current element is on the boundary specified by the given id
     * \param id  The boundary id to check
     */
    bool isOnBoundary( int id ) const override;

    /**
     * \brief     Check if the current element is in the given block
     * \details   Check if the current element is in the block specified by the given id
     * \param id  The block id to check
     */
    bool isInBlock( int id ) const override;

    //! Return the owner rank according to AMP_COMM_WORLD
    unsigned int globalOwnerRank() const override;


protected:
    // Default constructors
    TriangleMeshElement( const MeshElementID &id, const TriangleMesh<NG, NP> *mesh );

    // Reset the element data
    inline void resetElemId( const ElementID &id ) { d_globalID.resetElemID( id ); }

    //! Clone the iterator
    MeshElement *clone() const override;

    //! Get the vertices composing the element
    inline std::array<std::array<double, NP>, TYPE + 1> getVertexCoord() const;

    // The pointer to the current mesh
    const TriangleMesh<NG, NP> *d_mesh;

    // Friends
    friend class AMP::Mesh::TriangleMesh<NG, NP>;
    friend class AMP::Mesh::TriangleMeshIterator<NG, NP, TYPE>;

private:
    MeshElementID d_globalID;
};


} // namespace AMP::Mesh

#endif
