#ifndef included_AMP_libmeshMeshElement
#define included_AMP_libmeshMeshElement


#include "AMP/mesh/MeshElement.h"
#include "AMP/mesh/libmesh/libmeshMesh.h"

#include <memory>
#include <vector>

// libMesh includes
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/elem.h"


namespace AMP::Mesh {


class libmeshMesh;
class libmeshElemIterator;
class libmeshNodeIterator;


/**
 * \class libmeshMeshElement
 * \brief A derived class used to define a mesh element
 * \details  This class provides routines for accessing and using a mesh element.
 * A mesh element can be thought of as the smallest unit of a mesh.  It is of a type
 * of GeomType.  This class is derived to store a libMesh element.
 */
class libmeshMeshElement final : public MeshElement
{
public:
    //! Empty constructor for a MeshElement
    libmeshMeshElement();

    //! Copy constructor
    libmeshMeshElement( const libmeshMeshElement & );

    //! Assignment operator
    libmeshMeshElement &operator=( const libmeshMeshElement & );

    //! De-constructor for a MeshElement
    virtual ~libmeshMeshElement();

    //! Return the unique global ID of the element
    MeshElementID globalID() const override { return d_globalID; }

    //! Return the typeID of the underlying element
    typeID getTypeID() const override { return AMP::getTypeID<libmeshMeshElement>(); }

    //! Return the element class
    inline std::string elementClass() const override { return "libmeshMeshElement"; }

    //! Return the elements composing the current element
    void getElements( const GeomType type, ElementList &elements ) const override;

    //! Return the IDs of the elements composing the current element
    int getElementsID( const GeomType type, MeshElementID *ID ) const override;

    //! Return the elements neighboring the current element
    void getNeighbors( ElementList &neighbors ) const override;

    //! Return the volume of the current element (does not apply to vertices)
    double volume() const override;

    //! Return the normal to the current element (does not apply to all elements)
    Point norm() const override;

    //! Return the coordinates of all vertices composing the element
    Point coord() const override;

    /**
     * \brief     Return the centroid of the element
     * \details   This function returns the centroid of the element.  The
     *   centroid is defined as the average of the coordinates of the vertices.
     *   The centroid of a vertex is the vertex and will return the same result as coord().
     */
    Point centroid() const override;

    /**
     * \brief     Return true if the element contains the point
     * \details   This function checks if the given point is inside or
     *   within TOL of the given element.  If the current element is a vertex,
     *   this function checks if the point is with TOL of the vertex.
     * \param pos   The coordinates of the point to check.
     * \param TOL   The tolerance to use for the computation.
     */
    bool containsPoint( const Point &pos, double TOL = 1e-12 ) const override;

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

    //! Return the raw pointer to the element/node (if it exists)
    const void *get() const { return ptr_element; }


public:
    /** Default constructors
     * \param dim       Spatial dimension
     * \param type      Element type
     * \param element   Underlying libmesh element
     * \param mesh      Underlying mesh
     * \param rank      Rank of the current processor (must agree with libmesh->processor_id())
     * \param meshID    ID of the current mesh
     */
    libmeshMeshElement( int dim,
                        GeomType type,
                        void *element,
                        unsigned int rank,
                        MeshID meshID,
                        const libmeshMesh *mesh );
    libmeshMeshElement( int dim,
                        GeomType type,
                        std::shared_ptr<libMesh::Elem> element,
                        unsigned int rank,
                        MeshID meshID,
                        const libmeshMesh *mesh );

    //! Clone the iterator
    std::unique_ptr<MeshElement> clone() const override;


protected:                               // Internal data
    int d_dim;                           // The dimension of the mesh
    unsigned int d_rank;                 // The rank of the current processor
    void *ptr_element;                   // The underlying libmesh element properties (raw pointer)
    std::shared_ptr<libMesh::Elem> ptr2; // Optional smart pointer to the element (to hold a copy)
    const libmeshMesh *d_mesh;           // The pointer to the current mesh
    MeshID d_meshID;                     // The ID of the current mesh
    bool d_delete_elem;                  // Do we need to delete the libMesh element
    MeshElementID d_globalID;
};


} // namespace AMP::Mesh

#endif
