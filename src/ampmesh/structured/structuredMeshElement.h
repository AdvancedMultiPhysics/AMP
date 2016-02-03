#ifndef included_AMP_structuredMeshElement
#define included_AMP_structuredMeshElement

#include "ampmesh/structured/BoxMesh.h"
#include <vector>

namespace AMP {
namespace Mesh {


/**
 * \class structuredMeshElement
 * \brief A derived class used to define a mesh element
 * \details  This class provides routines for accessing and using a mesh element.
 * A mesh element can be thought of as the smallest unit of a mesh.  It is of a type
 * of GeomType.  This class is derived to store a libMesh element.
 */
class structuredMeshElement : public MeshElement
{
public:
    //! Empty constructor for a MeshElement
    structuredMeshElement();

    //! Copy constructor
    structuredMeshElement( const structuredMeshElement & );

    //! Assignment operator
    structuredMeshElement &operator=( const structuredMeshElement & );

    //! De-constructor for a MeshElement
    virtual ~structuredMeshElement();

    //! Return the elements composing the current element
    virtual std::vector<MeshElement> getElements( const GeomType type ) const override;

    //! Return the elements neighboring the current element
    virtual std::vector<MeshElement::shared_ptr> getNeighbors() const override;

    //! Return the volume of the current element (does not apply to verticies)
    virtual double volume() const override;

    //! Return the coordinates of the vertex (only applies to verticies)
    virtual std::vector<double> coord() const override;

    /**
     * \brief     Return the coordinate of the vertex
     * \details   This function returns the coordinates of the vertex
     *   in the given direction (only applies to verticies).
     *   Note: This is a faster access for obtaining a single coordinate
     * \param i     The direction requested.  Equivalent to coord()[i]
     */
    virtual double coord( int i ) const override;

    /**
     * \brief     Return true if the element contains the point
     * \details   This function checks if the given point is inside or
     *   within TOL of the given element.  If the current element is a vertex,
     *   this function checks if the point is with TOL of the vertex.
     * \param pos   The coordinates of the point to check.
     * \param TOL   The tolerance to use for the computation.
     */
    virtual bool containsPoint( const std::vector<double> &pos, double TOL = 1e-12 ) const override;

    //! Check if the element is on the surface
    virtual bool isOnSurface() const override;

    /**
     * \brief     Check if the current element is on the given boundary
     * \details   Check if the current element is on the boundary specified by the given id
     * \param id  The boundary id to check
     */
    virtual bool isOnBoundary( int id ) const override;

    /**
     * \brief     Check if the current element is in the given block
     * \details   Check if the current element is in the block specified by the given id
     * \param id  The block id to check
     */
    virtual bool isInBlock( int id ) const override;


    /**
     * \brief     Get the parents of the given element
     * \details   This function will get the parent elements of the current element
     * \param type  The desired type of the parents to get
     */
    virtual std::vector<MeshElement> getParents( GeomType type ) const;


    //! Return the index of the element
    BoxMesh::MeshElementIndex getIndex() const { return d_index; }


protected:
    /** Default constructor
     * \param index     Index for the current elements
     * \param mesh      Underlying mesh
     */
    structuredMeshElement( BoxMesh::MeshElementIndex index, const AMP::Mesh::BoxMesh *mesh );

    // Clone the iterator
    virtual MeshElement *clone() const override;

    // Internal data
    unsigned char d_dim;
    BoxMesh::MeshElementIndex d_index;
    const AMP::Mesh::BoxMesh *d_mesh;

    friend class AMP::Mesh::BoxMesh;
    friend class AMP::Mesh::structuredMeshIterator;
};
}
}

#endif
