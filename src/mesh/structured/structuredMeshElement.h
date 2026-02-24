#ifndef included_AMP_structuredMeshElement
#define included_AMP_structuredMeshElement

#include "AMP/mesh/structured/BoxMesh.h"

#include <vector>


namespace AMP::Mesh {


/**
 * \class structuredMeshElement
 * \brief A derived class used to define a mesh element
 * \details  This class provides routines for accessing and using a mesh element.
 * A mesh element can be thought of as the smallest unit of a mesh.  It is of a type
 * of GeomType.  This class is derived to store a libMesh element.
 */
class structuredMeshElement final : public MeshElement
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

    /** Default constructor
     * \param index     Index for the current elements
     * \param mesh      Underlying mesh
     */
    structuredMeshElement( const BoxMesh::MeshElementIndex &index, const AMP::Mesh::BoxMesh *mesh );

    //! Reset the internal data to an empty element
    void reset();

    //! Reset the internal data to the given element
    void reset( const BoxMesh::MeshElementIndex &index, const AMP::Mesh::BoxMesh *mesh );

    /**
     * \brief     Get the parents of the given element
     * \details   This function will get the parent elements of the current element
     * \param type  The desired type of the parents to get
     */
    ElementListPtr getParents( GeomType type ) const;

    //! Return the index of the element
    const auto &getIndex() const { return d_index; }


public: // Functions derived from MeshElement
    const typeID &getTypeID() const override;
    MeshElementID globalID() const override { return d_mesh->convert( d_index ); }
    inline std::string elementClass() const override { return "structuredMeshElement"; }
    ElementListPtr getElements( const GeomType ) const override;
    int getElementsID( const GeomType, MeshElementID * ) const override;
    ElementListPtr getNeighbors() const override;
    Point centroid() const override;
    double volume() const override;
    Point norm() const override;
    Point coord() const override final
    {
        Point x;
        x.setNdim( d_physicalDim );
        d_mesh->coord( d_index, x.data() );
        return x;
    }
    MeshPoint<double> nearest( const MeshPoint<double> &pos ) const override;
    double distance( const MeshPoint<double> &pos, const MeshPoint<double> &dir ) const override;
    bool containsPoint( const Point &pos, double TOL = 1e-12 ) const override;
    bool isOnSurface() const override;
    bool isOnBoundary( int id ) const override;
    bool isInBlock( int id ) const override;
    void getVertices( std::vector<Point> &vertices ) const override;
    using MeshElement::getElements;
    using MeshElement::getNeighbors;

public: // Expert interfaces
    int getNeighborIndex( BoxMesh::MeshElementIndex *index ) const;

protected:
    // Clone the iterator
    std::unique_ptr<MeshElement> clone() const override;

    // Internal data
    GeomType d_meshType;               // Mesh logical dimension
    unsigned char d_physicalDim;       // Mesh physical dimension
    BoxMesh::MeshElementIndex d_index; // Index of element
    const AMP::Mesh::BoxMesh *d_mesh;  // Pointer to mesh

    // Helper functions
    void getElementIndex( const GeomType type, int &N, BoxMesh::MeshElementIndex *index ) const;
    std::array<int8_t, 3> getBC() const;

    // Reset just the mesh element
    inline void reset( const BoxMesh::MeshElementIndex &index ) { d_index = index; }

    friend class AMP::Mesh::BoxMesh;
    friend class AMP::Mesh::structuredMeshIterator;
};


} // namespace AMP::Mesh


#endif
