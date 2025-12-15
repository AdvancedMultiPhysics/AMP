#ifndef included_AMP_libmeshElemIterator
#define included_AMP_libmeshElemIterator

#include "AMP/mesh/MeshIterator.h"
#include "AMP/mesh/libmesh/libmeshMeshElement.h"

// libMesh includes
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/elem.h"


namespace AMP::Mesh {


class libmeshMesh;


class libmeshElemIterator : public MeshIterator
{
public:
    //! Empty MeshIterator constructor
    libmeshElemIterator() = delete;

    //! Deconstructor
    virtual ~libmeshElemIterator() = default;

    //! Copy constructor
    libmeshElemIterator( const libmeshElemIterator & );

    //! Assignment operator
    libmeshElemIterator &operator=( const libmeshElemIterator & );

    // Increment
    MeshIterator &operator++() override;

    // Decrement
    MeshIterator &operator--() override;

    // Arithmetic operator+=
    MeshIterator &operator+=( int N ) override;

    // Check if two iterators are equal
    bool operator==( const MeshIterator &rhs ) const override;

    // Check if two iterators are not equal
    bool operator!=( const MeshIterator &rhs ) const override;

    // Return an iterator to the begining
    MeshIterator begin() const override;

    // Return an iterator to the begining
    MeshIterator end() const override;

    // Return the current libmeshMeshElement
    const libmeshMeshElement &current() const;

    // Return the current libMesh Elem
    libMesh::Elem *elem() const;

    using MeshIterator::operator+;
    using MeshIterator::operator+=;

protected:
    /** Default constructor
     * \param mesh      Pointer to the libMesh mesh
     * \param begin     Pointer to iterator with the begining position
     * \param end       Pointer to iterator with the end position
     */
    libmeshElemIterator( const AMP::Mesh::libmeshMesh *mesh,
                         const libMesh::Mesh::element_iterator &begin,
                         const libMesh::Mesh::element_iterator &end );

    /** Default constructor
     * \param mesh      Pointer to the libMesh mesh
     * \param begin     Pointer to iterator with the begining position
     * \param end       Pointer to iterator with the end position
     * \param pos       Pointer to iterator with the current position
     * \param size      Number of elements in the iterator (-1: unknown)
     * \param pos2      Index of the current position in the iterator (-1: unknown)
     */
    libmeshElemIterator( const AMP::Mesh::libmeshMesh *mesh,
                         const libMesh::Mesh::element_iterator &begin,
                         const libMesh::Mesh::element_iterator &end,
                         const libMesh::Mesh::element_iterator &pos,
                         int size,
                         int pos2 );

    //! Clone the iterator
    MeshIterator *clone() const override;

    friend class AMP::Mesh::libmeshMesh;

private:
    // Data members
    int d_dim;
    int d_rank;
    libMesh::Mesh::element_iterator d_begin2;
    libMesh::Mesh::element_iterator d_end2;
    libMesh::Mesh::element_iterator d_pos2;
    MeshID d_meshID;
    const libmeshMesh *d_mesh;
    libmeshMeshElement d_cur_element;

    void setCurrentElement();
};

} // namespace AMP::Mesh

#endif
