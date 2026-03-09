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


class libmeshElemIterator : public MeshIteratorBase
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

    //! Return the class name
    std::string className() const override { return "libmeshElemIterator"; }

    //! Set the position in the iterator
    void setPos( size_t ) override;

    // Increment
    MeshIteratorBase &operator++() override;

    // Decrement
    MeshIteratorBase &operator--() override;

    // Arithmetic operator+=
    MeshIteratorBase &operator+=( int N ) override;

    // Check if two iterators are equal
    bool operator==( const MeshIteratorBase &rhs ) const override;

    // Check if two iterators are not equal
    bool operator!=( const MeshIteratorBase &rhs ) const override;

    // Return an iterator to the begining
    MeshIterator begin() const override;

    //! Clone the iterator
    std::unique_ptr<MeshIteratorBase> clone() const override;

    // Return the current libmeshMeshElement
    const libmeshMeshElement &current() const;

    // Return the current libMesh Elem
    libMesh::Elem *elem() const;

    using MeshIteratorBase::operator==;
    using MeshIteratorBase::operator!=;


public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t fid ) const override;
    // libmeshElemIterator( int64_t fid, AMP::IO::RestartManager *manager );


public: // Advanced interfaces (use with caution)
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
