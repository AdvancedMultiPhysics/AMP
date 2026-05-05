#ifndef included_AMP_libmeshNodeIterator
#define included_AMP_libmeshNodeIterator

#include "AMP/mesh/MeshIterator.h"
#include "AMP/mesh/libmesh/libmeshMesh.h"
#include "AMP/mesh/libmesh/libmeshMeshElement.h"

// libMesh includes
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/elem.h"


namespace AMP::Mesh {


class libmeshNodeIterator : public MeshIteratorBase
{
public:
    //! Empty MeshIterator constructor
    libmeshNodeIterator() = delete;

    //! Deconstructor
    virtual ~libmeshNodeIterator() = default;

    //! Copy constructor
    libmeshNodeIterator( const libmeshNodeIterator & );

    //! Assignment operator
    libmeshNodeIterator &operator=( const libmeshNodeIterator & );

    //! Return the class name
    std::string className() const override { return "libmeshNodeIterator"; }

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

    using MeshIteratorBase::operator==;
    using MeshIteratorBase::operator!=;


public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t fid ) const override;
    // libmeshNodeIterator( int64_t fid, AMP::IO::RestartManager *manager );


public: // Advanced interfaces (use with caution)
    /** Default constructor
     * \param mesh      Pointer to the libMesh mesh
     * \param begin     Pointer to iterator with the begining position
     * \param end       Pointer to iterator with the end position
     */
    libmeshNodeIterator( const AMP::Mesh::libmeshMesh *mesh,
                         const libMesh::Mesh::node_iterator &begin,
                         const libMesh::Mesh::node_iterator &end );

    /** Default constructor
     * \param mesh      Pointer to the libMesh mesh
     * \param begin     Pointer to iterator with the begining position
     * \param end       Pointer to iterator with the end position
     * \param pos       Pointer to iterator with the current position
     * \param size      Number of elements in the iterator (-1: unknown)
     * \param pos2      Index of the current position in the iterator (-1: unknown)
     */
    libmeshNodeIterator( const AMP::Mesh::libmeshMesh *mesh,
                         const libMesh::Mesh::node_iterator &begin,
                         const libMesh::Mesh::node_iterator &end,
                         const libMesh::Mesh::node_iterator &pos,
                         int size,
                         int pos2 );


private:
    // Data members
    int d_dim;
    int d_rank;
    libMesh::Mesh::node_iterator d_begin2;
    libMesh::Mesh::node_iterator d_end2;
    libMesh::Mesh::node_iterator d_pos2;
    MeshID d_meshID;
    const AMP::Mesh::libmeshMesh *d_mesh;
    libmeshMeshElement d_cur_element;

    void setCurrentElement();
};

} // namespace AMP::Mesh

#endif
