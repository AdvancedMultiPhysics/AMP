#ifndef included_AMP_TriangleMeshIterators
#define included_AMP_TriangleMeshIterators


#include "AMP/mesh/MeshIterator.h"
#include "AMP/mesh/triangle/TriangleMeshElement.h"


namespace AMP::Mesh {


template<uint8_t NG>
class TriangleMesh;


template<uint8_t NG, uint8_t TYPE>
class TriangleMeshIterator final : public MeshIterator
{
public:
    //! Empty MeshIterator constructor
    TriangleMeshIterator();

    //! Deconstructor
    virtual ~TriangleMeshIterator() = default;

    //! Copy constructor
    TriangleMeshIterator( const TriangleMeshIterator & );

    //! Assignment operator
    TriangleMeshIterator &operator=( const TriangleMeshIterator & );

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

    //! Access the list of elements
    auto getList() const { return d_list; }

    using MeshIterator::operator+;
    using MeshIterator::operator+=;

protected:
    /** Default constructor
     * \param mesh      Pointer to the libMesh mesh
     * \param list      List of elements
     * \param pos       Pointer to iterator with the current position
     */
    explicit TriangleMeshIterator( const AMP::Mesh::TriangleMesh<NG> *mesh,
                                   std::shared_ptr<const std::vector<ElementID>> list,
                                   size_t pos = 0 );

    //! Clone the iterator
    MeshIterator *clone() const override;

    friend class AMP::Mesh::TriangleMesh<NG>;

protected:
    // Data members
    const AMP::Mesh::TriangleMesh<NG> *d_mesh;
    std::shared_ptr<const std::vector<ElementID>> d_list;
    TriangleMeshElement<NG, TYPE> d_cur_element;
};


} // namespace AMP::Mesh

#endif
