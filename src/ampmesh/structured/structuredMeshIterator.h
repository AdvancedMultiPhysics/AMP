#ifndef included_AMP_structuredMeshIterators
#define included_AMP_structuredMeshIterators

#include "AMP/ampmesh/MeshIterator.h"
#include "AMP/ampmesh/structured/BoxMesh.h"
#include "AMP/ampmesh/structured/structuredMeshElement.h"
#include <memory>

#include <array>


namespace AMP {
namespace Mesh {


class structuredMeshIterator final : public MeshIterator
{
public:
    //! Empty MultiVectorIterator constructor
    structuredMeshIterator();

    //! Range base constructor
    structuredMeshIterator( const BoxMesh::MeshElementIndex &first,
                            const BoxMesh::MeshElementIndex &last,
                            const AMP::Mesh::BoxMesh *mesh,
                            size_t pos = 0 );

    //! Element list constructor
    structuredMeshIterator( std::shared_ptr<const std::vector<BoxMesh::MeshElementIndex>> elements,
                            const AMP::Mesh::BoxMesh *mesh,
                            size_t pos = 0 );

    //! Deconstructor
    virtual ~structuredMeshIterator();

    //! Move constructor
    structuredMeshIterator( structuredMeshIterator && ) = default;

    //! Copy constructor
    structuredMeshIterator( const structuredMeshIterator & );

    //! Move operator
    structuredMeshIterator &operator=( structuredMeshIterator && ) = default;

    //! Assignment operator
    structuredMeshIterator &operator=( const structuredMeshIterator & );

    //! Increment
    virtual MeshIterator &operator++() override;

    //! Increment
    virtual MeshIterator operator++( int ) override;

    //! Decrement
    virtual MeshIterator &operator--() override;

    //! Decrement
    virtual MeshIterator operator--( int ) override;

    // Arithmetic operator+
    virtual MeshIterator operator+( int ) const override;

    // Arithmetic operator+=
    virtual MeshIterator &operator+=( int N ) override;

    //! Check if two iterators are equal
    virtual bool operator==( const MeshIterator &rhs ) const override;

    //! Check if two iterators are not equal
    virtual bool operator!=( const MeshIterator &rhs ) const override;

    //! Return an iterator to the begining
    virtual MeshIterator begin() const override;

    //! Return an iterator to the begining
    virtual MeshIterator end() const override;

    using MeshIterator::operator+;
    using MeshIterator::operator+=;

protected:
    //! Clone the iterator
    virtual MeshIterator *clone() const override;

    // Get the index given the position
    inline BoxMesh::MeshElementIndex getIndex( int pos ) const;

    // Get the elements in the iterator
    std::shared_ptr<const std::vector<BoxMesh::MeshElementIndex>> getElements() const;

    friend class AMP::Mesh::BoxMesh;

private:
    // Data members
    std::array<bool, 3> d_isPeriodic;
    std::array<int, 3> d_globalSize;
    BoxMesh::MeshElementIndex d_first;
    BoxMesh::MeshElementIndex d_last;
    std::shared_ptr<const std::vector<BoxMesh::MeshElementIndex>> d_elements;
    const AMP::Mesh::BoxMesh *d_mesh;
    mutable structuredMeshElement d_cur_element;

private:
    static constexpr uint32_t getTypeID()
    {
        return AMP::Utilities::hash_char( "structuredMeshIterator" );
    }
};


} // namespace Mesh
} // namespace AMP

#endif
