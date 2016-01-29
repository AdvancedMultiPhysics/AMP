#ifndef included_AMP_structuredMeshIterators
#define included_AMP_structuredMeshIterators

#include "ampmesh/MeshIterator.h"
#include "ampmesh/structured/BoxMesh.h"
#include "ampmesh/structured/structuredMeshElement.h"
#include "utils/shared_ptr.h"

namespace AMP {
namespace Mesh {


class structuredMeshIterator : public MeshIterator
{
public:
    //! Empty MultiVectorIterator constructor
    structuredMeshIterator();

    //! Default MultiVectorIterator constructor
    structuredMeshIterator( AMP::shared_ptr<std::vector<BoxMesh::MeshElementIndex>> elements,
                            const AMP::Mesh::BoxMesh *mesh,
                            size_t pos = 0 );

    //! Deconstructor
    virtual ~structuredMeshIterator();

    //! Copy constructor
    structuredMeshIterator( const structuredMeshIterator & );

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

    //! Dereference the iterator
    virtual MeshElement &operator*( void ) override;

    //! Dereference the iterator
    virtual MeshElement *operator->( void ) override;

    //! Return an iterator to the begining
    virtual MeshIterator begin() const override;

    //! Return an iterator to the begining
    virtual MeshIterator end() const override;

    //! Return the number of elements in the iterator
    virtual size_t size() const override;

    //! Return the current position (from the beginning) in the iterator
    virtual size_t position() const override;

    using MeshIterator::operator+;
    using MeshIterator::operator+=;

protected:
    //! Clone the iterator
    virtual MeshIterator *clone() const override;

    friend class AMP::Mesh::BoxMesh;

private:
    // Data members
    size_t d_pos;
    AMP::shared_ptr<std::vector<BoxMesh::MeshElementIndex>> d_elements;
    const AMP::Mesh::BoxMesh *d_mesh;
    structuredMeshElement d_cur_element;
};
}
}

#endif
