#ifndef included_AMP_structuredMeshIterators
#define included_AMP_structuredMeshIterators

#include "AMP/mesh/MeshIterator.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/structured/structuredMeshElement.h"
#include <memory>

#include <array>


namespace AMP::Mesh {


class structuredMeshIterator final : public MeshIterator
{
public:
    //! Empty structuredMeshIterator constructor
    structuredMeshIterator();

    //! Range base constructor
    structuredMeshIterator( const BoxMesh::MeshElementIndex &first,
                            const BoxMesh::MeshElementIndex &last,
                            const AMP::Mesh::BoxMesh *mesh,
                            size_t pos = 0 );

    //! Range base constructor
    structuredMeshIterator( const BoxMesh::MeshElementIndexIterator &it,
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

    //! Return the class name
    std::string className() const override { return "structuredMeshIterator"; }

    //! Increment
    MeshIterator &operator++() override;

    //! Decrement
    MeshIterator &operator--() override;

    // Arithmetic operator+=
    MeshIterator &operator+=( int N ) override;

    //! Check if two iterators are equal
    bool operator==( const MeshIterator &rhs ) const override;

    //! Check if two iterators are not equal
    bool operator!=( const MeshIterator &rhs ) const override;

    //! Return an iterator to the begining
    MeshIterator begin() const override;

    //! Return an iterator to the begining
    MeshIterator end() const override;

    using MeshIterator::operator+;
    using MeshIterator::operator+=;

public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t fid ) const override;
    structuredMeshIterator( int64_t fid, AMP::IO::RestartManager *manager );

public: // Advanced interfaces
    //! Clone the iterator
    MeshIterator *clone() const override;

    // Get the elements in the iterator
    std::shared_ptr<const std::vector<BoxMesh::MeshElementIndex>> getElements() const;

    // Get the current index
    BoxMesh::MeshElementIndex getCurrentIndex() const;

private:
    // Data members
    BoxMesh::MeshElementIndexIterator d_it;
    std::shared_ptr<const std::vector<BoxMesh::MeshElementIndex>> d_elements;
    const AMP::Mesh::BoxMesh *d_mesh;
    mutable structuredMeshElement d_cur_element;
};


} // namespace AMP::Mesh

#endif
