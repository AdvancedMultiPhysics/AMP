#ifndef included_AMP_TriangleMeshIterators
#define included_AMP_TriangleMeshIterators


#include "AMP/mesh/MeshIterator.h"
#include "AMP/mesh/triangle/TriangleMeshElement.h"


namespace AMP::Mesh {


template<uint8_t NG>
class TriangleMesh;


template<uint8_t NG>
class TriangleMeshIterator final : public MeshIteratorBase
{
public:
    //! Empty MeshIterator constructor
    TriangleMeshIterator();

    /** Default constructor
     * \param mesh      Pointer to the libMesh mesh
     * \param list      List of elements
     * \param pos       Pointer to iterator with the current position
     */
    explicit TriangleMeshIterator( const AMP::Mesh::TriangleMesh<NG> *mesh,
                                   std::shared_ptr<const std::vector<ElementID>> list,
                                   size_t pos = 0 );

    //! Deconstructor
    virtual ~TriangleMeshIterator() = default;

    //! Copy constructor
    TriangleMeshIterator( const TriangleMeshIterator & );

    //! Assignment operator
    TriangleMeshIterator &operator=( const TriangleMeshIterator & );

    //! Return the class name
    std::string className() const override;

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

    //! Access the list of elements
    auto getList() const { return d_list; }

    //! Clone the iterator
    std::unique_ptr<MeshIteratorBase> clone() const override;

    using MeshIteratorBase::operator==;
    using MeshIteratorBase::operator!=;


public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager * ) const override;
    void writeRestart( int64_t ) const override;
    TriangleMeshIterator( int64_t, AMP::IO::RestartManager * );


protected:
    // Data members
    const AMP::Mesh::TriangleMesh<NG> *d_mesh;
    std::shared_ptr<const std::vector<ElementID>> d_list;
    TriangleMeshElement<NG> d_cur_element;
};


} // namespace AMP::Mesh

#endif
