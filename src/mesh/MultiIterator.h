#ifndef included_AMP_MultiIterator
#define included_AMP_MultiIterator

#include "AMP/mesh/MeshIterator.h"

#include <memory>


namespace AMP::Mesh {


class MultiMesh;


/**
 * \class MultiIterator
 * \brief A class used to combine multiple iterators
 * \details  This class provides routines for iterating over a mesh.  More
 *  specifically, this class combines multiple iterators into one.  This
 *  is primarily needed for MultiMesh, but may be used for other applicaitons.
 */
class MultiIterator final : public MeshIteratorBase
{
public:
    using MeshIteratorPtr = std::unique_ptr<MeshIteratorBase>;


public:
    //! Empty MultiIterator constructor
    MultiIterator();

    //! Default MultiIterator constructor
    explicit MultiIterator( std::vector<MeshIterator> iterators, size_t pos = 0 );

    //! Deconstructor
    virtual ~MultiIterator();

    //! Move constructor
    MultiIterator( MultiIterator && );

    //! Copy constructor
    MultiIterator( const MultiIterator & ) = delete;

    //! Move operator
    MultiIterator &operator=( MultiIterator && );

    //! Assignment operator
    MultiIterator &operator=( const MultiIterator & ) = delete;

    //! Return the class name
    std::string className() const override { return "MultiIterator"; }

    //! Set the position in the iterator
    void setPos( size_t ) override;

    // Increment
    MeshIteratorBase &operator++() override;

    // Decrement
    MeshIteratorBase &operator--() override;

    // Arithmetic operator+=
    MeshIteratorBase &operator+=( int N ) override;

    //! Check if two iterators are equal
    bool operator==( const MeshIteratorBase &rhs ) const override;

    //! Check if two iterators are not equal
    bool operator!=( const MeshIteratorBase &rhs ) const override;

    //! Return an iterator to the begining
    MeshIterator begin() const override;

    //! Clone the iterator
    std::unique_ptr<MeshIteratorBase> clone() const override;

    using MeshIteratorBase::operator==;
    using MeshIteratorBase::operator!=;


public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t fid ) const override;
    MultiIterator( int64_t fid, AMP::IO::RestartManager *manager );


public: // Advanced interfaces (use with caution)
    explicit MultiIterator( std::vector<MeshIteratorBase *> &&iterators, size_t pos = 0 );


protected:
    friend class MultiMesh;


protected:
    // Data members
    size_t d_localPos    = 0;
    size_t d_iteratorNum = 0;
    std::vector<MeshIteratorBase *> d_iterators;
};


} // namespace AMP::Mesh

#endif
