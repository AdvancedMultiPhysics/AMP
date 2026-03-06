#ifndef included_AMP_MeshListIterator
#define included_AMP_MeshListIterator

#include "AMP/mesh/MeshIterator.h"
#include "AMP/utils/Utilities.h"

#include <iterator>
#include <memory>


namespace AMP::Mesh {


/**
 * \class MeshListIterator
 * \brief A class used to iterate over a set of mesh elements.
 * \details  This class provides routines for iterating over a set
 * of mesh elments that are in a std::vector.
 */
template<class TYPE = MeshElement>
class MeshListIterator final : public MeshIteratorBase
{
public:
    static MeshIterator create( std::shared_ptr<std::vector<TYPE>> elements, size_t pos = 0 );

    //! Empty MeshListIterator constructor
    MeshListIterator();

    //! Default MeshListIterator constructor
    MeshListIterator( std::shared_ptr<std::vector<TYPE>> elements, size_t pos = 0 );

    //! Deconstructor
    ~MeshListIterator() = default;

    //! Copy constructor
    MeshListIterator( const MeshListIterator & );

    //! Move constructor
    MeshListIterator( MeshListIterator && ) = delete;

    //! Assignment operator
    MeshListIterator &operator=( const MeshListIterator & );

    //! Move operator
    MeshListIterator &operator=( MeshListIterator && ) = delete;

    //! Return the class name
    std::string className() const override { return "MeshListIterator"; }

    //! Set the position in the iterator
    void setPos( size_t ) override;

    //! Increment
    MeshIteratorBase &operator++() override;

    //! Decrement
    MeshIteratorBase &operator--() override;

    // Arithmetic operator+=
    MeshIteratorBase &operator+=( int N ) override;

    //! Check if two iterators are equal
    bool operator==( const MeshIteratorBase &rhs ) const override;

    //! Check if two iterators are not equal
    bool operator!=( const MeshIteratorBase &rhs ) const override;

    //! Return an iterator to the begining
    MeshIterator begin() const override;

    using MeshIteratorBase::operator==;
    using MeshIteratorBase::operator!=;


public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager * ) const override;
    void writeRestart( int64_t ) const override;
    MeshListIterator( int64_t, AMP::IO::RestartManager * );


protected:
    //! Clone the iterator
    std::unique_ptr<MeshIteratorBase> clone() const override;


private:
    // A pointer to a std::vector containing the desired mesh elements
    std::shared_ptr<std::vector<TYPE>> d_elements;
};

} // namespace AMP::Mesh

#endif
