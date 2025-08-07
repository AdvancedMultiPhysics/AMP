#ifndef included_AMP_ArrayVectorData
#define included_AMP_ArrayVectorData

#include <string>

#include "AMP/utils/Array.h"
#include "AMP/utils/FunctionTable.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/vectors/data/GhostDataHelper.hpp"
#include "AMP/vectors/data/VectorData.h"


namespace AMP::LinearAlgebra {

/** \brief A core-local vector
 * \details This is a Vector that implements the Vector interface for an AMP::Array.
 */
template<typename T, typename FUN = FunctionTable, typename Allocator = AMP::HostAllocator<void>>
class ArrayVectorData : public GhostDataHelper<T, Allocator>
{
private:
    AMP::Array<T, FUN, Allocator> d_array;
    AMP_MPI d_comm;
    size_t d_offset = 0;
    ArraySize d_blockIndex;
    ArraySize d_globalArraySize;

public:
    /** \brief    Create a ArrayVector
     * \details  This is the factory method for the ArrayVector.  It returns the shared pointer
     * to be used in the code
     * \param    localSize  The number of elements in the vector on this processor
     */
    static std::shared_ptr<ArrayVectorData> create( const ArraySize &localSize );

    /** \brief    Create a ArrayVector
     * \details  This is the factory method for the ArrayVector.  It returns the shared pointer
     * to be used in the code
     * \param    localSize  The number of elements in the vector on this processor
     * \param    blockIndex  The global number of elements in the vector on this processor
     * \param    comm The variable associated with the new vector
     */
    static std::shared_ptr<ArrayVectorData>
    create( const ArraySize &localSize, const ArraySize &blockIndex, AMP_MPI comm );

    static std::shared_ptr<ArrayVectorData>
    create( const size_t localSize, std::shared_ptr<CommunicationList> commList, T *data );

    //! Empty constructor
    ArrayVectorData() = default;

    //! Destructor
    virtual ~ArrayVectorData() {}

    //! Copy constructor
    ArrayVectorData( const ArrayVectorData & ) = delete;

    /** \brief  Return the communicator this Vector spans
     */
    const AMP_MPI &getComm() const override { return d_comm; }

    //! resize the ArrayVector and reset the internal data structures
    void resize( const ArraySize &localDims );

    //! return a non-const reference to the internal data container
    Array<T, FUN, Allocator> &getArray( void ) { return d_array; }

    //! return a const reference to the internal data container
    const Array<T, FUN, Allocator> &getArray( void ) const { return d_array; }

    /** \brief Number of blocks of contiguous data in the Vector
     * \return Number of blocks in the Vector
     * \details  A vector is not necessarily contiguous in memory.  This method
     * returns the number of contiguous blocks in memory used by this vector
     */
    size_t numberOfDataBlocks() const override { return 1; }

    /** \brief Number of elements in a data block
     * \param[in] i  particular data block
     * \return The size of a particular block
     */
    size_t sizeOfDataBlock( size_t i = 0 ) const override { return i == 0 ? d_array.length() : 0; }

    /**\brief Copy data into this vector
     *\param[in] buf  Buffer to copy from
     * \param[in] id   typeID of raw data
     */
    void putRawData( const void *buf, const typeID &id ) override;

    /**\brief Copy data out of this vector
     *\param[out] buf  Buffer to copy to
     * \param[in] id   typeID of raw data
     *\details The Vector should be pre-allocated to the correct size (getLocalSize())
     */
    void getRawData( void *buf, const typeID &id ) const override;

    /**
     * \brief Set values in the vector by their local offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     * \param[in] id   typeID of raw data
     * \details This will set the owned values for this core.  All indices are
     * from 0.
     * \f$ \mathit{this}_{\mathit{indices}_i} = \mathit{vals}_i \f$
     */
    void setValuesByLocalID( size_t num,
                             const size_t *indices,
                             const void *vals,
                             const typeID &id ) override;

    /**
     * \brief Add values to vector entities by their local offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     * \param[in] id   typeID of raw data
     * \details This will set the owned values for this core.  All indices are
     * from 0.
     * \f$ \mathit{this}_{\mathit{indices}_i} = \mathit{this}_{\mathit{indices}_i} +
     * \mathit{vals}_i \f$
     */
    void addValuesByLocalID( size_t num,
                             const size_t *indices,
                             const void *vals,
                             const typeID &id ) override;

    /**
     * \brief Get local values in the vector by their lcoal offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[out] vals the values to place in the vector
     * \param[in] id   typeID of raw data
     * \details This will get any value owned by this core.
     */
    void getValuesByLocalID( size_t num,
                             const size_t *indices,
                             void *vals,
                             const typeID &id ) const override;

    /**\brief  A unique id for the underlying data allocation
     *\details This is a unique id that is associated with the data
     *   data allocation.  Views of a vector should preserve the id of
     *   the original vector.  Vectors that are not allocated, or contain
     *   multiple vectors (such as Multivector) should return 0.
     *   Note: this id is not consistent across multiple processors.
     */
    uint64_t getDataID() const override { return reinterpret_cast<uint64_t>( d_array.data() ); }

    //! Get the type name
    std::string VectorDataName() const override { return "ArrayVector"; }

protected:
    /** \brief Return a pointer to a particular block of memory in the
     * vector
     * \param i The block to return
     */
    void *getRawDataBlockAsVoid( size_t i ) override
    {
        AMP_ASSERT( i == 0 );
        return d_array.data();
    }

    /** \brief Return a pointer to a particular block of memory in the
     * vector
     * \param i The block to return
     */
    const void *getRawDataBlockAsVoid( size_t i ) const override
    {
        AMP_ASSERT( i == 0 );
        return d_array.data();
    }

    typeID getType( size_t ) const override { return getTypeID<T>(); }
    size_t sizeofDataBlockType( size_t ) const override { return sizeof( T ); }
    void swapData( VectorData & ) override;
    std::shared_ptr<VectorData> cloneData( const std::string &name = "" ) const override;
};

} // namespace AMP::LinearAlgebra

#include "AMP/vectors/data/ArrayVectorData.hpp"

#endif
