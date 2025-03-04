#ifndef included_AMP_VectorDataDefault
#define included_AMP_VectorDataDefault

#include "AMP/utils/UtilityMacros.h"
#include "AMP/utils/memory.h"
#include "AMP/vectors/data/VectorData.h"


namespace AMP::LinearAlgebra {


template<typename TYPE>
class VectorDataIterator;


/**
 * \brief  A class used to hold vector data
 * \details  VectorDataDefault is a default implementation of VectorData that stores
 * the local values as a single block of data on the CPU.
 */
template<typename TYPE = double, class Allocator = AMP::HostAllocator<void>>
class VectorDataDefault final : public VectorData
{
public: // Member types
    using value_type = TYPE;
    using scalarAllocator_t =
        typename std::allocator_traits<Allocator>::template rebind_alloc<TYPE>;

public: // Constructors
    VectorDataDefault( size_t start, size_t localSize, size_t globalSize );

    VectorDataDefault( const VectorDataDefault & ) = delete;

public: // Virtual functions
    //! Virtual destructor
    virtual ~VectorDataDefault();

    //! Get the type name
    std::string VectorDataName() const override;

    /** \brief Number of blocks of contiguous data in the Vector
     * \return Number of blocks in the Vector
     * \details  A vector is not necessarily contiguous in memory.  This method
     * returns the number of contiguous blocks in memory used by this vector
     */
    size_t numberOfDataBlocks() const override;

    /** \brief Number of elements in a data block
     * \param[in] i  particular data block
     * \return The size of a particular block
     */
    size_t sizeOfDataBlock( size_t i = 0 ) const override;


    /**\brief Copy data into this vector
     *\param[in] buf  Buffer to copy from
     * \param[in] id   typeID of raw data
     */
    void putRawData( const void *buf, const typeID &id ) override;

    /**\brief Copy data out of this vector
     * \param[out] buf  Buffer to copy to
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
     * \brief Get values to vector entities by their local offset
     * \param[in] num  number of values to get
     * \param[in] indices the indices of the values to get
     * \param[in] vals the values to place in the vector
     * \param[in] id   typeID of raw data
     * \details This will get the owned values for this core.  All indices are
     * from 0.
     * \f$ \mathit{this}_{\mathit{indices}_i} = \mathit{this}_{\mathit{indices}_i} +
     * \mathit{vals}_i \f$
     */
    void getValuesByLocalID( size_t num,
                             const size_t *indices,
                             void *vals,
                             const typeID &id ) const override;


public: // Advanced virtual functions
    /**\brief  A unique id for the underlying data allocation
     *\details This is a unique id that is associated with the data
     *   data allocation.  Views of a vector should preserve the id of
     *   the original vector.  Vectors that are not allocated, or contain
     *   multiple vectors (such as Multivector) should return 0.
     *   Note: this id is not consistent across multiple processors.
     */
    uint64_t getDataID() const override;

    /** \brief Return a pointer to a particular block of memory in the vector
     * \param i The block to return
     */
    void *getRawDataBlockAsVoid( size_t i ) override;

    /** \brief Return a pointer to a particular block of memory in the
     * vector
     * \param i        The block to return
     */
    const void *getRawDataBlockAsVoid( size_t i ) const override;

    /** \brief Return the result of sizeof(TYPE) for the given data block
     * \param i The block to return
     */
    size_t sizeofDataBlockType( size_t i ) const override;

    /** \brief Return the typeid of the given block
     * \param block    The block id to check
     */
    typeID getType( size_t block ) const override;

    /** \brief Swap the data with another VectorData object
     * \param rhs      The VectorData to swap with
     */
    void swapData( VectorData &rhs ) override;

    /** \brief Clone the data
     */
    std::shared_ptr<VectorData> cloneData( const std::string &name = "" ) const override;

public: // Non-virtual functions
    /** \brief Access the raw element
     * \param i        The element to return (local index)
     */
    TYPE &operator[]( size_t i );

    /** \brief Access the raw element
     * \param i        The element to return (local index)
     */
    const TYPE &operator[]( size_t i ) const;


public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t ) const override;
    VectorDataDefault( int64_t, AMP::IO::RestartManager * );

private:
    TYPE *d_data = nullptr;
    scalarAllocator_t d_alloc;
};


} // namespace AMP::LinearAlgebra


#endif
