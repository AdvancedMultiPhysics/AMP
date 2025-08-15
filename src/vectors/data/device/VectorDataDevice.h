#ifndef included_AMP_VectorDataDevice
#define included_AMP_VectorDataDevice

#include "AMP/vectors/data/VectorDataDefault.h"


namespace AMP::LinearAlgebra {


/**
 * \brief  A class used to hold vector data
 * \details  VectorDataDefault is a default implementation of VectorData that stores
 * the local values as a single block of data on the CPU.
 */
template<typename TYPE = double, class Allocator = AMP::HostAllocator<void>>
class VectorDataDevice final : public VectorDataDefault<TYPE, Allocator>
{
public: // Member types
    using value_type = TYPE;
    using scalarAllocator_t =
        typename std::allocator_traits<Allocator>::template rebind_alloc<TYPE>;
    using idxAllocator_t = typename std::allocator_traits<Allocator>::template rebind_alloc<size_t>;

public: // Constructors
    VectorDataDevice( size_t start, size_t localSize, size_t globalSize );

    VectorDataDevice( const VectorDataDevice & ) = delete;

public: // Virtual functions
    //! Virtual destructor
    virtual ~VectorDataDevice();

    //! Get the type name
    std::string VectorDataName() const override;

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

    /** \brief Clone the data
     */
    std::shared_ptr<VectorData> cloneData( const std::string &name = "" ) const override;

public: // Functions overloaded from GhostDataHelpers and in turn VectorData
    void fillGhosts( const Scalar & ) override;
    bool allGhostIndices( size_t N, const size_t *ndx ) const override;
    bool containsGlobalElement( size_t ) const override;
    void setGhostValuesByGlobalID( size_t, const size_t *, const void *, const typeID & ) override;
    void addGhostValuesByGlobalID( size_t, const size_t *, const void *, const typeID & ) override;
    void getGhostValuesByGlobalID( size_t, const size_t *, void *, const typeID & ) const override;
    void
    getGhostAddValuesByGlobalID( size_t, const size_t *, void *, const typeID & ) const override;

    using VectorData::addGhostValuesByGlobalID;
    using VectorData::getGhostAddValuesByGlobalID;
    using VectorData::getGhostValuesByGlobalID;
    using VectorData::setGhostValuesByGlobalID;

public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t ) const override;
    VectorDataDevice( int64_t, AMP::IO::RestartManager * );

private:
    void setScratchSpace( size_t N ) const;
    std::tuple<bool, size_t *, void *> copyToScratchSpace( size_t num,
                                                           const size_t *indices_,
                                                           const void *vals_,
                                                           const typeID &id ) const;

    mutable idxAllocator_t d_idx_alloc;
    mutable scalarAllocator_t d_scalar_alloc;

    mutable size_t d_scratchSize   = 0;
    mutable size_t *d_idx_scratch  = nullptr;
    mutable TYPE *d_scalar_scratch = nullptr;
};


} // namespace AMP::LinearAlgebra


#endif
