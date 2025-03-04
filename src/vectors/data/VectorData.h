#ifndef included_AMP_VectorData
#define included_AMP_VectorData

#include "AMP/utils/enable_shared_from_this.h"
#include "AMP/utils/typeid.h"
#include "AMP/vectors/CommunicationList.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/data/DataChangeFirer.h"

#include <memory>


namespace AMP::IO {
class RestartManager;
}


namespace AMP::LinearAlgebra {


template<typename TYPE>
class VectorDataIterator;


/**
 * \brief  A class used to hold vector data
 * \details  VectorData is a class to helping disassociate data storage
 *    and vector operations such as dot product, norms, etc.
 *    When indexing VectorData (and Vector) the local index runs from
 *       [0,getLocalSize()) and the global indexing (for the local elements)
 *       runs from [getLocalStartID(),getLocalStartID+getLocalSize())
 */

class VectorData : public DataChangeFirer, public AMP::enable_shared_from_this<VectorData>
{
public: // Get basic information
    //! Virtual destructor
    virtual ~VectorData() {}


    //! Get the type name
    virtual std::string VectorDataName() const = 0;


    /** \brief Number of blocks of contiguous data in the Vector
     * \return Number of blocks in the Vector
     * \details  A vector is not necessarily contiguous in memory.  This method
     * returns the number of contiguous blocks in memory used by this vector
     */
    virtual size_t numberOfDataBlocks() const = 0;

    /** \brief Number of elements in a data block
     * \param[in] i  particular data block
     * \return The size of a particular block
     */
    virtual size_t sizeOfDataBlock( size_t i = 0 ) const = 0;

    /**\brief Number of elements "owned" by this core
     *\return  Number of entries stored on this processor
     *\details  This function returns the number of locally stored values
     *    under contiguous indexing.
     */
    inline size_t getLocalSize() const { return d_localSize; }

    /**\brief Number of total entries in this vector across all cores
     *\return Number of entries stored across all cores in this
     */
    inline size_t getGlobalSize() const { return d_globalSize; }

    /**\brief get local start id core.
     *\return The first entry "owned" by this core
     */
    inline size_t getLocalStartID() const { return d_localStart; }

    /**\brief Number of entries "owned" by other cores stored on this
     * core.
     *\return Number of entries "owned" by other cores stored on this core
     */
    virtual size_t getGhostSize() const;

    /**\brief Check if any entries "owned" by other cores are stored on this core
     *\return True if any entries "owned" by other cores are stored on this core
     */
    virtual bool hasGhosts( void ) { return ( d_Ghosts != nullptr ); }

    /**\brief Gets vector of entries "owned" by other cores that are stored on this core
     *\return Vector of entries "owned" by other cores that are stored on this core
     */
    virtual std::vector<double> &getGhosts() const { return *d_Ghosts; }

    //! Return integer number of data components
    virtual size_t getNumberOfComponents() const;

    /** \brief Number of elements in the ith component
     * \param[in] i  particular data block
     * \return The number of elements in the ith component
     */
    virtual std::shared_ptr<VectorData> getComponent( size_t i = 0 );

    /** \brief Number of elements in the ith component
     * \param[in] i  particular data block
     * \return The number of elements in the ith component
     */
    virtual std::shared_ptr<const VectorData> getComponent( size_t i = 0 ) const;

    //! returns whether all data for the vector on a single process is contiguous
    virtual bool hasContiguousData() const { return true; }

public: // Get/Set data
    /**\brief Copy data into this vector
     *\param[in] buf  Buffer to copy from
     */
    template<class TYPE>
    void putRawData( const TYPE *buf );

    /**\brief Copy data out of this vector
     *\param[out] buf  Buffer to copy to
     *\details The Vector should be pre-allocated to the correct size (getLocalSize())
     */
    template<class TYPE>
    void getRawData( TYPE *buf ) const;

    /**
     * \brief Set values in the vector by their local offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     * \details This will set the owned values for this core.  All indices are
     * from 0.
     * \f$ \mathit{this}_{\mathit{indices}_i} = \mathit{vals}_i \f$
     */
    template<class TYPE>
    void setValuesByLocalID( size_t num, const size_t *indices, const TYPE *vals );

    /**
     * \brief Set owned values using global identifier
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     *
     * \f$ \mathit{this}_{\mathit{indices}_i} = \mathit{vals}_i \f$
     */
    template<class TYPE>
    void setLocalValuesByGlobalID( size_t num, const size_t *indices, const TYPE *vals );

    /**
     * \brief Set ghost values using global identifier
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     *
     * \f$ \mathit{this}_{\mathit{indices}_i} = \mathit{vals}_i \f$
     */
    template<class TYPE>
    void setGhostValuesByGlobalID( size_t num, const size_t *indices, const TYPE *vals );

    /**
     * \brief Set owned or shared values using global identifier
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     * \details Since the shared buffer and owned buffer are separate,
     * this function must sort the data by buffer before setting
     * values.
     */
    template<class TYPE>
    void setValuesByGlobalID( size_t num, const size_t *indices, const TYPE *vals );

    /**
     * \brief Add values to vector entities by their local offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     * \details This will set the owned values for this core.  All indices are
     * from 0.
     * \f$ \mathit{this}_{\mathit{indices}_i} = \mathit{this}_{\mathit{indices}_i} +
     * \mathit{vals}_i \f$
     */
    template<class TYPE>
    void addValuesByLocalID( size_t num, const size_t *indices, const TYPE *vals );

    /**
     * \brief Add owned values using global identifier
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     *
     * \f$ \mathit{this}_{\mathit{indices}_i} = \mathit{this}_{\mathit{indices}_i} +
     * \mathit{vals}_i \f$
     */
    template<class TYPE>
    void addLocalValuesByGlobalID( size_t num, const size_t *indices, const TYPE *vals );

    /**
     * \brief Add owned or shared values using global identifier
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     * \details Since the shared buffer and owned buffer are separate,
     * this function must sort the data by buffer before setting
     * values.
     */
    template<class TYPE>
    void addValuesByGlobalID( size_t num, const size_t *indices, const TYPE *vals );

    /**
     * \brief Add shared values using global identifier
     * \param[in] num  number of values to add
     * \param[in] indices the indices of the values to add
     * \param[in] vals the values to place in the vector
     */
    template<class TYPE>
    void addGhostValuesByGlobalID( size_t num, const size_t *indices, const TYPE *vals );


    /**
     * \brief get ghosted values to add to off-proc elements
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     * \details This will get the ghosted updates this processor has made.  All indices are
     * from global 0.
     */
    template<class TYPE>
    void getGhostAddValuesByGlobalID( size_t num, const size_t *indices, TYPE *vals ) const;


    /**
     * \brief Get local values in the vector by their global offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[out] vals the values to place in the vector
     * \details This will get any value used by this core.
     */
    template<class TYPE>
    void getValuesByLocalID( size_t num, const size_t *indices, TYPE *vals ) const;

    /**
     * \brief get values in the vector by their local offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[out] vals the values to place in the vector
     * \details This will get the owned values for this core.  All indices are
     * from 0.
     */
    template<class TYPE>
    void getValuesByGlobalID( size_t num, const size_t *indices, TYPE *vals ) const;

    /**
     * \brief Get local values in the vector by their global offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[out] vals the values to place in the vector
     * \details This will get any value owned by this core.
     */
    template<class TYPE>
    void getLocalValuesByGlobalID( size_t num, const size_t *indices, TYPE *vals ) const;

    /**
     * \brief Get ghost values in the vector by their global offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[out] vals the values to place in the vector
     * \details This will get any value owned by this core.
     */
    template<class TYPE>
    void getGhostValuesByGlobalID( size_t num, const size_t *indices, TYPE *vals ) const;


public: // Advanced (virtual) get/set values
    /**\brief Copy data into this vector
     * \param[in] buf  Buffer to copy from
     * \param[in] id   typeID of raw data
     */
    virtual void putRawData( const void *buf, const typeID &id ) = 0;

    /**\brief Copy data out of this vector
     * \param[out] buf  Buffer to copy to
     * \param[in] id   typeID of raw data
     * \details The Vector should be pre-allocated to the correct size (getLocalSize())
     */
    virtual void getRawData( void *buf, const typeID &id ) const = 0;

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
    virtual void
    setValuesByLocalID( size_t num, const size_t *indices, const void *vals, const typeID &id ) = 0;

    /**
     * \brief Set ghost values using global identifier
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     * \param[in] id   typeID of raw data
     *
     * \f$ \mathit{this}_{\mathit{indices}_i} = \mathit{vals}_i \f$
     */
    virtual void setGhostValuesByGlobalID( size_t num,
                                           const size_t *indices,
                                           const void *vals,
                                           const typeID &id );

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
    virtual void
    addValuesByLocalID( size_t num, const size_t *indices, const void *vals, const typeID &id ) = 0;

    /**
     * \brief Add shared values using global identifier
     * \param[in] num  number of values to add
     * \param[in] indices the indices of the values to add
     * \param[in] vals the values to place in the vector
     * \param[in] id   typeID of raw data
     */
    virtual void addGhostValuesByGlobalID( size_t num,
                                           const size_t *indices,
                                           const void *vals,
                                           const typeID &id );

    /**
     * \brief get ghosted values to add to off-proc elements
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[in] vals the values to place in the vector
     * \param[in] id   typeID of raw data
     * \details This will get the ghosted updates this processor has made.  All indices are
     * from global 0.
     */
    virtual void getGhostAddValuesByGlobalID( size_t num,
                                              const size_t *indices,
                                              void *vals,
                                              const typeID &id ) const;


    /**
     * \brief Get local values in the vector by their global offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[out] vals the values to place in the vector
     * \param[in] id   typeID of raw data
     * \details This will get any value used by this core.
     */
    virtual void
    getValuesByLocalID( size_t num, const size_t *indices, void *vals, const typeID &id ) const = 0;

    /**
     * \brief Get ghost values in the vector by their global offset
     * \param[in] num  number of values to set
     * \param[in] indices the indices of the values to set
     * \param[out] vals the values to place in the vector
     * \param[in] id   typeID of raw data
     * \details This will get any value owned by this core.
     */
    virtual void getGhostValuesByGlobalID( size_t num,
                                           const size_t *indices,
                                           void *vals,
                                           const typeID &id ) const;


public: // Advanced functions
    /**\brief  A unique id for the underlying data allocation
     *\details This is a unique id that is associated with the data
     *   data allocation.  Views of a vector should preserve the id of
     *   the original vector.  Vectors that are not allocated, or contain
     *   multiple vectors (such as Multivector) should return 0.
     *   Note: this id is not consistent across multiple processors.
     */
    virtual uint64_t getDataID() const = 0;

    /** \brief Return a pointer to a particular block of memory in the vector
     * \param i The block to return
     */
    virtual void *getRawDataBlockAsVoid( size_t i ) = 0;

    /** \brief Return a pointer to a particular block of memory in the
     * vector
     * \param i The block to return
     */
    virtual const void *getRawDataBlockAsVoid( size_t i ) const = 0;

    /** \brief Return the result of sizeof(TYPE) for the given data block
     * \param i The block to return
     */
    virtual size_t sizeofDataBlockType( size_t i ) const = 0;

    /** \brief Return the typeid of the given block
     * \param block    The block id to check
     */
    virtual typeID getType( size_t block ) const = 0;

    /** \brief Is the data of the given type
     * \param id       The typeid
     * \param block    The block id to check
     */
    inline bool isType( const typeID &id, size_t block ) const;

    /** \brief Swap the data with another VectorData object
     * \param rhs      The VectorData to swap with
     */
    virtual void swapData( VectorData &rhs ) = 0;

    /** \brief Clone the data
     */
    virtual std::shared_ptr<VectorData> cloneData( const std::string &name = "" ) const = 0;

    /** \brief Associate the ghost buffer of a Vector with this Vector
     * \param in  The Vector to share a ghost buffer with
     */
    void aliasGhostBuffer( std::shared_ptr<VectorData> in );

    /** \brief Check if the two VectorData objects are alias of each other
     * \details  This function checks if two VectorData objects are alias of each other.
     *     Two VectorData objects are alias if their data blocks are the same size and
     *     point to the same memory blocks.
     * \param[in] rhs  VectorData to compare
     * \see makeConsistent
     */
    virtual bool isAnAliasOf( const VectorData &rhs ) const;

    /** \brief reset the vector data object
     * \details At present this interface is primarily meant for vector data over
     * AMR hierarchies, where the number of AMR levels and vector data has to be
     * reallocated
     */
    virtual void reset();

public:
    /** \brief Write owned data to an std::ostream
     * \param[in] out  The output stream to write to.
     * \param[in] GIDoffset  A number to add to the global ID when writing information
     * \param[in] LIDoffset  A number to add to the local ID when writing information
     */
    virtual void
    dumpOwnedData( std::ostream &out, size_t GIDoffset = 0, size_t LIDoffset = 0 ) const;

    /** \brief Write data owned by other processors to an std::ostream
     * \param[in] out  The output stream to write to.
     * \param[in] offset  A number to add to the global ID when writing information
     */
    virtual void dumpGhostedData( std::ostream &out, size_t offset = 0 ) const;

public: // Virtual functions dealing with the update status
    /** \brief  Return the current update state of the Vector
     * \details  This returns the effective update state of the vector,
     *    including any vectors it contains.  The effective state is defined as:
     *       UNCHANGED - All data and sub vectors are unchanged
     *       LOCAL_CHANGED - Local data may be modified, sub vectors must either
     *                be UNCHANGED or LOCAL_CHANGED.
     *       ADDING - Local and ghost data may be modified through add operations,
     *                sub vectors must be UNCHANGED, LOCAL_CHANGED, or ADDING
     *       SETTING - Local and ghost data may be modified through set operations,
     *                sub vectors must be UNCHANGED, LOCAL_CHANGED, or SETTING
     *   If different subvectors have incompatible states ADDING and SETTING,
     *   this function will return MIXED
     *   This version returns the local state only and does not involve communication
     */
    virtual UpdateState getLocalUpdateStatus() const;


    /** \brief  Return the current update state of the Vector
     * \details  This returns the effective update state of the vector,
     *    including any vectors it contains.  The effective state is defined as:
     *       UNCHANGED - All data and sub vectors are unchanged
     *       LOCAL_CHANGED - Local data may be modified, sub vectors must either
     *                be UNCHANGED or LOCAL_CHANGED.
     *       ADDING - Local and ghost data may be modified through add operations,
     *                sub vectors must be UNCHANGED, LOCAL_CHANGED, or ADDING
     *       SETTING - Local and ghost data may be modified through set operations,
     *                sub vectors must be UNCHANGED, LOCAL_CHANGED, or SETTING
     *   If different subvectors have incompatible states ADDING and SETTING,
     *   this function will return MIXED
     *   This version returns the global state and requires a collective communication
     */
    UpdateState getGlobalUpdateStatus() const;


    /** \brief  Sets the current update state of the Vector
     * \details  This sets the update status of the vector.
     * This function should only be called by advanced users
     * \param[in] state  State of the vector to set
     */
    virtual void setUpdateStatus( UpdateState state );

    /**
     * \brief Update shared values on entire communicator
     * \param t The type of scatter used to compute values
     * \details  There are two algorithms used by makeConsistent
     * - If t = CONSISTENT_SET, then owned values are
     *   sent to processors that share the value.  Shared values are
     *   overwritten
     * - If t = CONSISTENT_ADD, then shared values are accumulated
     *   on the core that owns it and applied as determined, either
     *   add or set.  Then, the values are broadcast out.
     *
     * Generally, when adding to a vector, the GATHER_SCATTER should
     * be used to make consistent.  When setting entries in a vector
     * the BROADCAST should be used.
     */
    virtual void makeConsistent( ScatterType t );

    /**
     * \brief Update shared values on entire communicator
     * \details  This version will check the state of the vector and then
     *   call the appropriate version of makeConsistent
     */
    virtual void makeConsistent();

    //! Get the communicator
    virtual AMP_MPI getComm() const;

    virtual bool hasComm( void ) const { return ( d_CommList != nullptr ); }

    //! Get the CommunicationList for this Vector
    virtual std::shared_ptr<CommunicationList> getCommunicationList() const;

    /**\brief Set the CommunicationList for this Vector
     *\details  Setting the CommunicationList for a Vector may involve
     * reallocating ghost storage.
     */
    virtual void setCommunicationList( std::shared_ptr<CommunicationList> comm );

    virtual void
    print( std::ostream &os, const std::string &name = "A", const std::string &prefix = "" ) const;

public: // Non-virtual functions
    /**
     * \brief Return an iterator to the beginning of the data
     * \returns A VectorDataIterator
     * \details Since the Vector presents an interface to a contiguous
     *     block of data, it is natural for it to provide a random
     *     access iterator.
     * \warning The non-const version of the iterators will automatically
     *     leave the vector in a non-consistent state.  The user may
     *     be required to call makeConsistent.
     */
    template<class TYPE = double>
    inline VectorDataIterator<TYPE> begin();

    /// @copydoc VectorData::begin()
    template<class TYPE = double>
    inline VectorDataIterator<const TYPE> begin() const;

    /// @copydoc VectorData::begin()
    template<class TYPE = double>
    inline VectorDataIterator<const TYPE> constBegin() const;

    /**
     * \brief Return an iterator to the end of the data
     * \returns A VectorDataIterator
     * \details Since the Vector presents an interface to a contiguous
     *     block of data, it is natural for it to provide a random
     *     access iterator.
     * \warning The non-const version of the iterators will automatically
     *     leave the vector in a non-consistent state.  The user may
     *     be required to call makeConsistent.
     */
    template<class TYPE = double>
    inline VectorDataIterator<TYPE> end();

    /// @copydoc VectorData::end()
    template<class TYPE = double>
    inline VectorDataIterator<const TYPE> end() const;

    /// @copydoc VectorData::end()
    template<class TYPE = double>
    inline VectorDataIterator<const TYPE> constEnd() const;

    /** \brief Obtain a particular contiguous block of data cast to RETURN_TYPE
     * \tparam RETURN_TYPE  The pointer type of the return
     * \param[in] i  Which block
     * \return A contiguous array of type RETURN_TYPE
     */
    template<typename RETURN_TYPE>
    RETURN_TYPE *getRawDataBlock( size_t i = 0 );

    /** \brief Obtain a particular contiguous block of data cast to RETURN_TYPE
     * \tparam RETURN_TYPE  The pointer type of the return
     * \param[in] i  Which block
     * \return A const contiguous array of type RETURN_TYPE
     */
    template<typename RETURN_TYPE>
    const RETURN_TYPE *getRawDataBlock( size_t i = 0 ) const;

    /** \brief Check if the data is of the given type
     * \return true if all data is of the given type
     */
    template<typename TYPE>
    bool isType() const;

    /** \brief Check if the data is of the given type
     * \return true if the ith block of data is of the given type
     */
    template<typename TYPE>
    bool isBlockType( size_t i = 0 ) const;

    /** \brief Copy ghosted vlues to a vector
     * \param[in] rhs  Vector to copy ghost values from
     * \details  This ensures that ghosted values on this and rhs
     * are the same without a call to makeConsistent.
     * \see makeConsistent
     */
    void copyGhostValues( const VectorData &rhs );


public:
    /** \brief Notify listeners that data has changed in this vector.
     */
    virtual void dataChanged();

    /* \brief  Returns true if this vector has this element
     * \param[in]  GID  The global ID of the element
     */
    virtual bool containsGlobalElement( size_t GID );

    /**
     * \brief This method is used to implement the assemble interface
     * of PETSc.
     * \details  This method is empty except for instantiations of NativePetscVectorData
     */
    virtual void assemble() {}


public: // Non virtual functions
    /** \brief  Return the current update state of this Vector
     * \details  This returns the pointer to the update state
     *  of the current vector only (not vectors it contains).
     *  It should NOT be used by users.
     */
    std::shared_ptr<UpdateState> getUpdateStatusPtr() const;

    /** \brief  Tie the current update state to another
     * \details  This sets the pointer to the update state
     *  of the current vector only (not vectors it contains).
     *  It should NOT be used by users.
     * \param  rhs Pointer to share update state with
     */
    void setUpdateStatusPtr( std::shared_ptr<UpdateState> rhs );

    //! Get a unique id hash for the vector
    uint64_t getID() const;


public: // Write/read restart data
    /**
     * \brief    Register any child objects
     * \details  This function will register child objects with the manager
     * \param manager   Restart manager
     */
    virtual void registerChildObjects( AMP::IO::RestartManager *manager ) const;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    virtual void writeRestart( int64_t fid ) const;

    VectorData( int64_t fid, AMP::IO::RestartManager *manager );

protected:                   // Internal data
    size_t d_localSize  = 0; //! Number of local values
    size_t d_globalSize = 0; //! Number of global values
    size_t d_localStart = 0; //! Index of first local value

    //! The communication list for this vector
    std::shared_ptr<CommunicationList> d_CommList = nullptr;

    /** \brief  The current update state for a vector
     * \details A Vector can be in one of three states.
     *  This is the current state of the vector
     *  Because a vector can be composed of vectors,
     *  the update state needs to be shared between them.
     */
    std::shared_ptr<UpdateState> d_UpdateState = nullptr;

    // Ghost data
    std::shared_ptr<std::vector<double>> d_Ghosts    = nullptr;
    std::shared_ptr<std::vector<double>> d_AddBuffer = nullptr;

    // Friends
    friend class VectorOperations;


public:
    //! Default constructor
    VectorData();
    VectorData( const VectorData & ) = delete;

    VectorData( std::shared_ptr<CommunicationList> commList );
};


} // namespace AMP::LinearAlgebra


#include "AMP/vectors/data/VectorData.inline.h"


#endif
