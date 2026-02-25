#ifndef included_AMP_RestartManager
#define included_AMP_RestartManager

#include "AMP/IO/HDF.h"
#include "AMP/utils/TypeTraits.h"

#include <map>
#include <string>


namespace AMP::IO {


//! Class to manage reading/writing restart data
class RestartManager final
{
public: // User functions
    //! Create a writer for restart data
    RestartManager();

    //! Create a reader for restart data
    RestartManager( const std::string &filename );

    //! Copy constructor
    RestartManager( const RestartManager & ) = delete;

    //! Assignment operator
    RestartManager &operator=( const RestartManager & ) = delete;

    //! Move operator
    RestartManager( RestartManager && );

    //! Move assignment
    RestartManager &operator=( RestartManager && );

    //! Destructor
    ~RestartManager();

    //! Reset internal data
    void reset();

    /**
     * \brief  Write the data
     * \details  Write all of the data currently registered with the manager to the disk
     * and close file
     * @param[in] filename  Filename to use
     * @param[in] compress  Compression method to use
     */
    void write( const std::string &filename, Compression compress = Compression::None );

    /**
     * \brief  Read a restart file
     * \details  This will open a restart file for reading.
     *           Note: this will delete all internal data (see reset()) before opening the file.
     *           Note: the restart file will remain open (and locked) until reset() is called or
     *           this object goes out of scope.
     * @param[in] filename  Filename to use
     */
    void load( const std::string &filename );

    /**
     * \brief  Register data with the restart manager
     * \details This function registers an object with the restart manager
     * @param[in] name      Name to use for object
     * @param[in] data      Data to register
     */
    template<class TYPE>
    void registerData( const TYPE &data, const std::string &name );

    /**
     * \brief  Get data from the restart manager
     * \details This function will get a registered/loaded object from the restart manager
     * @param[in] name      Name to use for object
     */
    template<class TYPE>
    std::shared_ptr<TYPE> getData( const std::string &name );


public: // Developer functions
    /**
     * \brief  Register data with the restart manager
     * \details This function registers an object with the restart manager
     * @param[in] data      Data to register
     */
    template<class TYPE>
    uint64_t registerObject( const TYPE &data );

    /**
     * \brief  Register a communicator with the restart manager
     * \details This function registers a communicator (based on ranks) with the restart manager
     * @param[in] comm      Communicator to register
     */
    uint64_t registerComm( const AMP::AMP_MPI &comm );

    /**
     * \brief  Get data from the restart manager
     * \details This function will get a registered/loaded object from the restart manager
     * @param[in] hash      Object ID
     */
    template<class TYPE>
    std::shared_ptr<TYPE> getData( uint64_t hash );


    /**
     * \brief  Get the communicator from the restart manager
     * \details This function will get a registered/loaded object from the restart manager
     * @param[in] hash      Object ID
     */
    AMP_MPI getComm( uint64_t hash );


    /**
     * \brief  Register SAMRAI data with the restart manager
     * \details This function registers a SAMRAI object with the restart manager
     * @param[in] data      Data to register
     */
    template<class TYPE>
    uint64_t registerSAMRAIData( std::shared_ptr<const TYPE &> data );

    /**
     * \brief  Get SAMRAI data from the restart manager
     * \details This function will get a registered/loaded SAMRAI object from the restart manager
     * @param[in] hash      Object ID
     */
    template<class TYPE>
    std::shared_ptr<TYPE> getSAMRAIData( uint64_t hash );


    /**
     * \brief  Check if an object is registered
     * \details This function will check if data with the give id has already been registered
     * @param[in] hash      Object ID
     */
    bool isRegistered( uint64_t hash );


public:
    //! Base class for writing an object
    class DataStore
    {
    public:
        virtual ~DataStore() = default;
        inline uint64_t getHash() const { return d_hash; }
        virtual void write( hid_t fid, const std::string &name ) const = 0;

    protected:
        uint64_t d_hash = 0;
    };

    //! Class to store a single object to write/read
    template<class TYPE>
    class DataStoreType : public DataStore
    {
    public:
        DataStoreType( hid_t fid, uint64_t hash, RestartManager *manager );
        DataStoreType( std::shared_ptr<const TYPE>, RestartManager * );
        virtual ~DataStoreType() = default;
        void write( hid_t fid, const std::string &name ) const override;
        virtual std::shared_ptr<TYPE>
        read( hid_t fid, const std::string &name, RestartManager * ) const;
        auto getData() { return std::const_pointer_cast<TYPE>( d_data ); }

    protected:
        DataStoreType() = default;
        std::shared_ptr<const TYPE> d_data;
    };

    // Specialization for SAMRAI data object
    template<class TYPE>
    class SAMRAIDataStore : public DataStoreType<TYPE>
    {
    public:
        SAMRAIDataStore( hid_t fid, uint64_t hash, RestartManager *manager );
        SAMRAIDataStore( std::shared_ptr<const TYPE>, RestartManager * );
        virtual ~SAMRAIDataStore() = default;
        void write( hid_t fid, const std::string &name ) const override;
        std::shared_ptr<TYPE>
        read( hid_t fid, const std::string &name, RestartManager * ) const override;
    };
    using DataStorePtr = std::shared_ptr<DataStore>;


private: // Private functions
    template<class TYPE>
    RestartManager::DataStorePtr create( std::shared_ptr<const TYPE> );

    void writeCommData( const std::string &file, Compression compress );
    void readCommData( const std::string &file );
    static std::string hash2String( uint64_t );

private:                                     // Data members
    hid_t d_fid;                             // fid for reading
    std::map<uint64_t, DataStorePtr> d_data; // Object data
    std::map<std::string, uint64_t> d_names; // Names to associate with the hashes
    std::map<uint64_t, AMP_MPI> d_comms;     // Registered communicators
};


} // namespace AMP::IO


#include "AMP/IO/RestartManager.hpp"


#endif
