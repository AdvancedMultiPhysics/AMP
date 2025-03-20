#ifndef included_AMP_CommunicationList_h
#define included_AMP_CommunicationList_h

#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/ParameterBase.h"
#include <memory>

#include <vector>


namespace AMP::LinearAlgebra {

class VectorData;
class VectorIndexer;


/**
 * \class CommunicationListParameters
 * \brief Parameters class encapsulating the data necessary to instantiate a communication list.
 */
class CommunicationListParameters : public ParameterBase
{
public:
    //! The communicator over which the communication list is computed
    AMP_MPI d_comm;

    //! The number of local entities in the vector
    size_t d_localsize;

    //! The remote DOFs that we need to receive
    std::vector<size_t> d_remote_DOFs;

    //! Default constructor
    CommunicationListParameters();

    //! Copy constructor
    CommunicationListParameters( const CommunicationListParameters & );

    //! Assignment operator
    CommunicationListParameters &operator=( const CommunicationListParameters & ) = delete;
};


/**
 * \class CommunicationList
 * \brief What to send where and what to receive from where
 *
 * \details This interface provides communication routines to compute send and receive lists
 * for blocks of data with global indices.  For instance, a vector storing degrees of freedom
 * for nodes in a finite element analysis may share data with other cores in a parallel
 * computation.  This class tracks which local data need to be communicated with other cores
 * and which data should be received from those cores.
 */
class CommunicationList final
{
public:
    /**
     * \brief Construct a communication list
     * \param[in] params  A shared pointer to parameters for constructing the list
     * \details This will set the communicator for the communication list.  It will not
     * compute the communication lists.  Derived classes are expected to call
     * buildCommunicationArrays with appropriate data to compute the communication list
     */
    CommunicationList( std::shared_ptr<const CommunicationListParameters> params );

    /**
     * \brief  Construct a CommunicationList with no comunication
     * \param[in]  local  The number of local elements in the vector
     * \param[in]  comm   The AMP_MPI for the vector.
     * \details  Create a communication list with no communication.
     */
    CommunicationList( size_t local, const AMP_MPI &comm );

    /**
     * \brief Subset a communication list based on a VectorIndexer
     * \param[in] sub  A VectorIndexer pointer that describes a subset
     */
    std::shared_ptr<CommunicationList> subset( std::shared_ptr<VectorIndexer> sub );

    /**
     * \brief Retrieve the size of the buffer used to receive data from other processes
     * \details This is an alias of getGhostIDList().size()
     * \return The number of unowned entries on this core
     */
    size_t getVectorReceiveBufferSize() const;

    /**
     * \brief Retrieve the size of the buffer used to send data to other processes
     * \details This is an alias of getReplicatedIDList().size()
     * \return The number of owned entries on this core shared with other cores
     */
    size_t getVectorSendBufferSize() const;

    /**
     * \brief Retrieve list of global indices shared locally stored elsewhere
     * \return A vector of indices not owned by the core but are stored locally.
     */
    const std::vector<size_t> &getGhostIDList() const;

    /**
     * \brief Retrieve list of global indices stored here and shared elsewhere
     * \return A vector of indices owned by the core and shared on other cores.
     */
    const std::vector<size_t> &getReplicatedIDList() const;

    /**
     * \brief Retrieve number of DOFs received from each rank
     * \return A vector size of comm.getSize() containing the number
     *         of DOFs we will receive from each rank
     */
    const std::vector<int> &getReceiveSizes() const;

    /**
     * \brief Retrieve number of DOFs sent to each rank
     * \return A vector size of comm.getSize() containing the number
     *         of DOFs we will sent to each rank
     */
    const std::vector<int> &getSendSizes() const;

    /**
     * \brief Retrieve the partition of DOFs
     * \return A vector size of comm.getSize() containing the endDOF
     *        (getStartGID()+numLocalRows()) for each rank
     */
    const std::vector<size_t> &getPartition() const;

    /**
     * \brief Scatter data stored here to processors that share the data.
     * \param[in,out] vec  Data to set
     * \details  The convention is if data are set on different processes, then
     * the owner of the data has the correct value.  As such, in a scatter_set,
     * the owner of data scatters the data out which overwrites the data on cores
     * that share the data
     */
    void scatter_set( VectorData &vec ) const;

    /**
     * \brief Scatter data shared here to processors that own the data.
     * \param[in,out] vec  Data to add
     * \details  When adding data to a vector, any process that shares the data
     * can contribute to the value of the data.  Therefore, this will scatter data
     * that is shared to the core that owns it.  A call to scatter_add is generally
     * followed by a call to scatter_set to ensure parallel consistency.
     */
    void scatter_add( VectorData &vec ) const;

    /**
     * \brief  Return the first d.o.f. on this core
     * \return The first d.o.f. on this core
     */
    size_t getStartGID() const;

    /**
     * \brief  Return the total d.o.f. on entire communicator
     */
    size_t getTotalSize() const;

    /**
     * \brief  Return the number of local rows for this communication list
     * \return The number of local d.o.f. for this communication list
     */
    virtual size_t numLocalRows() const;

    /**
     * \brief  Return the local index of a shared datum.
     * \param[in] dof  The global index to get a local ghost id for
     * \details  It is assumed that data are stored in two arrays: an owned array and a shared
     * array.  This function returns the local offset of a shared datum into the shared array
     * \return The index into the shared array for the global index.
     */
    size_t getLocalGhostID( size_t dof ) const;

    /**
     * \brief  Return the communicator used for this communication list
     * \return The communicator.
     */
    const AMP_MPI &getComm() const;

    //! Get a unique id hash
    uint64_t getID() const;


public:
    //! Build the partition info from the local size
    static std::vector<size_t> buildPartition( AMP_MPI &comm, size_t N_local );


protected:
    // Build the communication lists
    void buildCommunicationArrays( const std::vector<size_t> &dofs );

    // Empty constructor
    CommunicationList();

private:
    AMP_MPI d_comm;                       // Communicator
    std::vector<size_t> d_ReceiveDOFList; // Sorted DOF receive lists
    std::vector<size_t> d_SendDOFList;    // Sorted DOF send lists
    std::vector<size_t> d_partition;      // Partition info
    std::vector<int> d_ReceiveSizes;      // Number of DOFs to receive from each rank
    std::vector<int> d_ReceiveDisp;       // Displacement for each rank into d_ReceiveDOFList
    std::vector<int> d_SendSizes;         // Number of DOFs to send from each rank
    std::vector<int> d_SendDisp;          // Displacement for each rank into d_SendDisplacements
};

} // namespace AMP::LinearAlgebra

#endif
