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
     *     compute the communication lists.  Derived classes are expected to call
     *     buildCommunicationArrays with appropriate data to compute the communication list
     *     This is a blocking call and must be called from all ranks.
     */
    CommunicationList( std::shared_ptr<const CommunicationListParameters> params );

    /**
     * \brief  Construct a CommunicationList with no comunication
     * \param[in]  local    The number of local elements in the vector
     * \param[in]  comm     The AMP_MPI for the vector.
     * \details  Create a communication list with no communication.
     *     This is a blocking call and must be called from all ranks.
     */
    CommunicationList( size_t local, const AMP_MPI &comm );

    /**
     * \brief  Construct a CommunicationList with comunication
     * \param[in]  comm     The AMP_MPI for the vector.
     * \param[in]  local    The number of local elements for each rank
     * \param[in]  remote   The remote DOFs that we need to receive on this rank
     * \details  Create a communication list (advanced interface)
     */
    CommunicationList( const AMP_MPI &comm, std::vector<size_t> local, std::vector<size_t> remote );

    /**
     * Constructor for creating CommunicationLists with no communication
     * from an existing CommunicationList object without incurring global communication
     * (meant for internal use only)
     */
    CommunicationList( const std::vector<size_t> &d_partition, const AMP_MPI &comm );

    /**
     * \brief Returns a shared_ptr to a CommunicationList with no data communication.
     */
    std::shared_ptr<CommunicationList> getNoCommunicationList();

    /**
     * \brief Subset a communication list based on a VectorIndexer
     * \param[in] sub  A VectorIndexer pointer that describes a subset
     *     This is a blocking call and must be called from all ranks.
     */
    std::shared_ptr<CommunicationList> subset( std::shared_ptr<VectorIndexer> sub );

    /**
     * \brief Retrieve the size of the buffer used to receive data from other processes
     * \details This is an alias of getGhostIDList().size()
     *     This is a potentially blocking call and must be called from all ranks.
     *     It is only blocking if initialize has not been called first.
     *     Users can explicitly call initialize() to avoid this.
     * \return The number of unowned entries on this rank
     */
    size_t getVectorReceiveBufferSize() const;

    /**
     * \brief Retrieve the size of the buffer used to send data to other processes
     * \details This is an alias of getReplicatedIDList().size()
     *     This is a potentially blocking call and must be called from all ranks.
     *     It is only blocking if initialize has not been called first.
     *     Users can explicitly call initialize() to avoid this.
     * \return The number of owned entries on this rank shared with other rank
     */
    size_t getVectorSendBufferSize() const;

    /**
     * \brief Retrieve list of global indices shared locally stored elsewhere
     * \return A vector of indices not owned by the rank but are stored locally.
     */
    const std::vector<size_t> &getGhostIDList() const;

    /**
     * \brief Retrieve list of global indices stored here and shared elsewhere
     * \details This will obtain the list of indices owned by this rank and shared on other rank.
     *     This is a potentially blocking call and must be called from all ranks.
     *     It is only blocking if initialize has not been called first.
     *     Users can explicitly call initialize() to avoid this.
     * \return A vector of indices owned by the rank and shared on other ranks.
     */
    const std::vector<size_t> &getReplicatedIDList() const;

    /**
     * \brief Retrieve number of DOFs received from each rank
     * \details Retrieve number of DOFs received from each rank
     *     This is a potentially blocking call and must be called from all ranks.
     *     It is only blocking if initialize has not been called first.
     *     Users can explicitly call initialize() to avoid this.
     * \return A vector size of comm.getSize() containing the number
     *         of DOFs we will receive from each rank
     */
    const std::vector<int> &getReceiveSizes() const;

    /**
     * \brief Retrieve number of DOFs sent to each rank
     * \details Retrieve number of DOFs sent to each rank
     *     This is a potentially blocking call and must be called from all ranks.
     *     It is only blocking if initialize has not been called first.
     *     Users can explicitly call initialize() to avoid this.
     * \return A vector size of comm.getSize() containing the number
     *         of DOFs we will sent to each rank
     */
    const std::vector<int> &getSendSizes() const;

    //! Get the receive displacements
    const std::vector<int> &getReceiveDisp() const;

    //! Get the send displacements
    const std::vector<int> &getSendDisp() const;

    /**
     * \brief Retrieve the partition of DOFs
     * \return A vector size of comm.getSize() containing the endDOF
     *        (getStartGID()+numLocalRows()) for each rank
     */
    const std::vector<size_t> &getPartition() const;

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
    size_t numLocalRows() const;

    /**
     * Clears the internal buffers so that they are empty
     */
    void clearBuffers();

    //! Returns true if any rank has remote data
    bool anyCommunication();

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
    // Initialize the internal data
    void initialize() const;

protected:
    // Empty constructor
    CommunicationList();

private:
    AMP_MPI d_comm;                            // Communicator
    mutable bool d_initialized   = false;      // Have we initialized all of the data
    mutable bool d_anyRankRemote = false;      // Does any rank have remote data
    std::vector<size_t> d_ReceiveDOFList;      // Sorted DOF receive lists
    std::vector<size_t> d_partition;           // Partition info
    mutable std::vector<size_t> d_SendDOFList; // Sorted DOF send lists
    mutable std::vector<int> d_ReceiveSizes;   // Number of DOFs to receive from each rank
    mutable std::vector<int> d_ReceiveDisp;    // Displacement for each rank
    mutable std::vector<int> d_SendSizes;      // Number of DOFs to send from each rank
    mutable std::vector<int> d_SendDisp;       // Displacement for each rank
};

} // namespace AMP::LinearAlgebra

#endif
