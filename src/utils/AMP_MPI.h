// This file includes a wrapper class for MPI functions
#ifndef included_AMP_MPI
#define included_AMP_MPI


#include <set>
#include <map>
#include <string>
#include <complex>
#include "utils/Utilities.h"

//! Define MPI objects
#ifdef USE_EXT_MPI
    #include "mpi.h"    // include mpi.h
#elif defined(USE_EXT_TRILINOS)
    #include "mpi.h"    // trilinos serial builds include mpi.h
#else
    typedef int MPI_Comm;
    typedef int MPI_Request;
    typedef void* MPI_Errhandler;
#endif
#define AMP_COMM_WORLD ((MPI_Comm)0xF4000010)
#define AMP_COMM_SELF  ((MPI_Comm)0xF4000001)
#define AMP_COMM_NULL  ((MPI_Comm)0xF4000000)



namespace AMP {


/**
 * \class AMP_MPI
 *
 * @brief Provides C++ wrapper around MPI routines.
 *
 * Class AMP_MPI groups common MPI routines into one globally-accessible
 * location.  It provides small, simple routines that are common in MPI code.
 * In some cases, the calling syntax has been simplified for convenience.
 * Moreover, there is no reason to include the preprocessor ifdef/endif
 * guards around these calls, since the MPI libraries are not called in
 * these routines if the MPI libraries are not being used (e.g., when
 * writing serial code).
 * Note: Many of the communication routines are templated on type.  When using
 * unknown types the reduce calls will fail, the send and gather calls should
 * succeed provided that the size of the data type object is a fixed size on
 * all processors.  sizeof(type) must be the same for all elements and processors.
 */
class AMP_MPI
{
public:


    /**
     *\brief  Empty constructor
     *\details  This creates an empty constructor that does not contain an MPI communicator.
     */
    AMP_MPI();


    //!  Empty destructor
    ~AMP_MPI();


    /**
     * \brief Constructor from existing AMP_MPI object
     * \details  This constructor creates a new AMP_MPI object from an existing AMP_MPI object.
     *   This does not create a new internal MPI_Comm, but uses the existing comm.
     * \param comm Existing AMP_MPI object
     */
    AMP_MPI(const AMP::AMP_MPI& comm);

    /**
     * \brief Constructor from existing MPI communicator
     * \details  This constructor creates a new AMP_MPI object from an existing MPI communicator.
     *    This does not create a new internal MPI_Comm, but uses the existing comm.
     *    Note that by default, AMP_MPI will not free the MPI_Comm object and the user is responsible 
     *      for free'ing the MPI_Comm when it is no longer used.  This behavior is controlled by the 
     *      optional manage argument.  
     * \param comm      Existing MPI communicator
     * \param manage    Do we want to manage the comm (free the MPI_Comm when this object leaves scope)
     */
    AMP_MPI( MPI_Comm comm, bool manage=false );


    /**
     * \brief Assignment operator
     * \details  This operator overloads the assignment to correctly copy an AMP_MPI object
     * \param comm Existing AMP_MPI object
     */
    AMP_MPI& operator=(const AMP::AMP_MPI& comm);


    /**
     * \brief Get the node name
     * \details  This function returns a unique name for each node.  
     *    It is a wrapper for MPI_Get_processor_name.
     */
    static std::string getNodeName( );


    //! Function to return the number of processors available
    static int getNumberOfProcessors();


    //! Function to return the affinity of the current process
    static std::vector<int> getProcessAffinity();


    //! Function to set the affinity of the current process
    static void setProcessAffinity( std::vector<int> procs );


    /**
     * \brief Load balance the processes within a node
     * \details  This function will redistribute the processes within a node using the
     *    process affinities to achieve the desired load balance.  
     *    Note: this is a global operation on the given comm, and it is STRONGLY 
     *    recommended to use COMM_WORLD.
     * \param comm      The communicator to use (Default is COMM_WORLD)
     * \param method    The desired load balance method to use:
     *                  1:  Adjust the affinities so all processes share the given processors.
     *                      This effectively allows the OS to handle the load balancing 
     *                      by migrating the processes as necessary.  This is recommended
     *                      for most users and use cases. (default)
     *                  2:  Adjust the affinities so that the fewest number of processes overlap.
     *                      This will try to give each process a unique set of processors while
     *                      ensuring that each process has at least N_min processes.
     * \param procs     An optional list of processors to use.  By default, setting this to an
     *                  empty vector will use all available processors on the given node.
     * \param N_min     The minimum number of processors for any process (-1 indicates all available processors).
     * \param N_max     The maximum number of processors for any process (-1 indicates all available processors).
     *                      
     */
    static void balanceProcesses( 
        const AMP_MPI comm = AMP_MPI(AMP_COMM_WORLD), 
        const int method = 1, 
        const std::vector<int>& procs = std::vector<int>(), 
        const int N_min = 1, 
        const int N_max = -1 
    );


    /**
     * \brief Split an existing communicator
     * \details  This creates a new AMP_MPI object by splitting an existing AMP_MPI object.
     *   See MPI_Comm_split for information on how the underlying split will occur.
     *   Note: the underlying MPI_Comm object will be free'd automatically when it is no longer 
     *   used by any MPI objects.
     * \param color  Control of subset assignment (nonnegative integer). 
     *               Processes with the same color are in the same new communicator .
     *               -1: processor will not be a member of any object (NULL object will be returned)
     * \param key    Control of rank assignment (integer).
     *               Note that, for a fixed color, the keys need not be unique. The processes will be sorted 
     *               in ascending order according to this key, then all the processes in a given color will 
     *               have the relative rank order as they did in their parent group. (See MPI_Comm_split) 
     */
    AMP_MPI split( int color, int key=-1 ) const;


    /**
     * \brief Split an existing communicator by node
     * \details  This creates a new AMP_MPI object by splitting an existing AMP_MPI object 
     *   by the node.  This will result in a separate MPI_Comm for each physical node.
     *   Internally this will use MPI_Get_processor_name to identify the nodes.
     *   Note: the underlying MPI_Comm object will be free'd automatically when it is no longer 
     *   used by any MPI objects)
     * \param key    Control of rank assignment (integer).
     *               Note that, for a fixed color, the keys need not be unique. The processes will be sorted 
     *               in ascending order according to this key, then all the processes in a given color will 
     *               have the relative rank order as they did in their parent group. (See MPI_Comm_split) 
     */
    AMP_MPI splitByNode( int key=-1 ) const;


    /**
     * \brief Duplicate an existing AMP_MPI communicator
     * \details  This creates a new AMP_MPI object by duplicating an existing AMP_MPI object.
     *   The resulting communicator will exist over the same processes, but have a different context. 
     *   Note: the underlying MPI_Comm object will be free'd automatically when it is no longer 
     *   used by any MPI objects.
     */
    AMP_MPI dup( ) const;


    /**
     * \brief Create a communicator from the intersection of two communicators
     * \details  This creates a new AMP_MPI object by intersecting two existing communicators.
     *   Any processors that do not contain the both communicators will receive a NULL communicator.
     *   There are 3 possible cases:
     *      The communicators are disjoint (a null communicator will be returned on all processors).
     *      One communicator is a sub communicator of another.  This will require communication on 
     *          the smaller communicator only.
     *      The communicators partially overlap.  This will require communication on the first communicator.
     */
    static AMP_MPI intersect( const AMP_MPI &comm1, const AMP_MPI &comm2 );


    /**
     * Check if the current communicator is NULL
     */
    bool isNull() const { return d_isNull; }


    /**
     *  Get the current MPI communicator.  
     *  Note: The underlying MPI_Comm object may be free'd by AMP_MPI when it is no 
     *  longer used by any AMP_MPI objects.  If the user has made a copy using the 
     *  getCommunicator routine, then it may be free'd without user knowledge.  The 
     *  user is responsible for checking if the communicator is valid, or keeping a 
     *  copy of the AMP_MPI object that provided the MPI_Communicator.
     */
    const MPI_Comm& getCommunicator() const { return communicator; }


    /**
     * \brief Overload operator ==
     * \details  Overload operator comm1 == comm2.  Two MPI objects are == if they share the same communicator.
     *   Note: this is a local operation.  
     */
    bool operator==(const AMP_MPI& ) const;


    /**
     * \brief Overload operator !=
     * \details  Overload operator comm1 != comm2.  Two MPI objects are != if they do not share the same communicator.
     *   Note: this is a local operation.  
     */
    bool operator!=(const AMP_MPI& ) const;


    /**
     * \brief Overload operator <
     * \details  Overload operator comm1 < comm2.  One MPI object is < another iff all the processors in the first
     *   object are also in the second.  Additionally, the second object must contain at least one processor
     *   that is not in the first object.  This is a collective operation, based on the first communicator.
     *   As a result all processors on the first communicator will return the same value, while any processors that are 
     *   not on the first communicator will return an unknown value.  Additionally, all processors on the first object MUST
     *   call this routine and will be synchronized through this call (there is an internal allReduce).  
     */
    bool operator<(const AMP_MPI& ) const;


    /**
     * \brief Overload operator <=
     * \details  Overload operator comm1 <= comm2.  One MPI object is <= another iff all the processors in the first
     *   object are also in the second.  This is a collective operation, based on the first communicator.
     *   As a result all processors on the first communicator will return the same value, while any processors that are 
     *   not on the first communicator will return an unknown value.  Additionally, all processors on the first object MUST
     *   call this routine and will be synchronized through this call (there is an internal allReduce).  
     */
    bool operator<=(const AMP_MPI& ) const;


    /**
     * \brief Overload operator >
     * \details  Overload operator comm1 > comm2.  One MPI object is > another iff all the processors in the second
     *   object are also in the first.  Additionally, the first object must contain at least one processor
     *   that is not in the second object.  This is a collective operation, based on the first communicator.
     *   As a result all processors on the first communicator will return the same value, while any processors that are 
     *   not on the first communicator will return an unknown value.  Additionally, all processors on the first object MUST
     *   call this routine and will be synchronized through this call (there is an internal allReduce).  
     */
    bool operator>(const AMP_MPI& ) const;


    /**
     * \brief Overload operator >=
     * \details  Overload operator comm1 >= comm2.  One MPI object is > another iff all the processors in the second
     *   object are also in the first.  Additionally, the first object must contain at least one processor
     *   that is not in the second object.  This is a collective operation, based on the first communicator.
     *   As a result all processors on the first communicator will return the same value, while any processors that are 
     *   not on the first communicator will return an unknown value.  Additionally, all processors on the first object MUST
     *   call this routine and will be synchronized through this call (there is an internal allReduce).  
     */
    bool operator>=(const AMP_MPI& ) const;


    /**
     * \brief Compare to another communicator
     * \details  This compares the current communicator to another communicator.  
     *   This returns 1 if the two communicators are equal (they share the same MPI communicator),
     *   2 if the contexts and groups are the same, 3 if different contexts but identical groups, 
     *   4 if different contexts but similar groups, and 0 otherwise.
     *   Note: this is a local operation.  
     */
    int compare(const AMP_MPI& ) const;


    /**
     * Return the processor rank (identifier) from 0 through the number of
     * processors minus one.
     */
    int getRank() const { return comm_rank; }


    /**
     * Return the number of processors.
     */
    int getSize() const { return comm_size; }


    /**
     * Return the maximum tag
     */
    int maxTag() const { return d_maxTag; }


    /**
     * \brief   Return a new tag
     * \details This routine will return an unused tag for communication.
     *   Note that this tag may match a user tag, but this function will 
     *   not return two duplicate tags.  This is a global operation.
     */
    int newTag();


    /**
     * Call MPI_Abort or exit depending on whether running with one or more 
     * processes and value set by function above, if called.  The default is
     * to call exit(-1) if running with one processor and to call MPI_Abort()
     * otherwise.  This function avoids having to guard abort calls in 
     * application code.  
     */
    void abort() const;


    /**
     * Set boolean flag indicating whether exit or abort is called when running
     * with one processor.  Calling this function influences the behavior of
     * calls to AMP_MPI::abort().  By default, the flag is true meaning that
     * abort() will be called.  Passing false means exit(-1) will be called. 
     */
    void setCallAbortInSerialInsteadOfExit(bool flag=true); 


    /**
     * \brief   Boolean all reduce
     * \details This function performs a boolean all reduce across all processors.
     *   It returns true iff all processor are true;
     * \param value  The input value for the all reduce
     */
    bool allReduce(const bool value) const;


    /**
     * \brief   Boolean any reduce
     * \details This function performs a boolean any reduce across all processors.
     *   It returns true if any processor is true;
     * \param value  The input value for the all reduce
     */
    bool anyReduce(const bool value) const;


    /**
     * \brief   Sum Reduce
     * \details This function performs a sum all reduce across all processor.
     *   It returns the sum across all processors;
     * \param value  The input value for the all reduce
     */
    template <class type>
    type sumReduce(const type value) const;


    /**
     * \brief   Sum Reduce
     * \details Perform an array sum Reduce across all nodes.  Each
     * processor contributes an array of values, and the
     * element-wise sum is returned in the same array.
     * \param x  The input/output array for the reduce
     * \param n  The number of values in the array (must match on all nodes)
     */
    template <class type>
    void sumReduce(type *x, const int n=1) const;


    /**
     * \brief   Sum Reduce
     * \details Perform an array sum Reduce across all nodes.  Each
     * processor contributes an array of values, and the
     * element-wise sum is returned in the same array.
     * \param x  The input array for the reduce
     * \param y  The output array for the reduce
     * \param n  The number of values in the array (must match on all nodes)
     */
    template <class type>
    void sumReduce(const type *x, type *y, const int n=1) const;


    /**
     * \brief   Min Reduce
     * \details This function performs a min all reduce across all processor.
     *   It returns the minimum value across all processors;
     * \param value  The input value for the all reduce
     */
    template <class type>
    type minReduce(const type value) const;


    /**
     * \brief   Sum Reduce
     * \details Perform an array min Reduce across all nodes.  Each
     * processor contributes an array of values, and the
     * element-wise minimum is returned in the same array.
     *
     * If a 'rank_of_min' argument is provided, it will set the array to the 
     * rank of process holding the minimum value.  Like the double argument,
     * the size of the supplied 'rank_of_min' array should be n.
     * \param x         The input/output array for the reduce
     * \param n         The number of values in the array (must match on all nodes)
     * \param rank_of_min  Optional array indicating the rank of the processor containing the minimum value
     */
    template <class type>
    void minReduce(type *x, const int n=1, int *rank_of_min=NULL) const;


    /**
     * \brief   Sum Reduce
     * \details Perform an array min Reduce across all nodes.  Each
     * processor contributes an array of values, and the
     * element-wise minimum is returned in the same array.
     *
     * If a 'rank_of_min' argument is provided, it will set the array to the 
     * rank of process holding the minimum value.  Like the double argument,
     * the size of the supplied 'rank_of_min' array should be n.
     * \param x         The input array for the reduce
     * \param y         The output array for the reduce
     * \param n         The number of values in the array (must match on all nodes)
     * \param rank_of_min  Optional array indicating the rank of the processor containing the minimum value
     */
    template <class type>
    void minReduce(const type *x, type *y, const int n=1, int *rank_of_min=NULL) const;


    /**
     * \brief   Max Reduce
     * \details This function performs a max all reduce across all processor.
     *   It returns the maximum value across all processors;
     * \param value     The input value for the all reduce
     */
    template <class type>
    type maxReduce(const type value) const;


    /**
     * \brief   Sum Reduce
     * \details Perform an array max Reduce across all nodes.  Each
     * processor contributes an array of values, and the
     * element-wise maximum is returned in the same array.
     *
     * If a 'rank_of_min' argument is provided, it will set the array to the 
     * rank of process holding the minimum value.  Like the double argument,
     * the size of the supplied 'rank_of_min' array should be n.
     * \param x         The input/output array for the reduce
     * \param n         The number of values in the array (must match on all nodes)
     * \param rank_of_max  Optional array indicating the rank of the processor containing the minimum value
     */
    template <class type>
    void maxReduce(type *x, const int n=1, int *rank_of_max=NULL) const;


    /**
     * \brief   Sum Reduce
     * \details Perform an array max Reduce across all nodes.  Each
     * processor contributes an array of values, and the
     * element-wise maximum is returned in the same array.
     *
     * If a 'rank_of_min' argument is provided, it will set the array to the 
     * rank of process holding the minimum value.  Like the double argument,
     * the size of the supplied 'rank_of_min' array should be n.
     * \param x         The input array for the reduce
     * \param y         The output array for the reduce
     * \param n         The number of values in the array (must match on all nodes)
     * \param rank_of_max  Optional array indicating the rank of the processor containing the minimum value
     */
    template <class type>
    void maxReduce(const type *x, type *y, const int n=1, int *rank_of_max=NULL) const;


    /**
     * \brief    Scan Sum Reduce
     * \details  Computes the sum scan (partial reductions) of data on a collection of processes.
     *   See MPI_Scan for more information.
     * \param x         The input array for the scan
     * \param y         The output array for the scan
     * \param n         The number of values in the array (must match on all nodes)
     */
    template <class type>
    void sumScan(const type *x, type *y, const int n=1) const;


    /**
     * \brief    Scan Min Reduce
     * \details  Computes the min scan (partial reductions) of data on a collection of processes.
     *   See MPI_Scan for more information.
     * \param x         The input array for the scan
     * \param y         The output array for the scan
     * \param n         The number of values in the array (must match on all nodes)
     */
    template <class type>
    void minScan(const type *x, type *y, const int n=1) const;


    /**
     * \brief    Scan Max Reduce
     * \details  Computes the max scan (partial reductions) of data on a collection of processes.
     *   See MPI_Scan for more information.
     * \param x         The input array for the scan
     * \param y         The output array for the scan
     * \param n     The number of values in the array (must match on all nodes)
     */
    template <class type>
    void maxScan(const type *x, type *y, const int n=1) const;


    /**
     * \brief   Broadcast
     * \details This function broadcasts a value from root to all processors
     * \param value     The input value for the broadcast.
     * \param root      The processor performing the broadcast
     */
    template <class type>
    type bcast(const type value, const int root) const;


    /**
     * \brief   Broadcast
     * \details This function broadcasts an array from root to all processors
     * \param value     The input/output array for the broadcast
     * \param n         The number of values in the array (must match on all nodes)
     * \param root      The processor performing the broadcast
     */
    template <class type>
    void bcast(type *value, const int n, const int root) const;


    /**
     * Perform a global barrier across all processors.
     */
    void barrier() const;


    /*!
     * @brief This function sends an MPI message with an array to another processor.
     *
     * If the receiving processor knows in advance the length 
     * of the array, use "send_length = false;"  otherwise, 
     * this processor will first send the length of the array, 
     * then send the data.  This call must be paired with a 
     * matching call to AMP_MPI::recv.
     *
     * @param buf       Pointer to array buffer with length integers.
     * @param length    Number of integers in buf that we want to send.
     * @param recv      Receiving processor number.
     * @param tag       Optional integer argument specifying an integer tag
     *                  to be sent with this message.  Default tag is 0.
     *                  The matching recv must share this tag.
     */
    template <class type>
    void send(const type *buf, const int length, const int recv, int tag=0) const;


    /*!
     * @brief This function sends an MPI message with an array of bytes
     * (MPI_BYTES) to receiving_proc_number.
     *
     * This call must be paired with a matching call to AMP_MPI::recvBytes.
     *
     * @param buf       Void pointer to an array of number_bytes bytes to send.
     * @param N_bytes   Integer number of bytes to send.
     * @param recv      Receiving processor number.
     * @param tag       Optional integer argument specifying an integer tag
     *                  to be sent with this message.  Default tag is 0.
     *                  The matching recv must share this tag.
     */
    void sendBytes(const void *buf, const int N_bytes, const int recv, int tag=0) const;


    /*!
     * @brief This function sends an MPI message with an array 
     *   to another processor using a non-blocking call.
     *   The receiving processor must know the length of the array.
     *   This call must be paired  with a matching call to AMP_MPI::Irecv.
     *
     * @param buf       Pointer to array buffer with length integers.
     * @param length    Number of integers in buf that we want to send.
     * @param recv_proc Receiving processor number.
     * @param tag       Integer argument specifying an integer tag
     *                  to be sent with this message.  
     */
    template <class type>
    MPI_Request Isend(const type *buf, const int length, const int recv_proc, const int tag) const;


    /*!
     * @brief This function sends an MPI message with an array of bytes
     *   (MPI_BYTES) to receiving_proc_number using a non-blocking call.
     *   The receiving processor must know the number of bytes to receive.
     *   This call must be paired with a matching call to AMP_MPI::IrecvBytes.
     *
     * @param buf       Void pointer to an array of number_bytes bytes to send.
     * @param N_bytes   Integer number of bytes to send.
     * @param recv_proc Receiving processor number.
     * @param tag       Integer argument specifying an integer tag
     *                  to be sent with this message.  
     */
    MPI_Request IsendBytes(const void *buf, const int N_bytes, const int recv_proc, const int tag) const;


   /*!
    * @brief This function receives an MPI message with a data
    * array from another processor.
    *
    * If this processor knows in advance the length of the array,
    * use "get_length = false;" otherwise we will get the return size.
    * This call must be paired with a matching call to AMP_MPI::send.
    *
    * @param buf        Pointer to integer array buffer with capacity of length integers.
    * @param length     If get_length==true: The number of elements to be received, otherwise
    *                   the maximum number of values that can be stored in buf.
    *                   On output the number of received elements.
    * @param send       Processor number of sender.
    * @param get_length Optional boolean argument specifying if we first 
    *                   need to check the message size to get the size of the array.
    *                   Default value is true.
    * @param tag        Optional integer argument specifying a tag which must be matched 
    *                   by the tag of the incoming message. Default tag is 0.
    */
    template <class type>
    void recv(type *buf, int &length, const int send, const bool get_length=false, int tag=0) const;


    /*!
     * @brief This function receives an MPI message with an array of
     * max size number_bytes (MPI_BYTES) from any processor.
     *
     * This call must be paired with a matching call to AMP_MPI::sendBytes.
     *
     * @param buf       Void pointer to a buffer of size number_bytes bytes.
     * @param N_bytes   Integer number specifying size of buf in bytes.
     * @param send      Integer number specifying size of buf in bytes.
     * @param tag       Optional integer argument specifying a tag which
     *   must be matched by the tag of the incoming message. Default
     *   tag is 0.
     */
    void recvBytes(void *buf, int &N_bytes, const int send, int tag=0) const;


   /*!
    * @brief This function receives an MPI message with a data
    * array from another processor using a non-blocking call.
    *
    * @param buf        Pointer to integer array buffer with capacity of length integers.
    * @param length     Maximum number of values that can be stored in buf.
    * @param send_proc  Processor number of sender.
    * @param tag        Optional integer argument specifying a tag which must
    *                   be matched by the tag of the incoming message.
    */
    template <class type>
    MPI_Request Irecv(type *buf, const int length, const int send_proc, const int tag) const;


    /*!
     * @brief This function receives an MPI message with an array of
     * max size number_bytes (MPI_BYTES) from any processor.
     *
     * This call must be paired with a matching call to AMP_MPI::sendBytes.
     *
     * @param buf       Void pointer to a buffer of size number_bytes bytes.
     * @param N_bytes   Integer number specifying size of buf in bytes.
     * @param send_proc Processor number of sender.
     * @param tag       Integer argument specifying a tag which must 
     *                  be matched by the tag of the incoming message. 
     */
    MPI_Request IrecvBytes(void *buf, const int N_bytes, const int send_proc, const int tag) const;


    /*!
     * Each processor sends every other processor a single value.
     * The x_out array should be preallocated to a length equal
     * to the number of processors.
     * @param x_in      Input value for allGather
     * @param x_out     Output array for allGather (must be preallocated to the size of the communicator)
     */
    template <class type>
    void allGather(const type x_in, type *x_out) const;


    /*!
     * Each processor sends an array of data to all other processors.
     * Each processor receives the values from all processors and gathers them 
     * to a single array.  If successful, the total number of received
     * elements will be returned.  
     * @param send_data     Input array
     * @param send_cnt      The number of values to send
     * @param recv_data     Output array of received values
     * @param recv_cnt      The number of values to receive from each processor (N).
     *                      If known, this should be provided as an input.  Otherwise
     *                      it is an optional output that will return the number of     
     *                      received values from each processor.
     * @param recv_disp     The displacement (relative to the start of the array)
     *                      from which to store the data received from processor i.
     *                      If known, this should be provided as an input.  Otherwise
     *                      it is an optional output that will return the starting location
     *                      (relative to the start of the array) for the received data from processor i.
     * @param known_recv    Are the received counts and displacements known.  
     *                      If the received sizes are known, then they must be provided, 
     *                      and an extra communication step is not necessary.  If the received
     *                      sizes are not known, then an extra communication step will occur internally
     *                      and the sizes and displacements will be returned (if desired).  
     */
    template <class type>
    int allGather(const type *send_data, const int send_cnt, type *recv_data, 
        int *recv_cnt=NULL, int* recv_disp=NULL, bool known_recv=false) const;


    /*!
     * This function combines sets from different processors to create a single master set
     * @param set       Input/Output std::set for the gather.
     */
    template <class type>
    void setGather( std::set<type> &set ) const;


    /*!
     * This function combines std::maps from different processors to create a single master std::map
     * If two or more ranks share the same key, the lowest rank will be used
     * @param map       Input/Output std::map for the gather.
     */
    template <class KEY, class DATA>
    void mapGather( std::map<KEY,DATA> &map ) const;


    /*!
     * Each processor sends an array of n values to each processor.
     * Each processor sends an array of n values to each processor.  
     * The jth block of data is sent from processor i to processor j and placed 
     * in the ith block on the receiving processor.  In the variable
     * description, N is the size of the communicator.  Note that this is a 
     * blocking global communication.    
     * @param n             The number of elements in each data block to send.
     * @param send_data     Input array (nxN)
     * @param recv_data     Output array of received values (nxN)
     */
    template <class type>
    void allToAll(const int n, const type *send_data, type *recv_data ) const;


    /*!
     * Each processor sends an array of data to the different processors.
     * Each processor may send any size array to any processor.  In the variable
     * description, N is the size of the communicator.  Note that this is a 
     * blocking global communication.  If successful, the total number of received
     * elements will be returned.  
     * @param send_data     Input array
     * @param send_cnt      The number of values to send to each processor (N)
     * @param send_disp     The displacement (relative to the start of the array)
     *                      from which to send to processor i
     * @param recv_data     Output array of received values
     * @param recv_cnt      The number of values to receive from each processor (N).
     *                      If known, this should be provided as an input.  Otherwise
     *                      it is an optional output that will return the number of     
     *                      received values from each processor.
     * @param recv_disp     The displacement (relative to the start of the array)
     *                      from which to send to processor i.
     *                      If known, this should be provided as an input.  Otherwise
     *                      it is an optional output that will return the starting location
     *                      (relative to the start of the array) for the received data from processor i.
     * @param known_recv    Are the received counts and displacements known.  
     *                      If the received sizes are known, then they must be provided, 
     *                      and an extra communication step is not necessary.  If the received
     *                      sizes are not know, then an extra communication step will occur internally
     *                      and the sizes and displacements will be returned (if desired).  
     */
    template <class type>
    int allToAll(const type *send_data, const int send_cnt[], const int send_disp[], 
                  type *recv_data, int *recv_cnt=NULL, int* recv_disp=NULL, bool known_recv=false) const;


    /*!
     * \brief   Wait for a communication to finish
     * \details Wait for a communication to finish.  
     *    Note: this does not require a communicator.
     * \param request    Communication request to wait for (returned for Isend or Irecv)
     */
    static void wait( MPI_Request request);


    /*!
     * \brief   Wait for any communication to finish.
     * \details This function waits for any of the given communication requests to finish.
     *    It returns the index of the communication request that finished.
     *    Note: this does not require a communicator.
     * \param count      Number of communications to check
     * \param request    Array of communication requests to wait for (returned for Isend or Irecv)
     */
    static int waitAny( int count, MPI_Request *request);


    /*!
     * \brief   Wait for all communications to finish.
     * \details This function waits for all of the given communication requests to finish.
     *    Note: this does not require a communicator.
     * \param count      Number of communications to check
     * \param request    Array of communication requests to wait for (returned for Isend or Irecv)
     */
    static void waitAll( int count, MPI_Request *request);


    /*!
     * \brief   Wait for some communications to finish.
     * \details This function waits for one (or more) communications to finish.  
     *    It returns an array of the indicies that have finished.
     *    Note: this does not require a communicator.
     * \param count      Number of communications to check
     * \param request    Array of communication requests to wait for (returned for Isend or Irecv)
     */
    static std::vector<int> waitSome( int count, MPI_Request *request );


    /*!
     * \brief   Nonblocking test for a message
     * \details This function performs a non-blocking test for a message.
     *    It will return the number of bytes in the message if a message with 
     *    the specified source and tag (on the current communicator) is available.
     *    Otherwise it will return -1.
     * \param source      source rank (-1: any source)
     * \param tag         tag (-1: any tag)
     */
    int Iprobe( int source=-1, int tag=-1) const;


    /*!
     * \brief   Blocking test for a message
     * \details This function performs a blocking test for a message.
     *    It will return the number of bytes in the message when a message with 
     *    the specified source and tag (on the current communicator) is available
     * \param source      source rank (-1: any source)
     * \param tag         tag (-1: any tag)
     */
    int probe( int source=-1, int tag=-1) const;


    /*!
     * \brief   Elapsed time
     * \details This function returns the elapsed time on the calling processor 
     *    since an arbitrary point in the past (seconds).  It is a wrapper to MPI_Wtime.  
     *    See "tick" for the timer resolution in seconds.
     *    The time may or may not be synchronized across processors depending on the MPI 
     *    implementation.  Refer to MPI documentation for the desired platform for more information.
     */
    static double time();


    /*!
     * \brief   Timer resolution
     * \details This function returns the timer resolution used by "time"
     */
    static double tick();


    /*!
     * \brief   Change the level of the internal timers
     * \details This function changes the level of the timers used to profile MPI
     * \param level         New level of the timers
     */
    static void changeProfileLevel(int level) { profile_level = level; }


private:
    // The internal MPI communicator
    MPI_Comm  communicator;
    
    // Is the communicator NULL
    bool d_isNull;

    // Do we want to call MPI_abort instead of exit
    bool call_abort_in_serial_instead_of_exit;

    // The rank and size of the communicator
    int comm_rank, comm_size;
    
    // The level for the profiles of MPI
    static int profile_level;

    // Some attributes
    int d_maxTag;
    int* volatile d_currentTag;

    /* How many AMP_MPI objects share the same underlying MPI communicator.
     * When the count goes to 0, the MPI comm will be free'd (assuming it was created
     * by an AMP_MPI object).  This may not be perfect, but is likely to be good enough.
     * Note that for thread safety, any access to this variable should be blocked for thread safety.
     * The value of count MUST be volatile to ensure the correct value is always used.
     */
    int* volatile d_count;

    // Add a variable for data alignment (necessary for some Intel builds)
    double tmp_alignment;

    /* We want to keep track of how many MPI_Comm objects we have created over time.
     * Like the count, for thread safety this should be blocked, however the most likely error
     * caused by not blocking is a slight error in the MPI count.  Since this is just for reference
     * we do not need to block (recognizing that the value may not be 100% accurate).
     */
    static volatile unsigned int N_MPI_Comm_created;
    static volatile unsigned int N_MPI_Comm_destroyed;

    // Private helper functions for templated MPI operations;
    template <class type>  void call_sumReduce(type *x, const int n=1) const;
    template <class type>  void call_sumReduce(const type *x, type *y, const int n=1) const;
    template <class type>  void call_minReduce(type *x, const int n=1, int *rank_of_min=NULL) const;
    template <class type>  void call_minReduce(const type *x, type *y, const int n=1, int *rank_of_min=NULL) const;
    template <class type>  void call_maxReduce(type *x, const int n=1, int *rank_of_max=NULL) const;
    template <class type>  void call_maxReduce(const type *x, type *y, const int n=1, int *rank_of_max=NULL) const;
    template <class type>  void call_bcast(type *x, const int n, const int root) const;
    template <class type>  void call_allGather(const type x_in, type *x_out) const;
    template <class type>  void call_allGather(const type *x_in, int size_in, type *x_out, int *size_out, int *disp_out) const;
    template <class type>  void call_sumScan(const type *x, type *y, int n=1) const;
    template <class type>  void call_minScan(const type *x, type *y, int n=1) const;
    template <class type>  void call_maxScan(const type *x, type *y, int n=1) const;
    template <class type>  void call_allToAll(const type *send_data, const int send_cnt[], 
        const int send_disp[], type *recv_data, const int *recv_cnt, const int *recv_disp) const;

};


// Include the default instantiations
// \cond HIDDEN_SYMBOLS
#include "AMP_MPI.I"
// \endcond

}

#endif
