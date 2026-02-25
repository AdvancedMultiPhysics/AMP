#ifndef included_AMP_GhostDataHelper
#define included_AMP_GhostDataHelper

#include "AMP/vectors/CommunicationList.h"
#include "AMP/vectors/data/VectorData.h"


namespace AMP::LinearAlgebra {


template<class TYPE = double, class Allocator = AMP::HostAllocator<void>>
class GhostDataHelper : public VectorData
{
public:
    using ScalarAllocator_t =
        typename std::allocator_traits<Allocator>::template rebind_alloc<TYPE>;
    using sizetAllocator_t =
        typename std::allocator_traits<Allocator>::template rebind_alloc<size_t>;
    using intAllocator_t = typename std::allocator_traits<Allocator>::template rebind_alloc<int>;

    GhostDataHelper();
    GhostDataHelper( std::shared_ptr<CommunicationList> );
    ~GhostDataHelper();

public: // Functions overloaded from VectorData
    bool hasGhosts() const override { return d_ghostSize > 0; }
    std::shared_ptr<CommunicationList> getCommunicationList() const override;
    void setCommunicationList( std::shared_ptr<CommunicationList> comm ) override;
    size_t getGhostSize() const override;
    void fillGhosts( const Scalar & ) override;
    void setNoGhosts() override;
    bool containsGlobalElement( size_t ) const override;
    void setGhostValuesByGlobalID( size_t num,
                                   const size_t *indices,
                                   const void *vals,
                                   const typeID &id ) override;
    void addGhostValuesByGlobalID( size_t num,
                                   const size_t *indices,
                                   const void *vals,
                                   const typeID &id ) override;
    void getGhostValuesByGlobalID( size_t num,
                                   const size_t *indices,
                                   void *vals,
                                   const typeID &id ) const override;
    void getGhostValuesByGlobalIDUnsorted( size_t num,
                                           const size_t *indices,
                                           void *vals,
                                           const typeID &id ) const;
    void getGhostAddValuesByGlobalID( size_t num,
                                      const size_t *indices,
                                      void *vals,
                                      const typeID &id ) const override;
    size_t getAllGhostValues( void *vals, const typeID &id ) const override;
    UpdateState getLocalUpdateStatus() const override;
    void setUpdateStatus( UpdateState state ) override;
    void setUpdateStatusPtr( std::shared_ptr<UpdateState> rhs ) override;
    std::shared_ptr<UpdateState> getUpdateStatusPtr() const override;
    void makeConsistent( ScatterType t ) override;
    void dataChanged() override;
    const AMP_MPI &getComm() const override;
    void dumpGhostedData( std::ostream &out, size_t offset ) const override;
    void copyGhostValues( const VectorData &rhs ) override;

    using VectorData::addGhostValuesByGlobalID;
    using VectorData::getGhostAddValuesByGlobalID;
    using VectorData::getGhostValuesByGlobalID;
    using VectorData::makeConsistent;
    using VectorData::setGhostValuesByGlobalID;

public: // Write/read restart data
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;
    void writeRestart( int64_t fid ) const override;
    GhostDataHelper( int64_t fid, AMP::IO::RestartManager *manager );

protected:
    void scatter_set();
    void scatter_add();
    void deallocateBuffers();
    void allocateBuffers( size_t len );
    virtual bool allGhostIndices( size_t N, const size_t *ndx ) const;

protected:
    std::shared_ptr<CommunicationList> d_CommList = nullptr;
    std::shared_ptr<UpdateState> d_UpdateState    = nullptr;

    //! size/length of ghost and add buffers
    size_t d_ghostSize = 0;

    ScalarAllocator_t d_alloc;

    std::vector<TYPE> d_Ghosts_h;
    std::vector<TYPE> d_SendRecv_h;
    std::vector<TYPE> d_AddBuffer_h;

    TYPE *d_Ghosts    = nullptr;
    TYPE *d_AddBuffer = nullptr;
    //! Buffers for sending/receiving data
    TYPE *d_SendRecv = nullptr;

    // cache the receive dof list from CommunicationList
    size_t *d_ReceiveDOFList = nullptr;

    //! number of local ids that are remote
    size_t d_numRemote = 0;

    sizetAllocator_t d_size_t_alloc;
    intAllocator_t d_int_alloc;

    //! list of local ids that are remote
    size_t *d_localRemote = nullptr;

    // cache various communication buffers
    int *d_sendSizes         = nullptr;
    int *d_recvSizes         = nullptr;
    int *d_sendDisplacements = nullptr;
    int *d_recvDisplacements = nullptr;

    // MPI communication tags
    int d_scatter_tag;
};


} // namespace AMP::LinearAlgebra

#endif
