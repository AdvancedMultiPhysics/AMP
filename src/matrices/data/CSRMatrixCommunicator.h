#ifndef included_AMP_CSRMatrixCommunicator_h
#define included_AMP_CSRMatrixCommunicator_h

#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/CommunicationList.h"

#include <map>
#include <set>
#include <vector>

namespace AMP::LinearAlgebra {

template<typename Config>
class CSRMatrixCommunicator
{
public:
    using gidx_t            = typename Config::gidx_t;
    using lidx_t            = typename Config::lidx_t;
    using scalar_t          = typename Config::scalar_t;
    using localmatrixdata_t = CSRLocalMatrixData<Config>;
    using allocator_type    = typename Config::allocator_type;
    static_assert( std::is_same_v<typename allocator_type::value_type, void> );
    using gidxAllocator_t =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<gidx_t>;
    using lidxAllocator_t =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<lidx_t>;
    using scalarAllocator_t =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<scalar_t>;

    CSRMatrixCommunicator() = default;
    CSRMatrixCommunicator( std::shared_ptr<CommunicationList> comm_list,
                           const bool flip_sendrecv = false )
        : d_comm( comm_list->getComm() ),
          d_send_called( false ),
          d_num_sources( 0 ),
          d_num_allowed_sources( 0 )
    {
        auto send_sizes = !flip_sendrecv ? comm_list->getSendSizes() : comm_list->getReceiveSizes();
        for ( int n = 0; n < d_comm.getSize(); ++n ) {
            if ( send_sizes[n] > 0 ) {
                d_allowed_dest.push_back( n );
            }
        }
        auto recv_sizes = !flip_sendrecv ? comm_list->getReceiveSizes() : comm_list->getSendSizes();
        for ( int n = 0; n < d_comm.getSize(); ++n ) {
            if ( recv_sizes[n] > 0 ) {
                d_num_allowed_sources++;
            }
        }
    }

    void sendMatrices( const std::map<int, std::shared_ptr<localmatrixdata_t>> &matrices );
    std::map<int, std::shared_ptr<localmatrixdata_t>>
    recvMatrices( gidx_t first_row, gidx_t last_row, gidx_t first_col, gidx_t last_col );

protected:
    void countSources( const std::map<int, std::shared_ptr<localmatrixdata_t>> &matrices );

    AMP_MPI d_comm;
    bool d_send_called;
    int d_num_sources;
    int d_num_allowed_sources;
    std::vector<int> d_allowed_dest;

    std::vector<AMP_MPI::Request> d_send_requests;

    // tags for each type of message to send/recv
    static constexpr int COMM_TEST = 5600;
    static constexpr int ROW_TAG   = 5601;
    static constexpr int COL_TAG   = 5602;
    static constexpr int COEFF_TAG = 5603;
};
} // namespace AMP::LinearAlgebra

#endif
