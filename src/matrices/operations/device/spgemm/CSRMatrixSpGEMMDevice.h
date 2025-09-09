#ifndef included_AMP_CSRMatrixSpGEMMDevice
#define included_AMP_CSRMatrixSpGEMMDevice

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/data/CSRMatrixCommunicator.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Memory.h"

#ifdef AMP_USE_CUDA
    #include "AMP/matrices/operations/device/spgemm/cuda/SpGEMM_Cuda.h"
#endif

#ifdef AMP_USE_HIP
    #include "AMP/matrices/operations/device/spgemm/hip/SpGEMM_Hip.h"
#endif

#include <map>
#include <memory>
#include <type_traits>
#include <vector>

namespace AMP::LinearAlgebra {

template<typename Config>
class CSRMatrixSpGEMMDevice
{
public:
    using allocator_type    = typename Config::allocator_type;
    using config_type       = Config;
    using matrixdata_t      = CSRMatrixData<Config>;
    using localmatrixdata_t = typename matrixdata_t::localmatrixdata_t;
    using lidx_t            = typename Config::lidx_t;
    using gidx_t            = typename Config::gidx_t;
    using scalar_t          = typename Config::scalar_t;

    static_assert( std::is_same_v<typename allocator_type::value_type, void> );

    enum class BlockType { DIAG, OFFD };

    CSRMatrixSpGEMMDevice() = default;
    CSRMatrixSpGEMMDevice( std::shared_ptr<matrixdata_t> A_,
                           std::shared_ptr<matrixdata_t> B_,
                           std::shared_ptr<matrixdata_t> C_ )
        : A( A_ ),
          B( B_ ),
          C( C_ ),
          A_diag( A->getDiagMatrix() ),
          A_offd( A->getOffdMatrix() ),
          B_diag( B->getDiagMatrix() ),
          B_offd( B->getOffdMatrix() ),
          C_diag( C->getDiagMatrix() ),
          C_offd( C->getOffdMatrix() ),
          d_num_rows( static_cast<lidx_t>( A->numLocalRows() ) ),
          comm( A->getComm() ),
          d_csr_comm( A->getRightCommList() )
    {
        AMP_DEBUG_INSIST(
            comm == B->getComm() && comm == C->getComm(),
            "CSRMatrixSpGEMMDevice: All three matrices must have the same communicator" );
    }

    ~CSRMatrixSpGEMMDevice() = default;

    void multiply();

    template<BlockType block_t>
    void multiply( std::shared_ptr<localmatrixdata_t> A_data,
                   std::shared_ptr<localmatrixdata_t> B_data,
                   std::shared_ptr<localmatrixdata_t> C_data );

protected:
    void setupBRemoteComm();
    void startBRemoteComm();
    void endBRemoteComm();

    void mergeDiag();
    void mergeOffd();

    // Matrix data of operands and output
    std::shared_ptr<matrixdata_t> A;
    std::shared_ptr<matrixdata_t> B;
    std::shared_ptr<matrixdata_t> C;

    // diag and offd blocks of input matrices
    std::shared_ptr<localmatrixdata_t> A_diag;
    std::shared_ptr<localmatrixdata_t> A_offd;
    std::shared_ptr<localmatrixdata_t> B_diag;
    std::shared_ptr<localmatrixdata_t> B_offd;

    // Matrix data formed from remote rows of B that get pulled to each process
    std::shared_ptr<localmatrixdata_t> BR_diag;
    std::shared_ptr<localmatrixdata_t> BR_offd;

    // Blocks of C matrix
    std::shared_ptr<localmatrixdata_t> C_diag;
    std::shared_ptr<localmatrixdata_t> C_offd;

    // number of local rows in A and C are the same, and many loops
    // run over this range
    lidx_t d_num_rows;

    // Communicator
    AMP_MPI comm;
    CSRMatrixCommunicator<Config> d_csr_comm;

    // To overlap comms and calcs it is easiest to form the output in four
    // blocks and merge them together at the end
    std::shared_ptr<localmatrixdata_t> C_diag_diag; // from A_diag * B_diag
    std::shared_ptr<localmatrixdata_t> C_diag_offd; // from A_diag * B_offd
    std::shared_ptr<localmatrixdata_t> C_offd_diag; // from A_offd * BR_diag
    std::shared_ptr<localmatrixdata_t> C_offd_offd; // from A_offd * BR_offd

    // The following all support the communication needed to build BRemote
    // these are worth preserving to allow repeated SpGEMMs to re-use the
    // structure with potentially new coefficients

    // struct to hold fields that are needed in both
    // the "source" and "destination" perspectives
    struct SpGEMMCommInfo {
        SpGEMMCommInfo() = default;
        SpGEMMCommInfo( int numrow_ ) : numrow( numrow_ ) {}
        // number of rows to send or receive
        int numrow;
        // ids of rows to send/receive
        std::vector<gidx_t> rowids;
        // number of non-zeros in those rows
        std::vector<lidx_t> rownnz;
    };

    // Source information, things expected from other ranks
    std::map<int, SpGEMMCommInfo> d_src_info;

    // Destination information, things sent to other ranks
    std::map<int, SpGEMMCommInfo> d_dest_info;

    std::map<int, std::shared_ptr<localmatrixdata_t>> d_send_matrices;
    std::map<int, std::shared_ptr<localmatrixdata_t>> d_recv_matrices;
};

} // namespace AMP::LinearAlgebra

#endif
