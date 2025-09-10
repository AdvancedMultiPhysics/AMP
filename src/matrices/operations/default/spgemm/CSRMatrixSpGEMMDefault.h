#ifndef included_AMP_CSRMatrixSpGEMMDefault
#define included_AMP_CSRMatrixSpGEMMDefault

#include "AMP/matrices/data/CSRMatrixCommunicator.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/utils/AMP_MPI.h"

#include <map>
#include <memory>
#include <vector>

namespace AMP::LinearAlgebra {

template<typename Config>
class CSRMatrixSpGEMMDefault
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

    CSRMatrixSpGEMMDefault() = default;
    CSRMatrixSpGEMMDefault( std::shared_ptr<matrixdata_t> A_,
                            std::shared_ptr<matrixdata_t> B_,
                            std::shared_ptr<matrixdata_t> C_,
                            bool overlap_comms_ )
        : A( A_ ),
          B( B_ ),
          C( C_ ),
          A_diag( A->getDiagMatrix() ),
          A_offd( A->getOffdMatrix() ),
          B_diag( B->getDiagMatrix() ),
          B_offd( B->getOffdMatrix() ),
          C_diag( C->getDiagMatrix() ),
          C_offd( C->getOffdMatrix() ),
          d_overlap_comms( ( A->getComm().getSize() > 1 ) && overlap_comms_ ),
          d_num_rows( static_cast<lidx_t>( A->numLocalRows() ) ),
          comm( A->getComm() ),
          d_csr_comm( A->getRightCommList() ),
          d_need_comms( true )
    {
        AMP_DEBUG_INSIST(
            comm == B->getComm() && comm == C->getComm(),
            "CSRMatrixSpGEMMDefault: All three matrices must have the same communicator" );
    }

    ~CSRMatrixSpGEMMDefault() = default;

    void symbolicMultiply();
    void numericMultiply();
    void numericMultiplyReuse();

protected:
    void symbolicMultiply_NonOverlapped();
    void numericMultiply_NonOverlapped();
    void symbolicMultiply_Overlapped();
    void numericMultiply_Overlapped();

    enum class Mode { SYMBOLIC, NUMERIC };
    enum class BlockType { DIAG, OFFD };

    template<Mode mode_t, BlockType block_t>
    void multiply( std::shared_ptr<localmatrixdata_t> A_data,
                   std::shared_ptr<localmatrixdata_t> B_data,
                   std::shared_ptr<localmatrixdata_t> C_data );

    template<Mode mode_t, BlockType block_t>
    void multiplyFused( std::shared_ptr<localmatrixdata_t> B_data,
                        std::shared_ptr<localmatrixdata_t> BR_data,
                        std::shared_ptr<localmatrixdata_t> C_data );

    template<BlockType block_t>
    void multiplyReuse( std::shared_ptr<localmatrixdata_t> A_data,
                        std::shared_ptr<localmatrixdata_t> B_data,
                        std::shared_ptr<localmatrixdata_t> C_data );

    void setupBRemoteComm();
    void startBRemoteComm();
    void endBRemoteComm();

    void mergeDiag();
    void mergeOffd();

    // default starting size for sparse accumulators
    static constexpr lidx_t SPACC_SIZE = 256;

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

    // flag for whether overlapped communication/computation should be done
    bool d_overlap_comms;

    // number of local rows in A and C are the same, and many loops
    // run over this range
    lidx_t d_num_rows;

    // Communicator
    AMP_MPI comm;
    CSRMatrixCommunicator<Config> d_csr_comm;
    bool d_need_comms;

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

    // Internal row accumlator classes
    template<typename col_t>
    struct DenseAccumulator {
        DenseAccumulator( int capacity_, gidx_t offset_ )
            : capacity( capacity_ ),
              offset( offset_ ),
              num_inserted( 0 ),
              total_inserted( 0 ),
              total_collisions( 0 ),
              total_probe_steps( 0 ),
              total_clears( 0 ),
              total_grows( 0 ),
              flags( capacity, -1 )
        {
            static_assert( std::is_same_v<col_t, gidx_t> || std::is_same_v<col_t, lidx_t> );
        }

        void insert_or_append( col_t col_idx );
        void insert_or_append( col_t col_idx, scalar_t val, col_t *col_space, scalar_t *val_space );
        void clear();
        lidx_t contains( col_t col_idx ) const;
        void set_flag( col_t col_idx, lidx_t k );

        static constexpr bool IsGlobal = std::is_same_v<gidx_t, col_t>;

        const lidx_t capacity;
        const col_t offset;
        lidx_t num_inserted;
        size_t total_inserted;
        size_t total_collisions;
        size_t total_probe_steps;
        size_t total_clears;
        size_t total_grows;
        std::vector<lidx_t> flags;
        std::vector<lidx_t> flag_inv;
        std::vector<col_t> cols;
    };

    template<typename col_t>
    struct SparseAccumulator {
        SparseAccumulator( int capacity_, gidx_t offset_ )
            : capacity( capacity_ ),
              offset( offset_ ),
              num_inserted( 0 ),
              total_inserted( 0 ),
              total_collisions( 0 ),
              total_probe_steps( 0 ),
              total_clears( 0 ),
              total_grows( 0 ),
              flags( capacity, 0xFFFF )
        {
            AMP_DEBUG_ASSERT( capacity > 1 );
            static_assert( std::is_same_v<col_t, gidx_t> || std::is_same_v<col_t, lidx_t> );
        }

        uint16_t hash( col_t col_idx ) const;
        void insert_or_append( col_t col_idx );
        void insert_or_append( col_t col_idx, scalar_t val, col_t *col_space, scalar_t *val_space );
        void clear();
        lidx_t contains( col_t col_idx ) const;
        void set_flag( col_t col_idx, lidx_t k );

        static constexpr bool IsGlobal = std::is_same_v<gidx_t, col_t>;

        uint16_t capacity;
        const gidx_t offset;
        uint16_t num_inserted;
        size_t total_inserted;
        size_t total_collisions;
        size_t total_probe_steps;
        size_t total_clears;
        size_t total_grows;
        std::vector<uint16_t> flags;
        std::vector<col_t> cols;

    private:
        void grow( col_t *col_space );
    };
};

} // namespace AMP::LinearAlgebra

#endif
