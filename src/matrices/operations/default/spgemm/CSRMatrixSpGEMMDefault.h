#ifndef included_AMP_CSRMatrixSpGEMMDefault
#define included_AMP_CSRMatrixSpGEMMDefault

#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/utils/AMP_MPI.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace AMP::LinearAlgebra {

template<typename Policy, class Allocator, class DiagMatrixData>
class CSRMatrixSpGEMMHelperDefault
{
    using CSRData = CSRMatrixData<Policy, Allocator, DiagMatrixData>;
    using lidx_t  = typename Policy::lidx_t;
    using gidx_t  = typename Policy::gidx_t;

public:
    CSRMatrixSpGEMMHelperDefault() = default;
    CSRMatrixSpGEMMHelperDefault( CSRData *A_, CSRData *B_, CSRData *C_ )
        : A( A_ ), B( B_ ), C( C_ ), comm( A->getComm() )
    {
        AMP_DEBUG_INSIST(
            comm == B->getComm() && comm == C->getComm(),
            "CSRMatrixSpGEMMHelperDefault: All three matrices must have the same communicator" );
    }

    ~CSRMatrixSpGEMMHelperDefault() = default;

    void symbolicMultiply();
    void numericMultiply();

protected:
    // helper for main symbolicMultiply function that acts on specfic
    // pairs of blocks in A and B
    template<class AMatrixData, class BMatrixData>
    void symbolicMultiply( std::shared_ptr<AMatrixData> A_data,
                           std::shared_ptr<BMatrixData> B_data,
                           const gidx_t col_diag_start,
                           const gidx_t col_diag_end,
                           const bool is_diag,
                           std::vector<std::set<gidx_t>> &C_Cols );

    // helper for main numericMultiply function that acts on specfic
    // pairs of blocks in A and B
    template<class AMatrixData, class BMatrixData, class CMatrixData>
    void numericMultiply( std::shared_ptr<AMatrixData> A_data,
                          std::shared_ptr<BMatrixData> B_data,
                          std::shared_ptr<CMatrixData> C_data );

    // This plans all communication needed for BRemote creation
    void createBRemoteCommInfo();

    // This uses comm info from createBRemoteCommInfo to
    // set pattern of BRemote and trigger all internal allocations
    void createBRemoteSymbolic();

    // This only (re-)fills the coefficients in BRemote
    // the comm info and symbolic creation must have already happened
    void fillBRemoteNumeric();

    // Matrix data of operands and output
    // these are non-owning pointers
    CSRData *A;
    CSRData *B;
    CSRData *C;

    // Communicator
    AMP_MPI comm;

    // Matrix data formed from remote rows of B that get pulled to each process
    // This is a single block for all columns in remote rows. It is much easier
    // to split these between C_diag and C_offd after receiving them
    std::shared_ptr<DiagMatrixData> BRemote;

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
        std::vector<lidx_t> rowids;
        // number of non-zeros in those rows
        std::vector<lidx_t> rownnz;
        // where to put each row in BRemote
        // only used by d_src_info
        std::vector<lidx_t> brow;
    };

    // Source information, things expected from other ranks
    std::map<int, SpGEMMCommInfo> d_src_info;

    // Destination information, things sent to other ranks
    std::map<int, SpGEMMCommInfo> d_dest_info;
};

} // namespace AMP::LinearAlgebra

#endif
