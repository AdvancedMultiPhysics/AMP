#include "AMP/matrices/data/hypre/HypreMatrixAdaptor.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"

#include <numeric>

#include "ProfilerApp.h"

#include "HYPRE_config.h"
#include "HYPRE_utilities.h"
#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"

namespace AMP::LinearAlgebra {


HypreMatrixAdaptor::HypreMatrixAdaptor( std::shared_ptr<MatrixData> matrixData )
{
    PROFILE( "HypreMatrixAdaptor::HypreMatrixAdaptor" );

    int ierr;
    char hypre_mesg[100];

    auto firstRow = static_cast<HYPRE_BigInt>( matrixData->beginRow() );
    auto lastRow  = static_cast<HYPRE_BigInt>( matrixData->endRow() - 1 );
    auto comm     = matrixData->getComm().getCommunicator();

    HYPRE_IJMatrixCreate( comm, firstRow, lastRow, firstRow, lastRow, &d_matrix );
    HYPRE_IJMatrixSetObjectType( d_matrix, HYPRE_PARCSR );
    HYPRE_IJMatrixSetMaxOffProcElmts( d_matrix, 0 );

    // For now all configurations turn off vendor spmv, but later there
    // may some cases where turning this on is a good idea
#if defined( HYPRE_USING_GPU ) && defined( HYPRE_USING_CUSPARSE ) && CUSPARSE_VERSION >= 11000
    // CUSPARSE_SPMV_ALG_DEFAULT doesn't provide deterministic results
    // hypre comment from test/ij.c
    // Note we crash without turning this off for Cuda 12.3-12.8 && hypre 2.31-33 with managed
    // memory
    HYPRE_Int spmv_use_vendor = 0;
#else
    HYPRE_Int spmv_use_vendor = 0;
#endif

    HYPRE_SetSpMVUseVendor( spmv_use_vendor );

    if ( matrixData->mode() < std::numeric_limits<std::uint16_t>::max() ) {
        LinearAlgebra::csrVisit( matrixData,
                                 [this]( auto csr_ptr ) { initializeHypreMatrix( csr_ptr ); } );
    } else {
        PROFILE( "HypreMatrixAdaptor::HypreMatrixAdaptor(deep copy)" );

        AMP_WARN_ONCE( "HypreMatrixAdaptor: Reverting to deep copy" );

        HYPRE_SetMemoryLocation( HYPRE_MEMORY_HOST );
        HYPRE_IJMatrixInitialize( d_matrix );

        // iterate over all rows
        for ( auto i = firstRow; i <= lastRow; ++i ) {

            std::vector<size_t> cols;
            std::vector<double> values;

            matrixData->getRowByGlobalID( i, cols, values );
            std::vector<HYPRE_BigInt> hypre_cols( cols.size() );
            std::copy( cols.begin(), cols.end(), hypre_cols.begin() );

            const int nrows   = 1;
            HYPRE_BigInt irow = i;
            HYPRE_Int ncols   = cols.size();
            auto data         = reinterpret_cast<const HYPRE_Real *>( values.data() );

            ierr =
                HYPRE_IJMatrixSetValues( d_matrix, nrows, &ncols, &irow, hypre_cols.data(), data );
            HYPRE_DescribeError( ierr, hypre_mesg );
        }

        HYPRE_IJMatrixAssemble( d_matrix );
    }
}

HypreMatrixAdaptor::~HypreMatrixAdaptor() { HYPRE_IJMatrixDestroy( d_matrix ); }

template<class Config>
void HypreMatrixAdaptor::initializeHypreMatrix( std::shared_ptr<CSRMatrixData<Config>> csrData )
{
    // The hypre vs amp ownership rules require elaboration.
    // We set the internal owns_data flags on the diag and offd
    // hypre blocks to false so that hypre (mostly) doesn't
    // deallocate things we own and pass in shallow-ly.
    // The row pointers (->i) and row nonzero counts (->rownnz)
    // are not guarded by that flag, so we let hypre allocate those
    // and copy into them. Luckily they are quite small relative to
    // everything else.
    // The parcsr matrix holding those blocks does *not* have
    // ownership turned off. We do want hypre to allocate the diag and
    // offd structs internally as well as all their comm info.
    // The offd column map is also owned by hypre, but since nnz is
    // zero at creation time we need to allocate it for them.

    PROFILE( "HypreMatrixAdaptor::initializeHypreMatrix" );

    using alloc_t          = typename Config::allocator_type;
    const auto csr_mem_loc = AMP::Utilities::getAllocatorMemoryType<alloc_t>();

    // Set the hypre memory space, matching the space of the input matrix
    // if possible.
    // If matching spaces is not possible then issue a warning to the user,
    // migrate the matrix, and call this function again. This lets the rest
    // of the adaptor work with matching spaces.
    HYPRE_MemoryLocation memory_location = HYPRE_MEMORY_HOST;
    if ( csr_mem_loc > AMP::Utilities::MemoryType::host ) {
        // Hypre builds with only unified memory or only pure device memory,
        // and does not support both in one installation. They only recognize
        // device from the outside and map it down internally.
#if defined( HYPRE_USING_DEVICE_MEMORY )
        if ( csr_mem_loc != AMP::Utilities::MemoryType::device ) {
            // hypre built for device memory but this took in a managed-space matrix
            AMP_WARN_ONCE(
                "HypreMatrixAdaptor: Hypre was with GPU device memory support but a managed memory "
                "matrix was passed in.\nMatrix will be migrated to device." );
            auto migrated      = csrData->template migrate<HypreConfig<alloc::device>>();
            d_csrdata_migrated = migrated; // hold reference ensuring raw pointers stay valid
            initializeHypreMatrix( migrated );
            return;
        }
        memory_location = HYPRE_MEMORY_DEVICE;
#elif defined( HYPRE_USING_UNIFIED_MEMORY )
        if ( csr_mem_loc != AMP::Utilities::MemoryType::managed ) {
            // hypre built for managed memory but this took in a device-space matrix
            AMP_WARN_ONCE(
                "HypreMatrixAdaptor: Hypre was with GPU managed memory support but a device memory "
                "matrix was passed in.\nMatrix will be migrated to managed memory." );
            auto migrated      = csrData->template migrate<HypreConfig<alloc::managed>>();
            d_csrdata_migrated = migrated; // hold reference ensuring raw pointers stay valid
            initializeHypreMatrix( migrated );
            return;
        }
        memory_location = HYPRE_MEMORY_DEVICE;
#else
        // neither flag defined means we stay with host memory, issue a warning and move on
        AMP_WARN_ONCE( "HypreMatrixAdaptor: Hypre was not built with GPU support but a device "
                       "matrix was passed in.\nMatrix will be migrated to host, consider "
                       "re-building Hypre with GPU support." );
        auto migrated      = csrData->template migrate<HypreConfig<alloc::host>>();
        d_csrdata_migrated = migrated; // hold reference ensuring raw pointers stay valid
        initializeHypreMatrix( migrated );
        return;
#endif
    }
    HYPRE_SetMemoryLocation( memory_location );

    // Manually create ParCSR and fill fields as needed
    // Roughly based on hypre_IJMatrixInitializeParCSR_v2 from IJMatrix_parcsr.c
    //   and the various functions that it calls
    hypre_IJMatrixCreateParCSR( d_matrix );
    hypre_ParCSRMatrix *par_matrix = static_cast<hypre_ParCSRMatrix *>( d_matrix->object );
    hypre_CSRMatrix *diag          = par_matrix->diag;
    hypre_CSRMatrix *off_diag      = par_matrix->offd;

    // Filling the contents manually should remove any need for aux matrix
    hypre_AuxParCSRMatrix *aux_mat = static_cast<hypre_AuxParCSRMatrix *>( d_matrix->translator );
    aux_mat->need_aux              = 0;

    // Verify that diag and off_diag are "empty"
    AMP_DEBUG_INSIST( diag->num_nonzeros == 0 && off_diag->num_nonzeros == 0,
                      "Hypre (off)diag matrix has nonzeros but shouldn't" );

    diag->memory_location     = memory_location;
    off_diag->memory_location = memory_location;

    // Hypre always frees the hypre_CSRMatrix->i and hypre_CSRMatrix->rownnz
    // fields regardless of ->owns_data. Calling matrix initialize will let
    // hypre do those allocations. ->big_j, ->j, and ->data should not get
    // allocated since ->num_nonzeros == 0 (see above).
    hypre_CSRMatrixInitialize( diag );
    hypre_CSRMatrixInitialize( off_diag );

    // extract fields from csrData
    const auto first_row    = static_cast<HYPRE_BigInt>( csrData->beginRow() );
    const auto last_row     = static_cast<HYPRE_BigInt>( csrData->endRow() - 1 );
    const auto nnz_total_d  = static_cast<HYPRE_BigInt>( csrData->numberOfNonZerosDiag() );
    const auto nnz_total_od = static_cast<HYPRE_BigInt>( csrData->numberOfNonZerosOffDiag() );
    const bool haveOffd     = csrData->hasOffDiag();
    const auto nrows        = last_row - first_row + 1;
    auto [rs_d, cols_d, cols_loc_d, coeffs_d]     = csrData->getDiagMatrix()->getDataFields();
    auto [rs_od, cols_od, cols_loc_od, coeffs_od] = csrData->getOffdMatrix()->getDataFields();

    AMP_INSIST( rs_d && cols_loc_d && coeffs_d, "diagonal block layout cannot be NULL" );

    // Fill in the ->i fields of diag and off_diag
    AMP::Utilities::Algorithms::copy_n( diag->i, rs_d, nrows + 1, csr_mem_loc );
    if ( haveOffd ) {
        AMP::Utilities::Algorithms::copy_n( off_diag->i, rs_od, nrows + 1, csr_mem_loc );
    } else {
        AMP::Utilities::Algorithms::zero_n( off_diag->i, nrows + 1, csr_mem_loc );
    }

    // This is where we tell hypre to stop owning any data
    hypre_CSRMatrixSetDataOwner( diag, 0 );
    hypre_CSRMatrixSetDataOwner( off_diag, 0 );

    // Now set diag/off_diag members to point at our data
    diag->big_j = NULL;
    diag->data  = reinterpret_cast<HYPRE_Real *>( coeffs_d );
    diag->j     = cols_loc_d;

    off_diag->big_j = NULL;
    off_diag->data  = reinterpret_cast<HYPRE_Real *>( coeffs_od );
    off_diag->j     = cols_loc_od;

    // Update metadata fields to match what we've inserted!
    diag->num_nonzeros     = nnz_total_d;
    off_diag->num_nonzeros = nnz_total_od;

    // Set colmap inside ParCSR and flag that assembly is already done
    // See comment above regarding ownership of these fields
    if ( haveOffd ) {
        auto colMap        = csrData->getOffdMatrix()->getColumnMap();
        off_diag->num_cols = csrData->getOffdMatrix()->numUniqueColumns();

        // always allocate and set host side offd map
        par_matrix->col_map_offd =
            hypre_TAlloc( HYPRE_BigInt, off_diag->num_cols, HYPRE_MEMORY_HOST );
        AMP::Utilities::Algorithms::copyCast( par_matrix->col_map_offd,
                                              AMP::Utilities::MemoryType::host,
                                              colMap,
                                              csr_mem_loc,
                                              off_diag->num_cols );

        // and do device map if needed
        if ( memory_location == HYPRE_MEMORY_DEVICE ) {
            par_matrix->device_col_map_offd =
                hypre_TAlloc( HYPRE_BigInt, off_diag->num_cols, HYPRE_MEMORY_DEVICE );
            AMP::Utilities::Algorithms::copyCast( par_matrix->device_col_map_offd,
                                                  csr_mem_loc,
                                                  colMap,
                                                  csr_mem_loc,
                                                  off_diag->num_cols );
        }
    }

    // Update ->rownnz fields, note that we don't own these
    hypre_CSRMatrixSetRownnz( diag );
    hypre_CSRMatrixSetRownnz( off_diag );

    // set assemble flag to indicate that we are done
    d_matrix->assemble_flag = 1;
}

} // namespace AMP::LinearAlgebra
