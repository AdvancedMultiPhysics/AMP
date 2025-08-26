#include "AMP/matrices/data/hypre/HypreMatrixAdaptor.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"

#include <numeric>

#include "ProfilerApp.h"

#include "HYPRE_utilities.h"
#include "_hypre_IJ_mv.h"
#include "_hypre_parcsr_mv.h"

namespace AMP::LinearAlgebra {

// TODO: inst with only hypre config types
#define CSR_INST( alloc )                                                                       \
    template void HypreMatrixAdaptor::initializeHypreMatrix<CSRMatrixData<HypreConfig<alloc>>>( \
        std::shared_ptr<CSRMatrixData<HypreConfig<alloc>>>, HYPRE_MemoryLocation );
CSR_INST( alloc::host )
#ifdef AMP_USE_DEVICE
CSR_INST( alloc::managed )
CSR_INST( alloc::device )
#endif
#undef CSR_INST

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


    // Attempt dynamic pointer casts to supported types
    // Config must match HypreConfig with one of the allocators
    // need to match supported allocators depending on device support
    auto csrDataHost =
        std::dynamic_pointer_cast<CSRMatrixData<HypreConfig<alloc::host>>>( matrixData );

#ifdef AMP_USE_DEVICE
    auto csrDataManaged =
        std::dynamic_pointer_cast<CSRMatrixData<HypreConfig<alloc::managed>>>( matrixData );
    auto csrDataDevice =
        std::dynamic_pointer_cast<CSRMatrixData<HypreConfig<alloc::device>>>( matrixData );

    // Hypre can *not* support pure device memory and managed (unified) memory
    // at the same time. If we have a matrix in the wrong space then migrate it
    #if defined( HYPRE_USING_DEVICE_MEMORY )
    if ( csrDataManaged ) {
        AMP_WARN_ONCE(
            "HypreMatrixAdaptor: Hypre not built with support for managed memory, matrix "
            "will be migrated to device." );
        d_csrdata_dev = csrDataManaged->migrate<HypreConfig<alloc::device>>(
            AMP::Utilities::Backend::Hip_Cuda );
        csrDataDevice =
            std::dynamic_pointer_cast<CSRMatrixData<HypreConfig<alloc::device>>>( d_csrdata_dev );
        csrDataManaged = nullptr;
    }
    #elif defined( HYPRE_USING_UNIFIED_MEMORY )
    if ( csrDataDevice ) {
        AMP_WARN_ONCE( "HypreMatrixAdaptor: Hypre not built with support for device memory, matrix "
                       "will be migrated to managed." );
        d_csrdata_dev = csrDataDevice->migrate<HypreConfig<alloc::managed>>(
            AMP::Utilities::Backend::Hip_Cuda );
        csrDataManaged =
            std::dynamic_pointer_cast<CSRMatrixData<HypreConfig<alloc::managed>>>( d_csrdata_dev );
        csrDataDevice = nullptr;
    }
    #endif

#else
    // Just default out these to nullptrs to make logic below simpler
    decltype( csrDataHost ) csrDataManaged = nullptr;
    decltype( csrDataHost ) csrDataDevice  = nullptr;
#endif

    if ( csrDataHost ) {
        initializeHypreMatrix( csrDataHost, HYPRE_MEMORY_HOST );
    } else if ( csrDataManaged ) {
        initializeHypreMatrix( csrDataManaged, HYPRE_MEMORY_DEVICE );
    } else if ( csrDataDevice ) {
        initializeHypreMatrix( csrDataDevice, HYPRE_MEMORY_DEVICE );
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

            const int nrows  = 1;
            const auto irow  = i;
            const auto ncols = cols.size();

            ierr = HYPRE_IJMatrixSetValues( d_matrix,
                                            nrows,
                                            (HYPRE_Int *) &ncols,
                                            (HYPRE_BigInt *) &irow,
                                            hypre_cols.data(),
                                            (const HYPRE_Real *) values.data() );
            HYPRE_DescribeError( ierr, hypre_mesg );
        }

        HYPRE_IJMatrixAssemble( d_matrix );
    }
}

HypreMatrixAdaptor::~HypreMatrixAdaptor()
{
    hypre_ParCSRMatrix *par_matrix = static_cast<hypre_ParCSRMatrix *>( d_matrix->object );
    // Now the standard IJMatrixDestroy can be called
    HYPRE_IJMatrixDestroy( d_matrix );
}

template<class csr_data_type>
void HypreMatrixAdaptor::initializeHypreMatrix( std::shared_ptr<csr_data_type> csrData,
                                                HYPRE_MemoryLocation memory_location )
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

    HYPRE_SetMemoryLocation( memory_location );

    // extract fields from csrData
    const auto first_row    = static_cast<HYPRE_BigInt>( csrData->beginRow() );
    const auto last_row     = static_cast<HYPRE_BigInt>( csrData->endRow() - 1 );
    const auto nnz_total_d  = static_cast<HYPRE_BigInt>( csrData->numberOfNonZerosDiag() );
    const auto nnz_total_od = static_cast<HYPRE_BigInt>( csrData->numberOfNonZerosOffDiag() );
    auto [rs_d, cols_d, cols_loc_d, coeffs_d]     = csrData->getDiagMatrix()->getDataFields();
    auto [rs_od, cols_od, cols_loc_od, coeffs_od] = csrData->getOffdMatrix()->getDataFields();
    const bool haveOffd                           = csrData->hasOffDiag();

    AMP_INSIST( rs_d && cols_loc_d && coeffs_d, "diagonal block layout cannot be NULL" );

    const auto nrows = last_row - first_row + 1;

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

    AMP_INSIST( AMP::Utilities::getMemoryType( rs_d ) == AMP::Utilities::getMemoryType( diag->i ),
                "HypreMatrixAdaptor::initializeHypreMatrix: Hypre and native representations "
                "must be in same memory space" );

    // Fill in the ->i fields of diag and off_diag
    AMP::Utilities::copy( static_cast<size_t>( nrows + 1 ), rs_d, diag->i );
    if ( haveOffd ) {
        AMP::Utilities::copy( static_cast<size_t>( nrows + 1 ), rs_od, off_diag->i );
    } else {
        AMP::Utilities::Algorithms<HYPRE_Int>::fill_n(
            off_diag->i, static_cast<size_t>( nrows + 1 ), 0 );
    }

    // This is where we tell hypre to stop owning any data
    hypre_CSRMatrixSetDataOwner( diag, 0 );
    hypre_CSRMatrixSetDataOwner( off_diag, 0 );

    // Now set diag/off_diag members to point at our data
    diag->big_j = NULL;
    diag->data  = coeffs_d;
    diag->j     = cols_loc_d;

    off_diag->big_j = NULL;
    off_diag->data  = coeffs_od;
    off_diag->j     = cols_loc_od;

    // Update metadata fields to match what we've inserted
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
        AMP::Utilities::copy(
            static_cast<size_t>( off_diag->num_cols ), colMap, par_matrix->col_map_offd );

        // and do device map if needed
        if ( memory_location == HYPRE_MEMORY_DEVICE ) {
            // Important: Hypre always stores this in pure device memory
            //            even if the rest is in unified (managed in amp lingo)
            par_matrix->device_col_map_offd =
                hypre_TAlloc( HYPRE_BigInt, off_diag->num_cols, HYPRE_MEMORY_DEVICE );
            AMP::Utilities::copy( static_cast<size_t>( off_diag->num_cols ),
                                  colMap,
                                  par_matrix->device_col_map_offd );
        }
    }

    // Update ->rownnz fields, note that we don't own these
    hypre_CSRMatrixSetRownnz( diag );
    hypre_CSRMatrixSetRownnz( off_diag );

    // set assemble flag to indicate that we are done
    d_matrix->assemble_flag = 1;
}

} // namespace AMP::LinearAlgebra
