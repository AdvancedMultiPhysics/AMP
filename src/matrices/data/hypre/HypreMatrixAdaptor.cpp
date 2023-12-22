#include "AMP/matrices/data/hypre/HypreMatrixAdaptor.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/utils/AMP_MPI.h"

namespace AMP::LinearAlgebra {

HypreMatrixAdaptor::HypreMatrixAdaptor( std::shared_ptr<MatrixData> matrixData )
{
    int ierr;
    char hypre_mesg[100];

    HYPRE_BigInt firstRow = static_cast<HYPRE_BigInt>( matrixData->beginRow() );
    HYPRE_BigInt lastRow  = static_cast<HYPRE_BigInt>( matrixData->endRow() - 1 );
    auto comm             = matrixData->getComm().getCommunicator();

    HYPRE_IJMatrixCreate( comm, firstRow, lastRow, firstRow, lastRow, &d_matrix );
    HYPRE_IJMatrixSetObjectType( d_matrix, HYPRE_PARCSR );

    HYPRE_IJMatrixSetMaxOffProcElmts( d_matrix, 0 );

    auto csrData = std::dynamic_pointer_cast<CSRMatrixData>( matrixData );
    if ( csrData ) {

        HYPRE_Int *nnz_per_row = nullptr;
        HYPRE_BigInt *csr_ja   = nullptr;
        HYPRE_Real *csr_aa     = nullptr;

        AMP_INSIST( nnz_per_row && csr_ja && csr_aa, "nnz_per_row, csr_ja, csr_aa cannot be NULL" );
        initializeHypreMatrix( firstRow, lastRow, nnz_per_row, csr_ja, csr_aa );
        AMP_ERROR( "Not implemented" );

    } else {

        // figure out how to incorporate this
        // HYPRE_IJMatrixSetRowSizes( d_matrix, nnz_per_row );

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

HypreMatrixAdaptor::~HypreMatrixAdaptor() { HYPRE_IJMatrixDestroy( d_matrix ); }

static void
set_row_ids_( HYPRE_BigInt const first_row, HYPRE_BigInt const nrows, HYPRE_BigInt *row_ids )
{
    AMP_ERROR( "Not implemented" );
}

void HypreMatrixAdaptor::initializeHypreMatrix( HYPRE_BigInt first_row,
                                                HYPRE_BigInt last_row,
                                                HYPRE_Int *const nnz_per_row,
                                                HYPRE_BigInt *const csr_ja,
                                                HYPRE_Real *const csr_aa )
{
    const auto nrows = last_row - first_row + 1;

    HYPRE_IJMatrixSetRowSizes( d_matrix, nnz_per_row );

    // The next 2 lines affect efficiency and should be resurrected at some point
    //  set_row_location_(d_first_row, d_last_row, nrows, nnz_per_row, csr_ia, csr_ja,
    //  number_of_local_cols, number_of_remote_cols ); HYPRE_IJMatrixSetDiagOffdSizes( d_matrix,
    //  number_of_local_cols.data(), number_of_remote_cols.data() );

    HYPRE_IJMatrixInitialize( d_matrix );

    HYPRE_BigInt *row_ids = nullptr;
    AMP_INSIST( row_ids, "row_ids cannot be NULL" );
    set_row_ids_( first_row, nrows, row_ids );
    HYPRE_IJMatrixSetValues( d_matrix, nrows, nnz_per_row, row_ids, csr_ja, csr_aa );
    HYPRE_IJMatrixAssemble( d_matrix );
}

} // namespace AMP::LinearAlgebra