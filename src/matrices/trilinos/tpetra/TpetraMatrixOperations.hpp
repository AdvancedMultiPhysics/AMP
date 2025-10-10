#include "AMP/matrices/trilinos/tpetra/TpetraMatrixOperations.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/trilinos/tpetra/TpetraMatrixData.h"
#include "AMP/vectors/trilinos/tpetra/TpetraVector.h"

DISABLE_WARNINGS
#include "TpetraExt_MatrixMatrix.hpp"
#include <Tpetra_CrsMatrix_def.hpp>
//#include <Tpetra_FECrsMatrix.h>
ENABLE_WARNINGS

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

static void VerifyTpetraReturn( int err, const char *func )
{
    std::stringstream error;
    error << func << ": " << err;
    if ( err < 0 )
        AMP_ERROR( error.str() );
    if ( err > 0 )
        AMP_ERROR( error.str() );
}

template<typename ST, typename LO, typename GO, typename NT>
static Tpetra::CrsMatrix<ST, LO, GO, NT> &getTpetra_CrsMatrix( MatrixData &A )
{
    auto data = dynamic_cast<TpetraMatrixData<ST, LO, GO, NT> *>( &A );
    AMP_ASSERT( data );
    return data->getTpetra_CrsMatrix();
}

template<typename ST, typename LO, typename GO, typename NT>
static const Tpetra::CrsMatrix<ST, LO, GO, NT> &getTpetra_CrsMatrix( MatrixData const &A )
{
    const auto data = dynamic_cast<const TpetraMatrixData<ST, LO, GO, NT> *>( &A );
    AMP_ASSERT( data );
    return data->getTpetra_CrsMatrix();
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::mult( std::shared_ptr<const Vector> in,
                                                   MatrixData const &A,
                                                   std::shared_ptr<Vector> out )
{
    PROFILE( "TpetraMatrixOperations<ST, LO, GO, NT>::mult" );
    AMP_ASSERT( in->getGlobalSize() == A.numGlobalColumns() );
    AMP_ASSERT( out->getGlobalSize() == A.numGlobalRows() );
    auto in_view       = TpetraVector::constView( in );
    auto out_view      = TpetraVector::view( out );
    const auto &in_vec = in_view->getTpetra_Vector();
    auto &out_vec      = out_view->getTpetra_Vector();
    getTpetra_CrsMatrix<ST, LO, GO, NT>( A ).apply( in_vec, out_vec );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::multTranspose( std::shared_ptr<const Vector> in,
                                                            MatrixData const &A,
                                                            std::shared_ptr<Vector> out )
{
    PROFILE( "TpetraMatrixOperations<ST, LO, GO, NT>::multTranspose" );
    AMP_ASSERT( in->getGlobalSize() == A.numGlobalColumns() );
    AMP_ASSERT( out->getGlobalSize() == A.numGlobalRows() );
    auto in_view  = TpetraVector::constView( in );
    auto out_view = TpetraVector::view( out );
    getTpetra_CrsMatrix<ST, LO, GO, NT>( A ).apply(
        in_view->getTpetra_Vector(), out_view->getTpetra_Vector(), Teuchos::TRANS );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::scale( AMP::Scalar alpha, MatrixData &A )
{
    getTpetra_CrsMatrix<ST, LO, GO, NT>( A ).scale( static_cast<ST>( alpha ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::setScalar( AMP::Scalar alpha, MatrixData &A )
{
    getTpetra_CrsMatrix<ST, LO, GO, NT>( A ).setAllToScalar( static_cast<ST>( alpha ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::zero( MatrixData &A )
{
    getTpetra_CrsMatrix<ST, LO, GO, NT>( A ).setAllToScalar( static_cast<ST>( 0.0 ) );
    A.disableModifications();
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::axpy( AMP::Scalar alpha,
                                                   const MatrixData &X,
                                                   MatrixData &Y )
{
    auto &tY = getTpetra_CrsMatrix<ST, LO, GO, NT>( Y );
    tY.resumeFill();
    Tpetra::MatrixMatrix::Add( getTpetra_CrsMatrix<ST, LO, GO, NT>( X ),
                               false,
                               static_cast<ST>( alpha ),
                               tY,
                               static_cast<ST>( 1.0 ) );
    tY.fillComplete();
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::setDiagonal( std::shared_ptr<const Vector> in,
                                                          MatrixData &A )
{
    // probably a much better way to do this
    using row_matrix_type           = Tpetra::RowMatrix<ST, LO, GO, NT>;
    using local_inds_host_view_type = typename row_matrix_type::local_inds_host_view_type;
    using values_host_view_type     = typename row_matrix_type::values_host_view_type;
    AMP_ASSERT( in && in->numberOfDataBlocks() == 1 );
    Kokkos::View<ST *> diag_vals( "diag tmp", in->getLocalSize() );
    in->getRawData<ST>( diag_vals.data() );

    auto &matrix = getTpetra_CrsMatrix<ST, LO, GO, NT>( A );

    //    matrix->resumeFill();

    // Get the current row's data
    for ( size_t row = 0; row < A.numLocalRows(); ++row ) {
        auto numCols = matrix.getNumEntriesInLocalRow( row );
        std::vector<LO> colInds( numCols );
        std::vector<ST> vals( numCols );
        local_inds_host_view_type colView( colInds.data(), numCols );
        values_host_view_type valsView( vals.data(), numCols );
        matrix.getLocalRowView( row, colView, valsView );

        // Find the diagonal entry within the row's column indices
        for ( size_t k = 0; k < colInds.size(); ++k ) {
            if ( colInds[k] == static_cast<LO>( row ) ) { // Check if it's the diagonal entry
                Teuchos::ArrayView<LO> replaceColInds( &colInds[k],
                                                       1 ); // Column index for replacement
                Teuchos::ArrayView<ST> replaceValues( &diag_vals[row],
                                                      1 ); // New value for replacement
                matrix.replaceLocalValues( row, replaceColInds, replaceValues );
                break;
            }
        }
    }

    //    matrix->fillComplete();
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::setIdentity( MatrixData &A )
{
    zero( A );
    int MyFirstRow = A.getLeftDOFManager()->beginDOF();
    int MyEndRow   = A.getLeftDOFManager()->endDOF();
    double one     = 1.0;
    for ( int i = MyFirstRow; i != MyEndRow; i++ ) {
        VerifyTpetraReturn(
            getTpetra_CrsMatrix<ST, LO, GO, NT>( A ).replaceGlobalValues( i, 1, &one, &i ),
            "setValuesByGlobalID" );
    }
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::extractDiagonal( MatrixData const &A,
                                                              std::shared_ptr<Vector> buf )
{
    auto view = TpetraVector::view( buf );
    getTpetra_CrsMatrix<ST, LO, GO, NT>( A ).getLocalDiagCopy( view->getTpetra_Vector() );
}

template<typename ST, typename LO, typename GO, typename NT>
AMP::Scalar TpetraMatrixOperations<ST, LO, GO, NT>::LinfNorm( MatrixData const & ) const
{
    AMP_ERROR( "Not implemented for Trilinos 16.0" );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::matMatMult( std::shared_ptr<MatrixData> A,
                                                         std::shared_ptr<MatrixData> B,
                                                         std::shared_ptr<MatrixData> C )
{
    Tpetra::MatrixMatrix::Multiply( getTpetra_CrsMatrix<ST, LO, GO, NT>( *A ),
                                    false,
                                    getTpetra_CrsMatrix<ST, LO, GO, NT>( *B ),
                                    false,
                                    getTpetra_CrsMatrix<ST, LO, GO, NT>( *C ),
                                    true );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::copy( const MatrixData &X, MatrixData &Y )
{
#if 1
    getTpetra_CrsMatrix<ST, LO, GO, NT>( Y ) = getTpetra_CrsMatrix<ST, LO, GO, NT>( X );
#else
    Tpetra::MatrixMatrix::Add( getTpetra_CrsMatrix<ST, LO, GO, NT>( X ),
                               false,
                               static_cast<ST>( 1.0 ),
                               getTpetra_CrsMatrix<ST, LO, GO, NT>( Y ),
                               0.0 );
#endif
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::scale( AMP::Scalar,
                                                    std::shared_ptr<const Vector>,
                                                    MatrixData & )
{
    AMP_ERROR( "Not implemented" );
}
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::scaleInv( AMP::Scalar,
                                                       std::shared_ptr<const Vector>,
                                                       MatrixData & )
{
    AMP_ERROR( "Not implemented" );
}
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::getRowSums( MatrixData const &,
                                                         std::shared_ptr<Vector> )
{
    AMP_ERROR( "Not implemented" );
}
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::getRowSumsAbsolute( MatrixData const &,
                                                                 std::shared_ptr<Vector> )
{
    AMP_ERROR( "Not implemented" );
}

} // namespace AMP::LinearAlgebra
