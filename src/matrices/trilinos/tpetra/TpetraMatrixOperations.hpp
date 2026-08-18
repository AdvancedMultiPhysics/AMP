#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/trilinos/tpetra/TpetraMatrixData.h"
#include "AMP/matrices/trilinos/tpetra/TpetraMatrixOperations.h"
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
    auto &crs_mat      = getTpetra_CrsMatrix<ST, LO, GO, NT>( A );
    auto dom_map       = crs_mat.getDomainMap();
    auto rng_map       = crs_mat.getRangeMap();
    auto in_view       = TpetraVector<ST, LO, GO, NT>::constView( in, dom_map );
    auto out_view      = TpetraVector<ST, LO, GO, NT>::view( out, rng_map );
    const auto &in_vec = in_view->getTpetra_Vector();
    auto &out_vec      = out_view->getTpetra_Vector();

    AMP_ASSERT( dom_map->getLocalNumElements() == A.numLocalColumns() );
    AMP_ASSERT( rng_map->getLocalNumElements() == A.numLocalRows() );
    AMP_ASSERT( dom_map->getGlobalNumElements() == A.numGlobalColumns() );
    AMP_ASSERT( rng_map->getGlobalNumElements() == A.numGlobalRows() );

    crs_mat.apply( in_vec, out_vec );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::multTranspose( std::shared_ptr<const Vector> in,
                                                            MatrixData const &A,
                                                            std::shared_ptr<Vector> out )
{
    PROFILE( "TpetraMatrixOperations<ST, LO, GO, NT>::multTranspose" );
    AMP_ASSERT( in->getGlobalSize() == A.numGlobalColumns() );
    AMP_ASSERT( out->getGlobalSize() == A.numGlobalRows() );
    auto &crs_mat = getTpetra_CrsMatrix<ST, LO, GO, NT>( A );
    auto in_view  = TpetraVector<ST, LO, GO, NT>::constView( in, crs_mat.getRangeMap() );
    auto out_view = TpetraVector<ST, LO, GO, NT>::view( out, crs_mat.getDomainMap() );
    crs_mat.apply( in_view->getTpetra_Vector(), out_view->getTpetra_Vector(), Teuchos::TRANS );
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

    auto &crs_mat = getTpetra_CrsMatrix<ST, LO, GO, NT>( A );
    // matrix.describe( *( Teuchos::getFancyOStream( Teuchos::rcpFromRef( std::cout ) ) ),
    //                  Teuchos::VERB_EXTREME );

    crs_mat.resumeFill();

    // Get the current row's data
    for ( size_t row = 0; row < A.numLocalRows(); ++row ) {
        local_inds_host_view_type colView;
        values_host_view_type valsView;
        crs_mat.getLocalRowView( row, colView, valsView );

        // Find the diagonal entry within the row's column indices
        for ( size_t k = 0; k < colView.size(); ++k ) {
            if ( colView[k] == static_cast<LO>( row ) ) { // Check if it's the diagonal entry
                Teuchos::ArrayView<LO> replaceColInds( const_cast<LO *>( &colView[k] ),
                                                       1 ); // Column index for replacement
                Teuchos::ArrayView<ST> replaceValues( &diag_vals[row],
                                                      1 ); // New value for replacement
                crs_mat.replaceLocalValues( row, replaceColInds, replaceValues );
                break;
            }
        }
    }

    crs_mat.fillComplete();
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::setIdentity( MatrixData &A )
{
    // this is wrong as is as it assumes where the diagonal entries reside
    zero( A );
    const auto MyFirstRow = static_cast<GO>( A.getLeftDOFManager()->beginDOF() );
    const auto MyEndRow   = static_cast<GO>( A.getLeftDOFManager()->endDOF() );
    const ST one          = 1.0;
    for ( GO i = MyFirstRow; i != MyEndRow; i++ ) {
        VerifyTpetraReturn(
            getTpetra_CrsMatrix<ST, LO, GO, NT>( A ).replaceGlobalValues( i, 1, &one, &i ),
            "setValuesByGlobalID" );
    }
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::extractDiagonal( MatrixData const &A,
                                                              std::shared_ptr<Vector> buf )
{
    const auto &crs_mat = getTpetra_CrsMatrix<ST, LO, GO, NT>( A );
    auto view           = TpetraVector<ST, LO, GO, NT>::view( buf, crs_mat.getRangeMap() );
    crs_mat.getLocalDiagCopy( view->getTpetra_Vector() );
}

template<typename ST, typename LO, typename GO, typename NT>
AMP::Scalar TpetraMatrixOperations<ST, LO, GO, NT>::LinfNorm( MatrixData const &A ) const
{
#if Trilinos_MAJOR_MINOR_VERSION < 160200
    AMP_ERROR( "Not implemented" );
    return 0;
#else
    return getTpetra_CrsMatrix<ST, LO, GO, NT>( A ).getNormInf();
#endif
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
    getTpetra_CrsMatrix<ST, LO, GO, NT>( Y ) = getTpetra_CrsMatrix<ST, LO, GO, NT>( X );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::scale( AMP::Scalar alpha,
                                                    std::shared_ptr<const Vector> D,
                                                    MatrixData &A )
{
    AMP_ASSERT( D->getGlobalSize() == A.numGlobalRows() );
    auto &crs_mat     = getTpetra_CrsMatrix<ST, LO, GO, NT>( A );
    auto D_view       = TpetraVector<ST, LO, GO, NT>::constView( D, crs_mat.getRangeMap() );
    const auto &D_vec = D_view->getTpetra_Vector();
    crs_mat.leftScale( D_vec );
    crs_mat.scale( static_cast<ST>( alpha ) );
}
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::scaleInv( AMP::Scalar,
                                                       std::shared_ptr<const Vector>,
                                                       MatrixData & )
{
    AMP_ERROR( "Not implemented" );
}
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::getRowSums( MatrixData const &A,
                                                         std::shared_ptr<Vector> buf )
{
    AMP_ASSERT( buf && buf->numberOfDataBlocks() == 1 );
    AMP_ASSERT( buf->isType<ST>( 0 ) );

    auto *rawVecData = buf->getRawDataBlock<ST>();

    // this is only correct for host code, we need to eventually get it working everywhere
    // and assumes all entries of a row are locally present
    // getNormInf only appears to be in some Trilinos version > 16.1.0 -- should do check
    using row_matrix_type           = Tpetra::RowMatrix<ST, LO, GO, NT>;
    using local_inds_host_view_type = typename row_matrix_type::local_inds_host_view_type;
    using values_host_view_type     = typename row_matrix_type::values_host_view_type;

    auto &matrix = getTpetra_CrsMatrix<ST, LO, GO, NT>( A );

    // Get the current row's data
    for ( LO row = 0; row < static_cast<LO>( A.numLocalRows() ); ++row ) {
        auto numCols = matrix.getNumEntriesInLocalRow( row );
        std::vector<LO> colInds( numCols );
        std::vector<ST> vals( numCols );
        local_inds_host_view_type colView( colInds.data(), numCols );
        values_host_view_type valsView( vals.data(), numCols );
        matrix.getLocalRowView( row, colView, valsView );
        rawVecData[row] = std::accumulate(
            vals.begin(), vals.end(), 0.0, []( const ST &a, const ST &b ) -> ST { return a + b; } );
    }
}
template<typename ST, typename LO, typename GO, typename NT>
void TpetraMatrixOperations<ST, LO, GO, NT>::getRowSumsAbsolute( MatrixData const &A,
                                                                 std::shared_ptr<Vector> buf,
                                                                 const bool )
{
    AMP_ASSERT( buf && buf->numberOfDataBlocks() == 1 );
    AMP_ASSERT( buf->isType<ST>( 0 ) );

    auto *rawVecData = buf->getRawDataBlock<ST>();

    // this is only correct for host code, we need to eventually get it working everywhere
    // and assumes all entries of a row are locally present
    // getNormInf only appears to be in some Trilinos version > 16.1.0 -- should do check
    using row_matrix_type           = Tpetra::RowMatrix<ST, LO, GO, NT>;
    using local_inds_host_view_type = typename row_matrix_type::local_inds_host_view_type;
    using values_host_view_type     = typename row_matrix_type::values_host_view_type;

    auto &matrix = getTpetra_CrsMatrix<ST, LO, GO, NT>( A );

    // Get the current row's data
    for ( LO row = 0; row < static_cast<LO>( A.numLocalRows() ); ++row ) {
        auto numCols = matrix.getNumEntriesInLocalRow( row );
        std::vector<LO> colInds( numCols );
        std::vector<ST> vals( numCols );
        local_inds_host_view_type colView( colInds.data(), numCols );
        values_host_view_type valsView( vals.data(), numCols );
        matrix.getLocalRowView( row, colView, valsView );
        rawVecData[row] =
            std::accumulate( vals.begin(), vals.end(), 0.0, []( const ST &a, const ST &b ) -> ST {
                return std::abs( a ) + std::abs( b );
            } );
    }
}

} // namespace AMP::LinearAlgebra
