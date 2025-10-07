#include "AMP/matrices/trilinos/epetra/EpetraMatrixOperations.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/trilinos/epetra/EpetraMatrixData.h"
#include "AMP/vectors/trilinos/epetra/EpetraVector.h"

DISABLE_WARNINGS
#include <EpetraExt_MatrixMatrix.h>
#include <EpetraExt_Transpose_RowMatrix.h>
#include <Epetra_FECrsMatrix.h>
ENABLE_WARNINGS

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

static void VerifyEpetraReturn( int err, const char *func )
{
    std::stringstream error;
    error << func << ": " << err;
    if ( err < 0 )
        AMP_ERROR( error.str() );
    if ( err > 0 )
        AMP_ERROR( error.str() );
}

static Epetra_CrsMatrix &getEpetra_CrsMatrix( MatrixData &A )
{
    auto data = dynamic_cast<EpetraMatrixData *>( &A );
    AMP_ASSERT( data );
    return data->getEpetra_CrsMatrix();
}

static const Epetra_CrsMatrix &getEpetra_CrsMatrix( MatrixData const &A )
{
    const auto data = dynamic_cast<const EpetraMatrixData *>( &A );
    AMP_ASSERT( data );
    return data->getEpetra_CrsMatrix();
}

void EpetraMatrixOperations::mult( std::shared_ptr<const Vector> in,
                                   MatrixData const &A,
                                   std::shared_ptr<Vector> out )
{
    PROFILE( "EpetraMatrixOperations::mult" );
    AMP_ASSERT( in->getGlobalSize() == A.numGlobalColumns() );
    AMP_ASSERT( out->getGlobalSize() == A.numGlobalRows() );
    auto in_view                = EpetraVector::constView( in );
    auto out_view               = EpetraVector::view( out );
    const Epetra_Vector &in_vec = in_view->getEpetra_Vector();
    Epetra_Vector &out_vec      = out_view->getEpetra_Vector();
    int err                     = getEpetra_CrsMatrix( A ).Multiply( false, in_vec, out_vec );
    VerifyEpetraReturn( err, "mult" );
}

void EpetraMatrixOperations::multTranspose( std::shared_ptr<const Vector> in,
                                            MatrixData const &A,
                                            std::shared_ptr<Vector> out )
{
    PROFILE( "EpetraMatrixOperations::multTranspose" );
    AMP_ASSERT( in->getGlobalSize() == A.numGlobalColumns() );
    AMP_ASSERT( out->getGlobalSize() == A.numGlobalRows() );
    auto in_view  = EpetraVector::constView( in );
    auto out_view = EpetraVector::view( out );
    int err       = getEpetra_CrsMatrix( A ).Multiply(
        true, in_view->getEpetra_Vector(), out_view->getEpetra_Vector() );
    VerifyEpetraReturn( err, "mult" );
}

void EpetraMatrixOperations::scale( AMP::Scalar alpha, MatrixData &A )
{
    VerifyEpetraReturn( getEpetra_CrsMatrix( A ).Scale( static_cast<double>( alpha ) ), "scale" );
}

void EpetraMatrixOperations::setScalar( AMP::Scalar alpha, MatrixData &A )
{
    VerifyEpetraReturn( getEpetra_CrsMatrix( A ).PutScalar( static_cast<double>( alpha ) ),
                        "setScalar" );
}

void EpetraMatrixOperations::zero( MatrixData &A )
{
    VerifyEpetraReturn( getEpetra_CrsMatrix( A ).PutScalar( 0.0 ), "setScalar" );
}

void EpetraMatrixOperations::axpy( AMP::Scalar alpha, const MatrixData &X, MatrixData &Y )
{
    EpetraExt::MatrixMatrix::Add( getEpetra_CrsMatrix( X ),
                                  false,
                                  static_cast<double>( alpha ),
                                  getEpetra_CrsMatrix( Y ),
                                  1.0 );
}

void EpetraMatrixOperations::setDiagonal( std::shared_ptr<const Vector> in, MatrixData &A )
{
    auto vec = EpetraVector::constView( in );
    VerifyEpetraReturn( getEpetra_CrsMatrix( A ).ReplaceDiagonalValues( vec->getEpetra_Vector() ),
                        "setDiagonal" );
}

void EpetraMatrixOperations::setIdentity( MatrixData &A )
{
    zero( A );
    int MyFirstRow = A.getLeftDOFManager()->beginDOF();
    int MyEndRow   = A.getLeftDOFManager()->endDOF();
    double one     = 1.0;
    for ( int i = MyFirstRow; i != MyEndRow; i++ ) {
        VerifyEpetraReturn( getEpetra_CrsMatrix( A ).ReplaceGlobalValues( i, 1, &one, &i ),
                            "setValuesByGlobalID" );
    }
}

void EpetraMatrixOperations::extractDiagonal( MatrixData const &A, std::shared_ptr<Vector> buf )
{
    auto view = EpetraVector::view( buf );
    VerifyEpetraReturn( getEpetra_CrsMatrix( A ).ExtractDiagonalCopy( view->getEpetra_Vector() ),
                        "extractDiagonal" );
}

AMP::Scalar EpetraMatrixOperations::LinfNorm( MatrixData const &A ) const
{
    return getEpetra_CrsMatrix( A ).NormInf();
}

void EpetraMatrixOperations::matMatMult( std::shared_ptr<MatrixData> A,
                                         std::shared_ptr<MatrixData> B,
                                         std::shared_ptr<MatrixData> C )
{
    int ierr = EpetraExt::MatrixMatrix::Multiply( getEpetra_CrsMatrix( *A ),
                                                  false,
                                                  getEpetra_CrsMatrix( *B ),
                                                  false,
                                                  getEpetra_CrsMatrix( *C ),
                                                  true );
    AMP_ASSERT( ierr == 0 );
}

void EpetraMatrixOperations::copy( const MatrixData &X, MatrixData &Y )
{
    EpetraExt::MatrixMatrix::Add(
        getEpetra_CrsMatrix( X ), false, 1.0, getEpetra_CrsMatrix( Y ), 0.0 );
}

void EpetraMatrixOperations::scale( AMP::Scalar, std::shared_ptr<const Vector>, MatrixData & )
{
    AMP_ERROR( "Not implemented" );
}
void EpetraMatrixOperations::scaleInv( AMP::Scalar, std::shared_ptr<const Vector>, MatrixData & )
{
    AMP_ERROR( "Not implemented" );
}
void EpetraMatrixOperations::getRowSums( MatrixData const &, std::shared_ptr<Vector> )
{
    AMP_ERROR( "Not implemented" );
}
void EpetraMatrixOperations::getRowSumsAbsolute( MatrixData const &, std::shared_ptr<Vector> )
{
    AMP_ERROR( "Not implemented" );
}

} // namespace AMP::LinearAlgebra
