#include "AMP/matrices/Matrix.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/operations/default/MatrixOperationsDefault.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/ParameterBase.h"

#include <iomanip>
#include <limits>

namespace AMP::LinearAlgebra {


/********************************************************
 * Constructors                                          *
 ********************************************************/
Matrix::Matrix( const Matrix & ) {}
Matrix::Matrix() {}
Matrix::Matrix( std::shared_ptr<MatrixParametersBase> ) {}
Matrix::Matrix( std::shared_ptr<MatrixData> data ) : d_matrixData( data ) {}
Matrix::~Matrix() {}


/********************************************************
 * multiply                                             *
 ********************************************************/
std::shared_ptr<Matrix> Matrix::matMatMult( shared_ptr A, shared_ptr B )
{
    if ( A->numGlobalColumns() != B->numGlobalRows() )
        AMP_ERROR( "Inner matrix dimensions must agree" );
    shared_ptr retVal;
    A->multiply( B, retVal );
    return retVal;
}

void Matrix::matMatMult( shared_ptr A, shared_ptr B, shared_ptr C )
{
    if ( A->numGlobalColumns() != B->numGlobalRows() )
        AMP_ERROR( "Inner matrix dimensions must agree" );
    A->multiply( B, C );
}


/********************************************************
 * axpy                                                  *
 ********************************************************/
void Matrix::axpy( AMP::Scalar alpha, std::shared_ptr<const Matrix> x )
{
    AMP_ASSERT( x );
    size_t N1 = x->numGlobalColumns();
    size_t N2 = this->numGlobalColumns();
    if ( N1 != N2 )
        AMP_ERROR( "Matrix sizes are not compatible" );
    axpy( alpha, *x );
}


/********************************************************
 * Print the matrix to a IO stream                       *
 ********************************************************/
std::ostream &operator<<( std::ostream &out, const Matrix &M_in )
{
    auto *M = (Matrix *) &M_in;
    // Print the matrix type (not supported yet)
    /*out << "Vector type: " << v.type() << "\n";
    if ( v.getVariable() )
    {
      out << "Variable name: " << v.getName() << "\n";
    }*/
    // Print the rank
    auto leftDOF   = M->getLeftDOFManager();
    auto rightDOF  = M->getRightDOFManager();
    auto leftComm  = leftDOF->getComm();
    auto rightComm = rightDOF->getComm();
    if ( leftComm == rightComm ) {
        int rank = leftComm.getRank();
        out << "Processor: " << rank << "\n";
    } else {
        int leftRank  = leftComm.getRank();
        int rightRank = rightComm.getRank();
        out << "Processor (left comm):  " << leftRank << "\n";
        out << "Processor (right comm): " << rightRank << "\n";
    }
    // Print some basic matrix info
    out << "\n"
        << "Global number of rows: " << M->numGlobalRows() << "\n"
        << "Global number of colums: " << M->numGlobalColumns() << "\n"
        << "Local number of rows: " << M->numLocalRows() << "\n"
        << "Local number of colums: " << M->numLocalColumns() << "\n";
    // Loop through each local row
    std::vector<size_t> cols;
    std::vector<double> values;
    out << "Compressed Matix: " << std::endl;
    for ( size_t row = leftDOF->beginDOF(); row < leftDOF->endDOF(); row++ ) {
        M->getRowByGlobalID( row, cols, values );
        out << "Row " << row << " (" << cols.size() << " entries):"
            << "\n";
        for ( size_t i = 0; i < cols.size(); i++ )
            out << "    M(" << row << "," << cols[i] << ") = " << values[i] << "\n";
    }
    return out;
}

void Matrix::mult( AMP::LinearAlgebra::Vector::const_shared_ptr in,
                   AMP::LinearAlgebra::Vector::shared_ptr out )
{
    AMP_ASSERT( in->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
    d_matrixOps->mult( in, *getMatrixData(), out );
}

void Matrix::multTranspose( AMP::LinearAlgebra::Vector::const_shared_ptr in,
                            AMP::LinearAlgebra::Vector::shared_ptr out )
{
    AMP_ASSERT( in->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
    d_matrixOps->multTranspose( in, *getMatrixData(), out );
    out->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}

void Matrix::scale( AMP::Scalar alpha ) { d_matrixOps->scale( alpha, *getMatrixData() ); }

void Matrix::scale( AMP::Scalar alpha, Vector::const_shared_ptr D )
{
    d_matrixOps->scale( alpha, D, *getMatrixData() );
}

void Matrix::scaleInv( AMP::Scalar alpha, Vector::const_shared_ptr D )
{
    d_matrixOps->scaleInv( alpha, D, *getMatrixData() );
}

void Matrix::axpy( AMP::Scalar alpha, const Matrix &X )
{
    if ( this->type() == X.type() )
        d_matrixOps->axpy( alpha, *( X.getMatrixData() ), *getMatrixData() );
    else
        MatrixOperationsDefault::axpy( alpha, *( X.getMatrixData() ), *getMatrixData() );
}

void Matrix::setScalar( AMP::Scalar alpha ) { d_matrixOps->setScalar( alpha, *getMatrixData() ); }

void Matrix::zero() { d_matrixOps->zero( *getMatrixData() ); }

void Matrix::setDiagonal( Vector::const_shared_ptr in )
{
    d_matrixOps->setDiagonal( in, *getMatrixData() );
}

void Matrix::setIdentity() { d_matrixOps->setIdentity( *getMatrixData() ); }

AMP::Scalar Matrix::LinfNorm() const { return d_matrixOps->LinfNorm( *getMatrixData() ); }

void Matrix::copy( std::shared_ptr<const Matrix> X )
{
    if ( X.get() == this )
        return;
    // Verify that A and B have compatible dimensions
    const auto globalKa = this->numGlobalColumns();
    const auto globalKb = X->numGlobalRows();
    const auto localKa  = this->numLocalColumns();
    const auto localKb  = X->numLocalRows();
    AMP_INSIST( globalKa == globalKb, "Matrix::copy got incompatible global dimensions" );
    AMP_INSIST( localKa == localKb, "Matrix::copy got incompatible local dimensions" );
    if ( this->type() == X->type() )
        d_matrixOps->copy( *X->getMatrixData(), *getMatrixData() );
    else
        MatrixOperationsDefault::copy( *X->getMatrixData(), *getMatrixData() );
}

void Matrix::copyCast( std::shared_ptr<const Matrix> X )
{
    d_matrixOps->copyCast( *X->getMatrixData(), *getMatrixData() );
}

std::uint16_t Matrix::mode() const { return std::numeric_limits<std::uint16_t>::max(); }

} // namespace AMP::LinearAlgebra
