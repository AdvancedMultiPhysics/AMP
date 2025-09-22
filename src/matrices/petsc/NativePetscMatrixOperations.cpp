#include "AMP/matrices/petsc/NativePetscMatrixOperations.h"
#include "AMP/matrices/petsc/NativePetscMatrixData.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/petsc/NativePetscVectorData.h"
#include "AMP/vectors/petsc/PetscVector.h"

#include "petscmat.h"
#include "petscvec.h"

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

// Get vector
static std::shared_ptr<Vec> getVec( std::shared_ptr<Vector> v )
{
    auto data = std::dynamic_pointer_cast<NativePetscVectorData>( v->getVectorData() );
    if ( data )
        return std::shared_ptr<Vec>( new Vec( data->getVec() ), []( auto ) {} );
    return std::shared_ptr<Vec>( new Vec( PETSC::getVec( v ) ), []( Vec *v ) { VecDestroy( v ); } );
}
static std::shared_ptr<Vec> getVec( std::shared_ptr<const Vector> v )
{
    return getVec( std::const_pointer_cast<Vector>( v ) );
}

static Mat getMat( MatrixData &m )
{
    auto data = dynamic_cast<NativePetscMatrixData *>( &m );
    AMP_ASSERT( data );
    return data->getMat();
}
static Mat getMat( const MatrixData &m )
{
    const auto data = dynamic_cast<const NativePetscMatrixData *>( &m );
    AMP_ASSERT( data );
    return const_cast<NativePetscMatrixData *>( data )->getMat();
}

void NativePetscMatrixOperations::mult( std::shared_ptr<const Vector> in,
                                        MatrixData const &A,
                                        std::shared_ptr<Vector> out )
{
    PROFILE( "NativePetscMatrixOperations::mult" );
    MatMult( getMat( A ), *getVec( in ), *getVec( out ) );
}

void NativePetscMatrixOperations::multTranspose( std::shared_ptr<const Vector> in,
                                                 MatrixData const &A,
                                                 std::shared_ptr<Vector> out )
{
    PROFILE( "NativePetscMatrixOperations::multTranspose" );
    MatMultTranspose( getMat( A ), *getVec( in ), *getVec( out ) );
}

void NativePetscMatrixOperations::scale( AMP::Scalar alpha, MatrixData &A )
{
    MatScale( getMat( A ), static_cast<PetscScalar>( alpha ) );
}

void NativePetscMatrixOperations::matMatMult( std::shared_ptr<MatrixData> Am,
                                              std::shared_ptr<MatrixData> Bm,
                                              std::shared_ptr<MatrixData> Cm )
{
    AMP_ASSERT( Am->numGlobalColumns() == Bm->numGlobalRows() );

    if ( getMat( *Cm ) != nullptr ) {
        AMP_WARN_ONCE( "NativePetscMatrixOperations::matMatMult does not support re-use of result "
                       "data yet. A new result matrix will be created." );
    }

    Mat resMat;

    MatProductCreate( getMat( *Am ), getMat( *Bm ), nullptr, &resMat );
    MatProductSetType( resMat, MATPRODUCT_AB );
    MatProductSetAlgorithm( resMat, MATPRODUCTALGORITHMDEFAULT );
    // MatProductSetAlgorithm( resMat, MATPRODUCTALGORITHMSCALABLE );
    // MatProductSetAlgorithm( resMat, MATPRODUCTALGORITHMSCALABLEFAST );
    // MatProductSetAlgorithm( resMat, MATPRODUCTALGORITHMOVERLAPPING );
    MatProductSetFill( resMat, 1.5 );
    MatProductSetFromOptions( resMat );
    {
        PROFILE( "NativePetscMatrixOperations::matMatMult (symbolic)" );
        MatProductSymbolic( resMat );
    }
    {
        PROFILE( "NativePetscMatrixOperations::matMatMult (numeric)" );
        MatProductNumeric( resMat );
    }
    MatProductClear( resMat );

    auto data = dynamic_cast<NativePetscMatrixData *>( Cm.get() );
    AMP_ASSERT( data );
    data->setMat( resMat );
}

void NativePetscMatrixOperations::axpy( AMP::Scalar alpha, const MatrixData &X, MatrixData &Y )
{
    AMP_ASSERT( X.numGlobalRows() == Y.numGlobalRows() );
    AMP_ASSERT( X.numGlobalColumns() == Y.numGlobalColumns() );
    MatAXPY( getMat( X ), static_cast<PetscReal>( alpha ), getMat( Y ), SAME_NONZERO_PATTERN );
}

void NativePetscMatrixOperations::setScalar( AMP::Scalar alpha_in, MatrixData &A )
{
    const auto alpha = static_cast<PetscScalar>( alpha_in );
    if ( alpha != 0.0 )
        AMP_ERROR( "Cannot perform operation on NativePetscMatrix yet!" );
    MatZeroEntries( getMat( A ) );
}

void NativePetscMatrixOperations::zero( MatrixData &A ) { MatZeroEntries( getMat( A ) ); }

void NativePetscMatrixOperations::setDiagonal( std::shared_ptr<const Vector> in, MatrixData &A )
{
    MatDiagonalSet( getMat( A ), *getVec( in ), INSERT_VALUES );
}

void NativePetscMatrixOperations::setIdentity( MatrixData &A )
{
    auto mat = getMat( A );
    MatZeroEntries( mat );
    MatShift( mat, 1.0 );
}

void NativePetscMatrixOperations::extractDiagonal( MatrixData const &A,
                                                   std::shared_ptr<Vector> buf )
{
    MatGetDiagonal( getMat( A ), *getVec( buf ) );
}

AMP::Scalar NativePetscMatrixOperations::LinfNorm( MatrixData const &A ) const
{
    PetscReal retVal;
    MatNorm( getMat( A ), NORM_INFINITY, &retVal );
    return retVal;
}

void NativePetscMatrixOperations::copy( const MatrixData &X, MatrixData &Y )
{
    AMP_ASSERT( X.numGlobalRows() == Y.numGlobalRows() );
    AMP_ASSERT( X.numGlobalColumns() == Y.numGlobalColumns() );
    MatCopy( getMat( X ), getMat( Y ), SAME_NONZERO_PATTERN );
}

void NativePetscMatrixOperations::scale( AMP::Scalar, std::shared_ptr<const Vector>, MatrixData & )
{
    AMP_ERROR( "Not implemented" );
}
void NativePetscMatrixOperations::scaleInv( AMP::Scalar, std::shared_ptr<const Vector>, MatrixData & )
{
    AMP_ERROR( "Not implemented" );
}
void NativePetscMatrixOperations::getRowSums( MatrixData const &, std::shared_ptr<Vector> )
{
    AMP_ERROR( "Not implemented" );
}
void NativePetscMatrixOperations::getRowSumsAbsolute( MatrixData const &, std::shared_ptr<Vector> )
{
    AMP_ERROR( "Not implemented" );
}

} // namespace AMP::LinearAlgebra
