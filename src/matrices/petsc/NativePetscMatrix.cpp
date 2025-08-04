#include "AMP/matrices/petsc/NativePetscMatrix.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/operations/default/MatrixOperationsDefault.h"
#include "AMP/matrices/petsc/NativePetscMatrixData.h"
#include "AMP/matrices/petsc/NativePetscMatrixOperations.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/petsc/NativePetscVectorData.h"
#include "AMP/vectors/petsc/PetscVector.h"

#include "ProfilerApp.h"

#include "petscmat.h"

namespace AMP::LinearAlgebra {

NativePetscMatrix::NativePetscMatrix()
{
    d_matrixData = std::make_shared<NativePetscMatrixData>();
    d_matrixOps  = std::make_shared<NativePetscMatrixOperations>();
}

NativePetscMatrix::NativePetscMatrix( Mat m, bool internally_created )
{
    d_matrixData = std::make_shared<NativePetscMatrixData>( m, internally_created );
    d_matrixOps  = std::make_shared<NativePetscMatrixOperations>();
}

NativePetscMatrix::NativePetscMatrix( std::shared_ptr<MatrixParametersBase> params )
    : Matrix( params )
{
    d_matrixData = std::make_shared<NativePetscMatrixData>( params );
    d_matrixOps  = std::make_shared<NativePetscMatrixOperations>();
}

NativePetscMatrix::NativePetscMatrix( std::shared_ptr<MatrixData> data ) : Matrix( data )
{
    d_matrixOps = std::make_shared<NativePetscMatrixOperations>();
}

NativePetscMatrix::~NativePetscMatrix() {}

void NativePetscMatrix::multiply( shared_ptr other_op, shared_ptr &result )
{
    PROFILE( "NativePetscMatrix::multiply" );

    auto other = std::dynamic_pointer_cast<NativePetscMatrix>( other_op );
    AMP_INSIST( other != nullptr, "Incompatible matrix types" );

    std::shared_ptr<Matrix> newMatrix = std::make_shared<NativePetscMatrix>();
    result.swap( newMatrix );

    d_matrixOps->matMatMult( d_matrixData, other_op->getMatrixData(), result->getMatrixData() );
}

Vector::shared_ptr NativePetscMatrix::extractDiagonal( Vector::shared_ptr v ) const
{
    Vector::shared_ptr retVal = v;
    if ( !retVal ) {
        retVal = this->getInputVector();
    } else if ( std::dynamic_pointer_cast<NativePetscVectorData>( v->getVectorData() ) ) {
        retVal = v;
    } else {
        AMP_ERROR( "NativePetscMatrix::extractDiagonal(): Not handled for vectors that are not "
                   "NativePetscVector" );
    }

    d_matrixOps->extractDiagonal( *d_matrixData, retVal );
    return retVal;
}

Vector::shared_ptr NativePetscMatrix::getInputVector() const
{
    return std::dynamic_pointer_cast<NativePetscMatrixData>( d_matrixData )->getInputVector();
}
Vector::shared_ptr NativePetscMatrix::getOutputVector() const
{
    return std::dynamic_pointer_cast<NativePetscMatrixData>( d_matrixData )->getOutputVector();
}
std::shared_ptr<Discretization::DOFManager> NativePetscMatrix::getRightDOFManager() const
{
    //    return getInputVector()->getDOFManager();
    return d_matrixData->getRightDOFManager();
}
std::shared_ptr<Discretization::DOFManager> NativePetscMatrix::getLeftDOFManager() const
{
    return d_matrixData->getLeftDOFManager();
    //    return getOutputVector()->getDOFManager();
}

std::shared_ptr<Matrix> NativePetscMatrix::clone() const
{
    return std::make_shared<NativePetscMatrix>( d_matrixData->cloneMatrixData() );
}

std::shared_ptr<Matrix> NativePetscMatrix::duplicateMat( Mat m )
{
    auto data = NativePetscMatrixData::duplicateMat( m );
    return std::make_shared<NativePetscMatrix>( data );
}

void NativePetscMatrix::copy( std::shared_ptr<const Matrix> X )
{
    if ( this->type() == X->type() ) {
        const auto xData =
            std::dynamic_pointer_cast<const NativePetscMatrixData>( X->getMatrixData() );
        AMP_ASSERT( xData );
        copyFromMat( std::const_pointer_cast<NativePetscMatrixData>( xData )->getMat() );
    } else {
        MatrixOperationsDefault::copy( *X->getMatrixData(), *getMatrixData() );
    }

    makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}

void NativePetscMatrix::copyFromMat( Mat m )
{
    std::dynamic_pointer_cast<NativePetscMatrixData>( d_matrixData )->copyFromMat( m );
}

} // namespace AMP::LinearAlgebra
