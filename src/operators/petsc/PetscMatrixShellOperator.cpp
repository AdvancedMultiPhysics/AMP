#include "AMP/operators/petsc/PetscMatrixShellOperator.h"
#include "AMP/matrices/petsc/NativePetscMatrix.h"
#include "AMP/vectors/petsc/PetscHelpers.h"

#include "petsc/private/vecimpl.h"

namespace AMP::Operator {

PetscMatrixShellOperator::PetscMatrixShellOperator(
    std::shared_ptr<const OperatorParameters> params )
    : LinearOperator( params )
{
    d_iMatLocalRowSize    = -1234;
    d_iMatLocalColumnSize = -5678;
}

void PetscMatrixShellOperator::setComm( const AMP_MPI &comm ) { d_comm = comm; }

void PetscMatrixShellOperator::setMatLocalRowSize( int val ) { d_iMatLocalRowSize = val; }

void PetscMatrixShellOperator::setMatLocalColumnSize( int val ) { d_iMatLocalColumnSize = val; }

void PetscMatrixShellOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                      AMP::LinearAlgebra::Vector::shared_ptr f )
{
    d_operator->apply( u, f );
}

void PetscMatrixShellOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    d_operator->reset( params );
}

std::shared_ptr<AMP::LinearAlgebra::Variable> PetscMatrixShellOperator::getOutputVariable() const
{
    return d_operator->getOutputVariable();
}

std::shared_ptr<AMP::LinearAlgebra::Variable> PetscMatrixShellOperator::getInputVariable() const
{
    return d_operator->getInputVariable();
}

void PetscMatrixShellOperator::setOperator( std::shared_ptr<Operator> op )
{
    d_operator = op;
    MatCreateShell( d_comm.getCommunicator(),
                    d_iMatLocalRowSize,
                    d_iMatLocalColumnSize,
                    PETSC_DETERMINE,
                    PETSC_DETERMINE,
                    this,
                    &d_mat );
    MatShellSetOperation(
        d_mat, MATOP_MULT, ( void ( * )() )( &( PetscMatrixShellOperator::mult ) ) );
    d_matrix.reset( new AMP::LinearAlgebra::NativePetscMatrix( d_mat, true ) );
}

PetscErrorCode PetscMatrixShellOperator::mult( Mat mat, Vec in, Vec out )
{
    void *ctx;
    MatShellGetContext( mat, &ctx );
    auto *op = reinterpret_cast<PetscMatrixShellOperator *>( ctx );

    auto inVec  = PETSC::getAMP( in );
    auto outVec = PETSC::getAMP( out );

    op->d_operator->apply( inVec, outVec );

    return ( 0 );
}
} // namespace AMP::Operator
