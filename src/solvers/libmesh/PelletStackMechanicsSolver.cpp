#include "AMP/solvers/libmesh/PelletStackMechanicsSolver.h"

namespace AMP::Solver {


PelletStackMechanicsSolver::PelletStackMechanicsSolver(
    std::shared_ptr<PelletStackMechanicsSolverParameters> params )
    : SolverStrategy( params ),
      d_pelletStackOp(
          std::dynamic_pointer_cast<AMP::Operator::PelletStackOperator>( d_pOperator ) ),
      d_columnSolver( params->d_columnSolver )
{
}


void PelletStackMechanicsSolver::resetOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    d_columnSolver->resetOperator( params );
}


void PelletStackMechanicsSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                        std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    AMP_ASSERT( ( f->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED ) ||
                ( f->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED ) );
    std::shared_ptr<const AMP::LinearAlgebra::Vector> fInternal = f;
    if ( d_pelletStackOp->useScaling() ) {
        if ( d_fbuffer1 == nullptr ) {
            d_fbuffer1 = f->clone();
        }
        d_fbuffer1->copyVector( f );
        d_pelletStackOp->applyUnscaling( d_fbuffer1 );
        fInternal = d_fbuffer1;
    }
    if ( d_pelletStackOp->useSerial() ) {
        solveSerial( fInternal, u );
    } else {
        solveScan( fInternal, u );
    }
}

void PelletStackMechanicsSolver::solveSerial( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                              std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    if ( d_fbuffer2 == nullptr ) {
        d_fbuffer2 = f->clone();
    }

    unsigned int totalNumberOfPellets = d_pelletStackOp->getTotalNumberOfPellets();

    int locPellIdx = d_pelletStackOp->getLocalIndexForPellet( 0 );

    if ( locPellIdx != -1 ) {
        auto currSolver = d_columnSolver->getSolver( locPellIdx );
        auto currOp     = currSolver->getOperator();
        auto subUvec    = currOp->subsetInputVector( u );
        auto subFvec    = currOp->subsetOutputVector( f );
        AMP_ASSERT( subUvec );
        AMP_ASSERT( subFvec );
        currSolver->apply( subFvec, subUvec );
    }

    for ( unsigned int pellId = 1; pellId < totalNumberOfPellets; pellId++ ) {
        std::shared_ptr<AMP::Database> emptyDb;
        auto pelletStackOpParams =
            std::make_shared<AMP::Operator::PelletStackOperatorParameters>( emptyDb );
        pelletStackOpParams->d_currentPellet = pellId;
        d_pelletStackOp->reset( pelletStackOpParams );
        d_pelletStackOp->residual( f, u, d_fbuffer2 );
        locPellIdx = d_pelletStackOp->getLocalIndexForPellet( pellId );
        if ( locPellIdx != -1 ) {
            auto currSolver = d_columnSolver->getSolver( locPellIdx );
            auto currOp     = currSolver->getOperator();
            auto subUvec    = currOp->subsetInputVector( u );
            auto subFvec    = currOp->subsetOutputVector( d_fbuffer2 );
            AMP_ASSERT( subUvec );
            AMP_ASSERT( subFvec );
            currSolver->apply( subFvec, subUvec );
        }
    } // end for pellId
}

void PelletStackMechanicsSolver::solveScan( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                            std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    d_columnSolver->apply( f, u );
    if ( d_pelletStackOp->onlyZcorrection() ) {
        AMP::LinearAlgebra::Vector::shared_ptr nullVec;
        d_pelletStackOp->apply( nullVec, u );
    } else {
        if ( d_fbuffer2 == nullptr ) {
            d_fbuffer2 = f->clone();
        }
        d_pelletStackOp->residual( f, u, d_fbuffer2 );
        d_columnSolver->apply( d_fbuffer2, u );
    }
}


} // namespace AMP::Solver
