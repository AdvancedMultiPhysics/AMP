#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/DiagonalSolver.h"
#include "AMP/solvers/SolverFactory.h"
#include "ProfilerApp.h"

namespace AMP::Solver {

/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/

template<typename T>
DiagonalSolver<T>::DiagonalSolver(
    std::shared_ptr<AMP::Solver::SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{
    AMP_ASSERT( parameters );

    // Initialize
    initialize( parameters );
}

template<typename T>
void DiagonalSolver<T>::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    d_pOperator = op;

    if ( d_pOperator ) {
        auto linearOp = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
        AMP_ASSERT( linearOp );
        auto matrix        = linearOp->getMatrix();
        d_pDiagonalInverse = matrix->extractDiagonal();
        AMP_ASSERT( d_pDiagonalInverse );
        d_pDiagonalInverse->reciprocal( *d_pDiagonalInverse );
    }
}

/****************************************************************
 *  Initialize                                                   *
 ****************************************************************/
template<typename T>
void DiagonalSolver<T>::initialize(
    std::shared_ptr<const AMP::Solver::SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );

    auto db = parameters->d_db;
    getFromInput( db );

    if ( parameters->d_pNestedSolver ) {
        d_pNestedSolver = parameters->d_pNestedSolver;
    } else {
        if ( d_bUsesNestedSolver ) {
            auto nestedName =
                db->getWithDefault<std::string>( "nested_solver_name", "NestedSolver" );
            auto outerDB = db->keyExists( nestedName ) ? db : parameters->d_global_db;
            if ( outerDB ) {
                auto nestedDB = outerDB->getDatabase( nestedName );
                auto innerParameters =
                    std::make_shared<AMP::Solver::SolverStrategyParameters>( nestedDB );
                innerParameters->d_global_db = parameters->d_global_db;
                innerParameters->d_pOperator = d_pOperator;
                d_pNestedSolver = AMP::Solver::SolverFactory::create( innerParameters );
                AMP_ASSERT( d_pNestedSolver );
            }
        }
    }

    registerOperator( d_pOperator );
}

// Function to get values from input
template<typename T>
void DiagonalSolver<T>::getFromInput( std::shared_ptr<AMP::Database> db )
{
    d_bUsesNestedSolver = db->getWithDefault<bool>( "uses_nested_solver", false );
}

/****************************************************************
 *  Solve                                                        *
 ****************************************************************/
template<typename T>
void DiagonalSolver<T>::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                               std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "DiagonalSolver<T>::apply" );

    // Always zero before checking stopping criteria for any reason
    d_iNumberIterations = 1;

    u->multiply( *d_pDiagonalInverse, *f );

    {
        PROFILE( "DiagonalSolver<T>:: u->makeConsistent" );
        u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }

    // always set this condition for now
    d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
}
} // namespace AMP::Solver
