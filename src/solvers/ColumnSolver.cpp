
#include "ColumnSolver.h"
#include "AMP/operators/ColumnOperatorParameters.h"

namespace AMP {
namespace Solver {


ColumnSolver::ColumnSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{
    AMP_ASSERT( parameters.get() != nullptr );
    std::shared_ptr<AMP::Database> db = parameters->d_db;
    d_IterationType       = db->getWithDefault<std::string>( "IterationType", "GaussSeidel" );
    d_resetColumnOperator = db->getWithDefault( "ResetColumnOperator", false );
}

void ColumnSolver::solve( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                          std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    // u->zero();

    if ( d_IterationType == "GaussSeidel" ) {
        GaussSeidel( f, u );
    } else if ( d_IterationType == "SymmetricGaussSeidel" ) {
        SymmetricGaussSeidel( f, u );
    } else {
        AMP::pout << "ERROR: Invalid iteration type specified " << std::endl;
    }
}

void ColumnSolver::GaussSeidel( std::shared_ptr<const AMP::LinearAlgebra::Vector> &f,
                                std::shared_ptr<AMP::LinearAlgebra::Vector> &u )
{
    for ( int it = 0; it < d_iMaxIterations; it++ ) {
        for ( auto &elem : d_Solvers ) {
            std::shared_ptr<AMP::Operator::Operator> op = elem->getOperator();
            AMP_INSIST( op.get() != nullptr,
                        "EROR: NULL Operator returned by SolverStrategy::getOperator" );

            std::shared_ptr<const AMP::LinearAlgebra::Vector> sf = op->subsetOutputVector( f );
            AMP_INSIST( sf.get() != nullptr,
                        "ERROR: subset on rhs f yields NULL vector in ColumnSolver::solve" );
            std::shared_ptr<AMP::LinearAlgebra::Vector> su = op->subsetInputVector( u );
            AMP_INSIST( su.get() != nullptr,
                        "ERROR: subset on solution u yields NULL vector in ColumnSolver::solve" );

            elem->solve( sf, su );
        }
    }
}

void ColumnSolver::SymmetricGaussSeidel( std::shared_ptr<const AMP::LinearAlgebra::Vector> &f,
                                         std::shared_ptr<AMP::LinearAlgebra::Vector> &u )
{
    for ( int it = 0; it < d_iMaxIterations; it++ ) {
        for ( auto &elem : d_Solvers ) {
            std::shared_ptr<AMP::Operator::Operator> op = elem->getOperator();
            AMP_INSIST( op.get() != nullptr,
                        "EROR: NULL Operator returned by SolverStrategy::getOperator" );

            std::shared_ptr<const AMP::LinearAlgebra::Vector> sf = op->subsetOutputVector( f );
            AMP_INSIST( sf.get() != nullptr,
                        "ERROR: subset on rhs f yields NULL vector in ColumnSolver::solve" );
            std::shared_ptr<AMP::LinearAlgebra::Vector> su = op->subsetInputVector( u );
            AMP_INSIST( su.get() != nullptr,
                        "ERROR: subset on solution u yields NULL vector in ColumnSolver::solve" );

            elem->solve( sf, su );
        }

        for ( int i = (int) d_Solvers.size() - 1; i >= 0; i-- ) {
            std::shared_ptr<AMP::Operator::Operator> op = d_Solvers[i]->getOperator();
            AMP_INSIST( op.get() != nullptr,
                        "EROR: NULL Operator returned by SolverStrategy::getOperator" );

            std::shared_ptr<const AMP::LinearAlgebra::Vector> sf = op->subsetOutputVector( f );
            AMP_INSIST( sf.get() != nullptr,
                        "ERROR: subset on rhs f yields NULL vector in ColumnSolver::solve" );
            std::shared_ptr<AMP::LinearAlgebra::Vector> su = op->subsetInputVector( u );
            AMP_INSIST( su.get() != nullptr,
                        "ERROR: subset on solution u yields NULL vector in ColumnSolver::solve" );

            d_Solvers[i]->solve( sf, su );
        }
    }
}

void ColumnSolver::setInitialGuess( std::shared_ptr<AMP::LinearAlgebra::Vector> initialGuess )
{
    for ( auto &elem : d_Solvers ) {
        elem->setInitialGuess( initialGuess );
    }
}

void ColumnSolver::append( std::shared_ptr<AMP::Solver::SolverStrategy> solver )
{
    AMP_INSIST( ( solver.get() != nullptr ),
                "AMP::Solver::ColumnSolver::append input argument is a NULL solver" );
    d_Solvers.push_back( solver );
}

void ColumnSolver::resetOperator( const std::shared_ptr<AMP::Operator::OperatorParameters> params )
{
    if ( d_resetColumnOperator ) {
        d_pOperator->reset( params );

        std::shared_ptr<SolverStrategyParameters> solverParams;

        for ( auto &elem : d_Solvers ) {
            elem->reset( solverParams );
        }
    } else {
        std::shared_ptr<AMP::Operator::ColumnOperatorParameters> columnParams =
            std::dynamic_pointer_cast<AMP::Operator::ColumnOperatorParameters>( params );
        AMP_INSIST( columnParams.get() != nullptr, "Dynamic cast failed!" );

        for ( unsigned int i = 0; i < d_Solvers.size(); i++ ) {
            d_Solvers[i]->resetOperator( ( columnParams->d_OperatorParameters )[i] );
        }
    }
}
} // namespace Solver
} // namespace AMP