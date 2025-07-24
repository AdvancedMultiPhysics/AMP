#include "AMP/solvers/ColumnSolver.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/ColumnOperatorParameters.h"
#include "AMP/solvers/SolverFactory.h"


namespace AMP::Solver {


ColumnSolver::ColumnSolver( std::shared_ptr<const SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{
    AMP_ASSERT( parameters );
    auto db               = parameters->d_db;
    d_IterationType       = db->getWithDefault<std::string>( "IterationType", "GaussSeidel" );
    d_resetColumnOperator = db->getWithDefault<bool>( "ResetColumnOperator", false );

    std::shared_ptr<AMP::Operator::ColumnOperator> columnOp;
    if ( d_pOperator ) {
        columnOp = std::dynamic_pointer_cast<AMP::Operator::ColumnOperator>( d_pOperator );
        AMP_INSIST( columnOp, "The operator for a ColumnSolver must be a ColumnOperator" );
    }

    if ( db->keyExists( "solvers" ) ) {
        auto solverNames = db->getVector<std::string>( "solvers" );
        for ( size_t i = 0; i < solverNames.size(); ++i ) {
            auto solverDB = db->getDatabase( solverNames[i] );
            auto params   = std::make_shared<AMP::Solver::SolverStrategyParameters>( solverDB );
            //          params->d_comm          = d_comm;
            // an initial guess should be made possible also in time for truly getting this to work
            //          params->d_pInitialGuess = initialGuess;
            params->d_global_db = db;
            if ( columnOp )
                params->d_pOperator = columnOp->getOperator( i );
            std::shared_ptr<SolverStrategy> solver = AMP::Solver::SolverFactory::create( params );
            append( solver );
        }
    }
}

void ColumnSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
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
            auto op = elem->getOperator();
            AMP_INSIST( op, "EROR: NULL Operator returned by SolverStrategy::getOperator" );

            auto sf = op->subsetOutputVector( f );
            AMP_INSIST( sf, "ERROR: subset on rhs f yields NULL vector in ColumnSolver::solve" );
            auto su = op->subsetInputVector( u );
            AMP_INSIST( su,
                        "ERROR: subset on solution u yields NULL vector in ColumnSolver::solve" );

            elem->apply( sf, su );
        }
    }
}

void ColumnSolver::SymmetricGaussSeidel( std::shared_ptr<const AMP::LinearAlgebra::Vector> &f,
                                         std::shared_ptr<AMP::LinearAlgebra::Vector> &u )
{
    for ( int it = 0; it < d_iMaxIterations; it++ ) {
        for ( auto &elem : d_Solvers ) {
            auto op = elem->getOperator();
            AMP_INSIST( op, "EROR: NULL Operator returned by SolverStrategy::getOperator" );

            auto sf = op->subsetOutputVector( f );
            AMP_INSIST( sf, "ERROR: subset on rhs f yields NULL vector in ColumnSolver::solve" );
            auto su = op->subsetInputVector( u );
            AMP_INSIST( su,
                        "ERROR: subset on solution u yields NULL vector in ColumnSolver::solve" );

            elem->apply( sf, su );
        }

        for ( int i = (int) d_Solvers.size() - 1; i >= 0; i-- ) {
            auto op = d_Solvers[i]->getOperator();
            AMP_INSIST( op, "EROR: NULL Operator returned by SolverStrategy::getOperator" );

            auto sf = op->subsetOutputVector( f );
            AMP_INSIST( sf, "ERROR: subset on rhs f yields NULL vector in ColumnSolver::solve" );
            auto su = op->subsetInputVector( u );
            AMP_INSIST( su,
                        "ERROR: subset on solution u yields NULL vector in ColumnSolver::solve" );

            d_Solvers[i]->apply( sf, su );
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
    AMP_INSIST( ( solver ), "AMP::Solver::ColumnSolver::append input argument is a NULL solver" );
    d_Solvers.push_back( solver );
}

void ColumnSolver::resetOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    if ( d_resetColumnOperator ) {
        d_pOperator->reset( params );

    } else {
        if ( params ) {
            auto columnParams =
                std::dynamic_pointer_cast<const AMP::Operator::ColumnOperatorParameters>( params );
            AMP_INSIST( columnParams, "Dynamic cast failed!" );

            for ( unsigned int i = 0; i < d_Solvers.size(); i++ ) {
                d_Solvers[i]->resetOperator( columnParams->d_OperatorParameters[i] );
            }
        } else {
            for ( auto &solver : d_Solvers )
                solver->resetOperator( {} );
        }
    }
}

} // namespace AMP::Solver
