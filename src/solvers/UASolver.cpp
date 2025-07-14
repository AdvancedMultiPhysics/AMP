#include "AMP/solvers/SolverFactory.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/UASolver.h"
#include "AMP/solvers/amg/Aggregation.h"
#include "AMP/solvers/amg/Stats.h"

namespace AMP::Solver {

UASolver::UASolver( std::shared_ptr<SolverStrategyParameters> params ) : SolverStrategy( params )
{
    AMP_ASSERT( params );

    getFromInput( params->d_db );

    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
}

void UASolver::getFromInput( std::shared_ptr<AMP::Database> db )
{
    d_max_levels     = db->getWithDefault<size_t>( "max_levels", 10 );
    d_min_coarse     = db->getWithDefault<size_t>( "min_coarse", 10 );
    d_num_relax_pre  = db->getWithDefault<size_t>( "num_relax_pre", 1 );
    d_num_relax_post = db->getWithDefault<size_t>( "num_relax_post", 1 );
    d_boomer_cg      = db->getWithDefault<bool>( "boomer_cg", false );
    d_kappa          = db->getWithDefault<size_t>( "kappa", 1 );
    d_kcycle_tol     = db->getWithDefault<float>( "kcycle_tol", 0 );

    d_coarsen_settings.strength_threshold = db->getWithDefault<float>( "strength_threshold", 0.25 );
    d_coarsen_settings.redist_coarsen_factor =
        db->getWithDefault<size_t>( "redist_coarsen_factor", 2 );
    d_coarsen_settings.min_local_coarse = db->getWithDefault<size_t>( "min_local_coarse", 5 );
    d_coarsen_settings.min_coarse       = db->getWithDefault<size_t>( "min_coarse", 10 );
    d_coarsen_settings.pairwise_passes  = db->getWithDefault<size_t>( "pairwise_passes", 2 );
    d_coarsen_settings.checkdd          = db->getWithDefault<bool>( "checkdd", true );

    auto pre_db        = db->getDatabase( "pre_relaxation" );
    d_pre_relax_params = std::make_shared<AMG::RelaxationParameters>( pre_db );

    auto post_db        = db->getDatabase( "post_relaxation" );
    d_post_relax_params = std::make_shared<AMG::RelaxationParameters>( post_db );

    AMP_INSIST( db->keyExists( "coarse_solver" ), "Key coarse_solver is missing!" );
    auto cg_solver_db = db->getDatabase( "coarse_solver" );
    AMP_INSIST( db->keyExists( "name" ), "Key name does not exist in coarse solver database" );
    d_coarse_solver_params = std::make_shared<SolverStrategyParameters>( cg_solver_db );
}

std::unique_ptr<SolverStrategy>
UASolver::create_relaxation( std::shared_ptr<AMP::Operator::LinearOperator> A,
                             std::shared_ptr<AMG::RelaxationParameters> params )
{
	auto rel_op = SolverFactory::create( params );
	rel_op->registerOperator( A );
	return rel_op;
}

void UASolver::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    auto linop = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op );
    AMP_INSIST( linop, "UASolver: operator must be linear" );
    auto mat = linop->getMatrix();
    AMP_INSIST( mat, "matrix cannot be NULL" );

    d_levels.clear();
    d_levels.emplace_back().A       = linop;
    d_levels.back().pre_relaxation  = create_relaxation( linop, d_pre_relax_params );
    d_levels.back().post_relaxation = create_relaxation( linop, d_post_relax_params );

    setup();
}


void UASolver::makeCoarseSolver()
{
    auto coarse_op                      = d_levels.back().A;
    d_coarse_solver_params->d_pOperator = coarse_op;
    d_coarse_solver_params->d_comm      = coarse_op->getMatrix()->getComm();
    d_coarse_solver = AMP::Solver::SolverFactory::create( d_coarse_solver_params );
}


void UASolver::setup()
{
    auto num_rows = []( std::shared_ptr<AMP::Operator::Operator> op ) {
        auto linop = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op );
        AMP_DEBUG_ASSERT( linop );
        auto matrix = linop->getMatrix();
        AMP_DEBUG_ASSERT( matrix );
        return matrix->numGlobalRows();
    };
    for ( size_t i = 0; i < d_max_levels; ++i ) {
        auto &fine_level = d_levels.back();
        auto [R, Ac, P]  = AMG::pairwise_coarsen( fine_level.A, d_coarsen_settings );
        if ( !Ac )
            break;

        d_levels.emplace_back().A = Ac;
        d_levels.back().R               = R;
        d_levels.back().P               = P;
        d_levels.back().pre_relaxation  = create_relaxation( Ac, d_pre_relax_params );
        d_levels.back().post_relaxation = create_relaxation( Ac, d_post_relax_params );
        d_levels.back().x               = Ac->getMatrix()->getRightVector();
        d_levels.back().b               = Ac->getMatrix()->getRightVector();

        if ( num_rows( Ac ) <= d_coarsen_settings.min_coarse )
            break;
    }

    makeCoarseSolver();
}

void UASolver::apply( std::shared_ptr<const LinearAlgebra::Vector> b,
                      std::shared_ptr<LinearAlgebra::Vector> x )
{
    x->zero(); // TODO: remove (for preconditioning)
    auto &A = *( d_levels[0].A );

    auto r = x->clone();
    A.residual( b, x, r );
    for ( int i = 0; i < d_iMaxIterations; ++i ) {
        kappa_kcycle( b, x, d_levels, *d_coarse_solver, d_kappa, d_kcycle_tol );
        A.residual( b, x, r );
        auto rnorm = r->L2Norm();
        if ( rnorm <= d_dRelativeTolerance )
            break;
    }
}

void UASolver::print_summary( std::size_t niters, float setup_time, float solve_time )
{
    AMG::print_summary( "summary", d_levels, *d_coarse_solver, niters, setup_time, solve_time );
}

} // namespace AMP::Solver
