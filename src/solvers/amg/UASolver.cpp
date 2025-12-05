#include "AMP/solvers/amg/UASolver.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/amg/Aggregation.h"
#include "AMP/solvers/amg/Relaxation.h"
#include "AMP/solvers/amg/Stats.h"
#include "AMP/solvers/amg/default/MIS2Aggregator.h"
#include "AMP/solvers/amg/default/SimpleAggregator.h"

namespace AMP::Solver::AMG {

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
    d_num_relax_pre  = db->getWithDefault<size_t>( "num_relax_pre", 1 );
    d_num_relax_post = db->getWithDefault<size_t>( "num_relax_post", 1 );
    d_boomer_cg      = db->getWithDefault<bool>( "boomer_cg", false );

    d_cycle_settings.kappa = db->getWithDefault<size_t>( "kappa", 1 );
    d_cycle_settings.tol   = db->getWithDefault<float>( "kcycle_tol", 0 );
    d_cycle_settings.type =
        KappaKCycle::parseType( db->getWithDefault<std::string>( "kcycle_type", "fcg" ) );

    d_coarsen_settings.strength_threshold = db->getWithDefault<float>( "strength_threshold", 0.25 );
    d_coarsen_settings.min_coarse_local   = db->getWithDefault<int>( "min_coarse_local", 10 );
    d_coarsen_settings.min_coarse         = db->getWithDefault<size_t>( "min_coarse_global", 100 );
    d_coarsen_settings.pairwise_passes    = db->getWithDefault<size_t>( "pairwise_passes", 2 );
    d_coarsen_settings.checkdd            = db->getWithDefault<bool>( "checkdd", true );

    d_implicit_RAP = d_cycle_settings.comm_free_interp =
        db->getWithDefault<bool>( "implicit_RAP", false );
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "UASolver: using " << ( ( d_implicit_RAP ) ? "implicit" : "explicit" )
                  << " RAP" << std::endl;
    }

    auto agg_type = db->getWithDefault<std::string>( "agg_type", "MIS2" );
    if ( !d_implicit_RAP || ( d_implicit_RAP && ( agg_type != "pairwise" ) ) ) {
        if ( agg_type == "simple" ) {
            d_aggregator = std::make_shared<SimpleAggregator>( d_coarsen_settings );
        } else if ( agg_type == "pairwise" ) {
            d_aggregator = std::make_shared<PairwiseAggregator>( d_coarsen_settings );
        } else {
            d_aggregator = std::make_shared<MIS2Aggregator>( d_coarsen_settings );
        }
    }

    auto pre_db        = db->getDatabase( "pre_relaxation" );
    d_pre_relax_params = std::make_shared<RelaxationParameters>( pre_db );

    auto post_db        = db->getDatabase( "post_relaxation" );
    d_post_relax_params = std::make_shared<RelaxationParameters>( post_db );

    AMP_INSIST( db->keyExists( "coarse_solver" ), "UASolver: Key coarse_solver is missing!" );
    auto cg_solver_db = db->getDatabase( "coarse_solver" );
    AMP_INSIST( db->keyExists( "name" ),
                "UASolver: Key name does not exist in coarse solver database" );
    d_coarse_solver_params = std::make_shared<SolverStrategyParameters>( cg_solver_db );
}

std::unique_ptr<SolverStrategy>
UASolver::create_relaxation( size_t lvl,
                             std::shared_ptr<AMP::Operator::LinearOperator> A,
                             std::shared_ptr<RelaxationParameters> params )
{
    auto rel_op = SolverFactory::create( params );
    rel_op->registerOperator( A );
    auto &op = *rel_op;
    dynamic_cast<Relaxation &>( op ).setLevel( lvl );
    return rel_op;
}

void UASolver::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    auto linop = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op );
    AMP_INSIST( linop, "UASolver: operator must be linear" );
    auto mat = linop->getMatrix();
    AMP_INSIST( mat, "UASolver: matrix cannot be NULL" );

    // verify this is actually a CSRMatrix
    const auto mode = mat->mode();
    AMP_INSIST( mode < std::numeric_limits<std::uint16_t>::max(),
                "UASolver::registerOperator: Must pass in linear operator in CSRMatrix format" );

    // determine the memory location from the mode
    const auto csr_mode = static_cast<LinearAlgebra::csr_mode>( mode );
    auto csr_alloc      = LinearAlgebra::get_alloc( csr_mode );
    if ( csr_alloc == LinearAlgebra::alloc::host ) {
        d_mem_loc = Utilities::MemoryType::host;
    } else if ( csr_alloc == LinearAlgebra::alloc::managed ) {
        d_mem_loc = Utilities::MemoryType::managed;
    } else if ( csr_alloc == LinearAlgebra::alloc::device ) {
        d_mem_loc = Utilities::MemoryType::device;
    } else {
        AMP_ERROR( "UASolver: Unrecognized memory location" );
    }

    d_levels.clear();
    d_levels.emplace_back().A       = std::make_shared<LevelOperator>( *linop );
    d_levels.back().pre_relaxation  = create_relaxation( 0, linop, d_pre_relax_params );
    d_levels.back().post_relaxation = create_relaxation( 0, linop, d_post_relax_params );
    d_levels.back().r               = linop->getMatrix()->createInputVector();
    d_levels.back().correction      = linop->getMatrix()->createInputVector();

    setup();
}


void UASolver::makeCoarseSolver()
{
    auto coarse_op                      = d_levels.back().A;
    d_coarse_solver_params->d_pOperator = coarse_op;
    d_coarse_solver_params->d_comm      = coarse_op->getMatrix()->getComm();
    d_coarse_solver = AMP::Solver::SolverFactory::create( d_coarse_solver_params );
}


coarse_ops_type UASolver::coarsen( std::shared_ptr<Operator::LinearOperator> Aop,
                                   const PairwiseCoarsenSettings &coarsen_settings,
                                   std::shared_ptr<Operator::OperatorParameters> op_params )
{
    if ( d_implicit_RAP ) {
        if ( !d_aggregator )
            return pairwise_coarsen( Aop, coarsen_settings );
        return aggregator_coarsen( Aop, *d_aggregator );
    }

    auto A  = Aop->getMatrix();
    auto P  = d_aggregator->getAggregateMatrix( A );
    auto R  = P->transpose();
    auto AP = LinearAlgebra::Matrix::matMatMult( A, P );
    auto Ac = LinearAlgebra::Matrix::matMatMult( R, AP );

    auto make_op = [=]( auto mat ) {
        auto op = std::make_shared<Operator::LinearOperator>( op_params );
        std::dynamic_pointer_cast<Operator::LinearOperator>( op )->setMatrix( mat );
        return op;
    };

    return { make_op( R ), make_op( Ac ), make_op( P ) };
}


void UASolver::setup()
{
    PROFILE( "UASolver::setup" );

    auto op_db = std::make_shared<Database>( "UASolver::Internal" );
    if ( d_mem_loc == Utilities::MemoryType::host ) {
        op_db->putScalar<std::string>( "memory_location", "host" );
    } else {
        AMP_ERROR( "UASolver: Only host memory is supported currently" );
    }
    auto op_params = std::make_shared<Operator::OperatorParameters>( op_db );

    auto coarse_too_small = [&]( std::shared_ptr<AMP::Operator::Operator> op ) {
        auto linop = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op );
        AMP_DEBUG_ASSERT( linop );
        auto matrix = linop->getMatrix();
        AMP_DEBUG_ASSERT( matrix );
        int nrows_local   = static_cast<int>( matrix->numLocalRows() );
        auto nrows_global = matrix->numGlobalRows();
        return nrows_global <= d_coarsen_settings.min_coarse ||
               matrix->getComm().anyReduce( nrows_local < d_coarsen_settings.min_coarse_local );
    };

    auto &fine_settings     = d_coarsen_settings;
    auto coarse_settings    = fine_settings;
    coarse_settings.checkdd = false; // checkdd only used on fine grid
    for ( size_t i = 0; i < d_max_levels; ++i ) {
        auto &fine_level = d_levels.back();
        auto [R, Ac, P] =
            coarsen( fine_level.A, ( i == 0 ? fine_settings : coarse_settings ), op_params );
        if ( !Ac )
            break;

        d_levels.emplace_back().A       = std::make_shared<LevelOperator>( *Ac );
        d_levels.back().R               = R;
        d_levels.back().P               = P;
        d_levels.back().pre_relaxation  = create_relaxation( i + 1, Ac, d_pre_relax_params );
        d_levels.back().post_relaxation = create_relaxation( i + 1, Ac, d_post_relax_params );
        d_levels.back().x               = Ac->getMatrix()->createInputVector();
        d_levels.back().b               = Ac->getMatrix()->createInputVector();
        d_levels.back().r               = Ac->getMatrix()->createInputVector();
        d_levels.back().correction      = Ac->getMatrix()->createInputVector();
        clone_workspace( d_levels.back(), *( d_levels.back().x ) );

        if ( coarse_too_small( Ac ) )
            break;
    }

    makeCoarseSolver();
    if ( d_iDebugPrintInfoLevel > 1 )
        print_summary( type(), d_levels, *d_coarse_solver );
}

void UASolver::apply( std::shared_ptr<const LinearAlgebra::Vector> b,
                      std::shared_ptr<LinearAlgebra::Vector> x )
{
    PROFILE( "UASolver::apply" );

    d_iNumberIterations   = 0;
    const bool need_norms = d_iMaxIterations > 1 || d_iDebugPrintInfoLevel > 1;
    auto r                = b->clone();

    const auto b_norm =
        need_norms ? static_cast<double>( b->L2Norm() ) : std::numeric_limits<double>::max();

    // Zero rhs implies zero solution, bail out early
    if ( b_norm == 0.0 ) {
        x->zero();
        d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
        d_dResidualNorm     = 0.0;
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "UASolver::apply: solution is zero" << std::endl;
        }
        return;
    }

    d_dInitialResidual = [&]() {
        if ( d_bUseZeroInitialGuess ) {
            x->zero();
            return b_norm;
        }

        d_pOperator->residual( b, x, r );
        return need_norms ? static_cast<double>( r->L2Norm() ) : std::numeric_limits<double>::max();
    }();

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "UASolver::apply: initial L2Norm of solution vector: " << x->L2Norm()
                  << std::endl;
        AMP::pout << "UASolver::apply: initial L2Norm of rhs vector: " << b_norm << std::endl;
        AMP::pout << "UASolver::apply: initial L2Norm of residual: " << d_dInitialResidual
                  << std::endl;
    }

    // return if the residual is already low enough
    // checkStoppingCriteria responsible for setting flags on convergence reason
    if ( checkStoppingCriteria( d_dInitialResidual ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "UASolver::apply: initial residual below tolerance" << std::endl;
        }
        return;
    }

    KappaKCycle cycle{ d_cycle_settings };
    auto current_res = d_dInitialResidual;
    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {
        cycle( b, x, d_levels, *d_coarse_solver );

        d_pOperator->residual( b, x, r );
        current_res =
            need_norms ? static_cast<double>( r->L2Norm() ) : std::numeric_limits<double>::max();
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "UA: iteration " << d_iNumberIterations << ", residual " << current_res
                      << std::endl;
        if ( checkStoppingCriteria( current_res ) )
            break;
    }

    // Store final residual norm and update convergence flags
    d_dResidualNorm = current_res;
    checkStoppingCriteria( current_res );

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "UASolver::apply: final L2Norm of solution: " << x->L2Norm() << std::endl;
        AMP::pout << "UASolver::apply: final L2Norm of residual: " << current_res << std::endl;
        AMP::pout << "UASolver::apply: iterations: " << d_iNumberIterations << std::endl;
        AMP::pout << "UASolver::apply: convergence reason: "
                  << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
    }
}

} // namespace AMP::Solver::AMG
