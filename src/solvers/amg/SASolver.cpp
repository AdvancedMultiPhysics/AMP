#include <cmath>
#include <limits>

#include "ProfilerApp.h"

#include "AMP/solvers/amg/Aggregator.hpp"
#include "AMP/solvers/amg/Relaxation.hpp"
#include "AMP/solvers/amg/SASolver.h"
#include "AMP/solvers/amg/Stats.h"
#include "AMP/solvers/amg/default/MIS2Aggregator.hpp"
#include "AMP/solvers/amg/default/SimpleAggregator.hpp"

namespace AMP::Solver::AMG {

SASolver::SASolver( std::shared_ptr<SolverStrategyParameters> params ) : SolverStrategy( params )
{
    AMP_ASSERT( params );
    getFromInput( params->d_db );
    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
}

void SASolver::getFromInput( std::shared_ptr<Database> db )
{
    // settings applicable to entire solver
    d_max_levels                      = db->getWithDefault<size_t>( "max_levels", 10 );
    d_min_coarse_local                = db->getWithDefault<int>( "min_coarse_local", 10 );
    d_min_coarse_global               = db->getWithDefault<size_t>( "min_coarse_global", 100 );
    d_cycle_settings.kappa            = db->getWithDefault<size_t>( "kappa", 1 );
    d_cycle_settings.tol              = db->getWithDefault<double>( "kcycle_tol", 0.0 );
    d_cycle_settings.comm_free_interp = false;
    d_cycle_settings.type =
        KappaKCycle::parseType( db->getWithDefault<std::string>( "kcycle_type", "fcg" ) );

    // get and setup coarse solver options
    AMP_INSIST( db->keyExists( "coarse_solver" ), "Key coarse_solver is missing!" );
    auto coarse_solver_db = db->getDatabase( "coarse_solver" );
    AMP_INSIST( db->keyExists( "name" ), "Key name does not exist in coarse solver database" );
    d_coarse_solver_params = std::make_shared<SolverStrategyParameters>( coarse_solver_db );

    // Apply default per-level options
    resetLevelOptions();
}

void SASolver::resetLevelOptions()
{
    d_coarsen_settings.strength_threshold =
        d_db->getWithDefault<float>( "strength_threshold", 0.25 );
    d_coarsen_settings.strength_measure =
        d_db->getWithDefault<std::string>( "strength_measure", "classical_min" );
    d_pair_coarsen_settings                 = d_coarsen_settings;
    d_pair_coarsen_settings.pairwise_passes = d_db->getWithDefault<size_t>( "pairwise_passes", 2 );
    d_pair_coarsen_settings.checkdd         = d_db->getWithDefault<bool>( "checkdd", true );
    d_num_smooth_prol                       = d_db->getWithDefault<int>( "num_smooth_prol", 1 );
    d_prol_trunc                            = d_db->getWithDefault<float>( "prol_trunc", 0 );
    d_prol_spec_lower                       = d_db->getWithDefault<float>( "prol_spec_lower", 0.5 );
    d_agg_type      = d_db->getWithDefault<std::string>( "agg_type", "simple" );
    d_pre_relax_db  = d_db->getDatabase( "pre_relaxation" );
    d_post_relax_db = d_db->getDatabase( "post_relaxation" );
}

void SASolver::setLevelOptions( const size_t lvl )
{
    // If a new set of level options is available
    // apply them over the defaults.
    // If no new options are available then keep existing
    // ones without falling back to defaults
    auto lvl_db_name =
        std::string( "level_options_" ) + AMP::Utilities::intToString( static_cast<int>( lvl ) );
    auto lvl_db = d_db->getDatabase( lvl_db_name );
    if ( lvl_db ) {
        resetLevelOptions();

        d_coarsen_settings.strength_threshold = lvl_db->getWithDefault<float>(
            "strength_threshold", d_coarsen_settings.strength_threshold );
        d_coarsen_settings.strength_measure = lvl_db->getWithDefault<std::string>(
            "strength_measure", d_coarsen_settings.strength_measure );
        d_pair_coarsen_settings                 = d_coarsen_settings;
        d_pair_coarsen_settings.pairwise_passes = lvl_db->getWithDefault<size_t>(
            "pairwise_passes", d_pair_coarsen_settings.pairwise_passes );
        d_pair_coarsen_settings.checkdd =
            lvl_db->getWithDefault<bool>( "checkdd", d_pair_coarsen_settings.checkdd );
        d_num_smooth_prol = lvl_db->getWithDefault<int>( "num_smooth_prol", d_num_smooth_prol );
        d_prol_trunc      = lvl_db->getWithDefault<float>( "prol_trunc", d_prol_trunc );
        d_prol_spec_lower = lvl_db->getWithDefault<float>( "prol_spec_lower", d_prol_spec_lower );
        d_agg_type        = lvl_db->getWithDefault<std::string>( "agg_type", d_agg_type );

        // only replace these DBs if they exist
        auto pre_relax_db = lvl_db->getDatabase( "pre_relaxation" );
        if ( pre_relax_db ) {
            d_pre_relax_db = pre_relax_db;
        }
        auto post_relax_db = lvl_db->getDatabase( "post_relaxation" );
        if ( post_relax_db ) {
            d_post_relax_db = post_relax_db;
        }
    }

    // create/replace aggregator
    if ( d_agg_type == "simple" ) {
        d_aggregator = std::make_shared<AMG::SimpleAggregator>( d_coarsen_settings );
    } else if ( d_agg_type == "pairwise" ) {
        d_aggregator = std::make_shared<PairwiseAggregator>( d_pair_coarsen_settings );
    } else {
        d_aggregator = std::make_shared<AMG::MIS2Aggregator>( d_coarsen_settings );
    }

    // create relaxation parameters
    // these are the only items with no defaults so these DBs must exist from somewhere
    AMP_INSIST( d_pre_relax_db && d_post_relax_db,
                "SASolver: pre_relaxation and post_relaxation parameters must be set" );
    d_pre_relax_params  = std::make_shared<AMG::RelaxationParameters>( d_pre_relax_db );
    d_post_relax_params = std::make_shared<AMG::RelaxationParameters>( d_post_relax_db );
}

void SASolver::registerOperator( std::shared_ptr<Operator::Operator> op )
{
    // store operator, destroy hierarchy, reset to fine level options
    d_pOperator = op;
    d_levels.clear();
    resetLevelOptions();
    setLevelOptions( 0 );

    // unwrap operator
    auto fine_op = std::dynamic_pointer_cast<Operator::LinearOperator>( op );
    AMP_INSIST( fine_op, "SASolver: operator must be linear" );
    auto mat = fine_op->getMatrix();
    AMP_INSIST( mat, "SASolver: matrix cannot be NULL" );

    // verify this is actually a CSRMatrix
    const auto mode = mat->mode();
    if ( mode == std::numeric_limits<std::uint16_t>::max() ) {
        AMP::pout << "Expected a CSRMatrix but received a matrix of type: " << mat->type()
                  << std::endl;
        AMP_ERROR( "SASolver::registerOperator: Must pass in linear operator in CSRMatrix format" );
    }

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
        AMP_ERROR( "Unrecognized memory location" );
    }

    // fill in finest level and setup remaining levels
    //    d_levels.emplace_back().A       = std::make_shared<LevelOperator>( *fine_op );
    auto op_db                = std::make_shared<Database>( "SASolver::Internal" );
    auto op_params            = std::make_shared<Operator::OperatorParameters>( op_db );
    d_levels.emplace_back().A = std::make_shared<LevelOperator>( op_params );
    d_levels.back().A->setMatrix( mat );
    d_levels.back().pre_relaxation  = createRelaxation( 0, fine_op, d_pre_relax_params );
    d_levels.back().post_relaxation = createRelaxation( 0, fine_op, d_post_relax_params );
    d_levels.back().r               = fine_op->getMatrix()->createOutputVector();
    d_levels.back().correction      = fine_op->getMatrix()->createInputVector();

    auto xVar = d_levels.back().correction->getVariable();
    auto bVar = d_levels.back().r->getVariable();
    d_levels.back().A->setVariables( xVar, bVar );

    setup( xVar, bVar );
}

std::unique_ptr<SolverStrategy>
SASolver::createRelaxation( size_t lvl,
                            std::shared_ptr<Operator::Operator> A,
                            std::shared_ptr<AMG::RelaxationParameters> params )
{
    auto rel_op = Solver::SolverFactory::create( params );
    rel_op->registerOperator( A );
    auto &op = *rel_op;
    dynamic_cast<Relaxation &>( op ).setLevel( lvl );
    return rel_op;
}

void SASolver::makeCoarseSolver()
{
    auto coarse_op                      = d_levels.back().A;
    d_coarse_solver_params->d_pOperator = coarse_op;
    d_coarse_solver_params->d_comm      = coarse_op->getMatrix()->getComm();
    d_coarse_solver                     = Solver::SolverFactory::create( d_coarse_solver_params );
    d_coarse_solver->registerOperator( coarse_op );
}

void SASolver::smoothP_JacobiL1( std::shared_ptr<LinearAlgebra::Matrix> A,
                                 std::shared_ptr<LinearAlgebra::Matrix> &P ) const
{
    // Get D as absolute row sums of A
    // ignore zero values since those rows won't matter anyway
    auto D = A->getRowSumsAbsolute( LinearAlgebra::Vector::shared_ptr(), true );

    // optimal weight for prescribed lower e-val estimate
    const double omega = 2.0 / ( 1.0 + d_prol_spec_lower );

    // Smooth P, swapping at end each time
    for ( int i = 0; i < d_num_smooth_prol; ++i ) {
        // First A * P
        auto P_smooth = LinearAlgebra::Matrix::matMatMult( A, P );

        // then apply -Dinv in-place
        P_smooth->scaleInv( -omega, D );

        // add back in P_tent
        P_smooth->axpy( 1.0, P );

        P.swap( P_smooth );
        P_smooth.reset();

        if ( d_prol_trunc > 0.0 ) {
            P->getMatrixData()->removeRange( -d_prol_trunc, d_prol_trunc );
        }
    }
    D.reset();
    auto Ds = P->getRowSums();
    P->scaleInv( 1.0, Ds );
}

void SASolver::setup( std::shared_ptr<LinearAlgebra::Variable> xVar,
                      std::shared_ptr<LinearAlgebra::Variable> bVar )
{
    PROFILE( "SASolver::setup" );

    auto op_db = std::make_shared<Database>( "SASolver::Internal" );
    if ( d_mem_loc == Utilities::MemoryType::host ) {
        op_db->putScalar<std::string>( "memory_location", "host" );
    } else {
        AMP_ERROR( "SASolver: Only host memory is supported currently" );
    }
    auto op_params = std::make_shared<Operator::OperatorParameters>( op_db );

    for ( size_t i = 0; i < d_max_levels; ++i ) {
        // Get matrix for current level
        auto A = d_levels.back().A->getMatrix();

        // aggregate on matrix to get tentative prolongator
        // then smooth and transpose to get P/R
        auto P = d_aggregator->getAggregateMatrix( A );
        smoothP_JacobiL1( A, P );
        auto R = P->transpose();

        // residual on current level needs comm list replaced by what R needs
        d_levels.back().r = R->createInputVector();

        // Find coarsened A
        auto AP = LinearAlgebra::Matrix::matMatMult( A, P );
        auto Ac = LinearAlgebra::Matrix::matMatMult( R, AP );

        const auto Ac_nrows_gbl = Ac->numGlobalRows();

        // from here the new level gets created, load any new options
        setLevelOptions( i + 1 );

        // create next level with coarsened matrix
        d_levels.emplace_back().A = std::make_shared<LevelOperator>( op_params );
        d_levels.back().A->setMatrix( Ac );
        d_levels.back().A->setVariables( xVar, bVar );

        // Attach restriction/prolongation operators for getting to/from new level
        d_levels.back().R = std::make_shared<Operator::LinearOperator>( op_params );
        std::dynamic_pointer_cast<Operator::LinearOperator>( d_levels.back().R )->setMatrix( R );
        std::dynamic_pointer_cast<Operator::LinearOperator>( d_levels.back().R )
            ->setVariables( bVar, xVar );

        d_levels.back().P = std::make_shared<Operator::LinearOperator>( op_params );
        std::dynamic_pointer_cast<Operator::LinearOperator>( d_levels.back().P )->setMatrix( P );
        std::dynamic_pointer_cast<Operator::LinearOperator>( d_levels.back().P )
            ->setVariables( bVar, xVar );

        // Relaxation operators for new level
        d_levels.back().pre_relaxation =
            createRelaxation( i + 1, d_levels.back().A, d_pre_relax_params );
        d_levels.back().post_relaxation =
            createRelaxation( i + 1, d_levels.back().A, d_post_relax_params );

        // in/out vectors for new level
        d_levels.back().x          = Ac->createInputVector();
        d_levels.back().b          = Ac->createInputVector();
        d_levels.back().r          = Ac->createInputVector();
        d_levels.back().correction = Ac->createInputVector();
        clone_workspace( d_levels.back(), *( d_levels.back().x ) );

        // if newest level is small enough break out
        // and make residual vector for coarsest level
        const auto Ac_nrows_loc = static_cast<int>( Ac->numLocalRows() );
        auto comm               = Ac->getComm();
        if ( Ac_nrows_gbl <= d_min_coarse_global ||
             comm.anyReduce( Ac_nrows_loc < d_min_coarse_local ) ) {
            break;
        }
    }

    makeCoarseSolver();

    if ( d_iDebugPrintInfoLevel > 2 ) {
        print_summary( type(), d_levels, *d_coarse_solver );
    }
}

void SASolver::apply( std::shared_ptr<const LinearAlgebra::Vector> b,
                      std::shared_ptr<LinearAlgebra::Vector> x )
{
    PROFILE( "SASolver::apply" );

    AMP_INSIST( x, "SASolver::apply Can't have null solution vector" );

    d_iNumberIterations = 0;
    const bool need_norms =
        d_iMaxIterations > 1 && ( d_dAbsoluteTolerance > 0.0 || d_dRelativeTolerance > 0.0 );
    auto r = b->clone();
    double current_res;

    const auto b_norm =
        need_norms ? static_cast<double>( b->L2Norm() ) : std::numeric_limits<double>::max();

    // Zero rhs implies zero solution, bail out early
    if ( b_norm == 0.0 ) {
        x->zero();
        d_ConvergenceStatus = SolverStatus::ConvergedOnAbsTol;
        d_dResidualNorm     = 0.0;
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "SASolver::apply: solution is zero" << std::endl;
        }
        return;
    }

    if ( d_bUseZeroInitialGuess ) {
        x->zero();
        current_res = b_norm;
    } else {
        d_pOperator->residual( b, x, r );
        current_res =
            need_norms ? static_cast<double>( r->L2Norm() ) : std::numeric_limits<double>::max();
    }
    d_dInitialResidual = current_res;

    if ( need_norms && d_iDebugPrintInfoLevel > 2 ) {
        const auto version = AMPManager::revision();
        AMP::pout << "SASolver: AMP version " << version[0] << "." << version[1] << "."
                  << version[2] << std::endl;
        AMP::pout << "SASolver: Memory location " << AMP::Utilities::getString( d_mem_loc )
                  << std::endl;
        AMP::pout << "SASolver: kappa " << d_cycle_settings.kappa << std::endl;
        AMP::pout << "SASolver: k-cycle type "
                  << KappaKCycle::krylovTypeName( d_cycle_settings.type ) << std::endl;
    }

    if ( need_norms && d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "SASolver::apply: initial L2Norm of solution vector " << x->L2Norm()
                  << std::endl;
        AMP::pout << "SASolver::apply: initial L2Norm of rhs vector " << b_norm << std::endl;
        AMP::pout << "SASolver::apply: initial L2Norm of residual " << current_res << std::endl;
    }

    // return if the residual is already low enough
    // checkStoppingCriteria responsible for setting flags on convergence reason
    if ( need_norms && checkStoppingCriteria( current_res ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "SASolver::apply: initial residual below tolerance" << std::endl;
        }
        return;
    }

    double prev_res = 0.0;
    KappaKCycle cycle{ d_cycle_settings };
    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {
        cycle( b, x, d_levels, *d_coarse_solver );

        d_pOperator->residual( b, x, r );
        current_res =
            need_norms ? static_cast<double>( r->L2Norm() ) : std::numeric_limits<double>::max();

        if ( need_norms && d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "SA: iteration " << d_iNumberIterations << ", residual " << current_res
                      << ", conv ratio ";
            if ( prev_res > 0.0 ) {
                AMP::pout << current_res / prev_res << std::endl;
            } else {
                AMP::pout << "--" << std::endl;
            }
            prev_res = current_res;
        }

        if ( need_norms && checkStoppingCriteria( current_res ) ) {
            break;
        }
    }

    // Store final residual norm and update convergence flags
    if ( need_norms ) {
        d_dResidualNorm = current_res;
        checkStoppingCriteria( current_res );

        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "SASolver::apply: final L2Norm of solution: " << x->L2Norm() << std::endl;
            AMP::pout << "SASolver::apply: final L2Norm of residual: " << current_res << std::endl;
            AMP::pout << "SASolver::apply: iterations: " << d_iNumberIterations << std::endl;
            AMP::pout << "SASolver::apply: convergence reason: "
                      << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
        }
    }
}

} // namespace AMP::Solver::AMG
