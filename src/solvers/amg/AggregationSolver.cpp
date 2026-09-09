#include <cmath>
#include <limits>

#include "ProfilerApp.h"

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/CSRVisit.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/amg/AggregationSolver.h"
#include "AMP/solvers/amg/Aggregator.hpp"
#include "AMP/solvers/amg/IntergridRedist.h"
#include "AMP/solvers/amg/MIS2Aggregator.hpp"
#include "AMP/solvers/amg/Relaxation.hpp"
#include "AMP/solvers/amg/SimpleAggregator.hpp"
#include "AMP/solvers/amg/Stats.h"

namespace AMP::Solver::AMG {

namespace {

struct InactiveCoarseSolver final : SolverStrategy {
    std::string type() const override { return "InactiveCoarseSolver"; }

    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector>,
                std::shared_ptr<AMP::LinearAlgebra::Vector> ) override
    {
    }
};

std::shared_ptr<LinearAlgebra::Vector>
createOperatorInputVector( const std::shared_ptr<Operator::Operator> &op )
{
    if ( !op )
        return nullptr;
    auto linear = std::dynamic_pointer_cast<Operator::LinearOperator>( op );
    if ( !linear )
        return op->createInputVector();
    auto mat = linear->getMatrix();
    auto vec = mat ? mat->createInputVector() : linear->createInputVector();
    if ( vec && linear->getInputVariable() )
        vec->setVariable( linear->getInputVariable() );
    return vec;
}

std::shared_ptr<LinearAlgebra::Vector>
createOperatorOutputVector( const std::shared_ptr<Operator::Operator> &op )
{
    if ( !op )
        return nullptr;
    auto linear = std::dynamic_pointer_cast<Operator::LinearOperator>( op );
    if ( !linear )
        return op->createOutputVector();
    auto mat = linear->getMatrix();
    auto vec = mat ? mat->createOutputVector() : linear->createOutputVector();
    if ( vec && linear->getOutputVariable() )
        vec->setVariable( linear->getOutputVariable() );
    return vec;
}

} // namespace

AggregationSolver::AggregationSolver( std::shared_ptr<SolverStrategyParameters> params )
    : AggregationSolver( params, "AggregationSolver" )
{
    finishConstruction();
}

AggregationSolver::AggregationSolver( std::shared_ptr<SolverStrategyParameters> params,
                                      std::string name )
    : SolverStrategy( params ), d_type( std::move( name ) )
{
    AMP_ASSERT( params );
}

void AggregationSolver::finishConstruction()
{
    getFromInput( d_db );
    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
}

void AggregationSolver::getFromInput( std::shared_ptr<Database> db )
{
    // settings applicable to entire solver
    d_max_levels           = db->getWithDefault<size_t>( "max_levels", 10 );
    d_min_coarse_local     = db->getWithDefault<int>( "min_coarse_local", 10 );
    d_min_coarse_global    = db->getWithDefault<size_t>( "min_coarse_global", 100 );
    d_cycle_settings.kappa = db->getWithDefault<size_t>( "kcycle_kappa", defaultKcycleKappa() );
    d_cycle_settings.tol   = db->getWithDefault<double>( "kcycle_tol", 0.0 );
    auto implicit_RAP      = db->getWithDefault<bool>( "implicit_RAP", false );
    d_cycle_settings.type =
        KappaKCycle::parseType( db->getWithDefault<std::string>( "kcycle_type", "fcg" ) );
    d_cycle_settings.trunc_depth =
        db->getWithDefault<size_t>( "kcycle_trunc_depth", std::numeric_limits<size_t>::max() );

    if ( db->getWithDefault<bool>( "save_to_file", false ) )
        d_flags.raise( flags::save_to_file );
    if ( db->getWithDefault<bool>( "save_to_file_on_ftc", false ) )
        d_flags.raise( flags::save_to_file_on_ftc );
    d_save_to_file_name = db->getWithDefault<std::string>( "save_to_file_name", type() );
    if ( db->getWithDefault<bool>( "propagate_nearnull", defaultPropagateNearNull() ) )
        d_flags.raise( flags::propagate_nearnull );
    if ( db->getWithDefault<bool>( "redistribute", false ) )
        d_flags.raise( flags::redistribute );

    // get and setup coarse solver options
    AMP_INSIST( db->keyExists( "coarse_solver" ), type() + ": Key coarse_solver is missing!" );
    auto coarse_solver_db = db->getDatabase( "coarse_solver" );
    AMP_INSIST( coarse_solver_db->keyExists( "name" ),
                type() + ": Key name does not exist in coarse solver database" );
    d_coarse_solver_params = std::make_shared<SolverStrategyParameters>( coarse_solver_db );

    // Apply default per-level options
    resetLevelOptions();

    if ( implicit_RAP ) {
        AMP_INSIST( supportsImplicitRAP(), type() + ": implicit_RAP is not valid for this solver" );
        AMP_INSIST( d_num_smooth_prol == 0,
                    type() + ": implicit_RAP requires num_smooth_prol = 0" );
        d_flags.raise( flags::implicit_RAP );
    }
    d_cycle_settings.comm_free_interp = implicit_RAP;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << type() << ": using " << ( ( implicit_RAP ) ? "implicit" : "explicit" )
                  << " RAP" << std::endl;
    }
}

void AggregationSolver::resetLevelOptions()
{
    d_coarsen_settings.strength_threshold =
        d_db->getWithDefault<float>( "strength_threshold", defaultStrengthThreshold() );
    d_coarsen_settings.strength_measure =
        d_db->getWithDefault<std::string>( "strength_measure", "symagg_min" );
    d_coarsen_settings.checkdd = d_db->getWithDefault<bool>( "checkdd", true );
    d_coarsen_settings.redist_coarsen_factor =
        d_db->getWithDefault<int>( "redist_coarsen_factor", 2 );
    d_pair_coarsen_settings                       = d_coarsen_settings;
    d_pair_coarsen_settings.redist_coarsen_factor = d_coarsen_settings.redist_coarsen_factor;
    d_pair_coarsen_settings.pairwise_passes = d_db->getWithDefault<size_t>( "pairwise_passes", 2 );
    d_num_smooth_prol = d_db->getWithDefault<int>( "num_smooth_prol", defaultNumSmoothProl() );
    d_prol_trunc      = d_db->getWithDefault<float>( "prol_trunc", 0.005 );
    d_prol_spec_lower = d_db->getWithDefault<float>( "prol_spec_lower", 0.5 );
    d_agg_type        = d_db->getWithDefault<std::string>( "agg_type", defaultAggType() );
    d_pre_relax_db    = d_db->getDatabase( "pre_relaxation" );
    d_post_relax_db   = d_db->getDatabase( "post_relaxation" );
}

void AggregationSolver::setLevelOptions( const size_t lvl )
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
        d_pair_coarsen_settings                       = d_coarsen_settings;
        d_pair_coarsen_settings.redist_coarsen_factor = d_coarsen_settings.redist_coarsen_factor;
        d_pair_coarsen_settings.pairwise_passes       = lvl_db->getWithDefault<size_t>(
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

    rebuildLevelComponents();
}

void AggregationSolver::initializeFineLevelOptions()
{
    resetLevelOptions();
    if ( usesLevelOptions() ) {
        setLevelOptions( 0 );
    } else {
        rebuildLevelComponents();
    }
}

void AggregationSolver::rebuildLevelComponents()
{
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
                type() + ": pre_relaxation and post_relaxation parameters must be set" );
    d_pre_relax_params  = std::make_shared<AMG::RelaxationParameters>( d_pre_relax_db );
    d_post_relax_params = std::make_shared<AMG::RelaxationParameters>( d_post_relax_db );
}

void AggregationSolver::registerOperator( std::shared_ptr<Operator::Operator> op )
{
    // unwrap operator
    auto fine_op = std::dynamic_pointer_cast<Operator::LinearOperator>( op );
    AMP_INSIST( fine_op, type() + ": operator must be linear" );
    auto mat = fine_op->getMatrix();
    AMP_INSIST( mat, type() + ": matrix cannot be NULL" );

    // verify this is actually a CSRMatrix
    const auto mode = mat->mode();
    if ( mode == std::numeric_limits<std::uint16_t>::max() ) {
        AMP::pout << "Expected a CSRMatrix but received a matrix of type: " << mat->type()
                  << std::endl;
        AMP_ERROR( type() +
                   "::registerOperator: Must pass in linear operator in CSRMatrix format" );
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
        AMP_ERROR( type() + ": Unrecognized memory location" );
    }

    // store operator, destroy hierarchy, reset to fine level options
    d_pOperator = op;
    d_levels.clear();
    initializeFineLevelOptions();

    // get in/out variables from given op so that all remaining
    // ops have compatible ones
    auto xVar = fine_op->getInputVariable();
    auto bVar = fine_op->getOutputVariable();

    // fill in finest level and setup remaining levels
    auto op_db                      = createInternalOperatorDb();
    auto op_params                  = std::make_shared<Operator::OperatorParameters>( op_db );
    d_levels.emplace_back().A       = makeFineLevelOperator( fine_op, op_params, xVar, bVar );
    auto relax_op                   = getRelaxationOperator( d_levels.back().A, fine_op );
    d_levels.back().pre_relaxation  = createRelaxation( 0, relax_op, d_pre_relax_params );
    d_levels.back().post_relaxation = createRelaxation( 0, relax_op, d_post_relax_params );
    d_levels.back().r               = createOperatorOutputVector( fine_op );
    d_levels.back().correction      = createOperatorInputVector( fine_op );

    setup( xVar, bVar );
}

std::unique_ptr<SolverStrategy>
AggregationSolver::createRelaxation( size_t lvl,
                                     std::shared_ptr<Operator::Operator> A,
                                     std::shared_ptr<AMG::RelaxationParameters> params )
{
    auto rel_op = Solver::SolverFactory::create( params );
    rel_op->registerOperator( A );
    auto &op = *rel_op;
    dynamic_cast<Relaxation &>( op ).setLevel( lvl );
    return rel_op;
}

void AggregationSolver::setLevelOperatorVariables(
    LevelOperator &level_op,
    std::shared_ptr<LinearAlgebra::Variable> xVar,
    std::shared_ptr<LinearAlgebra::Variable> bVar ) const
{
    if ( xVar && bVar ) {
        level_op.setVariables( xVar, bVar );
    }
}

void AggregationSolver::makeCoarseSolver()
{
    auto coarse_op = d_levels.back().A;
    if ( !coarse_op ) {
        d_coarse_solver = std::make_unique<InactiveCoarseSolver>();
        return;
    }
    d_coarse_solver_params->d_pOperator = coarse_op;
    d_coarse_solver_params->d_comm      = coarse_op->getMatrix()->getComm();
    d_coarse_solver                     = Solver::SolverFactory::create( d_coarse_solver_params );
    if ( shouldRegisterCoarseOperator() ) {
        d_coarse_solver->registerOperator( coarse_op );
    }
}

std::shared_ptr<Database> AggregationSolver::createInternalOperatorDb() const
{
    return std::make_shared<Database>( type() + "::Internal" );
}

PairwiseCoarsenSettings AggregationSolver::getCoarsenSettings( size_t ) const
{
    return d_pair_coarsen_settings;
}

std::shared_ptr<LevelOperator>
AggregationSolver::makeFineLevelOperator( std::shared_ptr<Operator::LinearOperator> fine_op,
                                          std::shared_ptr<Operator::OperatorParameters> op_params,
                                          std::shared_ptr<LinearAlgebra::Variable> xVar,
                                          std::shared_ptr<LinearAlgebra::Variable> bVar ) const
{
    auto level_op = std::make_shared<LevelOperator>( op_params );
    level_op->setMatrix( fine_op->getMatrix() );
    setLevelOperatorVariables( *level_op, xVar, bVar );
    return level_op;
}

std::shared_ptr<LevelOperator>
AggregationSolver::makeCoarseLevelOperator( std::shared_ptr<Operator::LinearOperator> coarse_op,
                                            std::shared_ptr<LinearAlgebra::Variable> xVar,
                                            std::shared_ptr<LinearAlgebra::Variable> bVar ) const
{
    auto level_op = std::make_shared<LevelOperator>( *coarse_op );
    setLevelOperatorVariables( *level_op, xVar, bVar );
    return level_op;
}

std::shared_ptr<Operator::Operator> AggregationSolver::getRelaxationOperator(
    std::shared_ptr<LevelOperator> level_op,
    std::shared_ptr<Operator::LinearOperator> source_op ) const
{
    return relaxationUsesSourceOperator() ? source_op : level_op;
}

void AggregationSolver::smoothP_JacobiL1( std::shared_ptr<LinearAlgebra::Matrix> A,
                                          std::shared_ptr<LinearAlgebra::Matrix> &P ) const
{
    if ( d_num_smooth_prol == 0 ) {
        return;
    }

    // Get D as absolute row sums of A
    // ignore zero values since those rows won't matter anyway
    auto D = A->getRowSumsAbsolute( LinearAlgebra::Vector::shared_ptr(), true );

    // special cases for 1 and 2 pass smoothing using optimized polynomials
    // of Dinv*A as smoothers, see DOI:10.1002/nla.775
    if ( d_num_smooth_prol == 1 ) {
        // one pass smoothing
        // P <- (I - w * Dinv * A)*P with w=4/3
        auto P_smooth = LinearAlgebra::Matrix::matMatMult( A, P ); // A*P
        P_smooth->scaleInv( -4.0 / 3.0, D );                       // -w*Dinv*A*P
        P_smooth->axpy( 1.0, P );                                  // P - w*Dinv*A*P
        // replace P with P_smooth and truncate if requested
        P.swap( P_smooth );
        P_smooth.reset();
        if ( d_prol_trunc > 0.0 ) {
            P->getMatrixData()->removeRange( -d_prol_trunc, d_prol_trunc );
        }
        return;
    } else if ( d_num_smooth_prol == 2 ) {
        // two pass smoothing
        // P <- (I - w * Dinv * A + q * (Dinv * A)^2 )*P with w=4, q=16/5
        // factor out as
        // P <- [I - w * Dinv * A * (I - q/w * Dinv * A)] * P
        // and compute in two steps. Handle inner term first
        auto P_1 = LinearAlgebra::Matrix::matMatMult( A, P ); // A*P
        P_1->scaleInv( -4.0 / 5.0, D );                       // -(q/w)Dinv*A*P
        P_1->axpy( 1.0, P );                                  // P' = (I - q/w * Dinv * A) * P

        auto P_2 = LinearAlgebra::Matrix::matMatMult( A, P_1 ); // A*P'
        P_1.reset();                                            // P' not needed any longer
        P_2->scaleInv( -4.0, D );                               // -w * Dinv * A * P'
        P_2->axpy( 1.0, P );                                    // P smoothed

        // replace P with P_smooth and truncate if requested
        P.swap( P_2 );
        P_2.reset();
        if ( d_prol_trunc > 0.0 ) {
            P->getMatrixData()->removeRange( -d_prol_trunc, d_prol_trunc );
        }
        return;
    }

    // More smoothing steps requested, just apply JL1 with estimate on
    // lower e-val
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
}

std::optional<AggregationSolver::redist_context>
AggregationSolver::redistributeIfNeeded( std::shared_ptr<LinearAlgebra::Matrix> &A ) const
{
    std::optional<redist_context> ret;

    if ( d_flags.raised( flags::redistribute ) ) {
        int nrows_local = static_cast<int>( A->numLocalRows() );
        auto comm       = A->getComm();
        if ( comm.anyReduce( nrows_local < d_min_coarse_local ) ) {
            auto redist_size = comm.getSize() / d_pair_coarsen_settings.redist_coarsen_factor;
            if ( redist_size < 1 ) {
                redist_size = 1;
            }
            auto plan = Utilities::GroupedRedistributionPlan( comm, redist_size );
            A   = LinearAlgebra::csrVisit( A, [&]( auto csr ) -> LinearAlgebra::Matrix::shared_ptr {
                return csr->redistribute( plan );
            } );
            ret = std::move( plan );
        }
    }

    return ret;
}

bool AggregationSolver::coarseTooSmall( std::shared_ptr<Operator::LinearOperator> op ) const
{
    AMP_DEBUG_ASSERT( op );
    auto matrix = op->getMatrix();
    AMP_DEBUG_ASSERT( matrix );
    int nrows_local           = static_cast<int>( matrix->numLocalRows() );
    auto nrows_global         = matrix->numGlobalRows();
    bool passed_global_thresh = nrows_global <= d_min_coarse_global;
    if ( d_flags.raised( flags::redistribute ) ) {
        return passed_global_thresh;
    }
    bool passed_local_thresh = matrix->getComm().anyReduce( nrows_local < d_min_coarse_local );
    return passed_global_thresh || passed_local_thresh;
}

coarse_ops_type
AggregationSolver::coarsen( std::shared_ptr<Operator::LinearOperator> Aop,
                            const PairwiseCoarsenSettings &coarsen_settings,
                            std::shared_ptr<Operator::OperatorParameters> op_params,
                            std::shared_ptr<LinearAlgebra::Vector> &nearNullVec ) const
{
    using direction = IntergridRedist::direction;

    auto make_op = [=]( auto mat ) {
        auto op = std::make_shared<Operator::LinearOperator>( op_params );
        op->setMatrix( mat );
        if ( Aop->getInputVariable() && Aop->getOutputVariable() ) {
            op->setVariables( Aop->getInputVariable(), Aop->getOutputVariable() );
        }
        return op;
    };

    auto make_transfer_op = [=]( auto mat,
                                 std::shared_ptr<LinearAlgebra::Variable> in_var,
                                 std::shared_ptr<LinearAlgebra::Variable> out_var ) {
        auto op = std::make_shared<Operator::LinearOperator>( op_params );
        op->setMatrix( mat );
        op->setVariables( std::move( in_var ), std::move( out_var ) );
        return op;
    };

    auto A            = Aop->getMatrix();
    auto maybe_redist = redistributeIfNeeded( A );
    auto fine_x       = createOperatorInputVector( Aop );
    auto fine_b       = createOperatorOutputVector( Aop );

    auto make_inactive_coarsening = [=]( const redist_context &context ) -> coarse_ops_type {
        auto make_inactive_intergrid =
            [=]( direction dir ) -> std::shared_ptr<AMP::Operator::Operator> {
            if ( dir == direction::down ) {
                return std::make_shared<IntergridRedist>(
                    op_params, dir, context, fine_b, nullptr );
            }
            return std::make_shared<IntergridRedist>(
                op_params, dir, context, std::shared_ptr<LinearAlgebra::Vector>{}, fine_x );
        };
        return { make_inactive_intergrid( direction::down ),
                 nullptr,
                 make_inactive_intergrid( direction::up ) };
    };

    if ( !A ) {
        AMP_ASSERT( maybe_redist.has_value() );
        return make_inactive_coarsening( maybe_redist.value() );
    }

    auto wrap_intergrid =
        [=, &maybe_redist]( std::shared_ptr<AMP::Operator::Operator> transfer,
                            direction dir ) -> std::shared_ptr<AMP::Operator::Operator> {
        if ( !maybe_redist.has_value() ) {
            return transfer;
        }

        auto interface_in =
            ( dir == direction::down ) ? fine_b : createOperatorInputVector( transfer );
        auto interface_out =
            ( dir == direction::down ) ? createOperatorOutputVector( transfer ) : fine_x;

        return std::make_shared<IntergridRedist>( op_params,
                                                  dir,
                                                  maybe_redist.value(),
                                                  std::move( transfer ),
                                                  std::move( interface_in ),
                                                  std::move( interface_out ) );
    };

    if ( d_flags.raised( flags::implicit_RAP ) ) {
        auto fine_op    = maybe_redist.has_value() ? make_op( A ) : Aop;
        auto [R, Ac, P] = ( d_agg_type == "pairwise" ) ?
                              pairwise_coarsen( fine_op, coarsen_settings ) :
                              aggregator_coarsen( fine_op, *d_aggregator );

        return { wrap_intergrid( R, direction::down ), Ac, wrap_intergrid( P, direction::up ) };
    }

    std::shared_ptr<LinearAlgebra::Matrix> P;
    if ( nearNullVec ) {
        std::shared_ptr<LinearAlgebra::Vector> coarseNearNullVec;
        std::tie( P, coarseNearNullVec ) = d_aggregator->getAggregateMatrix( A, nearNullVec );
        nearNullVec.swap( coarseNearNullVec );
    } else {
        P = d_aggregator->getAggregateMatrix( A );
    }

    smoothP_JacobiL1( A, P );
    auto R         = P->transpose();
    auto AP        = LinearAlgebra::Matrix::matMatMult( A, P );
    auto Ac        = LinearAlgebra::Matrix::matMatMult( R, AP );
    auto coarse_op = make_op( Ac );

    return { wrap_intergrid(
                 make_transfer_op( R, Aop->getOutputVariable(), coarse_op->getOutputVariable() ),
                 direction::down ),
             coarse_op,
             wrap_intergrid(
                 make_transfer_op( P, coarse_op->getInputVariable(), Aop->getInputVariable() ),
                 direction::up ) };
}

void AggregationSolver::setup( std::shared_ptr<LinearAlgebra::Variable> xVar,
                               std::shared_ptr<LinearAlgebra::Variable> bVar )
{
    PROFILE( "AggregationSolver::setup" );

    auto op_db     = createInternalOperatorDb();
    auto op_params = std::make_shared<Operator::OperatorParameters>( op_db );

    std::shared_ptr<LinearAlgebra::Vector> nearNullVec;
    if ( propagatesNearNull() ) {
        // For now assume that there is only one near-nullspace vector
        // and that it is a constant on the finest level.
        // This gets scattered into the tentative prolongator and
        // orthonormalized there.
        nearNullVec = createOperatorInputVector( d_levels.back().A );
        nearNullVec->setNoGhosts();
        nearNullVec->setToScalar( 1.0 );
    }

    for ( size_t i = 0; i < d_max_levels; ++i ) {
        auto &fine_level      = d_levels.back();
        auto coarsen_settings = getCoarsenSettings( i );
        auto [R, Ac, P]       = coarsen( fine_level.A, coarsen_settings, op_params, nearNullVec );
        /*
          The potential cases for R, Ac, P are as follows:
          1. redistribution active, rank active   -> R, Ac, and P not null
          2. redistribution active, rank inactive -> Ac null (R and P non null for communication)
          3. redistribution inactive              -> R, Ac, and P not null
         */

        if ( R ) {
            fine_level.r = createOperatorInputVector( R );
        }

        if ( usesLevelOptions() ) {
            setLevelOptions( i + 1 );
        }

        if ( !Ac ) { // redistributed and rank not active on this level
            AMP_INSIST( R && P,
                        type() +
                            ": R and P must not be null to handle redistribution communication." );
            // R and P handle communication at redistribution interface
            d_levels.emplace_back().R = R;
            d_levels.back().P         = P;
            break; // terminate coarsening on inactive rank
        }

        // create next level with coarsened matrix
        d_levels.emplace_back().A = makeCoarseLevelOperator( Ac, xVar, bVar );

        // Attach restriction/prolongation operators for getting to/from new level
        d_levels.back().R = R;
        d_levels.back().P = P;

        // Relaxation operators for new level
        auto relax_op                   = getRelaxationOperator( d_levels.back().A, Ac );
        d_levels.back().pre_relaxation  = createRelaxation( i + 1, relax_op, d_pre_relax_params );
        d_levels.back().post_relaxation = createRelaxation( i + 1, relax_op, d_post_relax_params );

        // in/out vectors for new level
        d_levels.back().x          = createOperatorInputVector( Ac );
        d_levels.back().b          = createOperatorOutputVector( Ac );
        d_levels.back().r          = createOperatorOutputVector( Ac );
        d_levels.back().correction = createOperatorInputVector( Ac );
        d_levels.back().work[0]    = d_levels.back().x->clone();
        d_levels.back().work[1]    = d_levels.back().b->clone();
        d_levels.back().work[2]    = d_levels.back().b->clone();
        d_levels.back().work[3]    = d_levels.back().x->clone();
        d_levels.back().work[4]    = d_levels.back().b->clone();

        // if newest level is small enough break out
        // and make residual vector for coarsest level
        if ( coarseTooSmall( Ac ) ) {
            break;
        }
    }

    makeCoarseSolver();

    if ( d_iDebugPrintInfoLevel > 2 ) {
        print_summary( type(), d_levels, *d_coarse_solver );
    }

    if ( savesToFile() ) {
        save_hierarchy( d_save_to_file_name, d_levels );
    }
}

void AggregationSolver::apply( std::shared_ptr<const LinearAlgebra::Vector> b,
                               std::shared_ptr<LinearAlgebra::Vector> x )
{
    PROFILE( "AggregationSolver::apply" );

    AMP_DEBUG_INSIST( x, type() + "::apply Can't have null solution vector" );

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
            AMP::pout << type() << "::apply: solution is zero" << std::endl;
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
        AMP::pout << type() << ": AMP version " << version[0] << "." << version[1] << "."
                  << version[2] << std::endl;
        AMP::pout << type() << ": Memory location " << AMP::Utilities::getString( d_mem_loc )
                  << std::endl;
        AMP::pout << type() << ": kappa " << d_cycle_settings.kappa << std::endl;
        AMP::pout << type() << ": k-cycle type "
                  << KappaKCycle::krylovTypeName( d_cycle_settings.type ) << std::endl;
    }

    if ( need_norms && d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << type() << "::apply: initial L2Norm of solution vector " << x->L2Norm()
                  << std::endl;
        AMP::pout << type() << "::apply: initial L2Norm of rhs vector " << b_norm << std::endl;
        AMP::pout << type() << "::apply: initial L2Norm of residual " << current_res << std::endl;
    }

    // return if the residual is already low enough
    // checkStoppingCriteria responsible for setting flags on convergence reason
    if ( need_norms && checkStoppingCriteria( current_res ) ) {
        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << type() << "::apply: initial residual below tolerance" << std::endl;
        }
        return;
    }

    double prev_res = current_res, iter2_res = 0.0;
    KappaKCycle cycle{ d_cycle_settings };
    for ( d_iNumberIterations = 1; d_iNumberIterations <= d_iMaxIterations;
          ++d_iNumberIterations ) {
        cycle( b, x, d_levels, *d_coarse_solver );

        d_pOperator->residual( b, x, r );
        current_res =
            need_norms ? static_cast<double>( r->L2Norm() ) : std::numeric_limits<double>::max();

        if ( need_norms && d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << type() << ": iteration " << d_iNumberIterations << ", residual "
                      << current_res << ", conv ratio ";
            if ( d_iNumberIterations == 2 ) {
                iter2_res = current_res;
            }
            AMP::pout << current_res / prev_res;
            prev_res = current_res;
            AMP::pout << ", avg conv ratio ";
            if ( iter2_res > 0.0 ) {
                AMP::pout << std::pow( current_res / iter2_res,
                                       1.0 / static_cast<double>( d_iNumberIterations - 1 ) )
                          << std::endl;
            } else {
                AMP::pout << "-----" << std::endl;
            }
        }

        if ( need_norms && checkStoppingCriteria( current_res ) ) {
            break;
        }
    }

    // Store final residual norm and update convergence flags
    if ( need_norms ) {
        d_dResidualNorm = current_res;
        checkStoppingCriteria( current_res );
        const bool ftc = ( d_ConvergenceStatus != SolverStatus::ConvergedOnAbsTol &&
                           d_ConvergenceStatus != SolverStatus::ConvergedOnRelTol );
        if ( savesToFileOnFailure() && ftc ) {
            save_hierarchy( d_save_to_file_name + "-FTC", d_levels );
        }

        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << type() << "::apply: final L2Norm of solution: " << x->L2Norm()
                      << std::endl;
            AMP::pout << type() << "::apply: final L2Norm of residual: " << current_res
                      << std::endl;
            AMP::pout << type() << "::apply: iterations: " << d_iNumberIterations << std::endl;
            AMP::pout << type() << "::apply: convergence reason: "
                      << SolverStrategy::statusToString( d_ConvergenceStatus ) << std::endl;
        }
    }
}

} // namespace AMP::Solver::AMG

namespace AMP::Utilities {

template struct Flags<AMP::Solver::AMG::AggregationSolver::flags>;

}
