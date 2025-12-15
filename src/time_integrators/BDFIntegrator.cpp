#include "AMP/time_integrators/BDFIntegrator.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/time_integrators/TimeOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"

#include <algorithm>
#include <cmath>
#include <iomanip>

#include "ProfilerApp.h"

namespace AMP::TimeIntegrator {

BDFIntegrator::BDFIntegrator(
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> params )
    : AMP::TimeIntegrator::ImplicitIntegrator( params )
{
    d_initialized = false;
    d_object_name = "BDFIntegrator";
    auto parameters =
        std::dynamic_pointer_cast<AMP::TimeIntegrator::TimeIntegratorParameters>( params );
    AMP_ASSERT( parameters );

    auto input_db = params->d_db;

    // read parameters from input
    getFromInput( input_db );

    initialize();
    d_initialized = true;
}

BDFIntegrator::~BDFIntegrator() {}

std::unique_ptr<AMP::TimeIntegrator::TimeIntegrator> BDFIntegrator::createTimeIntegrator(
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
{
    return std::unique_ptr<AMP::TimeIntegrator::TimeIntegrator>( new BDFIntegrator( parameters ) );
}

void BDFIntegrator::integratorSpecificInitialize( void )
{
    // used in keeping track of which integrator we are using
    d_integrator_names = { d_bdf_starting_integrator, "BDF2", "BDF3", "BDF4", "BDF5", "BDF6" };
    // used in computing the truncation error
    d_integrator_order = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    if ( d_bdf_starting_integrator == "BE" ) {
        d_integrator_order[0] = 1.0;
    } else if ( d_bdf_starting_integrator == "CN" ) {
        d_integrator_order[0] = 2.0;
    } else {
        AMP_ERROR( "Unknown integrator" );
    }

    // calculate what the max integrator index is based on the user input
    d_max_integrator_index = 0;
    for ( auto i = 0u; i < d_integrator_names.size(); ++i ) {
        if ( d_integrator_names[i] == d_implicit_integrator ) {
            d_max_integrator_index = i;
            break;
        } else {
            d_max_integrator_index++;
        }
    }

    AMP_ASSERT( d_max_integrator_index < d_integrator_names.size() );

    std::vector<std::string> time_level_names = { "v(n)",   "v(n-1)", "v(n-2)", "v(n-3)",
                                                  "v(n-4)", "v(n-5)", "v(n-6)" };
    // setup the vectors for the time integrator that are important
    // to keep track of during regridding
    for ( auto i = 0u; i <= d_max_integrator_index + 1; ++i ) {
        d_vector_names.push_back( time_level_names[i] );
    }

    d_vector_names.push_back( "BDFIntegratorSourceTerm" );

    if ( d_use_predictor ) {
        d_vector_names.push_back( "Predictor" );
        d_vector_names.push_back( "TimeDerivative" );
        d_vector_names.push_back( "Current Function Vector" );
        if ( d_predictor_type == "ab2" ) {
            d_vector_names.push_back( "Old Time Derivative Vector" );
        }
    }
    d_vector_names.push_back( "Scratch Function Vector" );
    if ( d_implicit_integrator != "BE" ) {
        d_vector_names.push_back( "Old Function Vector" );
    }

    std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>> vecs;

    for ( size_t i = 0u; i < d_vector_names.size(); ++i ) {
        auto vec = d_solution_vector->clone();
        vec->zero();
        vecs.push_back( vec );
    }

    AMP_ASSERT( vecs.size() == d_vector_names.size() );

    d_prev_solutions.resize( d_max_integrator_index + 2 );
    d_integrator_steps.resize( d_max_integrator_index + 2 );

    for ( auto i = 0u; i <= d_max_integrator_index + 1; ++i ) {
        d_prev_solutions[i] = vecs[i];
    }

    size_t i = d_max_integrator_index + 2;

    d_integrator_source_vector = vecs[i];
    i++;

    if ( d_use_predictor ) {
        d_predictor_vector = vecs[i];
        i++;
        d_timederivative_vector = vecs[i];
        i++;
        d_current_function_vector = vecs[i];
        i++;
        if ( d_predictor_type == "ab2" ) {
            d_old_td_vector = vecs[i];
            i++;
        }
    }

    d_scratch_function_vector = vecs[i];
    i++;

    if ( d_implicit_integrator != "BE" ) {
        d_prev_function_vector = vecs[i];
    }

    // copy the initial conditions into the previous solution vector
    // NOTE: previously only copying one prev sol -- checkk!!!
    if ( d_ic_vector != nullptr ) {
        for ( auto i = 0u; i <= d_max_integrator_index + 1; ++i ) {
            d_prev_solutions[i]->copyVector( d_ic_vector );
        }
    }

    d_scratch_vector = d_solution_vector->clone();
    d_scratch_vector->zero();

    auto params = std::make_shared<AMP::Solver::SolverStrategyParameters>();
    params->d_vectors.push_back( d_solution_vector );
}

void BDFIntegrator::getFromInput( std::shared_ptr<AMP::Database> db )
{
    if ( db ) {
        if ( !d_initialized )
            d_first_initial_dt = db->getWithDefault<double>( "first_initial_dt", d_initial_dt );

        if ( db->keyExists( "variable_names" ) ) {
            d_var_names = db->getVector<std::string>( "variable_names" );
        }

        if ( db->keyExists( "implicit_integrator" ) ) {

            d_implicit_integrator = db->getString( "implicit_integrator" );
            if ( ( d_implicit_integrator == "BDF1" ) ||
                 ( d_implicit_integrator == "Backward Euler" ) )
                d_implicit_integrator = "BE";

        } else {
            AMP_ERROR( d_object_name + " -- Key data `implicit_integrator' missing in input." );
        }

        if ( d_implicit_integrator == "BE" ) {
            // start with BE itself
            d_bdf_starting_integrator = "BE";
        } else {
            // when BDF2 starts up another one step time integration scheme is required for the
            // first step the options are BE and CN
            d_bdf_starting_integrator =
                db->getWithDefault<std::string>( "bdf_starting_integrator", "CN" );
        }

        if ( ( d_bdf_starting_integrator != "BE" ) && ( d_bdf_starting_integrator != "CN" ) ) {
            AMP_ERROR( d_object_name +
                       " -- Key data `d_bdf_starting_integrator' valid values are BE and CN" );
        }

        if ( db->keyExists( "timestep_selection_strategy" ) ) {
            d_timestep_strategy = db->getString( "timestep_selection_strategy" );
        } else {
            AMP_ERROR( d_object_name +
                       " -- Key data `timestep_selection_strategy' missing in input." );
        }

        d_use_predictor = db->getWithDefault<bool>( "use_predictor", true );

        d_use_initial_predictor =
            db->getWithDefault<bool>( "use_initial_predictor", d_use_predictor );

        if ( d_use_predictor ) {
            d_predictor_type = db->getWithDefault<std::string>( "predictor_type", "" );
            setPredictorType( d_predictor_type );
            // override and set to true if we are using the predictor
            //            d_calculateTimeTruncError = true;
        }

        d_time_atol = db->getWithDefault<double>( "truncation_error_rtol", 1e-09 );
        d_time_rtol = db->getWithDefault<double>( "truncation_error_atol", 1e-06 );

        d_auto_component_scaling = db->getWithDefault<bool>( "auto_component_scaling", true );

        if ( d_timestep_strategy != "constant" ) {


            if ( d_timestep_strategy == "truncationErrorStrategy" ) {

                d_use_bdf1_estimator_on_regrid =
                    db->getWithDefault<bool>( "use_bdf1_estimator_on_regrid", true );

                d_enable_picontrol_regrid_steps =
                    db->getWithDefault<int>( "enable_picontrol_regrid_steps", 3 );

                d_bdf1_eps_regrid_steps = db->getWithDefault<int>( "bdf1_eps_regrid_steps", 10 );

                d_combine_timestep_estimators =
                    db->getWithDefault<bool>( "combine_timestep_estimators", false );
                d_control_timestep_variation =
                    db->getWithDefault<bool>( "control_timestep_variation", false );

                // these bounds are based on the paper by Emmrich, 2008 for nonlinear evolution
                // equations //
                d_DtCutLowerBound = db->getWithDefault<double>( "dt_cut_lower_bound", 0.58754407 );
                d_DtGrowthUpperBound = db->getWithDefault<double>( "dt_growth_upper_bound", 1.702 );

                d_use_pi_controller = db->getWithDefault<bool>( "use_pi_controller", true );

                if ( d_use_pi_controller ) {
                    d_pi_controller_type =
                        db->getWithDefault<std::string>( "pi_controller_type", "PC.4.7" );
                }

                // override and set to true if we are using the truncation error strategy
                d_calculateTimeTruncError = true;
                d_use_predictor           = true;
            }
            if ( d_timestep_strategy == "final constant" ) {

                // these bounds are based on the paper by Emmrich, 2008 for nonlinear evolution
                // equations //
                //            d_DtCutLowerBound    = db->getWithDefault<double>(
                //            "dt_cut_lower_bound", 0.58754407 );
                d_DtCutLowerBound =
                    db->getWithDefault<double>( "dt_cut_lower_bound", 0.9 ); // to match old for now
                d_DtGrowthUpperBound = db->getWithDefault<double>( "dt_growth_upper_bound", 1.702 );
                if ( !d_initialized )
                    d_final_constant_timestep_current_step =
                        db->getWithDefault<int>( "final_constant_timestep_current_step", 1 );
            }

            if ( !d_calculateTimeTruncError ) {
                // keep the next line above the choice of truncation error strategy so overriding
                // can happen if the truncation error strategy is chosen
                d_calculateTimeTruncError =
                    db->getWithDefault<bool>( "calculate_time_trunc_error", false );
            }
        }

        if ( d_timestep_strategy == "final constant" ) {
            d_number_of_time_intervals = db->getWithDefault<int>( "number_of_time_intervals", 100 );
            d_number_initial_fixed_steps =
                db->getWithDefault<int>( "number_initial_fixed_steps", 0 );
        }

        // keep this towards the end so if the d_calculateTimeTruncError has to be overridden it is
        // done
        if ( d_calculateTimeTruncError ) {
            // we have to use a predictor if we are going to calculate truncation errors
            d_use_predictor = true;
            d_timeTruncationErrorNormType =
                db->getWithDefault<std::string>( "timeTruncationErrorNormType", "maxNorm" );
            d_time_error_scaling =
                db->getWithDefault<std::string>( "time_error_scaling", "fixed_scaling" );
            if ( d_time_error_scaling == "fixed_scaling" ) {
                if ( db->keyExists( "problem_fixed_scaling" ) ) {
                    d_problem_scales = db->getVector<double>( "problem_fixed_scaling" );
                } else {
                    AMP_ERROR( d_object_name +
                               " -- Key data `problem_fixed_scaling' missing in input." );
                }
            }
        }
    }
}

void BDFIntegrator::setPredictorType( const std::string &predictor )
{
    d_predictor_type = predictor;
    // set defaults if necessary
    if ( d_predictor_type == "" ) {
        if ( ( d_implicit_integrator == "BDF2" ) || ( d_implicit_integrator == "BDF3" ) ||
             ( d_implicit_integrator == "BDF4" ) || ( d_implicit_integrator == "BDF5" ) ||
             ( d_implicit_integrator == "BDF6" ) ) {
            // this is the only predictor for these at present
            d_predictor_type = "leapfrog";
        } else if ( d_implicit_integrator == "CN" ) {
            d_predictor_type = "ab2";
        } else if ( d_implicit_integrator == "BE" ) {
            d_predictor_type = "forward_euler";
        } else {
            d_predictor_type = "previous_solution";
        }
    }

    d_initial_predictor_type = "forward_euler";
}

double BDFIntegrator::getGamma( void )
{
    if ( !d_time_history_initialized ) {
        setTimeHistoryScalings();
        d_time_history_initialized = true;
    }
    return d_gamma;
}

std::vector<double> BDFIntegrator::getTimeHistoryScalings( void )
{
    if ( !d_time_history_initialized ) {
        setTimeHistoryScalings();
        d_time_history_initialized = true;
    }
    return d_a;
}

void BDFIntegrator::setTimeHistoryScalings( void )
{
    const auto &current_integrator = d_integrator_names[d_integrator_index];

    std::vector<double> h( d_integrator_index + 1 );

    // the h vector stores the scalar coefficients that will be used in
    // calculating the time
    // derivative
    h[0] = d_current_dt;
    for ( size_t i = 1; i <= d_integrator_index; ++i ) {
        h[i] = h[i - 1] + d_integrator_steps[i - 1];
    }

    if ( current_integrator == "BE" ) {

        AMP_ASSERT( d_integrator_index == 0 );
        d_a.resize( 1 );
        d_gamma = d_current_dt;
        d_a[0]  = 1.0;

    } else if ( current_integrator == "CN" ) {

        AMP_ASSERT( d_integrator_index == 0 );
        const double eps   = 0.0;
        const double alpha = 0.5 * d_current_dt + eps;
        //        const double beta  = 0.5 * d_current_dt - eps;

        d_a.resize( 1 );
        d_gamma = alpha;
        d_a[0]  = 1.0;

    } else if ( current_integrator == "BDF2" ) {

        AMP_ASSERT( d_integrator_index == 1 );

        const double h1    = h[0];
        const double h2    = h[1];
        const double alpha = h1 + h2;
        const double a0    = ( h1 * h2 ) / alpha;
        const double a1    = ( h2 * h2 ) / ( alpha * ( h2 - h1 ) );
        const double a2    = ( h1 * h1 ) / ( alpha * ( h1 - h2 ) );

        d_a.resize( 2 );

        d_gamma = a0;
        d_a[0]  = a1;
        d_a[1]  = a2;

    } else if ( current_integrator == "BDF3" ) {

        AMP_ASSERT( d_integrator_index == 2 );

        const double h1    = h[0];
        const double h2    = h[1];
        const double h3    = h[2];
        const double n1    = h2 * h3;
        const double n2    = h3 * h1;
        const double n3    = h1 * h2;
        const double alpha = n1 + n2 + n3;
        const double a0    = ( h1 * h2 * h3 ) / alpha;
        const double a1    = ( n1 * n1 ) / ( alpha * ( h2 - h1 ) * ( h3 - h1 ) );
        const double a2    = ( n2 * n2 ) / ( alpha * ( h1 - h2 ) * ( h3 - h2 ) );
        const double a3    = ( n3 * n3 ) / ( alpha * ( h1 - h3 ) * ( h2 - h3 ) );

        d_a.resize( 3 );

        d_gamma = a0;
        d_a[0]  = a1;
        d_a[1]  = a2;
        d_a[2]  = a3;

    } else if ( current_integrator == "BDF4" ) {

        AMP_ASSERT( d_integrator_index == 3 );

        const double h1    = h[0];
        const double h2    = h[1];
        const double h3    = h[2];
        const double h4    = h[3];
        const double n1    = h2 * h3 * h4;
        const double n2    = h3 * h4 * h1;
        const double n3    = h4 * h1 * h2;
        const double n4    = h1 * h2 * h3;
        const double alpha = n1 + n2 + n3 + n4;

        const double a0 = ( h1 * h2 * h3 * h4 ) / alpha;
        const double a1 = ( n1 * n1 ) / ( alpha * ( h2 - h1 ) * ( h3 - h1 ) * ( h4 - h1 ) );
        const double a2 = ( n2 * n2 ) / ( alpha * ( h1 - h2 ) * ( h3 - h2 ) * ( h4 - h2 ) );
        const double a3 = ( n3 * n3 ) / ( alpha * ( h1 - h3 ) * ( h2 - h3 ) * ( h4 - h3 ) );
        const double a4 = ( n4 * n4 ) / ( alpha * ( h1 - h4 ) * ( h2 - h4 ) * ( h3 - h4 ) );

        d_a.resize( 4 );

        d_gamma = a0;
        d_a[0]  = a1;
        d_a[1]  = a2;
        d_a[2]  = a3;
        d_a[3]  = a4;

    } else if ( current_integrator == "BDF5" ) {

        AMP_ASSERT( d_integrator_index == 4 );

        const double h1    = h[0];
        const double h2    = h[1];
        const double h3    = h[2];
        const double h4    = h[3];
        const double h5    = h[4];
        const double n1    = h2 * h3 * h4 * h5;
        const double n2    = h3 * h4 * h5 * h1;
        const double n3    = h4 * h5 * h1 * h2;
        const double n4    = h5 * h1 * h2 * h3;
        const double n5    = h1 * h2 * h3 * h4;
        const double alpha = n1 + n2 + n3 + n4 + n5;

        const double a0 = ( h1 * h2 * h3 * h4 * h5 ) / alpha;
        const double a1 =
            ( n1 * n1 ) / ( alpha * ( h2 - h1 ) * ( h3 - h1 ) * ( h4 - h1 ) * ( h5 - h1 ) );
        const double a2 =
            ( n2 * n2 ) / ( alpha * ( h1 - h2 ) * ( h3 - h2 ) * ( h4 - h2 ) * ( h5 - h2 ) );
        const double a3 =
            ( n3 * n3 ) / ( alpha * ( h1 - h3 ) * ( h2 - h3 ) * ( h4 - h3 ) * ( h5 - h3 ) );
        const double a4 =
            ( n4 * n4 ) / ( alpha * ( h1 - h4 ) * ( h2 - h4 ) * ( h3 - h4 ) * ( h5 - h4 ) );
        const double a5 =
            ( n5 * n5 ) / ( alpha * ( h1 - h5 ) * ( h2 - h5 ) * ( h3 - h5 ) * ( h4 - h5 ) );

        d_a.resize( 5 );

        d_gamma = a0;
        d_a[0]  = a1;
        d_a[1]  = a2;
        d_a[2]  = a3;
        d_a[3]  = a4;
        d_a[4]  = a5;

    } else if ( current_integrator == "BDF6" ) {

        AMP_ASSERT( d_integrator_index == 5 );

        const double h1 = h[0];
        const double h2 = h[1];
        const double h3 = h[2];
        const double h4 = h[3];
        const double h5 = h[4];
        const double h6 = h[5];

        const double n1    = h2 * h3 * h4 * h5 * h6;
        const double n2    = h3 * h4 * h5 * h6 * h1;
        const double n3    = h4 * h5 * h6 * h1 * h2;
        const double n4    = h5 * h6 * h1 * h2 * h3;
        const double n5    = h6 * h1 * h2 * h3 * h4;
        const double n6    = h1 * h2 * h3 * h4 * h5;
        const double alpha = n1 + n2 + n3 + n4 + n5 + n6;

        const double a0 = ( h1 * h2 * h3 * h4 * h5 * h6 ) / alpha;
        const double a1 = ( n1 * n1 ) / ( alpha * ( h2 - h1 ) * ( h3 - h1 ) * ( h4 - h1 ) *
                                          ( h5 - h1 ) * ( h6 - h1 ) );
        const double a2 = ( n2 * n2 ) / ( alpha * ( h1 - h2 ) * ( h3 - h2 ) * ( h4 - h2 ) *
                                          ( h5 - h2 ) * ( h6 - h2 ) );
        const double a3 = ( n3 * n3 ) / ( alpha * ( h1 - h3 ) * ( h2 - h3 ) * ( h4 - h3 ) *
                                          ( h5 - h3 ) * ( h6 - h3 ) );
        const double a4 = ( n4 * n4 ) / ( alpha * ( h1 - h4 ) * ( h2 - h4 ) * ( h3 - h4 ) *
                                          ( h5 - h4 ) * ( h6 - h4 ) );
        const double a5 = ( n5 * n5 ) / ( alpha * ( h1 - h5 ) * ( h2 - h5 ) * ( h3 - h5 ) *
                                          ( h4 - h5 ) * ( h6 - h5 ) );
        const double a6 = ( n6 * n6 ) / ( alpha * ( h1 - h6 ) * ( h2 - h6 ) * ( h3 - h6 ) *
                                          ( h4 - h6 ) * ( h5 - h6 ) );

        d_a.resize( 6 );

        d_gamma = a0;
        d_a[0]  = a1;
        d_a[1]  = a2;
        d_a[2]  = a3;
        d_a[3]  = a4;
        d_a[4]  = a5;
        d_a[5]  = a6;

    } else {
        AMP_ERROR( "Only BE, BDF2-6, and CN integrators implemented" );
    }

    auto timeOperator = std::dynamic_pointer_cast<AMP::TimeIntegrator::TimeOperator>( d_operator );
    if ( timeOperator ) {
        timeOperator->setTimeOperatorScaling( d_gamma );
    } else {
        if ( d_fTimeScalingFnPtr )
            d_fTimeScalingFnPtr( d_gamma );
    }
}

void BDFIntegrator::computeIntegratorSourceTerm( void )
{
    auto f = d_integrator_source_vector;

    f->zero();

    // note that we have to subtract since the signs are the opposite of what should be there!!
    // should be fixed in future!!
    for ( size_t i = 0; i <= d_integrator_index; ++i ) {
        f->axpy( -d_a[i], *d_prev_solutions[i], *f );
    }

    const auto &current_integrator = d_integrator_names[d_integrator_index];

    // add in a time dependent source, g, for problems of the form u_t = f(u)+g
    // or for MGRIT it adds in the FAS correction
    if ( d_pSourceTerm ) {
        const auto alpha = ( current_integrator == "CN" ) ? -2.0 * d_gamma : -d_gamma;
        f->axpy( alpha, *d_pSourceTerm, *f );
    }

    if ( current_integrator == "CN" ) {

        AMP_ASSERT( d_integrator_index == 0 );
        const double eps = 0.0;
        //        const double alpha = 0.5 * d_current_dt + eps;
        const double beta = 0.5 * d_current_dt - eps;

        // set the source term to -( u^{n}+(dt/2-\eps)f(u^n) )
        f->axpy( -beta, *d_prev_function_vector, *f );
    }
}

void BDFIntegrator::printVectorComponentNorms(
    const std::shared_ptr<AMP::LinearAlgebra::Vector> &vec,
    const std::string &prefix,
    const std::string &postfix,
    const std::string &norm )
{
    const size_t nComponents = vec->getNumberOfComponents();

    for ( size_t i = 0u; i < nComponents; ++i ) {

        // subset for each component (cheap)
        auto component_vector = vec->subsetVectorForComponent( i );
        double norm_val       = 0.0;
        if ( norm == "L2Norm" ) {
            norm_val = static_cast<double>( component_vector->L2Norm() );
        } else if ( norm == "maxNorm" ) {
            norm_val = static_cast<double>( component_vector->maxNorm() );
        } else {
            AMP_ERROR( "Unknown known type" );
        }

        std::string var_name =
            ( d_var_names.size() == (unsigned int) nComponents ) ? d_var_names[i] : "var";
        AMP::pout << std::setprecision( 16 ) << norm << prefix << var_name << postfix << norm_val
                  << std::endl;
    }
}

/*
*************************************************************************
*                                                                       *
* Return the next time increment through which to advance the solution. *
*                                                                       *
*************************************************************************
*/

double BDFIntegrator::getNextDtTruncationError( const bool good_solution, const int solver_retcode )
{
    PROFILE( "getNextDt-truncation", 1 );
    if ( good_solution ) {
        PROFILE( "getNextDt-truncation-good", 2 );

        d_current_dt = estimateDtWithTruncationErrorEstimates( d_current_dt, good_solution );

        if ( d_max_dt < d_current_dt ) {
            d_current_dt = d_max_dt;
        } else {
            if ( d_iDebugPrintInfoLevel > 0 )
                AMP::pout << "Truncation error limited timestep" << std::endl;
        }

    } else {
        PROFILE( "getNextDt-truncation-bad", 2 );
        // the rejection could be due to the truncation error being too large or
        // a solution step failing

        if ( solver_retcode == 0 ) {
            // solution method failure, decrease step by max allowed
            // both the cases of truncation error and efficiency are overridden here
            if ( d_iDebugPrintInfoLevel > 0 ) {
                AMP::pout << std::setprecision( 16 )
                          << "The solution process failed. Timestep is being decreased by "
                             "max allowable factor:: "
                          << d_DtCutLowerBound << std::endl;
            }
            d_current_dt = d_DtCutLowerBound * d_current_dt;
        } else {
            // truncation error is too high
            d_current_dt = estimateDtWithTruncationErrorEstimates( d_current_dt, good_solution );
        }
    }

    // check to make sure the predictor vector is a valid vector,
    // if not scale back
    //   AMP::pout << "Check in getNextDt to see if predictor is valid" << std::endl;
    evaluatePredictor();
    //   AMP::pout << "Finished call to predictor in getNextDt" << std::endl;

    bool validVector = d_operator->isValidVector( d_predictor_vector );

    if ( !validVector ) {
        PROFILE( "getNextDt-truncation-not_valid", 2 );
        AMP::pout << "The predictor is not valid" << std::endl;
        int iNumberOfPredictorPreCheckAttempts = 10;

        for ( int i = 0; i < iNumberOfPredictorPreCheckAttempts; i++ ) {
            AMP::pout << "WARNING:: Attempting to scale predictor, attempt number " << i
                      << std::endl;
            // the vector is scaled to be 50% of the previous vector
            // NOTE: that each time it is indeed 50% of the previous vector, scaled or unscaled
            d_current_dt = 0.5 * d_current_dt;

            evaluatePredictor();

            validVector = d_operator->isValidVector( d_predictor_vector );

            if ( validVector ) {
                AMP::pout << "Predictor was successful in scaling back after " << i << " attempts"
                          << std::endl;
                break;
            } else {
                AMP::pout << "WARNING: predictor was not successful in scaling back after " << i
                          << " attempts" << std::endl;
            }
        }
    }
    return d_current_dt;
}

double BDFIntegrator::getNextDtConstant( const bool, const int ) { return d_initial_dt; }

double BDFIntegrator::getNextDtPredefined( const bool good_solution, const int solver_retcode )
{
    return getNextDtConstant( good_solution, solver_retcode );
}

double BDFIntegrator::getNextDtFinalConstant( const bool, const int )
{
    if ( d_final_constant_timestep_current_step <= d_number_initial_fixed_steps ) {
        d_current_dt = d_initial_dt;
        ++d_final_constant_timestep_current_step;
    } else if ( d_final_constant_timestep_current_step <
                d_number_of_time_intervals + d_number_initial_fixed_steps ) {
        d_current_dt =
            d_first_initial_dt +
            ( (double) d_final_constant_timestep_current_step - d_number_initial_fixed_steps ) /
                ( (double) d_number_of_time_intervals ) * ( d_max_dt - d_first_initial_dt );
        ++d_final_constant_timestep_current_step;
    } else {
        d_current_dt = d_max_dt;
    }
    return d_current_dt;
}

double BDFIntegrator::integratorSpecificGetNextDt( const bool good_solution,
                                                   const int solver_retcode )
{
    PROFILE( "getNextDt" );
    // store the current dt somewhere as this has to be transferred to d_old_dt after the various
    // estimators are done
    const double d_tmp_dt = d_current_dt;

    if ( d_timestep_strategy == "truncationErrorStrategy" ) {
        d_current_dt = getNextDtTruncationError( good_solution, solver_retcode );
    } else {
        PROFILE( "getNextDt-default", 1 );
        if ( good_solution ) {
            static bool dtLimitedForCheckPoint;
            static double dtBeforeCheckPoint;
            if ( d_timestep_strategy == "predefined" ) {
                d_current_dt = getNextDtPredefined( good_solution, solver_retcode );
            } else {
                if ( dtLimitedForCheckPoint ) {
                    d_current_dt = dtBeforeCheckPoint;
                } else if ( d_timestep_strategy == "constant" ) {
                    d_current_dt = getNextDtConstant( good_solution, solver_retcode );
                } else if ( d_timestep_strategy == "final constant" ) {
                    d_current_dt = getNextDtFinalConstant( good_solution, solver_retcode );
                } else {
                    AMP_ERROR( "Unknown time step control strategy selected:  " +
                               d_timestep_strategy );
                }
            }

        } else {

            if ( solver_retcode == 0 ) {
                // solution method failure, decrease step by d_DtCutLowerBound*d_current_dt
                if ( d_iDebugPrintInfoLevel > 0 ) {
                    AMP::pout << std::setprecision( 16 )
                              << "The implicit solver failed. Timestep is being decreased by "
                                 "max allowable factor:: "
                              << d_DtCutLowerBound << std::endl;
                }
            }
            d_current_dt = d_DtCutLowerBound * d_tmp_dt;
        }
    }

    // now set the old dt once it has been used
    if ( good_solution ) {
        d_old_dt = d_tmp_dt;
    }

    d_current_dt = std::min( std::min( d_current_dt, d_max_dt ), d_final_time - d_current_time );

    return ( d_current_dt );
}

void BDFIntegrator::evaluatePredictor()
{
    PROFILE( "evaluatePredictor" );
    if ( ( d_implicit_integrator == "BDF2" ) || ( d_implicit_integrator == "BDF3" ) ||
         ( d_implicit_integrator == "BDF4" ) || ( d_implicit_integrator == "BDF5" ) ||
         ( d_implicit_integrator == "BDF6" ) ) {
        // the commented out version can be used if using BE at the first step after regrid
        //      if(d_first_step || ((!d_is_after_regrid)&& (d_timesteps_after_regrid<2)))
        if ( d_first_step ) {
            AMP_INSIST( ( d_bdf_starting_integrator == "CN" ) ||
                            ( d_bdf_starting_integrator == "BE" ),
                        "ERROR: Valid options for d_bdf_starting_integrator are CN or BE" );
            // use forward Euler as a predictor for both BE and CN for now
            evaluateForwardEulerPredictor();
        } else {
            if ( d_predictor_type == "leapfrog" ) {
                evaluateLeapFrogPredictor();
            } else if ( d_predictor_type == "bdf_interpolant" ) {
                evaluateBDFInterpolantPredictor();
            } else {
                AMP_ERROR( "ERROR: Valid options for BDF predictor are leapfrog  and  "
                           "bdf_interpolant currently" );
            }
        }
    } else if ( d_implicit_integrator == "CN" ) {
        AMP_INSIST( d_predictor_type == "ab2",
                    "ERROR: Valid option for Crank-Nicolson predictor is only ab2 currently" );
        if ( d_first_step )
            evaluateForwardEulerPredictor();
        else
            evaluateAB2Predictor();
    } else if ( d_implicit_integrator == "BE" ) {
        evaluateForwardEulerPredictor();
    } else {
        AMP_ERROR( "ERROR: Valid option for time integrator are BDF2-6, CN, or BE currently" );
    }

    if ( !d_operator->isValidVector( d_predictor_vector ) ) {
        // do a constant extrapolation in time for the
        // predictor
        d_predictor_vector->copyVector( d_prev_solutions[0] );
    }
}

void BDFIntegrator::evaluateAB2Predictor()
{
    PROFILE( "evaluateAB2Predictor" );

    const double dt_ratio = d_current_dt / d_old_dt;
    const double alpha    = d_current_dt * ( 2.0 + dt_ratio ) / 2.0;
    const double beta     = -d_current_dt * dt_ratio / 2.0;

    d_predictor_vector->linearSum( alpha, *d_current_function_vector, beta, *d_old_td_vector );
    d_predictor_vector->add( *d_predictor_vector, *d_prev_solutions[0] );
}

// the leapfrog estimator is based on Gresho and Sani
void BDFIntegrator::evaluateLeapFrogPredictor()
{
    PROFILE( "evaluateLeapFrogPredictor" );

    if ( d_iDebugPrintInfoLevel > 4 ) {
        printVectorComponentNorms( d_prev_solutions[1], " of old ", ": ", "L2Norm" );
    }

    const double dt_ratio = d_current_dt / d_old_dt;
    // these are the values for BDF2
    double alpha = dt_ratio * dt_ratio;
    double beta  = 1.0 - alpha;
    double gamma = ( 1.0 + dt_ratio ) * d_current_dt;

    if ( d_iDebugPrintInfoLevel > 4 ) {
        printVectorComponentNorms(
            d_timederivative_vector, " of time derivative of ", ": ", "L2Norm" );
        AMP::pout << std::setprecision( 16 ) << "LeapFrog Predictor: alpha " << alpha << ", beta "
                  << beta << ", gamma " << gamma << std::endl;
    }

    d_predictor_vector->linearSum( alpha, *d_prev_solutions[1], beta, *d_prev_solutions[0] );
    d_predictor_vector->axpy( gamma, *d_timederivative_vector, *d_predictor_vector );

    if ( d_iDebugPrintInfoLevel > 4 ) {
        printVectorComponentNorms( d_predictor_vector, " of ", " predictor: ", "L2Norm" );
    }
}

void BDFIntegrator::evaluateBDFInterpolantPredictor() { AMP_ERROR( "Not implemented" ); }

// the leapfrog estimator is based on Gresho and Sani
void BDFIntegrator::evaluateForwardEulerPredictor()
{
    PROFILE( "evaluateForwardEulerPredictor" );

    if ( d_iDebugPrintInfoLevel > 4 ) {
        printVectorComponentNorms( d_prev_solutions[0], " of old ", ": ", "L2Norm" );
    }

    // these are the values for BDF2
    double alpha = 0.0;
    double beta  = 1.0 - alpha;
    double gamma = d_current_dt;

    // at the very first step use forward Euler as a predictor. For this we
    // need the time derivative at the initial time. While in general not a
    // good idea we use the function evaluation to approximate the time derivative
    if ( d_first_step ) {
        d_scratch_vector->copyVector( d_prev_solutions[0] );
        auto timeOperator =
            std::dynamic_pointer_cast<AMP::TimeIntegrator::TimeOperator>( d_operator );
        if ( timeOperator ) {
            timeOperator->applyRhs( d_scratch_vector, d_current_function_vector );
        } else {
            // it is assumed the user supplied operator is calculating (u^{n+1}-\gamma
            // f(u^{n+1}) so we do the necessary operations to back calculate f(u^{n+1})
            d_operator->apply( d_scratch_vector, d_current_function_vector );
            d_current_function_vector->subtract( *d_current_function_vector, *d_scratch_vector );
            d_current_function_vector->scale( -1.0 / d_gamma );
        }

        d_timederivative_vector->copyVector( d_current_function_vector );

        // modify alpha, beta, gamma for forward Euler
        alpha = 0.0;
        beta  = 1.0 - alpha;
        gamma = d_current_dt;
    }

    if ( d_iDebugPrintInfoLevel > 4 ) {
        printVectorComponentNorms(
            d_timederivative_vector, " of time derivative of ", ": ", "L2Norm" );
        AMP::pout << std::setprecision( 16 ) << "Forward Euler Predictor: alpha " << alpha
                  << ", beta " << beta << ", gamma " << gamma << std::endl;
    }

    d_predictor_vector->copyVector( d_prev_solutions[0] );
    d_predictor_vector->axpy( gamma, *d_timederivative_vector, *d_predictor_vector );

    if ( d_iDebugPrintInfoLevel > 4 ) {
        printVectorComponentNorms( d_predictor_vector, " of ", "predictor: ", "L2Norm" );
    }
}

/*
*************************************************************************
*                                                                       *
* Set the initial guess for the time advanced solution at the start     *
* of the nonlinear iteration.                                           *
*                                                                       *
*************************************************************************
*/
void BDFIntegrator::setInitialGuess( const bool first_step,
                                     const double current_time,
                                     const double,
                                     const double )
{
    PROFILE( "setInitialGuess" );
    (void) current_time;

    d_first_step = first_step;

    if ( !d_is_after_regrid ) {

        const auto &current_integrator = d_integrator_names[d_integrator_index];

        // we compute f(u_{n-1}) and store it for the timestep here
        if ( current_integrator == "CN" ) {
            AMP_ASSERT( d_prev_solutions[0] );
            AMP_ASSERT( d_prev_function_vector );
            d_scratch_vector->copyVector( d_prev_solutions[0] );
            d_scratch_vector->makeConsistent();

            auto timeOperator =
                std::dynamic_pointer_cast<AMP::TimeIntegrator::TimeOperator>( d_operator );
            if ( timeOperator ) {
                timeOperator->applyRhs( d_scratch_vector, d_prev_function_vector );
            } else {
                // it is assumed the user supplied operator is calculating (u^{n+1}-\gamma
                // f(u^{n+1}) so we do the necessary operations to back calculate f(u^{n+1})
                d_operator->apply( d_scratch_vector, d_prev_function_vector );
                d_prev_function_vector->subtract( *d_prev_function_vector, *d_scratch_vector );
                d_prev_function_vector->scale( -1.0 / d_gamma );
            }
            //	    AMP::pout << "applyRhs: L2Norm " << d_prev_function_vector->L2Norm() <<
            // std::endl;
        }

        if ( d_use_predictor ) {
            if ( d_use_constant_time_interpolation ||
                 ( d_first_step && ( d_use_initial_predictor == false ) ) ) {

                if ( d_iDebugPrintInfoLevel > 0 ) {
                    AMP::pout << "First step" << std::endl;
                }

                // for the very first step simply do a constant extrapolation in time for the
                // predictor
                d_predictor_vector->copyVector( d_prev_solutions[0] );

                // this flag is used so that a constant extrapolation is used for the step
                // immediately after a regrid
                if ( d_use_constant_time_interpolation )
                    d_use_constant_time_interpolation = false;

            } else {
                evaluatePredictor();
            }

            // we call reInitializeVector because it postprocesses a vector
            // to ensure it is a valid vector for E & T values. The routine
            // itself should be renamed as clearly there are multiple
            // places where it should be used
            d_operator->reInitializeVector( d_predictor_vector );

            d_solution_vector->copyVector( d_predictor_vector );

            if ( d_iDebugPrintInfoLevel > 4 ) {
                printVectorComponentNorms( d_solution_vector, " of new ", ": ", "L2Norm" );
            }

        } else {
            /*
             * Use previous timestep as initial guess, by simply copying
             * from the current context to the new context.
             */
            d_solution_vector->copyVector( d_prev_solutions[0] );
        }
        d_solution_vector->makeConsistent();
    }

    computeIntegratorSourceTerm();
}

/*
*************************************************************************
*                                                                       *
* Check the computed solution and return true if it is acceptable;      *
* otherwise return false.                                               *
*                                                                       *
*************************************************************************
*/
bool BDFIntegrator::integratorSpecificCheckNewSolution( const int solver_retcode )
{
    PROFILE( "integratorSpecificCheckNewSolution" );
    bool checkPassed = false;

    // the first check is whether the solver passed or failed
    if ( solver_retcode == 1 ) {
        if ( d_iDebugPrintInfoLevel > 4 ) {
            AMP::pout << "Solver checks: PASSED" << std::endl;
        }

        if ( d_calculateTimeTruncError ) {
            calculateTemporalTruncationError();
        }

        // the next check is whether we met the temporal truncation error criteria
        if ( d_timestep_strategy == "truncationErrorStrategy" ) {
            checkPassed = ( d_timeTruncationErrorEstimate <= 1.0 ) ? true : false;

            if ( checkPassed ) {
                d_current_stepaccepts++;

                if ( d_current_steprejects > 1 ) {
                    // set a bool to alert the time step controller there were previous
                    // successive rejects when there are successive rejects the standard
                    // deadbeat controller is used
                    d_prevSuccessiveRejects = true;
                } else {
                    d_prevSuccessiveRejects = false;
                }

                // reset the current step rejected flag
                d_current_steprejects = 0;

                if ( d_iDebugPrintInfoLevel > 4 ) {
                    AMP::pout << "Time truncation error checks: PASSED" << std::endl;
                }
            } else {
                if ( d_iDebugPrintInfoLevel > 4 ) {
                    AMP::pout << "Time truncation error checks: FAILED" << std::endl;
                }

                d_total_steprejects++;
                d_current_steprejects++;
                // reset the counter for currently accepted steps
                d_current_stepaccepts = 0;
            }
        } else {
            checkPassed = true;
        }
    } else {
        // a negative value indicates the solver did not converge
        d_solver_converged_reason = -1;
        d_total_steprejects++;
        d_current_steprejects++;
        // reset the counter for currently accepted steps
        d_current_stepaccepts = 0;
    }

    if ( d_log_statistics ) {
        int stepStatus = checkPassed ? 1 : 0;
        d_step_accepted.push_back( stepStatus );
    }

    return checkPassed == 0 ? false : true;
}

void BDFIntegrator::estimateBDF2TimeDerivative( void )
{
    PROFILE( "estimateBDF2TimeDerivative" );
    // we use the approach suggested in Gresho and Sani, Pg 805 to estimate
    // what the time derivative is
    const double dtt   = d_current_dt / d_old_dt;
    const double alpha = ( 2.0 * dtt + 1.0 ) / ( ( dtt + 1.0 ) * d_current_dt );
    const double beta  = -( dtt + 1.0 ) / d_current_dt;
    const double gamma = ( dtt * dtt ) / ( ( dtt + 1.0 ) * d_current_dt );

    d_timederivative_vector->linearSum( alpha, *d_solution_vector, beta, *d_prev_solutions[0] );
    d_timederivative_vector->axpy( gamma, *d_prev_solutions[1], *d_timederivative_vector );
}

/**
   Use the approach suggested in Gresho and Sani, Pg 267 to estimate
   what the time derivative is for CN
*/
void BDFIntegrator::estimateCNTimeDerivative( void )
{
    PROFILE( "estimateCNTimeDerivative" );
    const double alpha = -1.0;
    const double beta  = 2.0 / d_current_dt;
    const double gamma = -1.0;

    d_timederivative_vector->axpy( alpha, *d_prev_solutions[0], *d_solution_vector );
    d_timederivative_vector->linearSum(
        beta, *d_timederivative_vector, gamma, *d_prev_function_vector );
}

/**
   Use the approach suggested in Gresho and Sani, Pg 267 to estimate
   what the time derivative is for CN
*/
void BDFIntegrator::estimateBETimeDerivative( void )
{
    PROFILE( "estimateBETimeDerivative" );

    const double alpha = -1.0;
    const double beta  = 1.0 / d_current_dt;

    d_timederivative_vector->axpy( alpha, *d_prev_solutions[0], *d_solution_vector );
    d_timederivative_vector->scale( beta, *d_timederivative_vector );
}


void BDFIntegrator::estimateTimeDerivative( void )
{
    PROFILE( "estimateTimeDerivative" );

    const auto &current_integrator = d_integrator_names[d_integrator_index];

    if ( current_integrator == "BDF2" ) {
        estimateBDF2TimeDerivative();
    } else if ( ( current_integrator == "BDF3" ) || ( current_integrator == "BDF4" ) ||
                ( current_integrator == "BDF5" ) || ( current_integrator == "BDF6" ) ) {
        AMP_WARN_ONCE(
            "BDF 3-6 methods being used with low order BDF2 estimate of time derivative" );
        estimateBDF2TimeDerivative();
    } else if ( current_integrator == "CN" ) {
        estimateBETimeDerivative();
        // the CN estimate currently evaluates f(u^{n-1}) which introduces significant error
        //        estimateCNTimeDerivative();
    } else if ( current_integrator == "BE" ) {
        estimateBETimeDerivative();
    } else {
        AMP_ERROR(
            "ERROR: Unknown time time integrator, valid options are BDF2, CN or BE currently" );
    }
}

/*
*************************************************************************
*                                                                       *
* Update solution quantities after computing an acceptable time         *
* advanced solution.                                                    *
*                                                                       *
*************************************************************************
*/
void BDFIntegrator::integratorSpecificUpdateSolution( const double new_time )
{
    PROFILE( "integratorSpecificUpdateSolution" );

    d_new_time = d_current_time = new_time;

    // increment the counter for the number of timesteps after regrid
    d_timesteps_after_regrid++;

    // the swapping of the function values must come
    // before the swapping of the variables as we need the variables
    // in computing the time derivative
    if ( d_use_predictor ) {
        if ( d_predictor_type == "ab2" ) {
            d_old_td_vector->swapVectors( d_current_function_vector );
        }

        estimateTimeDerivative();
    }

    if ( ( d_implicit_integrator == "BE" ) || ( d_implicit_integrator == "CN" ) ) {

        d_prev_solutions[0]->copyVector( d_solution_vector );

    } else {

        if ( d_integrator_index < d_max_integrator_index ) {
            d_integrator_index++;
        }

        // update solutions at previous time levels by swapping
        for ( auto i = d_integrator_index + 1; i >= 1; --i ) {
            d_prev_solutions[i]->swapVectors( d_prev_solutions[i - 1] );
            d_integrator_steps[i] = d_integrator_steps[i - 1];
        }

        // update solution at current time level by copy
        d_prev_solutions[0]->copyVector( d_solution_vector );
    }

    d_integrator_steps[0] = d_current_dt;

    if ( d_iDebugPrintInfoLevel > 2 ) {
        printVectorComponentNorms( d_prev_solutions[0], " of current ", ": ", "L2Norm" );
    }

    // the fact that an updateSolution has happened implies that atleast
    // one step is completed. However, SAMRAI only updates the status
    // after the call to getNextDt. So going into getNextDt and in particular
    // into evaluatePredictor after the first step the status is incorrectly
    // signalled (from one point of view) as being the first step causing
    // problems. To avoid that we set it explicitly here
    d_first_step = false;
}

/*
*************************************************************************
*                                                                       *
* Return coefficient multiplying dt in integrator-specific nonlinear    *
* function.                                                             *
*                                                                       *
*************************************************************************
*/

double BDFIntegrator::getTimeOperatorScaling( void ) { return d_gamma; }

double BDFIntegrator::estimateDtWithTruncationErrorEstimates( double current_dt,
                                                              bool good_solution )
{
    PROFILE( "estimateDtWithTruncationErrorEstimates", 1 );
    /*
     * Compute a new time step based on truncation error estimates
     * The truncation error estimate comes from a private communication with M. Pernice
     * and is based on an AB2 predictor and BDF2 corrector
     */

    if ( ( d_implicit_integrator != "BE" ) && ( d_implicit_integrator != "BDF2" ) &&
         ( d_implicit_integrator != "BDF3" ) && ( d_implicit_integrator != "BDF4" ) &&
         ( d_implicit_integrator != "BDF5" ) && ( d_implicit_integrator != "BDF6" ) &&
         ( d_implicit_integrator != "CN" ) ) {
        AMP_ERROR( "Unknown time integrator, current implementation is for BDF1-6 and CN" );
    }

    // the truncation error estimate should already have been calculated while checking the
    // solution
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << std::setprecision( 16 )
                  << "Truncation error estimate is: " << d_timeTruncationErrorEstimate << std::endl;
        //   AMP::pout << "Cube root of truncation error estimate is: " <<
        //   std::pow(d_timeTruncationErrorEstimate,1.0/3.0) << std::endl;
    }

    double dtFactor = 0.0;

    // exponent for truncation error
    const double p = d_integrator_order[d_integrator_index];

    // When the PI controller is being used the code will force a switch to the deadbeat
    // controller if any of the following happens
    // 1. There is a failure in the nonlinear solution process
    // 2. There were previous successive rejections of the timestep
    // 3. The number of timesteps after regrid is less than the number of timesteps before PI
    // control is re-enabled after a regrid
    const auto m_eps = std::numeric_limits<double>::epsilon();

    if ( d_use_pi_controller && ( good_solution ) && ( !d_prevSuccessiveRejects ) &&
         ( d_timesteps_after_regrid >= d_enable_picontrol_regrid_steps ) ) {

        // the safety factor limits tries to pull the computed/predicted timestep a bit back
        const double safetyFactor = 0.8;
        //   const double safetyFactor     = 0.9;
        if ( d_pi_controller_type == "H211b" ) {
            const double b = 4.0;
            dtFactor =
                std::pow( safetyFactor / ( std::max( d_timeTruncationErrorEstimate, m_eps ) ),
                          1.0 / ( b * ( p + 1.0 ) ) );
            dtFactor *=
                std::pow( safetyFactor / ( std::max( d_prevTimeTruncationErrorEstimate, m_eps ) ),
                          1.0 / ( b * ( p + 1.0 ) ) );
            dtFactor *= std::pow( d_alpha, -0.25 );

        } else if ( d_pi_controller_type == "PC.4.7" ) {
            dtFactor =
                std::pow( safetyFactor / ( std::max( d_timeTruncationErrorEstimate, m_eps ) ),
                          0.4 / ( p + 1.0 ) );
            dtFactor = dtFactor * std::pow( d_timeErrorEstimateRatio, 0.7 / ( p + 1.0 ) ) * d_alpha;

        } else if ( d_pi_controller_type == "PC11" ) {
            dtFactor =
                std::pow( safetyFactor / ( std::max( d_timeTruncationErrorEstimate, m_eps ) ),
                          1.0 / ( p + 1.0 ) );
            dtFactor = dtFactor * std::pow( d_timeErrorEstimateRatio, 1.0 / ( p + 1.0 ) ) * d_alpha;

        } else if ( d_pi_controller_type == "Deadbeat" ) {
            dtFactor =
                std::pow( safetyFactor / ( std::max( d_timeTruncationErrorEstimate, m_eps ) ),
                          1.0 / ( p + 1.0 ) );
        } else {
            AMP_ERROR( "ERROR: Unknown PI controller type" );
        }

        if ( d_iDebugPrintInfoLevel > 0 ) {
            AMP::pout << "Error controller: " << d_pi_controller_type << std::endl;
        }
    } else {
        // the safety factor for the deadbeat when recovering from step size rejects
        // or immediately after a regrid should be very conservative
        // the safety factor limits tries to pull the computed/predicted timestep a bit back
        //        const double safetyFactor = 0.5;
        const double safetyFactor = 0.8;

        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "Error controller: Deadbeat" << std::endl;
        }

        if ( ( d_iDebugPrintInfoLevel > 2 ) && ( !good_solution ) ) {
            AMP::pout << "Reason: Step rejected, high truncation error" << std::endl;
        }

        if ( ( d_iDebugPrintInfoLevel > 2 ) &&
             ( d_timesteps_after_regrid < d_enable_picontrol_regrid_steps ) ) {
            AMP::pout << "Reason: " << d_timesteps_after_regrid << " step(s) after regrid"
                      << std::endl;
        }

        if ( d_prevSuccessiveRejects && ( d_iDebugPrintInfoLevel > 2 ) ) {
            AMP::pout << "Reason: Successive step rejections" << std::endl;
        }

        // commented section was being used with BE after regrid
        //       if( (d_timesteps_after_regrid <2 ) && (d_bdf_starting_integrator=="BE"))
        //     {
        //       p=1;
        //     }

        dtFactor = std::pow( safetyFactor / ( std::max( d_timeTruncationErrorEstimate, m_eps ) ),
                             1.0 / ( p + 1.0 ) );

        // when regridding has taken place and a resolve is done the potential
        // exists for a situation where the convergence rate of the solve on the
        // new mesh is excellent and the truncation error estimate is
        // very small. This leads to the error estimator trying to dramatically
        // increase the timestep which often leads to negative values of the energy
        // To prevent this situation immediately after regrid we do not allow the timestep
        // to increase
        if ( ( d_integrator_step > 1 ) &&
             ( d_timesteps_after_regrid < d_enable_picontrol_regrid_steps ) &&
             ( dtFactor > 1.0 ) ) {
            //       dtFactor = 1.1;
            dtFactor = 1.0;
        }
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << std::setprecision( 16 )
                  << "Time local error estimate: " << d_timeTruncationErrorEstimate << std::endl;
        AMP::pout << std::setprecision( 16 )
                  << "Ratio of time local errors: " << d_timeErrorEstimateRatio << std::endl;
    }

    // compute how far away from 1 we are, if it's a small fraction away from
    // 1 (in this case 0.1) don't change the timestep

    if ( d_control_timestep_variation && ( !d_use_pi_controller ) ) {
        const double dtFactorVar = fabs( 1.0 - dtFactor );

        if ( dtFactorVar < 0.1 ) {
            dtFactor = 1.0;
        }

        if ( d_iDebugPrintInfoLevel > 2 ) {
            AMP::pout << std::setprecision( 16 ) << "Current dt: " << current_dt << std::endl;
            AMP::pout << std::setprecision( 16 )
                      << "dtFactor after checking for small variations: " << dtFactor << std::endl;
        }
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << std::setprecision( 16 ) << "Final dtFactor: " << dtFactor << std::endl;
    }

    current_dt = current_dt * dtFactor;

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << std::setprecision( 16 ) << "New dt: " << current_dt << std::endl;
    }

    return current_dt;
}

double BDFIntegrator::calculateLTEScalingFactor()
{
    double errorFactor = 0.0;

    // immediately after regrid only interpolated values are available for the solution
    // at the previous two timesteps. Also only an interpolated value of the time derivative
    // is available. An option is to use a lower order time estimator such as for BDF1.
    // The formula used is from the book by Hunsdorfer
    if ( ( d_use_bdf1_estimator_on_regrid ) &&
         ( d_timesteps_after_regrid < d_bdf1_eps_regrid_steps ) && ( d_integrator_step > 3 ) ) {
        // debugging
        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "Using BDF1 estimator after regrid " << std::endl;
        }

        // immediately on regrid when we do a resolve we are still using BDF2
        // Trompert and Verwer approach
        errorFactor = 2.0; // fairly arbitrary factor, used because the error estimator is
        // coarse and typically under-estimates
        errorFactor = errorFactor * d_current_dt;
    } else {
        if ( d_first_step ) {
            if ( d_bdf_starting_integrator == "BE" ) {
                //          errorFactor = 3.0;
                errorFactor = 0.5;
            } else if ( d_bdf_starting_integrator == "CN" ) {
                // immediately on regrid when we do a resolve we are still using BDF2
                // Trompert and Verwer approach
                errorFactor = 1.5; // fairly arbitrary factor, used because the error estimator
                // is coarse and typically under-estimates
                errorFactor = errorFactor * d_current_dt;
                //          AMP_ERROR("ERROR: CN predictor not implemented for BDF2 starting");
            } else {
                AMP_ERROR( "ERROR: Unknown BDF2 starting predictor" );
            }
        } else {
            // check!! the expressions given in Gresho and Sani appear to be wrong!
            if ( d_predictor_type == "leapfrog" ) {
                // there might be an error here!!
                //      errorFactor = 1.0/((1+alpha)*(1.0+std::pow(alpha/(1.0+alpha),2) ) );

                errorFactor = ( 1.0 + d_alpha ) / ( 2.0 + 3.0 * d_alpha );
            } else if ( d_predictor_type == "ab2" ) {
                // compute error factor from M. Pernice communication for AB2 and BDF2
                errorFactor = 2.0 / ( 6.0 - 1.0 / ( std::pow( d_alpha + 1, 2.0 ) ) );
            } else if ( d_predictor_type == "forward_euler" ) {
                errorFactor = 0.5;
            } else {
                AMP_ERROR( "ERROR: Unknown BDF2 predictor" );
            }
        }

        if ( d_iDebugPrintInfoLevel > 2 ) {
            AMP::pout << std::setprecision( 16 ) << "Predictor type: " << d_predictor_type
                      << ",  errorFactor " << errorFactor << std::endl;
            AMP::pout << std::setprecision( 16 ) << "Alpha = dt_new/dt_old = " << d_alpha
                      << std::endl;
        }
    }

    return errorFactor;
}

void BDFIntegrator::calculateScaledLTENorm( std::shared_ptr<AMP::LinearAlgebra::Vector> x,
                                            std::shared_ptr<AMP::LinearAlgebra::Vector> y,
                                            std::vector<double> &norms )
{
    const size_t nComponents = x->getNumberOfComponents();

    if ( d_time_error_scaling == "fixed_resolution" ) {

        d_scratch_vector->scale( d_time_rtol, *x );
        d_scratch_vector->addScalar( *d_scratch_vector, d_time_atol );

    } else if ( d_time_error_scaling == "fixed_scaling" ) {

        d_scratch_vector->zero();

        for ( size_t i = 0u; i < nComponents; ++i ) {
            // subset for each component (cheap)
            auto scratch_vector = d_scratch_vector->subsetVectorForComponent( i );
            auto sol_vector     = x->subsetVectorForComponent( i );
            scratch_vector->addScalar( *sol_vector, d_problem_scales[i] );
        }

        d_scratch_vector->scale( d_time_rtol, *d_scratch_vector );

    } else {
        AMP_ERROR( "Unknown time error scaling option" );
    }

    d_scratch_function_vector->subtract( *x, *y );
    d_scratch_function_vector->divide( *d_scratch_function_vector, *d_scratch_vector );

    for ( size_t i = 0u; i < nComponents; ++i ) {

        // subset for each component (cheap)
        auto component_vector = d_scratch_function_vector->subsetVectorForComponent( i );

        // the L2 Norm tends to under-estimate the error for the AMR calculations
        // but might be useful immediately after regrid as the max norm will probably
        // over predict at that point. The code for switching between norms immediately
        // after regrid has been deleted but we might try to experiment with it later
        // BP
        if ( d_timeTruncationErrorNormType == "maxNorm" ) {
            norms[i] = static_cast<double>( component_vector->maxNorm() );
        } else {
            norms[i] = static_cast<double>( component_vector->L2Norm() );
        }
    }

    if ( d_iDebugPrintInfoLevel > 4 ) {
        AMP::pout << std::setprecision( 16 ) << "Scaled LTE Norms ";
        for ( const auto &v : norms )
            AMP::pout << "  " << v;
        AMP::pout << std::endl;
    }
}

void BDFIntegrator::calculateTemporalTruncationError()
{

    PROFILE( "calculateTemporalTruncationError" );
    const size_t nComponents = d_solution_vector->getNumberOfComponents();
    std::vector<double> truncErrorEstimate( nComponents, 1.0 );

    // We will perform the check for all timesteps. The only exception
    // is potentially for the very first time
    if ( ( d_integrator_step > 0 ) || ( d_first_step && d_use_initial_predictor ) ) {
        /*
         * Compute a new time step based on truncation error estimates
         * One of the truncation error estimate comes from a private communication with M.
         * Pernice and is based on an AB2 predictor and BDF2 corrector
         */
        if ( ( d_implicit_integrator != "BE" ) && ( d_implicit_integrator != "BDF2" ) &&
             ( d_implicit_integrator != "BDF3" ) && ( d_implicit_integrator != "BDF4" ) &&
             ( d_implicit_integrator != "BDF5" ) && ( d_implicit_integrator != "BDF6" ) &&
             ( d_implicit_integrator != "CN" ) ) {
            AMP_ERROR(
                "Unknown time integrator, current implementation is for BDF1-6, Crank-Nicolson" );
        }

        d_alpha            = d_current_dt / d_old_dt;
        double errorFactor = 0.0;

        if ( !d_scratch_vector ) {
            d_scratch_vector = d_solution_vector->clone();
        }

        // debugging
        if ( d_iDebugPrintInfoLevel > 2 ) {

            printVectorComponentNorms( d_solution_vector, " of ", " new: ", "L2Norm" );
            printVectorComponentNorms( d_prev_solutions[0], " of ", " current: ", "L2Norm" );
            printVectorComponentNorms( d_predictor_vector, " of ", " predictor: ", "L2Norm" );
        }

        errorFactor = calculateLTEScalingFactor();

        // immediately after regrid only interpolated values are available for the solution
        // at the previous two timesteps. Also only an interpolated value of the time derivative
        // is available. An option is to use a lower order time estimator such as for BDF1.
        // The formula used is from the book by Hunsdorfer
        std::vector<double> normOfError( nComponents, 0.0 );

        if ( ( d_use_bdf1_estimator_on_regrid ) &&
             ( d_timesteps_after_regrid < d_bdf1_eps_regrid_steps ) && ( d_integrator_step > 3 ) ) {

            //  AMP::pout << "Timesteps after regrid " << d_timesteps_after_regrid <<
            //  std::endl;

            // compute the norm of the approximate error
            // currently the max norm is used, an alternative norm might be appropriate
            calculateScaledLTENorm( d_solution_vector, d_prev_solutions[0], normOfError );

            std::vector<double> t1( nComponents );

            for ( size_t i = 0u; i < nComponents; ++i ) {
                t1[i] = normOfError[i] * errorFactor;
            }

            if ( d_timesteps_after_regrid > 1 ) {

                // use predictor and corrector solutions for E & T to estimate LTE
                calculateScaledLTENorm( d_solution_vector, d_predictor_vector, normOfError );

                std::vector<double> t2( nComponents );

                for ( size_t i = 0u; i < nComponents; ++i ) {
                    t2[i] = normOfError[i] * errorFactor;
                }

                double wt = ( double( d_bdf1_eps_regrid_steps - d_timesteps_after_regrid ) ) /
                            ( double( d_bdf1_eps_regrid_steps ) );
                //  AMP::pout << "Weight factor " << wt << ", timesteps after regrid " <<
                //  d_timesteps_after_regrid << std::endl;
                for ( size_t i = 0u; i < nComponents; ++i ) {
                    truncErrorEstimate[i] = wt * t1[i] + ( 1.0 - wt ) * t2[i];
                }

            } else {

                for ( size_t i = 0u; i < nComponents; ++i ) {
                    truncErrorEstimate[i] = normOfError[i] * errorFactor;
                }
            }
        } else {

            std::shared_ptr<AMP::LinearAlgebra::Vector> y;

            if ( d_first_step && ( d_bdf_starting_integrator == "CN" ) ) {
                y = d_prev_solutions[0];
            } else {
                // compute difference between predictor and corrector solutions for E & T
                y = d_predictor_vector;
            }

            calculateScaledLTENorm( d_solution_vector, y, normOfError );

            for ( size_t i = 0u; i < nComponents; ++i ) {
                truncErrorEstimate[i] = normOfError[i] * errorFactor;
            }
        }

        if ( d_iDebugPrintInfoLevel > 2 ) {

            AMP::pout << std::setprecision( 16 ) << "Error factor: " << errorFactor << std::endl;

            for ( size_t i = 0u; i < nComponents; ++i ) {
                std::string var_name =
                    ( d_var_names.size() == nComponents ) ? d_var_names[i] : "var";
                AMP::pout << std::setprecision( 16 ) << "Norm of error " << var_name << ": "
                          << normOfError[i] << std::endl;
            }

            for ( size_t i = 0u; i < nComponents; ++i ) {
                std::string var_name =
                    ( d_var_names.size() == nComponents ) ? d_var_names[i] : "var";
                AMP::pout << std::setprecision( 16 ) << "Truncation error estimate " << var_name
                          << ": " << truncErrorEstimate[i] << std::endl;
            }

            if ( d_iDebugPrintInfoLevel > 2 ) {
                AMP::pout << std::setprecision( 16 )
                          << "Old truncation error estimate prior to update "
                          << d_prevTimeTruncationErrorEstimate << std::endl;
                AMP::pout << std::setprecision( 16 ) << "Truncation error estimate prior to update "
                          << d_timeTruncationErrorEstimate << std::endl;
            }
        }

        // the time error ratio is only used for a PI based controller
        d_prevTimeTruncationErrorEstimate = d_timeTruncationErrorEstimate;
        // store the truncation error estimate in time over E and T
        // it will be used in the getNextDt routine
        d_timeTruncationErrorEstimate =
            *( std::max_element( truncErrorEstimate.begin(), truncErrorEstimate.end() ) );

        d_timeTruncationErrorEstimate =
            std::max( std::numeric_limits<double>::epsilon(), d_timeTruncationErrorEstimate );

        //  AMP::pout << "Using the truncation error estimate!!" << std::endl;

        if ( d_iDebugPrintInfoLevel > 2 ) {
            AMP::pout << std::setprecision( 16 ) << "Old truncation error estimate post update "
                      << d_prevTimeTruncationErrorEstimate << std::endl;
            AMP::pout << std::setprecision( 16 ) << "Truncation error estimate post update "
                      << d_timeTruncationErrorEstimate << std::endl;
            for ( size_t i = 0u; i < nComponents; ++i ) {
                std::string var_name =
                    ( d_var_names.size() == nComponents ) ? d_var_names[i] : "var";
                AMP::pout << std::setprecision( 16 ) << var_name << " error: " << normOfError[i]
                          << std::endl;
            }

            AMP::pout << std::setprecision( 16 ) << "errorFactor: " << errorFactor << std::endl;
            AMP::pout << std::setprecision( 16 ) << "Current dt: " << d_current_dt << std::endl;
            AMP::pout << std::setprecision( 16 ) << "Old dt: " << d_old_dt << std::endl;
        }

        // the time error ratio is only used for a PI based controller
        d_timeErrorEstimateRatio =
            d_prevTimeTruncationErrorEstimate / d_timeTruncationErrorEstimate;

        if ( d_iDebugPrintInfoLevel > 2 ) {
            AMP::pout << std::setprecision( 16 )
                      << "Ratio of time local errors: " << d_timeErrorEstimateRatio << std::endl;
        }
    } else {
        // for the first time step pretend we met the truncation error estimate
        d_prevTimeTruncationErrorEstimate = 1.0;
        d_timeTruncationErrorEstimate     = 1.0;
        d_timeErrorEstimateRatio          = 1.0;
    }

    if ( d_log_statistics ) {

        if ( d_LTE.size() == 0 ) {
            d_LTE.resize( nComponents );
        }

        for ( size_t i = 0u; i < nComponents; ++i ) {
            d_LTE[i].push_back( truncErrorEstimate[i] );
        }
    }
}

void BDFIntegrator::setIterationCounts( const int nli, const int li )
{
    d_nonlinear_iterations = nli;
    d_linear_iterations    = li;

    if ( d_log_statistics ) {
        d_times.push_back( d_current_time );
        d_timesteps.push_back( d_current_dt );
        d_nonlinearIterations.push_back( nli );
        d_linearIterations.push_back( li );
    }
}

void BDFIntegrator::printStatistics( std::ostream &os )
{
    if ( d_log_statistics ) {

        // we can only print if the stats were recorded
        os << "Total number of rejected timesteps: " << d_total_steprejects << std::endl;

        std::string spct = "          ";
        std::string spcn = "        ";
        os << std::endl;
        if ( d_calculateTimeTruncError ) {
            os << std::scientific << "Iter" << spct << "Time" << spct << "Timestep" << spct
               << "Accepted" << spct << "LTE (E)" << spct << "LTE (T)" << spct << "NLI" << spct
               << "LI" << std::endl;
            unsigned int N = d_timesteps.size();
            for ( unsigned int i = 0; i < N; ++i ) {
                os << i << spcn << d_times[i] << spcn << d_timesteps[i] << spcn
                   << d_step_accepted[i] << spcn << d_LTE[0][i] << spcn << d_LTE[1][i] << spcn
                   << d_nonlinearIterations[i] << spcn << d_linearIterations[i] << std::endl;
            }
        } else {
            os << std::scientific << "Iter" << spct << "Time" << spct << "Timestep" << spct
               << "Accepted" << spct << "NLI" << spct << "LI" << std::endl;
            unsigned int N = d_timesteps.size();
            for ( unsigned int i = 0; i < N; ++i ) {
                os << i << spcn << d_times[i] << spcn << d_timesteps[i] << spcn
                   << d_step_accepted[i] << spcn << d_nonlinearIterations[i] << spcn
                   << d_linearIterations[i] << std::endl;
            }
        }
        os << std::endl;
        os << "======================================================================" << std::endl;
    }
}

void BDFIntegrator::setRegridStatus( bool is_after_regrid ) { d_is_after_regrid = is_after_regrid; }

// functions for regrid
void BDFIntegrator::registerVectorsForMemoryManagement( void )
{
    if ( !d_vectors_registered_for_mgmt ) {

        if ( d_registerVectorForManagement ) {

            d_registerVectorForManagement( d_solution_vector );

            for ( auto i = 0u; i <= d_max_integrator_index + 1; ++i ) {
                d_registerVectorForManagement( d_prev_solutions[i] );
            }

            d_registerVectorForManagement( d_integrator_source_vector );

            if ( d_use_predictor ) {
                d_registerVectorForManagement( d_predictor_vector );
                d_registerVectorForManagement( d_timederivative_vector );
                d_registerVectorForManagement( d_current_function_vector );
                if ( d_predictor_type == "ab2" ) {
                    d_registerVectorForManagement( d_old_td_vector );
                }
            }

            d_registerVectorForManagement( d_scratch_function_vector );

            if ( d_implicit_integrator != "BE" ) {
                d_registerVectorForManagement( d_prev_function_vector );
            }
        }

        d_vectors_registered_for_mgmt = true;
    }
}

/*
*************************************************************************
*
* Reset cached information that depends on the mesh configuration.
*
*************************************************************************
*/
void BDFIntegrator::reset(
    std::shared_ptr<const AMP::TimeIntegrator::TimeIntegratorParameters> params )
{
    PROFILE( "BDFIntegrator::reset" );
    ImplicitIntegrator::reset( params );

    if ( params )
        BDFIntegrator::getFromInput( params->d_db );

    registerVectorsForMemoryManagement();

    if ( !d_reset_after_restart ) {
        // reset the number of timesteps after regrid to zero
        d_timesteps_after_regrid = 0;
    }

    d_scratch_vector->reset();
    d_scratch_function_vector->reset();

    if ( d_implicit_integrator != "BE" ) {
        d_prev_function_vector->reset();
    }

    d_integrator_source_vector->reset();

    if ( d_use_predictor ) {
        d_current_function_vector->reset();

        if ( d_predictor_type == "ab2" ) {
            d_old_td_vector->reset();
        }
    }

    if ( d_solution_vector ) {
        d_solution_vector->reset();
    }

    for ( auto i = 0u; i <= d_max_integrator_index; ++i ) {
        d_prev_solutions[i]->reset();
    }

    if ( d_use_predictor ) {

        d_predictor_vector->reset();
        d_timederivative_vector->reset();
    }

    if ( d_operator ) {
        std::shared_ptr<AMP::Operator::OperatorParameters> opParams;
        d_operator->reset( opParams );
        d_operator->reInitializeVector( d_solution_vector );
        // experimental, should all previous vectors be also cleaned up??
        for ( auto i = 0u; i <= d_max_integrator_index; ++i ) {
            d_operator->reInitializeVector( d_prev_solutions[i] );
        }

        // we re-evaluate the rhs using the current variables (not the new)
        // so as to restore information lost ie f(u_n) which will be needed
        // at the next step
        if ( d_use_predictor && ( !d_reset_after_restart ) ) {

            if ( d_predictor_type == "ab2" ) {
                // right now I am going to leave this as is
                // but probably the right approach would be to interpolate
                // the time derivative if we are following the Gresho/Sani approach
                // evaluate f(u_{n-1})
                d_scratch_vector->copyVector( d_prev_solutions[1] );
                auto timeOperator =
                    std::dynamic_pointer_cast<AMP::TimeIntegrator::TimeOperator>( d_operator );
                if ( timeOperator ) {
                    timeOperator->applyRhs( d_scratch_vector, d_old_td_vector );
                } else {
                    // it is assumed the user supplied operator is calculating (u^{n+1}-\gamma
                    // f(u^{n+1}) so we do the necessary operations to back calculate f(u^{n+1})
                    d_operator->apply( d_scratch_vector, d_old_td_vector );
                    d_old_td_vector->subtract( *d_old_td_vector, *d_scratch_vector );
                    d_old_td_vector->scale( -1.0 / d_gamma );
                }
            }

            // re-evaluates the AB2 or Leapfrog predictor
            // it is important to re-evaluate because the truncation error estimate will
            // be calculated off of the predicted values
            evaluatePredictor();
        }
    }
    d_reset_after_restart = false;
}

void BDFIntegrator::setMultiPhysicsScalings( void )
{
    if ( d_auto_component_scaling ) {
        if ( !d_solution_scaling )
            d_solution_scaling = d_solution_vector->clone();
        if ( !d_function_scaling )
            d_function_scaling = d_solution_vector->clone();
        d_solution_scaling->zero();
        d_function_scaling->zero();

        if ( !d_scratch_function_vector )
            d_scratch_function_vector = d_function_scaling->clone();
        d_scratch_function_vector->zero();

        auto mv = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( d_solution_vector );
        if ( d_iDebugPrintInfoLevel >= 2 ) {
            if ( mv ) {
                const auto n = mv->getNumberOfSubvectors();
                for ( size_t i = 0u; i < n; ++i ) {
                    const auto s    = mv->getVector( i );
                    const auto name = s->getVariable()->getName();
                    auto snorm      = static_cast<double>( s->L2Norm() );
                    AMP::pout << "BDFIntegrator::setMultiPhysicsScalings() Input solution vector "
                                 "component: "
                              << name << "  " << snorm << std::endl;
                }
            }
        }

        auto op = std::dynamic_pointer_cast<TimeOperator>( d_operator );
        if ( op ) {
            // this should be only called after a call to setInitialGuess for a timestep
            op->applyRhs( d_solution_vector, d_function_scaling );
            d_function_scaling->axpy( -d_gamma, *d_function_scaling, *d_solution_vector );
        } else {
            d_operator->apply( d_solution_vector, d_function_scaling );
        }
        d_function_scaling->add( *d_function_scaling, *d_integrator_source_vector );

        auto mf = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( d_function_scaling );
        auto ms = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( d_solution_scaling );

        if ( mf && ms && mv ) {
            const auto n = mf->getNumberOfSubvectors();
            std::vector<double> rnorms( n );
            std::vector<double> snorms( n );
            for ( size_t i = 0u; i < n; ++i ) {
                // residual scaling
                const auto r     = mf->getVector( i );
                const auto s     = ms->getVector( i );
                const auto sol   = mv->getVector( i );
                const auto rnorm = static_cast<double>( r->L2Norm() );
                const auto snorm = static_cast<double>( sol->L2Norm() );
                rnorms[i]        = ( rnorm == 0.0 ) ? 1.0 : rnorm;
                snorms[i]        = ( snorm == 0.0 ) ? 1.0 : snorm;
            }
            double resNorm{ 0.0 }, solNorm{ 0.0 };
            for ( size_t i = 0u; i < n; ++i ) {
                resNorm += rnorms[i] * rnorms[i];
                solNorm += snorms[i] * snorms[i];
            }
            const auto &vnorm = ( resNorm > solNorm ) ? rnorms : snorms;
            // for now both are set to the same
            for ( size_t i = 0u; i < n; ++i ) {
                const auto r = mf->getVector( i );
                const auto s = ms->getVector( i );
                r->setToScalar( vnorm[i] );
                s->setToScalar( vnorm[i] );
            }
        } else {
            // residual scaling
            auto fnorm = static_cast<double>( d_function_scaling->L2Norm() );
            fnorm      = ( fnorm == 0.0 ) ? 1.0 : fnorm;
            auto snorm = static_cast<double>( d_solution_vector->L2Norm() );
            snorm      = ( snorm == 0.0 ) ? 1.0 : snorm;
            auto norm  = ( fnorm > snorm ) ? fnorm : snorm;
            // for now both are set to the same
            d_function_scaling->setToScalar( norm );
            d_solution_scaling->setToScalar( norm );
        }
        setComponentScalings( d_solution_scaling, d_function_scaling );
    }
}

// provide a default implementation
int BDFIntegrator::integratorSpecificAdvanceSolution(
    const double dt,
    const bool first_step,
    std::shared_ptr<AMP::LinearAlgebra::Vector> in,
    std::shared_ptr<AMP::LinearAlgebra::Vector> out )
{
    AMP_ASSERT( in != out );
    AMP_ASSERT( stepsRemaining() && ( d_current_time < d_final_time ) );

    d_current_dt = dt;

    d_prev_solutions[0]->copyVector( in );

    setInitialGuess( first_step, d_current_time, d_current_dt, d_old_dt );

    if ( !d_scratch_function_vector )
        d_scratch_function_vector = in->clone();

    setMultiPhysicsScalings();

    if ( !d_scratch_vector )
        d_scratch_vector = in->clone();

    if ( d_solution_scaling ) {
        d_solution_vector->divide( *d_solution_vector, *d_solution_scaling );
        d_solution_vector->makeConsistent();
    }

    if ( !d_scratch_function_vector )
        d_scratch_function_vector = in->clone();

    if ( d_function_scaling ) {
        d_scratch_function_vector->divide( *d_integrator_source_vector, *d_function_scaling );
    } else {
        d_scratch_function_vector->copyVector( d_integrator_source_vector );
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "Scaled incoming TI solution component norms: ";
        auto mv = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( d_scratch_vector );
        if ( mv ) {
            const auto n = mv->getNumberOfSubvectors();
            for ( size_t i = 0u; i < n; ++i ) {
                const auto v    = mv->getVector( i );
                const auto name = v->getVariable()->getName();
                AMP::pout << name << " " << static_cast<double>( v->L2Norm() ) << " ";
            }
        } else {
            const auto name = d_scratch_vector->getVariable()->getName();
            AMP::pout << name << " " << static_cast<double>( d_scratch_vector->L2Norm() ) << " ";
        }
        AMP::pout << std::endl;
    }

    if ( !d_solver ) {
        createSolver();
        d_solver->setComponentScalings( d_solution_scaling, d_function_scaling );
    } else {
        d_solver->reset( {} );
    }

    d_scratch_function_vector->scale( -1.0 );

    d_solver->apply( d_scratch_function_vector, d_solution_vector );

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "Scaled outgoing TI solution component norms: ";
        auto mv = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( d_scratch_vector );
        if ( mv ) {
            const auto n = mv->getNumberOfSubvectors();
            for ( size_t i = 0u; i < n; ++i ) {
                const auto v    = mv->getVector( i );
                const auto name = v->getVariable()->getName();
                AMP::pout << name << " " << static_cast<double>( v->L2Norm() ) << " ";
            }
        } else {
            const auto name = d_scratch_vector->getVariable()->getName();
            AMP::pout << name << " " << static_cast<double>( d_scratch_vector->L2Norm() ) << " ";
        }
        AMP::pout << std::endl;
    }
    const auto convStatus = d_solver->getConvergenceStatus();
    d_solver_retcode =
        convStatus <= AMP::Solver::SolverStrategy::SolverStatus::ConvergedUserCondition ? 1 : 0;

    if ( d_solution_scaling )
        d_solution_vector->multiply( *d_solution_vector, *d_solution_scaling );

    out->copyVector( d_solution_vector );

    return d_solver_retcode;
}

/********************************************************
 *  Restart operations                                   *
 ********************************************************/
void BDFIntegrator::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    ImplicitIntegrator::registerChildObjects( manager );
}

void BDFIntegrator::writeRestart( int64_t fid ) const
{
    d_pParameters->d_db->putScalar<int>( "final_constant_timestep_current_step",
                                         d_final_constant_timestep_current_step,
                                         {},
                                         AMP::Database::Check::Overwrite );
    d_pParameters->d_db->putScalar<double>(
        "first_initial_dt", d_first_initial_dt, {}, AMP::Database::Check::Overwrite );
    ImplicitIntegrator::writeRestart( fid );
}

BDFIntegrator::BDFIntegrator( int64_t fid, AMP::IO::RestartManager *manager )
    : ImplicitIntegrator( fid, manager )
{
    d_initialized = false;
    d_object_name = "BDFIntegrator";
    BDFIntegrator::getFromInput( d_pParameters->d_db );
    BDFIntegrator::initialize();
    d_initialized = true;
}

} // namespace AMP::TimeIntegrator
