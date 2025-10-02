#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/SolverFactory.h"
#ifdef AMP_USE_TRILINOS_NOX
    #include "AMP/solvers/trilinos/nox/TrilinosNOXSolverParameters.h"
#endif
#include <memory>
#include <string>


namespace AMP::Solver::Test {


std::unique_ptr<AMP::Database> SolverParameters::getParameters( const std::string &solver,
                                                                bool use_nested )
{
    if ( solver == "CG" ) {
        return getCGParameters( use_nested );
    } else if ( solver == "GMRES" ) {
        return getGMRESParameters( use_nested );
    } else if ( solver == "FGMRES" ) {
        return getFGMRESParameters( use_nested );
    } else if ( solver == "BiCGSTAB" ) {
        return getBiCGSTABParameters( use_nested );
    } else if ( solver == "TFQMR" ) {
        return getTFQMRParameters( use_nested );
    } else if ( solver == "PetscFGMRES" ) {
        return getPetscFGMRESParameters( use_nested );
    } else if ( solver == "BoomerAMG" ) {
        return getBoomerAMGParameters( use_nested );
    } else if ( solver == "HyprePCG" ) {
        return getHyprePCGParameters( use_nested );
    } else if ( solver == "HypreGMRES" ) {
        return getHypreGMRESParameters( use_nested );
    } else if ( solver == "HypreBiCGSTAB" ) {
        return getHypreBiCGSTABParameters( use_nested );
    } else if ( solver == "ML" ) {
        return getMLParameters( use_nested );
    } else if ( solver == "MueLu" ) {
        return getMueLuParameters( use_nested );
    } else {
        AMP_ERROR( "Unknown solver" );
    }
}

std::unique_ptr<AMP::Database> SolverParameters::getCGParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "CGSolver" );
    db->putScalar<bool>( "uses_preconditioner", use_nested );
    db->putScalar<double>( "absolute_tolerance", 1.0e-12 );
    db->putScalar<double>( "relative_tolerance", 1.0e-12 );
    db->putScalar<int>( "print_info_level", 1 );
    db->putScalar<int>( "max_iterations", 100 );
    if ( use_nested )
        db->putScalar<std::string>( "pc_solver_name", "Preconditioner" );
    return db;
}

std::unique_ptr<AMP::Database> SolverParameters::getGMRESParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "GMRESSolver" );
    db->putScalar<bool>( "uses_preconditioner", use_nested );
    db->putScalar<double>( "absolute_tolerance", 1.0e-12 );
    db->putScalar<double>( "relative_tolerance", 1.0e-12 );
    db->putScalar<int>( "print_info_level", 1 );
    db->putScalar<int>( "max_iterations", 100 );
    if ( use_nested )
        db->putScalar<std::string>( "pc_solver_name", "Preconditioner" );
    return db;
}

std::unique_ptr<AMP::Database> SolverParameters::getFGMRESParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "GMRESSolver" );
    db->putScalar<bool>( "flexible_gmres", true );
    db->putScalar<bool>( "uses_preconditioner", use_nested );
    db->putScalar<double>( "absolute_tolerance", 1.0e-12 );
    db->putScalar<double>( "relative_tolerance", 1.0e-12 );
    db->putScalar<int>( "print_info_level", 1 );
    db->putScalar<int>( "max_iterations", 100 );
    if ( use_nested )
        db->putScalar<std::string>( "pc_solver_name", "Preconditioner" );
    return db;
}

std::unique_ptr<AMP::Database> SolverParameters::getBiCGSTABParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "BiCGSTABSolver" );
    db->putScalar<bool>( "uses_preconditioner", use_nested );
    db->putScalar<double>( "absolute_tolerance", 1.0e-12 );
    db->putScalar<double>( "relative_tolerance", 1.0e-12 );
    db->putScalar<int>( "print_info_level", 1 );
    db->putScalar<int>( "max_iterations", 100 );
    if ( use_nested )
        db->putScalar<std::string>( "pc_solver_name", "Preconditioner" );
    return db;
}

std::unique_ptr<AMP::Database> SolverParameters::getTFQMRParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "TFQMRSolver" );
    db->putScalar<bool>( "uses_preconditioner", use_nested );
    db->putScalar<bool>( "compute_residual", true );
    db->putScalar<double>( "absolute_tolerance", 1.0e-12 );
    db->putScalar<double>( "relative_tolerance", 1.0e-12 );
    db->putScalar<int>( "print_info_level", 1 );
    db->putScalar<int>( "max_iterations", 100 );
    if ( use_nested )
        db->putScalar<std::string>( "pc_solver_name", "Preconditioner" );
    return db;
}

std::unique_ptr<AMP::Database> SolverParameters::getPetscFGMRESParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "PetscKrylovSolver" );
    db->putScalar<std::string>( "ksp_type", "fgmres" );
    db->putScalar<std::string>( "pc_type", "shell" );
    db->putScalar<std::string>( "pc_side", "RIGHT" );
    db->putScalar<std::string>( "KSPOptions",
                                "-ksp_monitor -ksp_converged_reason -ksp_max_it 25 -ksp_rtol "
                                "1.0e-12 -ksp_atol 1.0e-12" );
    db->putScalar<bool>( "uses_preconditioner", use_nested );
    db->putScalar<double>( "absolute_tolerance", 1.0e-12 );
    db->putScalar<double>( "relative_tolerance", 1.0e-12 );
    db->putScalar<int>( "print_info_level", 1 );
    db->putScalar<int>( "max_iterations", 100 );
    if ( use_nested )
        db->putScalar<std::string>( "pc_solver_name", "Preconditioner" );
    return db;
}

std::unique_ptr<AMP::Database> SolverParameters::getHyprePCGParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "HyprePCGSolver" );
    db->putScalar<bool>( "uses_preconditioner", use_nested );
    db->putScalar<double>( "absolute_tolerance", 1.0e-12 );
    db->putScalar<double>( "relative_tolerance", 1.0e-12 );
    db->putScalar<int>( "print_info_level", 1 );
    db->putScalar<int>( "max_iterations", 100 );
    if ( use_nested )
        db->putScalar<std::string>( "pc_solver_name", "Preconditioner" );
    return db;
}


std::unique_ptr<AMP::Database> SolverParameters::getHypreGMRESParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "HyprePCGSolver" );
    db->putScalar<bool>( "uses_preconditioner", use_nested );
    db->putScalar<double>( "absolute_tolerance", 1.0e-12 );
    db->putScalar<double>( "relative_tolerance", 1.0e-12 );
    db->putScalar<int>( "print_info_level", 1 );
    db->putScalar<int>( "max_iterations", 100 );
    db->putScalar<int>( "max_krylov_dimension", 150 );
    if ( use_nested )
        db->putScalar<std::string>( "pc_solver_name", "Preconditioner" );
    return db;
}


std::unique_ptr<AMP::Database> SolverParameters::getHypreBiCGSTABParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "HyprePCGSolver" );
    db->putScalar<bool>( "uses_preconditioner", use_nested );
    db->putScalar<double>( "absolute_tolerance", 1.0e-12 );
    db->putScalar<double>( "relative_tolerance", 1.0e-12 );
    db->putScalar<int>( "print_info_level", 1 );
    db->putScalar<int>( "max_iterations", 100 );
    if ( use_nested )
        db->putScalar<std::string>( "pc_solver_name", "Preconditioner" );
    return db;
}

std::unique_ptr<AMP::Database> SolverParameters::getBoomerAMGParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "BoomerAMGSolver" );
    db->putScalar<int>( "max_iterations", use_nested ? 1 : 25 );
    db->putScalar<int>( "min_coarse_size", 10 );
    db->putScalar<int>( "relax_type", 16 );
    db->putScalar<int>( "coarsen_type", 8 );
    db->putScalar<int>( "interp_type", 18 );
    db->putScalar<int>( "cycle_type", 1 );
    db->putScalar<int>( "relax_order", 0 );
    db->putScalar<double>( "strong_threshold", 0.5 );
    return db;
}

std::unique_ptr<AMP::Database> SolverParameters::getMLParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "TrilinosMLSolver" );
    db->putScalar<int>( "max_iterations", use_nested ? 1 : 25 );
    db->putScalar<bool>( "problem_symmetric", true );
    db->putScalar<std::string>( "smoother_preorpost", "both" );
    db->putScalar<int>( "smoother_sweeps", 2 );
    db->putScalar<int>( "coarse_maxsize", 10 );
    return db;
}

std::unique_ptr<AMP::Database> SolverParameters::getMueLuParameters( bool use_nested )
{
    auto db = std::make_unique<AMP::Database>( "LinearSolver" );
    db->putScalar<std::string>( "name", "TrilinosMueLuSolver" );
    db->putScalar<int>( "max_iterations", use_nested ? 1 : 25 );
    db->putScalar<bool>( "problem_symmetric", true );
    db->putScalar<std::string>( "smoother_preorpost", "both" );
    db->putScalar<int>( "coarse_max_size", 100 );
    auto smoother_db = std::make_unique<AMP::Database>( "smoother_params" );
    smoother_db->putScalar<std::string>( "relaxation_type", "Gauss-Seidel" );
    smoother_db->putScalar<int>( "relaxation_sweeps", 2 );
    db->putDatabase( "smoother_params", std::move( smoother_db ) );
    return db;
}


std::shared_ptr<AMP::Solver::SolverStrategy>
buildSolver( const std::string &solver_name,
             std::shared_ptr<AMP::Database> input_db,
             const AMP::AMP_MPI &comm,
             std::shared_ptr<AMP::LinearAlgebra::Vector> initialGuess,
             std::shared_ptr<AMP::Operator::Operator> op )
{

    AMP_INSIST( input_db->keyExists( solver_name ), "Key " + solver_name + " is missing!" );

    auto db = input_db->getDatabase( solver_name );
    AMP_INSIST( db->keyExists( "name" ), "Key name does not exist in solver database" );
    auto type_name = db->getScalar<std::string>( "name" );
    // temporary hack for NOX since this is testing infrastructure
    std::shared_ptr<AMP::Solver::SolverStrategyParameters> parameters;
    if ( type_name == "TrilinosNOXSolver" ) {
#ifdef AMP_USE_TRILINOS_NOX
        parameters = std::make_shared<AMP::Solver::TrilinosNOXSolverParameters>( db );
#else
        AMP_ERROR( "AMP built without support for Trilinos NOX" );
#endif
    } else {
        parameters = std::make_shared<AMP::Solver::SolverStrategyParameters>( db );
    }

    parameters->d_pOperator     = op;
    parameters->d_comm          = comm;
    parameters->d_pInitialGuess = initialGuess;
    parameters->d_db            = db;
    parameters->d_global_db     = input_db;

    return AMP::Solver::SolverFactory::create( parameters );
}


} // namespace AMP::Solver::Test
