#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/amg/SASolver.h"
#include "AMP/solvers/amg/Stats.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.h"

#include <chrono>
#include <iomanip>
#include <memory>
#include <string>

#ifdef AMP_USE_HYPRE
    #include "HYPRE_config.h"
#endif

std::tuple<std::shared_ptr<AMP::Operator::LinearOperator>,
           std::shared_ptr<AMP::LinearAlgebra::Vector>,
           std::shared_ptr<AMP::LinearAlgebra::Vector>>
constructLinearSystem( const std::string &physicsFileName )
{
    PROFILE( "DRIVER::linearThermalPhysics" );

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( physicsFileName );

    auto neutronicsOp_db = input_db->getDatabase( "NeutronicsOperator" );
    auto volumeOp_db     = input_db->getDatabase( "VolumeIntegralOperator" );

    // Create the Mesh
    const auto mesh = createMesh( input_db );

    auto PowerInWattsVec = constructNeutronicsPowerSource( input_db, mesh );

    // Set appropriate acceleration backend
    auto op_db = input_db->getDatabase( "DiffusionBVPOperator" );
    // Create the Thermal BVP Operator
    auto diffusionOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "DiffusionBVPOperator", input_db ) );

    auto linearOp               = diffusionOperator->getVolumeOperator();
    auto TemperatureInKelvinVec = linearOp->createOutputVector();
    auto RightHandSideVec       = linearOp->createInputVector();

    auto boundaryOpCorrectionVec = RightHandSideVec->clone();

    // Add the boundary conditions corrections
    auto boundaryOp = diffusionOperator->getBoundaryOperator();
    boundaryOp->addRHScorrection( boundaryOpCorrectionVec );
    RightHandSideVec->subtract( *PowerInWattsVec, *boundaryOpCorrectionVec );

    return std::make_tuple( linearOp, TemperatureInKelvinVec, RightHandSideVec );
}

bool compare_expected( AMP::UnitTest &ut,
                       const AMP::Solver::AMG::HierarchyStats &a,
                       const AMP::Solver::AMG::HierarchyStats &b )
{
    const float ftol = 1e-3;
    bool failed      = false;

    auto fail = [&]( const char *name, auto e, auto g ) {
        std::stringstream ss;
        ss << "FAILED: " << name << " " << e << " vs " << g;
        ut.failure( ss.str() );
        failed = true;
    };
    if ( std::abs( a.operator_complexity - b.operator_complexity ) >= ftol )
        fail( "operator complexity", a.operator_complexity, b.operator_complexity );

    if ( std::abs( a.grid_complexity - b.grid_complexity ) >= ftol )
        fail( "grid complexity", a.grid_complexity, b.grid_complexity );

    if ( a.levels.size() != b.levels.size() )
        fail( "number of levels", a.levels.size(), b.levels.size() );

    using level_t = AMP::Solver::AMG::HierarchyStats::level_type;
    for ( std::size_t i = 0; i < a.levels.size(); ++i ) {
        if ( ![&]( const level_t &al, const level_t &bl ) {
                 auto cmp = []( const char *name, auto a, auto b ) {
                     if ( a != b )
                         AMP::pout << name << ": " << a << " vs " << b << std::endl;
                 };
                 cmp( "comm_size", al.comm_size, bl.comm_size );
                 cmp( "nrows", al.nrows, bl.nrows );
                 cmp( "nnz", al.nnz, bl.nnz );
                 cmp( "max_local_rows", al.max_local_rows, bl.max_local_rows );
                 cmp( "min_local_rows", al.min_local_rows, bl.min_local_rows );
                 return al.solver_name == bl.solver_name && al.comm_size == bl.comm_size &&
                        al.nrows == bl.nrows && al.nnz == bl.nnz &&
                        al.max_local_rows == bl.max_local_rows &&
                        al.min_local_rows == bl.min_local_rows;
             }( a.levels[i], b.levels[i] ) ) {
            ut.failure( "FAILED: level data on level " + std::to_string( i ) );
            failed = true;
        }
    }

    return !failed;
}

void statsTest( AMP::UnitTest *ut,
                const std::string &inputFileName,
                std::tuple<std::shared_ptr<AMP::Operator::LinearOperator>,
                           std::shared_ptr<AMP::LinearAlgebra::Vector>,
                           std::shared_ptr<AMP::LinearAlgebra::Vector>> linearSystem )
{
    std::string accelerationBackend{ "serial" };
    std::string memoryLocation{ "host" };

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( inputFileName );
    input_db->print( AMP::plog );

    auto [linearOperator, sol, rhs] = linearSystem;

    auto &comm     = rhs->getComm();
    auto solver_db = input_db->getDatabase( "LinearSolver" );
    solver_db->putScalar( "MemoryLocation", memoryLocation );
    auto backend = AMP::Utilities::backendFromString( accelerationBackend );

    linearOperator->getMatrix()->setBackend( backend );

    auto linearSolver =
        AMP::Solver::Test::buildSolver( "LinearSolver", input_db, comm, nullptr, linearOperator );
    auto amg = std::dynamic_pointer_cast<AMP::Solver::AMG::SASolver>( linearSolver );
    AMP_INSIST( amg, "testAMGStats: preconditioner must be AMG" );

    AMP::Solver::AMG::HierarchyStats expected{ 1.12446,
                                               1.0458,
                                               { { "SASolver", 2, 4913, 117649, 2601, 2312 },
                                                 { "SASolver", 2, 216, 14562, 108, 108 },
                                                 { "BoomerAMGSolver", 2, 9, 81, 5, 4 } } };

    auto stats =
        AMP::Solver::AMG::collect_statistics( amg->type(), amg->levels(), amg->getCoarseSolver() );
    AMP::Solver::AMG::print_summary( amg->type(), amg->levels(), amg->getCoarseSolver() );

    auto is_expected = compare_expected( *ut, stats, expected );
    if ( is_expected )
        ut->passes( "AMG Hierarchy Stats match expected values." );
}

void runTestOnInputs( AMP::UnitTest *ut,
                      const std::string &physicsInput,
                      const std::string &generalInput )
{
    auto linearSystem = constructLinearSystem( physicsInput );
    statsTest( ut, generalInput, linearSystem );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string physicsInput = "input_LinearThermalRobinOperator";
    std::string generalInput = "input_testLinearSolvers-LinearThermalRobin-SASolver-BoomerAMG";
    AMP_INSIST( AMP::LinearAlgebra::getDefaultMatrixType() == "CSRMatrix",
                "CSRMatrix required for AMG Stats." );
#ifdef AMP_USE_HYPRE
    runTestOnInputs( &ut, physicsInput, generalInput );
#endif
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
