#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iomanip>

#include "AMP/solvers/testHelpers/testSolverHelpers.h"

void linearElasticTest( AMP::UnitTest *ut, const std::string &inputFileName )
{
    std::string input_file = inputFileName;
    AMP::pout << "Running linearElasticTest with input " << input_file << std::endl;

    AMP::AMP_MPI comm( AMP_COMM_WORLD );

    std::string log_file = "output_" + input_file;

    AMP::logOnlyNodeZero( log_file );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // create the Mesh
    const auto mesh = createMesh( input_db );

    auto bvpOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "MechanicsBVPOperator", input_db ) );

    auto var = bvpOperator->getOutputVariable();

    auto dirichletVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );
    // This has an in-place apply. So, it has an empty input variable and
    // the output variable is the same as what it is operating on.
    dirichletVecOp->setVariable( var );

    auto dofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto mechSolVec = AMP::LinearAlgebra::createVector( dofMap, var, true );
    auto mechRhsVec = mechSolVec->clone();
    auto mechResVec = mechSolVec->clone();

    mechSolVec->setToScalar( 0.5 );
    mechRhsVec->setToScalar( 0.0 );
    mechResVec->setToScalar( 0.0 );

    dirichletVecOp->apply( nullVec, mechRhsVec );

    AMP::pout << "RHS Norm: " << mechRhsVec->L2Norm() << std::endl;
    AMP::pout << "Initial Solution Norm: " << mechSolVec->L2Norm() << std::endl;

    bvpOperator->residual( mechRhsVec, mechSolVec, mechResVec );

    double initResidualNorm = static_cast<double>( mechResVec->L2Norm() );
    AMP::pout << "Initial Residual Norm: " << initResidualNorm << std::endl;

    auto mlSolver =
        AMP::Solver::Test::buildSolver( "LinearSolver", input_db, comm, nullptr, bvpOperator );

    mlSolver->setZeroInitialGuess( false );

    mlSolver->apply( mechRhsVec, mechSolVec );

    bvpOperator->residual( mechRhsVec, mechSolVec, mechResVec );

    double finalResidualNorm = static_cast<double>( mechResVec->L2Norm() );

    AMP::pout << "Final Residual Norm: " << finalResidualNorm << std::endl;

    // BP: convergence assumes convergence rate of ~0.6 and 40 iterations
    if ( finalResidualNorm > ( 1.0e-8 * initResidualNorm ) ) {
        ut->failure( mlSolver->type() + " fails to solve a linear elasticity problem" );
    } else {
        ut->passes( mlSolver->type() + " successfully solves a linear elasticity problem" );
    }

    input_db.reset();
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> files;

    PROFILE_ENABLE();

    if ( argc > 1 ) {

        files.emplace_back( argv[1] );

    } else {
#ifdef AMP_USE_HYPRE
        files.emplace_back( "input_testBoomerAMGSolver-LinearElasticityOperator-1" );
#endif
#ifdef AMP_USE_TRILINOS_ML
        files.emplace_back( "input_testTrilinosMLSolver-LinearElasticityOperator-1" );
#endif
#ifdef AMP_USE_TRILINOS_MUELU
        files.emplace_back( "input_testTrilinosMueLuSolver-LinearElasticityOperator-1" );
#endif
    }

    {
        PROFILE( "DRIVER::main(test loop)" );
        for ( auto &file : files ) {
            linearElasticTest( &ut, file );
        }
    }

    ut.report();

    // build unique profile name to avoid collisions
    std::ostringstream ss;
    ss << "testLinearElasticity_r" << std::setw( 3 ) << std::setfill( '0' )
       << AMP::AMPManager::getCommWorld().getSize();

    PROFILE_SAVE( ss.str() );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
