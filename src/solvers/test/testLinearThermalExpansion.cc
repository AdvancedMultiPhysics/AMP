
#include "utils/InputManager.h"
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"

#include "utils/shared_ptr.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "discretization/simpleDOF_Manager.h"
#include "vectors/VectorBuilder.h"

#include "utils/Writer.h"

#include "operators/ColumnOperator.h"
#include "operators/LinearBVPOperator.h"
#include "operators/OperatorBuilder.h"
#include "operators/mechanics/ConstructLinearMechanicsRHSVector.h"

#include "solvers/petsc/PetscKrylovSolver.h"
#include "solvers/petsc/PetscKrylovSolverParameters.h"
#include "solvers/trilinos/ml/TrilinosMLSolver.h"

void myTest( AMP::UnitTest *ut, std::string exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero( log_file );

    AMP::shared_ptr<AMP::InputDatabase> input_db( new AMP::InputDatabase( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile( input_file, input_db );
    input_db->printClassData( AMP::plog );

#ifdef USE_EXT_SILO
    // Create the silo writer and register the data
    AMP::Utilities::Writer::shared_ptr siloWriter = AMP::Utilities::Writer::buildWriter( "Silo" );
#endif

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    AMP::shared_ptr<AMP::Database> mesh_db = input_db->getDatabase( "Mesh" );
    AMP::shared_ptr<AMP::Mesh::MeshParameters> meshParams(
        new AMP::Mesh::MeshParameters( mesh_db ) );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    AMP::Mesh::Mesh::shared_ptr meshAdapter = AMP::Mesh::Mesh::buildMesh( meshParams );

    AMP::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel;
    AMP::shared_ptr<AMP::Operator::LinearBVPOperator> bvpOperator =
        AMP::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                meshAdapter, "MechanicsBVPOperator", input_db, elementPhysicsModel ) );

    AMP::pout << "Constructed BVP operator" << std::endl;

    AMP::LinearAlgebra::Variable::shared_ptr dispVar = bvpOperator->getOutputVariable();
    AMP::LinearAlgebra::Variable::shared_ptr tempVar( new AMP::LinearAlgebra::Variable( "temp" ) );

    AMP::Discretization::DOFManager::shared_ptr tempDofMap =
        AMP::Discretization::simpleDOFManager::create( meshAdapter, AMP::Mesh::GeomType::Vertex, 1, 1, true );

    AMP::Discretization::DOFManager::shared_ptr dispDofMap =
        AMP::Discretization::simpleDOFManager::create( meshAdapter, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    AMP::LinearAlgebra::Vector::shared_ptr mechSolVec =
        AMP::LinearAlgebra::createVector( dispDofMap, dispVar, true );
    AMP::LinearAlgebra::Vector::shared_ptr mechRhsVec = mechSolVec->cloneVector();
    AMP::LinearAlgebra::Vector::shared_ptr mechResVec = mechSolVec->cloneVector();

    AMP::LinearAlgebra::Vector::shared_ptr currTempVec =
        AMP::LinearAlgebra::createVector( tempDofMap, tempVar, true );
    AMP::LinearAlgebra::Vector::shared_ptr prevTempVec = currTempVec->cloneVector();

    mechSolVec->setToScalar( 0.0 );
    mechResVec->setToScalar( 0.0 );

    currTempVec->setToScalar( 500.0 );
    prevTempVec->setToScalar( 300.0 );

    AMP::shared_ptr<AMP::Database> temperatureRhsDatabase =
        input_db->getDatabase( "TemperatureRHS" );

    computeTemperatureRhsVector( meshAdapter,
                                 temperatureRhsDatabase,
                                 tempVar,
                                 dispVar,
                                 currTempVec,
                                 prevTempVec,
                                 mechRhsVec );
    bvpOperator->modifyRHSvector( mechRhsVec );

    AMP::shared_ptr<AMP::Database> linearSolver_db = input_db->getDatabase( "LinearSolver" );
    AMP::shared_ptr<AMP::Database> pcSolver_db = linearSolver_db->getDatabase( "Preconditioner" );
    AMP::shared_ptr<AMP::Solver::TrilinosMLSolverParameters> pcSolverParams(
        new AMP::Solver::TrilinosMLSolverParameters( pcSolver_db ) );
    pcSolverParams->d_pOperator = bvpOperator;
    AMP::shared_ptr<AMP::Solver::TrilinosMLSolver> pcSolver(
        new AMP::Solver::TrilinosMLSolver( pcSolverParams ) );

    AMP::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> linearSolverParams(
        new AMP::Solver::PetscKrylovSolverParameters( linearSolver_db ) );
    linearSolverParams->d_pOperator       = bvpOperator;
    linearSolverParams->d_comm            = AMP_COMM_WORLD;
    linearSolverParams->d_pPreconditioner = pcSolver;
    AMP::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver(
        new AMP::Solver::PetscKrylovSolver( linearSolverParams ) );

    linearSolver->solve( mechRhsVec, mechSolVec );

#ifdef USE_EXT_SILO
    siloWriter->registerVector( mechSolVec, meshAdapter, AMP::Mesh::GeomType::Vertex, "Solution" );
#endif

#ifdef USE_EXT_SILO
    siloWriter->writeFile( exeName, 1 );
    meshAdapter->displaceMesh( mechSolVec );
    siloWriter->writeFile( exeName, 2 );
#endif

    ut->passes( exeName );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName = "testLinearThermalExpansion";

    myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
