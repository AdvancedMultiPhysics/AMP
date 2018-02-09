#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>

#include "AMP/solvers/libmesh/PelletStackHelpers.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"

#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/InputManager.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/Writer.h"


void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

#ifdef USE_EXT_SILO
    // Create the silo writer and register the data
    AMP::Utilities::Writer::shared_ptr siloWriter = AMP::Utilities::Writer::buildWriter( "Silo" );
#endif

    AMP::shared_ptr<AMP::InputDatabase> global_input_db(
        new AMP::InputDatabase( "global_input_db" ) );
    AMP::InputManager::getManager()->parseInputFile( input_file, global_input_db );
    global_input_db->printClassData( AMP::plog );

    unsigned int NumberOfLoadingSteps = global_input_db->getInteger( "NumberOfLoadingSteps" );
    bool usePointLoad                 = global_input_db->getBool( "USE_POINT_LOAD" );
    bool useThermalLoad               = global_input_db->getBool( "USE_THERMAL_LOAD" );

    AMP_INSIST( global_input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    AMP::shared_ptr<AMP::Database> mesh_db = global_input_db->getDatabase( "Mesh" );
    AMP::shared_ptr<AMP::Mesh::MeshParameters> meshParams(
        new AMP::Mesh::MeshParameters( mesh_db ) );
    meshParams->setComm( globalComm );
    AMP::Mesh::Mesh::shared_ptr manager = AMP::Mesh::Mesh::buildMesh( meshParams );

    AMP::shared_ptr<AMP::Operator::CoupledOperator> coupledOp;
    AMP::shared_ptr<AMP::Operator::ColumnOperator> linearColumnOperator;
    AMP::shared_ptr<AMP::Operator::PelletStackOperator> pelletStackOp;
    helperCreateAllOperatorsForPelletMechanics(
        manager, globalComm, global_input_db, coupledOp, linearColumnOperator, pelletStackOp );

    AMP::LinearAlgebra::Vector::shared_ptr solVec, rhsVec, scaledRhsVec;
    helperCreateVectorsForPelletMechanics( manager, coupledOp, solVec, rhsVec, scaledRhsVec );

    if ( usePointLoad ) {
        helperBuildPointLoadRHSForPelletMechanics( global_input_db, coupledOp, rhsVec );
    } else {
        rhsVec->zero();
    }

    AMP::LinearAlgebra::Vector::shared_ptr initialTemperatureVec, finalTemperatureVec;
    if ( useThermalLoad ) {
        helperCreateTemperatureVectorsForPelletMechanics(
            manager, initialTemperatureVec, finalTemperatureVec );
    }

    if ( useThermalLoad ) {
        double initialTemp = global_input_db->getDouble( "InitialTemperature" );
        initialTemperatureVec->setToScalar( initialTemp );
        helperSetReferenceTemperatureForPelletMechanics( coupledOp, initialTemperatureVec );
    }

    solVec->zero();
    helperApplyBoundaryCorrectionsForPelletMechanics( coupledOp, solVec, rhsVec );

    AMP::shared_ptr<AMP::Database> nonlinearSolver_db =
        global_input_db->getDatabase( "NonlinearSolver" );
    AMP::shared_ptr<AMP::Database> linearSolver_db =
        nonlinearSolver_db->getDatabase( "LinearSolver" );
    AMP::shared_ptr<AMP::Database> pelletStackSolver_db =
        linearSolver_db->getDatabase( "PelletStackSolver" );

    AMP::shared_ptr<AMP::Solver::SolverStrategy> pelletStackSolver;
    helperBuildStackSolverForPelletMechanics(
        pelletStackSolver_db, pelletStackOp, linearColumnOperator, pelletStackSolver );

    AMP::shared_ptr<AMP::Solver::PetscSNESSolverParameters> nonlinearSolverParams(
        new AMP::Solver::PetscSNESSolverParameters( nonlinearSolver_db ) );
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = coupledOp;
    nonlinearSolverParams->d_pInitialGuess = solVec;
    AMP::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearSolver(
        new AMP::Solver::PetscSNESSolver( nonlinearSolverParams ) );

    AMP::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver =
        nonlinearSolver->getKrylovSolver();
    linearSolver->setPreconditioner( pelletStackSolver );

#ifdef USE_EXT_SILO
    siloWriter->registerVector( solVec, manager, AMP::Mesh::GeomType::Vertex, "Displacement" );
#endif

    for ( unsigned int step = 0; step < NumberOfLoadingSteps; step++ ) {
        AMP::pout << "########################################" << std::endl;
        AMP::pout << "The current loading step is " << ( step + 1 ) << std::endl;

        double scaleValue =
            ( static_cast<double>( step + 1 ) ) / ( static_cast<double>( NumberOfLoadingSteps ) );
        scaledRhsVec->scale( scaleValue, rhsVec );

        if ( useThermalLoad ) {
            double initialTemp = global_input_db->getDouble( "InitialTemperature" );
            double finalTemp   = global_input_db->getDouble( "FinalTemperature" );
            double deltaTemp =
                initialTemp + ( ( static_cast<double>( step + 1 ) ) * ( finalTemp - initialTemp ) /
                                ( static_cast<double>( NumberOfLoadingSteps ) ) );
            finalTemperatureVec->setToScalar( deltaTemp );
            helperSetFinalTemperatureForPelletMechanics( coupledOp, finalTemperatureVec );
        }

        AMP::LinearAlgebra::Vector::shared_ptr resVec = solVec->cloneVector();
        resVec->zero();
        coupledOp->residual( scaledRhsVec, solVec, resVec );
        AMP::pout << "initial, rhsVec: " << scaledRhsVec->L2Norm() << std::endl;
        AMP::pout << "initial, solVec: " << solVec->L2Norm() << std::endl;
        AMP::pout << "initial, resVec: " << resVec->L2Norm() << std::endl;
        nonlinearSolver->solve( scaledRhsVec, solVec );
        AMP::pout << "solved,  rhsVec: " << scaledRhsVec->L2Norm() << std::endl;
        AMP::pout << "solved,  solVec: " << solVec->L2Norm() << std::endl;
        coupledOp->residual( scaledRhsVec, solVec, resVec );
        AMP::pout << "final,   rhsVec: " << scaledRhsVec->L2Norm() << std::endl;
        AMP::pout << "final,   solVec: " << solVec->L2Norm() << std::endl;
        AMP::pout << "final,   resVec: " << resVec->L2Norm() << std::endl;

#ifdef USE_EXT_SILO
        siloWriter->writeFile( exeName, step );
#endif

        helperResetNonlinearOperatorForPelletMechanics( coupledOp );
    } // end for step

    ut->passes( exeName );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    int inp = 1;
    if ( argc > 1 ) {
        inp = atoi( argv[1] );
    }

    char exeName[200];
    sprintf( exeName, "testPelletStackMechanicsSolver-%d", inp );

    myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
