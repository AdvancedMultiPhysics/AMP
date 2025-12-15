#include "AMP/IO/PIO.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/solvers/libmesh/PelletStackHelpers.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>


using namespace AMP::Operator::PelletMechanics;


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto db = AMP::Database::parseInputFile( input_file );
    db->print( AMP::plog );

    int NumberOfLoadingSteps = db->getScalar<int>( "NumberOfLoadingSteps" );
    bool usePointLoad        = db->getScalar<bool>( "USE_POINT_LOAD" );
    bool useThermalLoad      = db->getScalar<bool>( "USE_THERMAL_LOAD" );

    // Load the mesh
    AMP_INSIST( db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( globalComm );
    auto manager = AMP::Mesh::MeshFactory::create( meshParams );

    // Create the nonlinear operators
    auto n2nmaps                 = createMaps( manager, db );
    auto pelletStackOp           = createStackOperator( manager, n2nmaps, db );
    auto nonlinearColumnOperator = createNonlinearColumnOperator( pelletStackOp, db );
    auto coupledOp               = createCoupledOperator( n2nmaps, nonlinearColumnOperator );
    setFrozenVectorForMaps( manager, coupledOp );

    // Create the vectors
    AMP::LinearAlgebra::Vector::shared_ptr solVec, rhsVec, scaledRhsVec;
    createVectors( manager, coupledOp, solVec, rhsVec, scaledRhsVec );

    if ( usePointLoad ) {
        buildPointLoadRHS( db, coupledOp, rhsVec );
    } else {
        rhsVec->zero();
    }

    AMP::LinearAlgebra::Vector::shared_ptr initialTemperatureVec, finalTemperatureVec;
    if ( useThermalLoad ) {
        createTemperatureVectors( manager, initialTemperatureVec, finalTemperatureVec );
    }

    if ( useThermalLoad ) {
        auto initialTemp = db->getScalar<double>( "InitialTemperature" );
        initialTemperatureVec->setToScalar( initialTemp );
        setReferenceTemperature( coupledOp, initialTemperatureVec );
    }

    solVec->zero();
    applyBoundaryCorrections( coupledOp, solVec, rhsVec );

    auto nonlinearSolver_db   = db->getDatabase( "NonlinearSolver" );
    auto linearSolver_db      = nonlinearSolver_db->getDatabase( "LinearSolver" );
    auto pelletStackSolver_db = linearSolver_db->getDatabase( "PelletStackSolver" );

    // Create the linear operators
    auto linearColumnOperator = createLinearColumnOperator( nonlinearColumnOperator );

    // Create the solvers
    auto pelletStackSolver =
        buildStackSolver( pelletStackSolver_db, pelletStackOp, linearColumnOperator );

    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = coupledOp;
    nonlinearSolverParams->d_pInitialGuess = solVec;
    auto nonlinearSolver = std::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams );

    auto linearSolver = nonlinearSolver->getKrylovSolver();
    linearSolver->setNestedSolver( pelletStackSolver );

    for ( int step = 0; step < NumberOfLoadingSteps; step++ ) {
        AMP::pout << "########################################" << std::endl;
        AMP::pout << "The current loading step is " << ( step + 1 ) << std::endl;

        double scaleValue =
            ( static_cast<double>( step + 1 ) ) / ( static_cast<double>( NumberOfLoadingSteps ) );
        scaledRhsVec->scale( scaleValue, *rhsVec );

        if ( useThermalLoad ) {
            auto initialTemp = db->getScalar<double>( "InitialTemperature" );
            auto finalTemp   = db->getScalar<double>( "FinalTemperature" );
            double deltaTemp =
                initialTemp + ( ( static_cast<double>( step + 1 ) ) * ( finalTemp - initialTemp ) /
                                ( static_cast<double>( NumberOfLoadingSteps ) ) );
            finalTemperatureVec->setToScalar( deltaTemp );
            setFinalTemperature( coupledOp, finalTemperatureVec );
        }

        auto resVec = solVec->clone();
        resVec->zero();
        coupledOp->residual( scaledRhsVec, solVec, resVec );
        AMP::pout << "initial, rhsVec: " << scaledRhsVec->L2Norm() << std::endl;
        AMP::pout << "initial, solVec: " << solVec->L2Norm() << std::endl;
        AMP::pout << "initial, resVec: " << resVec->L2Norm() << std::endl;
        nonlinearSolver->apply( scaledRhsVec, solVec );
        AMP::pout << "solved,  rhsVec: " << scaledRhsVec->L2Norm() << std::endl;
        AMP::pout << "solved,  solVec: " << solVec->L2Norm() << std::endl;
        coupledOp->residual( scaledRhsVec, solVec, resVec );
        AMP::pout << "final,   rhsVec: " << scaledRhsVec->L2Norm() << std::endl;
        AMP::pout << "final,   solVec: " << solVec->L2Norm() << std::endl;
        AMP::pout << "final,   resVec: " << resVec->L2Norm() << std::endl;

        resetNonlinearOperator( coupledOp );
    } // end for step

    ut->passes( exeName );
}

int testPelletStackMechanicsSolver( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    const char *exeName[] = { "testPelletStackMechanicsSolver-1",
                              "testPelletStackMechanicsSolver-2" };
    for ( auto exe : exeName )
        myTest( &ut, exe );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
