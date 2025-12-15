#include "AMP/IO/PIO.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/operators/subchannel/FlowFrapconJacobian.h"
#include "AMP/operators/subchannel/FlowFrapconOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/libmesh/Flow1DSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static void flowTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    AMP::logAllNodes( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );

    // Get the Mesh database and create the mesh parameters
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto database   = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( database );
    meshParams->setComm( globalComm );

    // Create the meshes from the input database
    auto manager = AMP::Mesh::MeshFactory::create( meshParams );
    auto mesh    = manager->Subset( "bar" );


    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    // CREATE THE FLOW OPERATOR
    AMP_INSIST( input_db->keyExists( "FlowFrapconOperator" ),
                "Key ''FlowFrapconOperator'' is missing!" );

    auto flowDatabase = input_db->getDatabase( "FlowFrapconOperator" );
    auto flowOperator = std::dynamic_pointer_cast<AMP::Operator::FlowFrapconOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "FlowFrapconOperator", input_db ) );

    auto inputVariable  = flowOperator->getInputVariable();
    auto outputVariable = flowOperator->getOutputVariable();

    auto solVec  = AMP::LinearAlgebra::createSimpleVector<double>( 10, outputVariable );
    auto cladVec = AMP::LinearAlgebra::createSimpleVector<double>( 10, inputVariable );

    auto rhsVec = AMP::LinearAlgebra::createSimpleVector<double>( 10, outputVariable );
    auto resVec = AMP::LinearAlgebra::createSimpleVector<double>( 10, outputVariable );
    auto tmpVec = AMP::LinearAlgebra::createSimpleVector<double>( 10, inputVariable );

    auto flowJacobian = std::dynamic_pointer_cast<AMP::Operator::FlowFrapconJacobian>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "FlowFrapconJacobian", input_db ) );

    //    MANUFACTURE THE INPUT SOLUTION
    //     FROM PRESCRIBED FLOW SOLUTION

    // int cnt=0;
    double Tin = 300;
    double Cp, De, G, K, Re, Pr, heff, dz, nP;

    Cp = ( flowDatabase )->getScalar<double>( "Heat_Capacity" );
    De = ( flowDatabase )->getScalar<double>( "Channel_Diameter" );
    G  = ( flowDatabase )->getScalar<double>( "Mass_Flux" );
    K  = ( flowDatabase )->getScalar<double>( "Conductivity" );
    Re = ( flowDatabase )->getScalar<double>( "Reynolds" );
    Pr = ( flowDatabase )->getScalar<double>( "Prandtl" );
    nP = ( flowDatabase )->getScalar<double>( "numpoints" );

    heff = ( 0.023 * K / De ) * std::pow( Re, 0.8 ) * std::pow( Pr, 0.4 );
    dz   = 2. / nP;

    std::cout << "Original Flow Solution  " << std::endl;
    for ( size_t i = 0; i < 10; i++ ) {
        double val = Tin + i * 30;
        tmpVec->setValuesByLocalID( 1, &i, &val );
        //	  solVec->setValueByLocalID(i, Tin + i*30);
        std::cout << " @i : " << i << " is " << tmpVec->getValueByLocalID( i );
    }
    std::cout << std::endl;

    std::cout << "Original Norm: " << tmpVec->L2Norm() << std::endl;

    size_t idx = 0;
    double val = 300.0;
    cladVec->setValuesByLocalID( 1, &idx, &val );
    for ( size_t i = 1; i < 10; i++ ) {
        double Tc = tmpVec->getValueByLocalID( i ) +
                    ( tmpVec->getValueByLocalID( i ) - tmpVec->getValueByLocalID( i - 1 ) ) *
                        ( ( Cp * G * De ) / ( 4. * heff * dz ) );
        cladVec->setValuesByLocalID( 1, &i, &Tc );
    }

    std::cout << "Imposed Clad Solution  " << std::endl;
    for ( int i = 0; i < 10; i++ ) {
        std::cout << " @i : " << i << " is " << cladVec->getValueByLocalID( i );
    }
    std::cout << std::endl;

    flowOperator->setVector( cladVec );

    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );

    auto mv_view_solVec = AMP::LinearAlgebra::MultiVector::view( solVec, globalComm );
    auto mv_view_rhsVec = AMP::LinearAlgebra::MultiVector::view( rhsVec, globalComm );
    auto mv_view_tmpVec = AMP::LinearAlgebra::MultiVector::view( tmpVec, globalComm );

    flowOperator->residual( rhsVec, solVec, resVec );

    // initialize the nonlinear solver
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );

    // change the next line to get the correct communicator out
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = flowOperator;
    nonlinearSolverParams->d_pInitialGuess = mv_view_tmpVec;

    auto nonlinearSolver = std::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams );

    flowOperator->residual( rhsVec, solVec, resVec );

    AMP::pout << "Initial Residual Norm: " << resVec->L2Norm() << std::endl;

    nonlinearSolver->setZeroInitialGuess( true );

    nonlinearSolver->apply( mv_view_rhsVec, mv_view_solVec );

    flowOperator->residual( rhsVec, solVec, resVec );

    std::cout << "Final Residual Norm: " << resVec->L2Norm() << std::endl;

    std::cout << "Final Flow Solution " << std::endl;
    for ( int i = 0; i < 10; i++ ) {
        std::cout << " @i : " << i << " is " << solVec->getValueByLocalID( i );
    }
    std::cout << std::endl;

    if ( resVec->L2Norm() > 0.01 ) {
        ut->failure( "Manufactured Solution verification test for 1D flow operator." );
    } else {
        ut->passes( "Manufactured Solution verification test for 1D flow operator." );
    }

    input_db.reset();

    ut->passes( exeName );
}


int testFlowSolution( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    flowTest( &ut, "testFlowSolution" );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
