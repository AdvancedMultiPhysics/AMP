#include "AMP/IO/PIO.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/OperatorBuilder.h"
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
    auto input_db          = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP::logAllNodes( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto manager = AMP::Mesh::MeshFactory::create( params );
    auto mesh    = manager->Subset( "bar" );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    // CREATE THE FLOW OPERATOR
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
    rhsVec->zero();

    auto flowJacobian = std::dynamic_pointer_cast<AMP::Operator::FlowFrapconJacobian>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "FlowFrapconJacobian", input_db ) );

    //  MANUFACTURE THE INPUT SOLUTION FROM PRESCRIBED FLOW SOLUTION
    double Tin  = 300;
    auto Cp     = flowDatabase->getScalar<double>( "Heat_Capacity" );
    auto De     = flowDatabase->getScalar<double>( "Channel_Diameter" );
    auto G      = flowDatabase->getScalar<double>( "Mass_Flux" );
    auto K      = flowDatabase->getScalar<double>( "Conductivity" );
    auto Re     = flowDatabase->getScalar<double>( "Reynolds" );
    auto Pr     = flowDatabase->getScalar<double>( "Prandtl" );
    auto nP     = flowDatabase->getScalar<double>( "numpoints" );
    double heff = ( 0.023 * K / De ) * std::pow( Re, 0.8 ) * std::pow( Pr, 0.4 );
    double dz   = 2. / nP;

    double val;
    std::cout << "Original Flow Solution  " << std::endl;
    for ( size_t i = 0; i < 10; i++ ) {
        val = Tin + i * 30;
        tmpVec->setValuesByLocalID( 1, &i, &val );
        //	  solVec->setValueByLocalID(i, Tin + i*30);
        std::cout << " @i : " << i << " is " << tmpVec->getValueByLocalID( i );
    }
    std::cout << std::endl;

    std::cout << "Original Norm: " << tmpVec->L2Norm() << std::endl;

    size_t idx = 0;
    val        = 300.0;
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

    solVec->setValuesByLocalID( 1, &idx, &val );
    solVec->makeConsistent();

    auto mv_view_solVec = AMP::LinearAlgebra::MultiVector::view( solVec, globalComm );
    auto mv_view_rhsVec = AMP::LinearAlgebra::MultiVector::view( rhsVec, globalComm );
    auto mv_view_tmpVec = AMP::LinearAlgebra::MultiVector::view( tmpVec, globalComm );

    flowOperator->setVector( cladVec );
    flowJacobian->setVector( cladVec );

    flowOperator->residual( rhsVec, solVec, resVec );
    flowJacobian->reset( flowOperator->getParameters( "Jacobian", mv_view_solVec ) );
    flowJacobian->residual( rhsVec, solVec, resVec );

    // initialize the jacobian solver
    auto JacobianSolver_db = input_db->getDatabase( "Flow1DSolver" );
    auto flowSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( JacobianSolver_db );
    flowSolverParams->d_pOperator = flowJacobian;
    auto flowJacobianSolver       = std::make_shared<AMP::Solver::Flow1DSolver>( flowSolverParams );

    // initialize the nonlinear solver
    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = flowOperator;
    nonlinearSolverParams->d_pInitialGuess = mv_view_tmpVec;
    auto nonlinearSolver = std::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams );

    // register the preconditioner with the Jacobian free Krylov solver
    auto linearSolver = nonlinearSolver->getKrylovSolver();
    linearSolver->setNestedSolver( flowJacobianSolver );

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

    double norm = static_cast<double>( resVec->L2Norm() );
    if ( norm > 0.01 ) {
        ut->failure( "Manufactured Solution verification test for 1D flow operator." );
    } else {
        ut->passes( "Manufactured Solution verification test for 1D flow operator." );
    }

    input_db.reset();

    ut->passes( exeName );
}


int testFlowSolver( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    flowTest( &ut, "testFlowSolver" );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
