#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/libmesh/MassLinearFEOperator.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/time_integrators/sundials/IDATimeIntegrator.h"
#include "AMP/time_integrators/sundials/IDATimeOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static inline double fun( double x, double y, double z )
{
    return ( 750.0 + 10000.0 * ( 0.5 + x ) * ( 0.5 - x ) * ( 0.5 + y ) * ( 0.5 - y ) * ( 0.5 + z ) *
                         ( 0.5 - z ) );
}


static void IDATimeIntegratorTest( AMP::UnitTest *ut )
{
    std::string input_file = "input_testIDA-NonlinearBVPOperator-1";
    std::string log_file   = "output_testIDA-NonlinearBVPOperator-1";

    AMP::logOnlyNodeZero( log_file );


    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto db = AMP::Database::parseInputFile( input_file );
    db->print( AMP::plog );

    AMP_INSIST( db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode          = 1;
    int DOFsPerElement       = 8;
    int nodalGhostWidth      = 1;
    int gaussPointGhostWidth = 1;
    bool split               = true;

    auto nodalDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, gaussPointGhostWidth, DOFsPerElement, split );

    // create a nonlinear BVP operator for nonlinear BVP operator
    AMP_INSIST( db->keyExists( "NonlinearOperator" ), "key missing!" );

    auto nonlinearOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "NonlinearOperator", db ) );

    auto outputVar = nonlinearOperator->getOutputVariable();
    // create a linear BVP operator
    auto linearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "LinearOperator", db ) );

    // create a mass linear BVP operator
    auto massOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "MassLinearOperator", db ) );

    //  create neutronics source
    AMP_INSIST( db->keyExists( "NeutronicsOperator" ), "Key ''NeutronicsOperator'' is missing!" );
    auto neutronicsOp_db  = db->getDatabase( "NeutronicsOperator" );
    auto neutronicsParams = std::make_shared<AMP::Operator::OperatorParameters>( neutronicsOp_db );
    auto neutronicsOperator = std::make_shared<AMP::Operator::NeutronicsRhs>( neutronicsParams );

    auto SpecificPowerVar = neutronicsOperator->getOutputVariable();
    auto SpecificPowerVec = AMP::LinearAlgebra::createVector( gaussPointDofMap, SpecificPowerVar );

    // create the following shared pointers for ease of use
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    neutronicsOperator->apply( nullVec, SpecificPowerVec );

    //  Integrate Nuclear Rhs over Density * GeomType::Cell //

    AMP_INSIST( db->keyExists( "VolumeIntegralOperator" ), "key missing!" );

    auto sourceOperator = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "VolumeIntegralOperator", db ) );

    // Create the power (heat source) vector.
    auto powerInWattsVar = sourceOperator->getOutputVariable();
    auto powerInWattsVec = AMP::LinearAlgebra::createVector( nodalDofMap, powerInWattsVar );
    powerInWattsVec->zero();

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply( SpecificPowerVec, powerInWattsVec );

    // create vectors for initial conditions (IC) and time derivative at IC
    auto initialCondition      = AMP::LinearAlgebra::createVector( nodalDofMap, outputVar );
    auto initialConditionPrime = AMP::LinearAlgebra::createVector( nodalDofMap, outputVar );
    auto f                     = AMP::LinearAlgebra::createVector( nodalDofMap, outputVar );

    // set initial conditions, initialize created vectors
    auto node     = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto end_node = node.end();
    int counter   = 0;
    for ( ; node != end_node; ++node ) {
        counter++;

        std::vector<size_t> bndGlobalIds;
        nodalDofMap->getDOFs( node->globalID(), bndGlobalIds );

        auto pt   = node->coord();
        double px = pt[0];
        double py = pt[1];
        double pz = pt[2];

        double val = fun( px, py, pz );

        for ( auto &bndGlobalId : bndGlobalIds ) {
            initialCondition->setValuesByGlobalID( 1, &bndGlobalId, &val );
            // ** please do not set the time derivative to be non-zero!!
            // ** as this causes trouble with the boundary - BP, 07/16/2010
            const double zero = 0.0;
            initialConditionPrime->setValuesByGlobalID( 1, &bndGlobalId, &zero );
        }
    }
    initialCondition->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    initialConditionPrime->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    std::cout << "With Counter " << counter << " Max initial temp " << initialCondition->max()
              << " Min initial temp " << initialCondition->min() << std::endl;

    // create a copy of the rhs which can be modified at each time step (maybe)
    f->copyVector( powerInWattsVec );
    // modify the rhs to take into account boundary conditions
    nonlinearOperator->modifyRHSvector( f );
    nonlinearOperator->modifyInitialSolutionVector( initialCondition );

    // create a linear time operator
    auto timeOperator_db = std::make_shared<AMP::Database>( "TimeOperatorDatabase" );
    timeOperator_db->putScalar<double>( "CurrentDt", 0.01 );
    timeOperator_db->putScalar<std::string>( "name", "TimeOperator" );
    timeOperator_db->putScalar<bool>( "bLinearMassOperator", true );
    timeOperator_db->putScalar<bool>( "bLinearRhsOperator", false );
    timeOperator_db->putScalar<double>( "ScalingFactor", 1.0 / 0.01 );
    timeOperator_db->putScalar<double>( "CurrentTime", .0 );

    auto timeOperatorParameters =
        std::make_shared<AMP::TimeIntegrator::TimeOperatorParameters>( timeOperator_db );
    timeOperatorParameters->d_pRhsOperator  = linearOperator;
    timeOperatorParameters->d_pMassOperator = massOperator;
    // timeOperatorParameters->d_pMassOperator = massLinearOperator;
    auto linearTimeOperator =
        std::make_shared<AMP::TimeIntegrator::LinearTimeOperator>( timeOperatorParameters );

    auto residualVec = AMP::LinearAlgebra::createVector( nodalDofMap, outputVar );

    linearOperator->apply( initialCondition, residualVec );
    std::cout << "Residual Norm of linearTimeOp apply : " << residualVec->L2Norm() << std::endl;

    massOperator->apply( initialCondition, residualVec );
    std::cout << "Residual Norm of linearTimeOp apply : " << residualVec->L2Norm() << std::endl;

    linearTimeOperator->apply( initialCondition, residualVec );
    std::cout << "Residual Norm of linearTimeOp apply : " << residualVec->L2Norm() << std::endl;

    // create a preconditioner

    // get the ida database
    AMP_INSIST( db->keyExists( "IDATimeIntegrator" ), "Key ''IDATimeIntegrator'' is missing!" );
    auto ida_db         = db->getDatabase( "IDATimeIntegrator" );
    auto pcSolver_db    = ida_db->getDatabase( "Preconditioner" );
    auto pcSolverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( pcSolver_db );
    pcSolverParams->d_pOperator = linearTimeOperator;

    if ( !pcSolverParams ) {
        ut->failure( "Testing SolverStrategyParameters's constructor: FAIL" );
    } else {
        ut->passes( "Testing SolverStrategyParameters's constructor: PASS" );
    }

    auto pcSolver = std::make_shared<AMP::Solver::TrilinosMLSolver>( pcSolverParams );

    if ( !pcSolver ) {
        ut->failure( "Testing TrilinosMLSolver's constructor: FAIL" );
    } else {
        ut->passes( "Testing TrilinosMLSolver's constructor: PASS" );
    }

    // create the IDA time integrator
    auto time_Params = std::make_shared<AMP::TimeIntegrator::IDATimeIntegratorParameters>( ida_db );

    if ( !time_Params ) {
        ut->failure( "Testing IDATimeIntegratorParameters' Constructor" );
    } else {
        ut->passes( "Testing IDATimeIntegratorParameters' Constructor" );
    }

    time_Params->d_pMassOperator = massOperator;
    // time_Params->d_pMassOperator = massLinearOperator;
    time_Params->d_operator      = nonlinearOperator;
    time_Params->d_pNestedSolver = pcSolver;

    time_Params->d_ic_vector       = initialCondition;
    time_Params->d_ic_vector_prime = initialConditionPrime;

    time_Params->d_pSourceTerm = f;
    time_Params->d_name        = "IDATimeIntegratorParameters";

    std::cout << "Before IDATimeIntegrator" << std::endl;
    auto pIDATimeIntegrator =
        std::make_shared<AMP::TimeIntegrator::IDATimeIntegrator>( time_Params );

    if ( !pIDATimeIntegrator ) {
        ut->failure( "Testing IDATimeIntegrator's constructor" );
    } else {
        ut->passes( "Tested IDATimeIntegrator's constructor" );
    }

    // ---------------------------------------------------------------------------------------
    // step in time
    double current_time = 0;
    int j               = 1;
    while ( pIDATimeIntegrator->getCurrentTime() < pIDATimeIntegrator->getFinalTime() ) {
        auto v = pIDATimeIntegrator->getSolution();
        int retval =
            pIDATimeIntegrator->advanceSolution( pIDATimeIntegrator->getCurrentDt(), false, v, v );
        // pIDATimeIntegrator->updateSolution();
        current_time = pIDATimeIntegrator->getCurrentTime();

        std::cout << j++ << "-th timestep" << std::endl;
        if ( retval == 0 ) {
            ut->passes( "Testing IDATimeIntegrator's advanceSolution. PASS!!" );
        } else {
            ut->failure( "Tested IDATimeIntegrator's advanceSolution. FAIL!!" );
        }

        auto max = pIDATimeIntegrator->getSolution()->max();
        auto min = pIDATimeIntegrator->getSolution()->min();

        std::cout << "current_time = " << current_time << std::endl;
        std::cout << "max val of the current solution = " << max << std::endl;
        std::cout << "min val of the current solution = " << min << std::endl;
    }

    auto pSolution = pIDATimeIntegrator->getSolution();

    if ( ut->NumFailLocal() == 0 ) {
        ut->passes( "testIDATimeIntegrator successful" );
    }
}


//---------------------------------------------------------------------------//

int testIDA_NonlinearBVPOperator( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    IDATimeIntegratorTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}

//---------------------------------------------------------------------------//
//                        end of SundialsVectorTest.cc
//---------------------------------------------------------------------------//
