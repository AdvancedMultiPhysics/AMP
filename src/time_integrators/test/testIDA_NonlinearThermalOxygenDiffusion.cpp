#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/FickSoretNonlinearFEOperator.h"
#include "AMP/operators/libmesh/MassLinearFEOperator.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/operators/mechanics/MechanicsLinearElement.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/time_integrators/ColumnTimeOperator.h"
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
#include "AMP/vectors/VectorSelector.h"

#include <memory>
#include <string>


static inline double init_thermal( double x, double y, double z )
{
    return 750.0 + 10000.0 * ( 0.5 + x ) * ( 0.5 - x ) * ( 0.5 + y ) * ( 0.5 - y ) * ( 0.5 + z ) *
                       ( 0.5 - z );
}
static inline double init_oxygen( double x, double y, double z )
{
    return 0.01 + ( 0.5 + x ) * ( 0.5 - x ) * ( 0.5 + y ) * ( 0.5 - y ) * ( 0.5 + z ) * ( 0.5 - z );
}


static void IDATimeIntegratorTest( AMP::UnitTest *ut )
{
    std::string input_file = "input_testIDA-NonlinearThermalOxygenDiffusion-1";
    std::string log_file   = "output_testIDA-NonlinearThermalOxygenDiffusion-1";

    AMP::logOnlyNodeZero( log_file );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode          = 1;
    int DOFsPerElement       = 8;
    int nodalGhostWidth      = 1;
    int gaussPointGhostWidth = 1;
    bool split               = true;
    auto nodalDofMap         = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, gaussPointGhostWidth, DOFsPerElement, split );

    // create two nonlinear thermal BVP operators
    AMP_INSIST( input_db->keyExists( "NonlinearThermalOperator" ), "key missing!" );
    AMP_INSIST( input_db->keyExists( "NonlinearOxygenOperator" ), "key missing!" );

    auto nonlinearThermalOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "NonlinearThermalOperator", input_db ) );

    auto nonlinearOxygenOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "NonlinearOxygenOperator", input_db ) );

    // create a column rhs operator object with the nonlinear thermal in it for use in the nonlinear
    // problem definition
    auto columnNonlinearRhsOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    columnNonlinearRhsOperator->append( nonlinearThermalOperator );
    columnNonlinearRhsOperator->append( nonlinearOxygenOperator );

    // create linear BVP operators
    auto linearThermalOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "LinearThermalOperator", input_db ) );
    auto linearOxygenOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "LinearOxygenOperator", input_db ) );
    // create a column rhs operator object with the linear thermal in it for use in the linear
    // problem definition
    auto columnLinearRhsOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    columnLinearRhsOperator->append( linearThermalOperator );
    columnLinearRhsOperator->append( linearOxygenOperator );

    // create a mass linear BVP operator
    auto massThermalOp = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "MassThermalOperator", input_db ) );
    auto massOxygenOp = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "MassOxygenOperator", input_db ) );

    // create a column mass operator object for use in the nonlinear and linear problem definition
    auto columnMassOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    columnMassOperator->append( massThermalOp );
    columnMassOperator->append( massOxygenOp );

    // create a  time operator for use in the preconditioner
    auto timeOperator_db = std::make_shared<AMP::Database>( "TimeOperatorDatabase" );
    timeOperator_db->putScalar( "CurrentDt", 0.01 );
    timeOperator_db->putScalar( "name", "TimeOperator" );
    timeOperator_db->putScalar( "bLinearMassOperator", true );
    timeOperator_db->putScalar( "bLinearRhsOperator", false );
    timeOperator_db->putScalar( "ScalingFactor", 1.0 / 0.01 );

    auto timeOperatorParameters =
        std::make_shared<AMP::TimeIntegrator::TimeOperatorParameters>( timeOperator_db );
    timeOperatorParameters->d_pRhsOperator  = columnLinearRhsOperator;
    timeOperatorParameters->d_pMassOperator = columnMassOperator;
    timeOperatorParameters->d_Mesh          = mesh;
    auto columnLinearTimeOperator =
        std::make_shared<AMP::TimeIntegrator::ColumnTimeOperator>( timeOperatorParameters );

    // create vectors for initial conditions (IC) and time derivative at IC
    auto thermalVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearThermalOperator->getVolumeOperator() );
    auto oxygenVolumeOperator =
        std::dynamic_pointer_cast<AMP::Operator::FickSoretNonlinearFEOperator>(
            nonlinearOxygenOperator->getVolumeOperator() );

    // note that the input variable for the time integrator and time operator will be a
    // multivariable
    auto inputVar = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "inputVariable" );
    inputVar->add( thermalVolumeOperator->getOutputVariable() );
    inputVar->add( oxygenVolumeOperator->getOutputVariable() );

    auto outputVar = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVariable>(
        columnNonlinearRhsOperator->getOutputVariable() );

    auto initialCondition      = AMP::LinearAlgebra::createVector( nodalDofMap, outputVar );
    auto initialConditionPrime = AMP::LinearAlgebra::createVector( nodalDofMap, outputVar );
    auto f                     = AMP::LinearAlgebra::createVector( nodalDofMap, outputVar );

    f->zero();

    //  create neutronics source
    AMP_INSIST( input_db->keyExists( "NeutronicsOperator" ),
                "Key ''NeutronicsOperator'' is missing!" );
    auto neutronicsOp_db  = input_db->getDatabase( "NeutronicsOperator" );
    auto neutronicsParams = std::make_shared<AMP::Operator::OperatorParameters>( neutronicsOp_db );
    neutronicsParams->d_Mesh = mesh;
    auto neutronicsOperator  = std::make_shared<AMP::Operator::NeutronicsRhs>( neutronicsParams );

    auto SpecificPowerVar = neutronicsOperator->getOutputVariable();
    auto SpecificPowerVec = AMP::LinearAlgebra::createVector( gaussPointDofMap, SpecificPowerVar );

    // create the following shared pointers for ease of use
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    neutronicsOperator->apply( nullVec, SpecificPowerVec );

    // Integrate Nuclear Rhs over Density * GeomType::Cell //
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    auto sourceOperator = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "VolumeIntegralOperator", input_db ) );

    // Create the power (heat source) vector.
    auto powerInWattsVar = sourceOperator->getOutputVariable();
    auto powerInWattsVec = AMP::LinearAlgebra::createVector( nodalDofMap, powerInWattsVar );
    powerInWattsVec->zero();

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply( SpecificPowerVec, powerInWattsVec );
    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // set initial conditions, initialize created vectors

    auto node     = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto end_node = node.end();

    AMP::LinearAlgebra::VS_Mesh vectorSelector1( mesh );
    AMP::LinearAlgebra::VS_Mesh vectorSelector2( mesh );

    auto thermalIC = initialCondition->select( vectorSelector1 );
    auto oxygenIC  = initialCondition->select( vectorSelector2 );
    int counter    = 0;
    for ( ; node != end_node; ++node ) {
        counter++;

        std::vector<size_t> gid;
        nodalDofMap->getDOFs( node->globalID(), gid );

        double px = ( node->coord() )[0];
        double py = ( node->coord() )[1];
        double pz = ( node->coord() )[2];

        double tval = init_thermal( px, py, pz );
        double oval = init_oxygen( px, py, pz );
        std::cout << "tval = " << tval << std::endl;
        std::cout << "oval = " << oval << std::endl;

        std::cout << "counter = " << counter << "bndGlobalIds.size() = " << gid.size() << std::endl;
        for ( auto &elem : gid ) {
            thermalIC->setValuesByGlobalID( 1, &elem, &tval );
            oxygenIC->setValuesByGlobalID( 1, &elem, &oval );
        } // end for i
    }     // end for node

    // ** please do not set the time derivative to be non-zero!!
    // ** as this causes trouble with the boundary - BP, 07/16/2010
    initialConditionPrime->zero();

    auto thermalRhs = f->select( vectorSelector1 );
    // create a copy of the rhs which can be modified at each time step (maybe)
    thermalRhs->copyVector( powerInWattsVec );
    // modify the rhs to take into account boundary conditions
    nonlinearThermalOperator->modifyRHSvector( f );
    nonlinearThermalOperator->modifyInitialSolutionVector( initialCondition );
    nonlinearOxygenOperator->modifyRHSvector( f );
    nonlinearOxygenOperator->modifyInitialSolutionVector( initialCondition );

    // create a preconditioner

    // get the ida database
    AMP_INSIST( input_db->keyExists( "IDATimeIntegrator" ),
                "Key ''IDATimeIntegrator'' is missing!" );
    auto ida_db = input_db->getDatabase( "IDATimeIntegrator" );
    // initialize the column preconditioner which is a diagonal block preconditioner
    auto columnPreconditioner_db = ida_db->getDatabase( "Preconditioner" );
    auto columnPreconditionerParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( columnPreconditioner_db );
    if ( !columnPreconditionerParams ) {
        ut->failure( "Testing SolverStrategyParameters's constructor: FAIL" );
    } else {
        ut->passes( "Testing SolverStrategyParameters's constructor: PASS" );
    }

    columnPreconditionerParams->d_pOperator = columnLinearTimeOperator;
    auto columnPreconditioner =
        std::make_shared<AMP::Solver::ColumnSolver>( columnPreconditionerParams );

    auto thermalPreconditioner_db = columnPreconditioner_db->getDatabase( "thermalPreconditioner" );
    auto oxygenPreconditioner_db  = columnPreconditioner_db->getDatabase( "oxygenPreconditioner" );
    auto thermalPreconditionerParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( thermalPreconditioner_db );
    auto oxygenPreconditionerParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( oxygenPreconditioner_db );
    thermalPreconditionerParams->d_pOperator = columnLinearTimeOperator->getOperator( 0 );
    oxygenPreconditionerParams->d_pOperator  = columnLinearTimeOperator->getOperator( 1 );
    auto linearThermalPreconditioner =
        std::make_shared<AMP::Solver::TrilinosMLSolver>( thermalPreconditionerParams );
    auto linearOxygenPreconditioner =
        std::make_shared<AMP::Solver::TrilinosMLSolver>( oxygenPreconditionerParams );

    columnPreconditioner->append( linearThermalPreconditioner );
    columnPreconditioner->append( linearOxygenPreconditioner );

    if ( !columnPreconditioner ) {
        ut->failure( "Testing column preconditioner's constructor: FAIL" );
    } else {
        ut->passes( "Testing column preconditioner's constructor: PASS" );
    }

    // create the IDA time integrator
    auto time_Params = std::make_shared<AMP::TimeIntegrator::IDATimeIntegratorParameters>( ida_db );

    if ( !time_Params ) {
        ut->failure( "Testing IDATimeIntegratorParameters' Constructor" );
    } else {
        ut->passes( "Testing IDATimeIntegratorParameters' Constructor" );
    }

    time_Params->d_pMassOperator = columnMassOperator;
    time_Params->d_operator      = columnNonlinearRhsOperator;
    // time_Params->d_pNestedSolver = columnPreconditioner;

    time_Params->d_ic_vector       = initialCondition;
    time_Params->d_ic_vector_prime = initialConditionPrime;

    time_Params->d_pSourceTerm = f;
    // time_Params->d_object_name = "IDATimeIntegratorParameters";

    std::cout << "Before IDATimeIntegrator" << std::endl;
    auto pIDATimeIntegrator =
        std::make_shared<AMP::TimeIntegrator::IDATimeIntegrator>( time_Params );

    if ( !pIDATimeIntegrator ) {
        ut->failure( "Testing IDATimeIntegrator's constructor" );
    } else {
        ut->passes( "Tested IDATimeIntegrator's constructor" );
    }

    // step in time
    int j = 1;
    while ( pIDATimeIntegrator->getCurrentTime() < pIDATimeIntegrator->getFinalTime() ) {
        auto v = pIDATimeIntegrator->getSolution();
        int retval =
            pIDATimeIntegrator->advanceSolution( pIDATimeIntegrator->getCurrentDt(), false, v, v );
        // pIDATimeIntegrator->updateSolution();
        double current_time = pIDATimeIntegrator->getCurrentTime();

        std::cout << j++ << "-th timestep" << std::endl;
        if ( retval == 0 ) {
            ut->passes( "Testing IDATimeIntegrator's advanceSolution. PASS!!" );
        } else {
            ut->failure( "Tested IDATimeIntegrator's advanceSolution. FAIL!!" );
        }

        auto currentSolution = pIDATimeIntegrator->getSolution();
        auto currentThermalSolution =
            currentSolution->subsetVectorForVariable( thermalVolumeOperator->getOutputVariable() );
        auto currentOxygenSolution =
            currentSolution->subsetVectorForVariable( oxygenVolumeOperator->getOutputVariable() );

        auto maxT = currentThermalSolution->max();
        auto minT = currentThermalSolution->min();
        auto maxO = currentOxygenSolution->max();
        auto minO = currentOxygenSolution->min();

        std::cout << "current_time = " << current_time << std::endl;
        std::cout << "max val of the current thermal solution = " << maxT << std::endl;
        std::cout << "min val of the current thermal solution = " << minT << std::endl;
        std::cout << "max val of the current oxygen solution = " << maxO << std::endl;
        std::cout << "min val of the current oxygen solution = " << minO << std::endl;
    }


    AMP::AMPManager::shutdown();

    if ( ut->NumFailLocal() == 0 ) {
        ut->passes( "testIDATimeIntegrator successful" );
    }
}


//---------------------------------------------------------------------------//

int testIDA_NonlinearThermalOxygenDiffusion( int argc, char *argv[] )
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
