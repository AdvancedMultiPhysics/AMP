#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/ColumnBoundaryOperator.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrectionParameters.h"
#include "AMP/operators/boundary/libmesh/RobinMatrixCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/operators/map/MapOperatorParameters.h"
#include "AMP/operators/map/libmesh/Map1Dto3D.h"
#include "AMP/operators/map/libmesh/Map3Dto1D.h"
#include "AMP/operators/subchannel/FlowFrapconOperator.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <memory>
#include <string>


static void thermalContactTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP::logAllNodes( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    //  Create the Mesh.
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto manager = AMP::Mesh::MeshFactory::create( mgrParams );
    auto mesh1   = manager->Subset( "pellet" );
    auto mesh2   = manager->Subset( "clad" );

    // Create a DOF manager for a nodal vector
    using AMP::Discretization::simpleDOFManager;
    auto Vertex            = AMP::Mesh::GeomType::Vertex;
    auto Cell              = AMP::Mesh::GeomType::Cell;
    auto nodalDofMap       = simpleDOFManager::create( manager, Vertex, 1, 1 );
    auto nodalDofMap1      = simpleDOFManager::create( mesh1, Vertex, 1, 1 );
    auto nodalDofMap2      = simpleDOFManager::create( mesh2, Vertex, 1, 1 );
    auto gaussPointDofMap1 = simpleDOFManager::create( mesh1, Cell, 1, 8 );
    AMP::LinearAlgebra::VS_Mesh vectorSelector1( mesh1 );
    AMP::LinearAlgebra::VS_Mesh vectorSelector2( mesh2 );

    auto TemperatureVar = std::make_shared<AMP::LinearAlgebra::Variable>( "Temperature" );

    auto intguess = input_db->getWithDefault<double>( "InitialGuess", 400 );

    auto TemperatureInKelvin = AMP::LinearAlgebra::createVector( nodalDofMap, TemperatureVar );
    TemperatureInKelvin->setToScalar( intguess );


    //   CREATE THE NONLINEAR THERMAL OPERATOR 1
    AMP_INSIST( input_db->keyExists( "NonlinearThermalOperator1" ), "key missing!" );
    auto nonlinearThermalDatabase1 = input_db->getDatabase( "NonlinearThermalOperator1" );
    auto nonlinearThermalOperator1 = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh1, "NonlinearThermalOperator1", input_db ) );

    // initialize the input variable
    auto thermalVolumeOperator1 =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearThermalOperator1->getVolumeOperator() );

    // initialize the output variable
    auto outputVariable1 = thermalVolumeOperator1->getOutputVariable();

    auto TemperatureInKelvinVec1 = TemperatureInKelvin->select( vectorSelector1 );
    auto RightHandSideVec1 = AMP::LinearAlgebra::createVector( nodalDofMap1, outputVariable1 );
    auto ResidualVec1      = AMP::LinearAlgebra::createVector( nodalDofMap1, outputVariable1 );

    // CREATE THE LINEAR THERMAL OPERATOR 1
    auto linearThermalOperator1 = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh1, "LinearThermalOperator1", input_db ) );

    // CREATE THE NEUTRONICS SOURCE
    AMP_INSIST( input_db->keyExists( "NeutronicsOperator" ),
                "Key ''NeutronicsOperator'' is missing!" );
    auto neutronicsOp_db  = input_db->getDatabase( "NeutronicsOperator" );
    auto neutronicsParams = std::make_shared<AMP::Operator::OperatorParameters>( neutronicsOp_db );
    neutronicsParams->d_Mesh = mesh1;
    auto neutronicsOperator  = std::make_shared<AMP::Operator::NeutronicsRhs>( neutronicsParams );

    auto SpecificPowerVar = neutronicsOperator->getOutputVariable();
    auto SpecificPowerVec = AMP::LinearAlgebra::createVector( gaussPointDofMap1, SpecificPowerVar );

    neutronicsOperator->apply( nullptr, SpecificPowerVec );

    //  Integrate Nuclear Rhs over Desnity * GeomType::Cell //
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    auto sourceOperator = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh1, "VolumeIntegralOperator", input_db ) );

    // Create the power (heat source) vector.
    auto PowerInWattsVar = sourceOperator->getOutputVariable();
    auto PowerInWattsVec = AMP::LinearAlgebra::createVector( nodalDofMap1, PowerInWattsVar );
    PowerInWattsVec->zero();

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply( SpecificPowerVec, PowerInWattsVec );

    AMP_INSIST( input_db->keyExists( "NonlinearSolver" ), "Key ''NonlinearSolver'' is missing!" );

    auto nonlinearSolver_db1 = input_db->getDatabase( "NonlinearSolver" );
    auto linearSolver_db1    = nonlinearSolver_db1->getDatabase( "LinearSolver" );

    // initialize the nonlinear solver
    auto nonlinearSolverParams1 =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db1 );

    // change the next line to get the correct communicator out
    nonlinearSolverParams1->d_comm          = globalComm;
    nonlinearSolverParams1->d_pOperator     = nonlinearThermalOperator1;
    nonlinearSolverParams1->d_pInitialGuess = TemperatureInKelvinVec1;

    auto nonlinearSolver1 =
        std::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams1 );

    auto thermalPreconditioner_db1 = linearSolver_db1->getDatabase( "Preconditioner" );
    auto thermalPreconditionerParams1 =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( thermalPreconditioner_db1 );
    thermalPreconditionerParams1->d_pOperator = linearThermalOperator1;
    auto linearThermalPreconditioner1 =
        std::make_shared<AMP::Solver::TrilinosMLSolver>( thermalPreconditionerParams1 );

    // register the preconditioner with the Jacobian free Krylov solver
    auto linearSolver1 = nonlinearSolver1->getKrylovSolver();
    linearSolver1->setNestedSolver( linearThermalPreconditioner1 );
    nonlinearThermalOperator1->residual( RightHandSideVec1, TemperatureInKelvinVec1, ResidualVec1 );

    // CREATE THE CONTACT GAP OPERATOR
    AMP_INSIST( input_db->keyExists( "GapOperator" ), "Key ''GapOperator'' is missing!" );
    auto gapDatabase = input_db->getDatabase( "GapOperator" );

    auto heff        = ( gapDatabase )->getScalar<double>( "Convective_Coefficient" );
    auto gapVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "Gap" );

    // CREATE THE LINEAR THERMAL OPERATOR 2
    AMP_INSIST( input_db->keyExists( "LinearThermalOperator2" ), "key missing!" );

    auto linearThermalDatabase2 = input_db->getDatabase( "LinearThermalOperator2" );
    auto linearThermalOperator2 = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh2, "LinearThermalOperator2", input_db ) );

    auto thermalVolumeOperator2 =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>(
            linearThermalOperator2->getVolumeOperator() );

    // initialize the output variable
    auto outputVariable2 = thermalVolumeOperator2->getOutputVariable();

    auto TemperatureInKelvinVec2 = TemperatureInKelvin->select( vectorSelector2 );
    auto RightHandSideVec2 = AMP::LinearAlgebra::createVector( nodalDofMap2, outputVariable2 );
    auto ResidualVec2      = AMP::LinearAlgebra::createVector( nodalDofMap2, outputVariable2 );

    auto mlSolverParams2 =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db1 );
    mlSolverParams2->d_pOperator = linearThermalOperator2;
    auto mlSolver2 = std::make_shared<AMP::Solver::TrilinosMLSolver>( mlSolverParams2 );
    mlSolver2->setZeroInitialGuess( true );

    //-------------------------------------
    auto variableFluxVec1 = AMP::LinearAlgebra::createVector( nodalDofMap1, TemperatureVar );
    auto scratchTempVec1  = AMP::LinearAlgebra::createVector( nodalDofMap1, TemperatureVar );
    variableFluxVec1->setToScalar( 0.0 );

    auto variableFluxVec2 = AMP::LinearAlgebra::createVector( nodalDofMap2, TemperatureVar );
    auto scratchTempVec2  = AMP::LinearAlgebra::createVector( nodalDofMap2, TemperatureVar );
    variableFluxVec2->setToScalar( 0.0 );

    //-------------------------------------

    auto map3dto1d_db1    = input_db->getDatabase( "MapPelletto1D" );
    auto map3dto1dParams1 = std::make_shared<AMP::Operator::MapOperatorParameters>( map3dto1d_db1 );
    map3dto1dParams1->d_MapMesh = mesh1;
    auto map1ToLowDim           = std::make_shared<AMP::Operator::Map3Dto1D>( map3dto1dParams1 );

    auto map1dto3d_db1    = input_db->getDatabase( "Map1DtoClad" );
    auto map1dto3dParams1 = std::make_shared<AMP::Operator::MapOperatorParameters>( map1dto3d_db1 );
    map1dto3dParams1->d_MapMesh = mesh2;
    //-------------------------------------
    ut->passes( "Everything up till constructing 1Dto3D passes." );
    //-------------------------------------
    auto map1ToHighDim = std::make_shared<AMP::Operator::Map1Dto3D>( map1dto3dParams1 );

    map1ToLowDim->setZLocations( map1ToHighDim->getZLocations() );

    auto map3dto1d_db2    = input_db->getDatabase( "MapCladto1D" );
    auto map3dto1dParams2 = std::make_shared<AMP::Operator::MapOperatorParameters>( map3dto1d_db2 );
    map3dto1dParams2->d_MapMesh = mesh2;
    auto map2ToLowDim           = std::make_shared<AMP::Operator::Map3Dto1D>( map3dto1dParams2 );

    auto map1dto3d_db2    = input_db->getDatabase( "Map1DtoPellet" );
    auto map1dto3dParams2 = std::make_shared<AMP::Operator::MapOperatorParameters>( map1dto3d_db2 );
    map1dto3dParams2->d_MapMesh = mesh1;
    auto map2ToHighDim          = std::make_shared<AMP::Operator::Map1Dto3D>( map1dto3dParams2 );

    map2ToLowDim->setZLocations( map2ToHighDim->getZLocations() );

    // From flow operator test
    auto mapcladflow_db = input_db->getDatabase( "MapCladtoFlow" );
    auto mapcladflowParams =
        std::make_shared<AMP::Operator::MapOperatorParameters>( mapcladflow_db );
    mapcladflowParams->d_MapMesh = mesh2;
    auto mapCladToFlow           = std::make_shared<AMP::Operator::Map3Dto1D>( mapcladflowParams );

    auto mapflowclad_db = input_db->getDatabase( "MapFlowtoClad" );
    auto mapflowcladParams =
        std::make_shared<AMP::Operator::MapOperatorParameters>( mapflowclad_db );
    mapflowcladParams->d_MapMesh = mesh2;
    auto mapFlowToClad           = std::make_shared<AMP::Operator::Map1Dto3D>( mapflowcladParams );

    mapCladToFlow->setZLocations( mapFlowToClad->getZLocations() );

    unsigned int flowVecSize = mapFlowToClad->getNumZlocations();

    //     CREATE THE FLOW OPERATOR
    AMP_INSIST( input_db->keyExists( "FlowFrapconOperator" ),
                "Key ''FlowFrapconOperator'' is missing!" );

    auto flowDatabase = input_db->getDatabase( "FlowFrapconOperator" );
    flowDatabase->putScalar( "numPoints", flowVecSize );
    auto flowOperator = std::dynamic_pointer_cast<AMP::Operator::FlowFrapconOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh2, "FlowFrapconOperator", input_db ) );

    flowOperator->setZLocations( mapFlowToClad->getZLocations() );

    auto inputVariable  = flowOperator->getInputVariable();
    auto outputVariable = flowOperator->getOutputVariable();

    // double Cp, De, G, K, Re, Pr, hclad, dz, Tc, Tin;
    double De, K, Re, Pr, hclad;

    // Cp = flowDatabase->getScalar<double>("Heat_Capacity");
    De = flowDatabase->getScalar<double>( "Channel_Diameter" );
    // G = flowDatabase->getScalar<double>("Mass_Flux");
    K  = flowDatabase->getScalar<double>( "Conductivity" );
    Re = flowDatabase->getScalar<double>( "Reynolds" );
    Pr = flowDatabase->getScalar<double>( "Prandtl" );
    // Tin  = flowDatabase->getScalar<double>("Temp_Inlet");

    hclad = ( 0.023 * K / De ) * std::pow( Re, 0.8 ) * std::pow( Pr, 0.4 );

    // dz = 0.0127/flowVecSize ;

    auto solVec = AMP::LinearAlgebra::createSimpleVector<double>( flowVecSize, inputVariable );
    auto rhsVec = AMP::LinearAlgebra::createSimpleVector<double>( flowVecSize, outputVariable );
    auto resVec = AMP::LinearAlgebra::createSimpleVector<double>( flowVecSize, outputVariable );
    auto vecLag = AMP::LinearAlgebra::createSimpleVector<double>( flowVecSize, outputVariable );

    flowOperator->setVector( solVec );

    resVec->setToScalar( 350 );
    //-------------------------------------

    auto robinRHSVec = AMP::LinearAlgebra::createVector(
        nodalDofMap2, thermalVolumeOperator2->getInputVariable() );

    //------------------------------------------

    auto boundaryOp1 = nonlinearThermalOperator1->getBoundaryOperator();

    auto robinBoundaryOp1 =
        std::dynamic_pointer_cast<AMP::Operator::BoundaryOperator>( boundaryOp1 );

    auto boundaryDatabase1 =
        input_db->getDatabase( nonlinearThermalDatabase1->getString( "BoundaryOperator" ) );
    auto robinboundaryDatabase1 = boundaryDatabase1;

    robinboundaryDatabase1->putScalar( "constant_flux", false );
    robinboundaryDatabase1->putScalar( "skip_matrix_correction", true );
    auto correctionParameters1 = std::make_shared<AMP::Operator::NeumannVectorCorrectionParameters>(
        robinboundaryDatabase1 );

    //------------------------------------------

    auto boundaryOp2 = linearThermalOperator2->getBoundaryOperator();
    auto robinBoundaryOp2 =
        std::dynamic_pointer_cast<AMP::Operator::ColumnBoundaryOperator>( boundaryOp2 )
            ->getBoundaryOperator( 0 );
    auto robinBoundaryOp3 =
        std::dynamic_pointer_cast<AMP::Operator::ColumnBoundaryOperator>( boundaryOp2 )
            ->getBoundaryOperator( 1 );

    auto boundaryDatabase2 =
        input_db->getDatabase( nonlinearThermalDatabase1->getString( "BoundaryOperator" ) );
    auto robinboundaryDatabase2 = input_db->getDatabase( "RobinMatrixCorrection1" );
    auto robinboundaryDatabase3 = input_db->getDatabase( "RobinMatrixCorrection2" );

    robinboundaryDatabase2->putScalar( "constant_flux", false );
    robinboundaryDatabase2->putScalar( "skip_matrix_correction", true );
    auto correctionParameters2 =
        std::make_shared<AMP::Operator::RobinMatrixCorrectionParameters>( robinboundaryDatabase2 );

    robinboundaryDatabase3->putScalar( "constant_flux", false );
    robinboundaryDatabase3->putScalar( "skip_matrix_correction", true );
    auto correctionParameters3 =
        std::make_shared<AMP::Operator::RobinMatrixCorrectionParameters>( robinboundaryDatabase3 );


    //-------------------------------------

    size_t gapVecCladSize = map1ToHighDim->getNumZlocations();
    auto gapVecClad = AMP::LinearAlgebra::createSimpleVector<double>( gapVecCladSize, gapVariable );

    size_t gapVecPelletSize = map2ToHighDim->getNumZlocations();
    auto gapVecPellet =
        AMP::LinearAlgebra::createSimpleVector<double>( gapVecPelletSize, gapVariable );

    int cnt      = 0;
    auto vecLag1 = AMP::LinearAlgebra::createVector( nodalDofMap1, outputVariable1 );
    vecLag1->copyVector( TemperatureInKelvinVec1 );
    auto vecLag2 = AMP::LinearAlgebra::createVector( nodalDofMap2, outputVariable2 );
    vecLag2->copyVector( TemperatureInKelvinVec2 );

    bool testPassed = false;

    int maxIt = input_db->getWithDefault<int>( "max_iterations", 100 );

    while ( cnt < maxIt ) {
        cnt++;

        RightHandSideVec1->zero();
        RightHandSideVec2->zero();

        RightHandSideVec1->copyVector( PowerInWattsVec );
        AMP::pout << "PowerInWattsVec norm  inside loop = " << RightHandSideVec1->L2Norm() << "\n";

        map2ToLowDim->apply( TemperatureInKelvinVec2, gapVecPellet );
        map2ToHighDim->apply( gapVecPellet, scratchTempVec1 );

        scratchTempVec1->scale( heff );
        variableFluxVec1->copyVector( scratchTempVec1 );

        correctionParameters1->d_variableFlux = variableFluxVec1;
        robinBoundaryOp1->reset( correctionParameters1 );

        AMP::pout << "Variable flux1 norm inside loop : " << variableFluxVec1->L2Norm()
                  << std::endl;

        nonlinearThermalOperator1->modifyRHSvector( RightHandSideVec1 );
        nonlinearThermalOperator1->modifyInitialSolutionVector( TemperatureInKelvinVec1 );
        nonlinearSolver1->apply( RightHandSideVec1, TemperatureInKelvinVec1 );
        nonlinearThermalOperator1->residual(
            RightHandSideVec1, TemperatureInKelvinVec1, ResidualVec1 );

        AMP::pout << "Norm of TemperatureInKelvinVec1: " << TemperatureInKelvinVec1->L2Norm()
                  << std::endl;

        //------------------------------------------------------------

        mapCladToFlow->residual( nullptr, TemperatureInKelvinVec2, solVec );
        while ( true ) {
            flowOperator->residual( rhsVec, solVec, resVec );
            if ( ( resVec->L2Norm() - vecLag->L2Norm() ).abs() < .000005 * vecLag->L2Norm() )
                break;
            else
                AMP::pout << "for iteration cnt = " << cnt << " --> " << vecLag->L2Norm() << " "
                          << resVec->L2Norm() << std::endl;

            AMP::pout << "Intermediate Flow Solution " << std::endl;
            for ( unsigned int i = 0; i < flowVecSize; i++ ) {
                AMP::pout << " @i : " << i << " is " << resVec->getValueByLocalID( i );
            }
            AMP::pout << std::endl;
            vecLag->copyVector( resVec );
        }

        mapFlowToClad->residual( nullptr, resVec, robinRHSVec );

        robinRHSVec->scale( hclad );
        correctionParameters3->d_variableFlux = robinRHSVec;
        robinBoundaryOp3->reset( correctionParameters3 );


        //-----------------------------------------------
        map1ToLowDim->apply( TemperatureInKelvinVec1, gapVecClad );

        AMP::pout << "Norm of solVec after map1toLowDim: " << gapVecClad->L2Norm() << std::endl;

        map1ToHighDim->apply( gapVecClad, scratchTempVec2 );

        AMP::pout << "Norm of scratch2: " << scratchTempVec2->L2Norm() << std::endl;

        scratchTempVec2->scale( heff );
        variableFluxVec2->copyVector( scratchTempVec2 );

        correctionParameters2->d_variableFlux = variableFluxVec2;
        robinBoundaryOp2->reset( correctionParameters2 );

        AMP::pout << "Variable flux2 norm inside loop : " << variableFluxVec2->L2Norm()
                  << std::endl;

        linearThermalOperator2->modifyRHSvector( RightHandSideVec2 );
        linearThermalOperator2->residual(
            RightHandSideVec2, TemperatureInKelvinVec2, ResidualVec2 );
        mlSolver2->apply( RightHandSideVec2, TemperatureInKelvinVec2 );

        //------------------------------------------------------------

        AMP::pout << "Residual Norm on Pellet after " << cnt
                  << " iteration is : " << ResidualVec1->L2Norm() << std::endl;
        AMP::pout << "Residual Norm on Clad after " << cnt
                  << " iteration is : " << ResidualVec2->L2Norm() << std::endl;

        vecLag2->subtract( *TemperatureInKelvinVec2, *vecLag2 );

        if ( vecLag2->L2Norm() < 1.e-6 ) {
            testPassed = true;
            break;
        } else {
            AMP::pout << "for iteration cnt = " << cnt << " --> " << vecLag1->L2Norm() << " "
                      << vecLag2->L2Norm() << std::endl;
        }
        AMP::pout << std::endl;

        vecLag1->copyVector( TemperatureInKelvinVec1 );
        vecLag2->copyVector( TemperatureInKelvinVec2 );
    }

    //-------------------------------------

    if ( testPassed ) {
        ut->passes( "Seggregated solve of Composite Operator using control loop of Nonlinear "
                    "Thermal+Robin->Map->Gap->Map->Nonlinear Thermal+Robin ." );
    } else {
        ut->failure( "Seggregated solve of Composite Operator using control loop of Nonlinear "
                     "Thermal+Robin->Map->Gap->Map->Nonlinear Thermal+Robin ." );
    }

    input_db.reset();

    ut->passes( exeName );
}

int testNonlinearThermalContact_1DFlow( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    thermalContactTest( &ut, "testNonlinearThermalContactFlowPicard" );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
