#include "AMP/ampmesh/Mesh.h"
#include "AMP/ampmesh/MeshParameters.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/ColumnBoundaryOperator.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrectionParameters.h"
#include "AMP/operators/boundary/libmesh/RobinMatrixCorrection.h"
#include "AMP/operators/boundary/libmesh/RobinVectorCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"
#include "AMP/operators/libmesh/MassLinearElement.h"
#include "AMP/operators/libmesh/MassLinearFEOperator.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/operators/map/Map1Dto3D.h"
#include "AMP/operators/map/Map3Dto1D.h"
#include "AMP/operators/map/MapOperatorParameters.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolverParameters.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolverParameters.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/Writer.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static void thermalContactTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP::PIO::logAllNodes( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Create the Mesh.
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto manager = AMP::Mesh::Mesh::buildMesh( mgrParams );

    // Create a DOF manager for a nodal vector
    int DOFsPerNode          = 1;
    int DOFsPerElement       = 8;
    int nodalGhostWidth      = 1;
    int gaussPointGhostWidth = 1;
    bool split               = true;
    auto nodalDofMap         = AMP::Discretization::simpleDOFManager::create(
        manager, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

    auto meshAdapter1 = manager->Subset( "pellet" );
    auto meshAdapter2 = manager->Subset( "clad" );
    auto nodalDofMap1 = AMP::Discretization::simpleDOFManager::create(
        meshAdapter1, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    auto nodalDofMap2 = AMP::Discretization::simpleDOFManager::create(
        meshAdapter2, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    auto gaussPointDofMap1 = AMP::Discretization::simpleDOFManager::create(
        meshAdapter1, AMP::Mesh::GeomType::Volume, gaussPointGhostWidth, DOFsPerElement, split );
    AMP::LinearAlgebra::VS_Mesh vectorSelector1( meshAdapter1 );
    AMP::LinearAlgebra::VS_Mesh vectorSelector2( meshAdapter2 );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    auto TemperatureVar = std::make_shared<AMP::LinearAlgebra::Variable>( "Temperature" );

    double intguess = input_db->getWithDefault<double>( "InitialGuess", 400 );

    auto TemperatureInKelvin = AMP::LinearAlgebra::createVector( nodalDofMap, TemperatureVar );
    TemperatureInKelvin->setToScalar( intguess );


    //   CREATE THE NONLINEAR THERMAL OPERATOR 1
    AMP_INSIST( input_db->keyExists( "NonlinearThermalOperator1" ), "key missing!" );
    std::shared_ptr<AMP::Operator::ElementPhysicsModel> thermalTransportModel1;
    auto nonlinearThermalDatabase1 = input_db->getDatabase( "NonlinearThermalOperator1" );
    auto nonlinearThermalOperator1 = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter1, "NonlinearThermalOperator1", input_db, thermalTransportModel1 ) );

    // initialize the input variable
    auto thermalVolumeOperator1 =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearThermalOperator1->getVolumeOperator() );

    // initialize the output variable
    auto outputVariable1 = thermalVolumeOperator1->getOutputVariable();

    auto TemperatureInKelvinVec1 =
        TemperatureInKelvin->select( vectorSelector1, TemperatureVar->getName() );
    auto RightHandSideVec1 = AMP::LinearAlgebra::createVector( nodalDofMap1, outputVariable1 );
    auto ResidualVec1      = AMP::LinearAlgebra::createVector( nodalDofMap1, outputVariable1 );

    // CREATE THE LINEAR THERMAL OPERATOR 1
    std::shared_ptr<AMP::Operator::ElementPhysicsModel> transportModel1;
    std::shared_ptr<AMP::Operator::LinearBVPOperator> linearThermalOperator1 =
        std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                meshAdapter1, "LinearThermalOperator1", input_db, thermalTransportModel1 ) );

    // CREATE THE NEUTRONICS SOURCE
    AMP_INSIST( input_db->keyExists( "NeutronicsOperator" ),
                "Key ''NeutronicsOperator'' is missing!" );
    auto neutronicsOp_db = input_db->getDatabase( "NeutronicsOperator" );
    auto neutronicsParams =
        std::make_shared<AMP::Operator::NeutronicsRhsParameters>( neutronicsOp_db );
    neutronicsParams->d_Mesh = meshAdapter1;
    auto neutronicsOperator  = std::make_shared<AMP::Operator::NeutronicsRhs>( neutronicsParams );

    auto SpecificPowerVar = neutronicsOperator->getOutputVariable();
    auto SpecificPowerVec = AMP::LinearAlgebra::createVector( gaussPointDofMap1, SpecificPowerVar );

    neutronicsOperator->apply( nullVec, SpecificPowerVec );

    //  Integrate Nuclear Rhs over Desnity * GeomType::Volume //
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    std::shared_ptr<AMP::Operator::ElementPhysicsModel> stransportModel;
    auto sourceOperator = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter1, "VolumeIntegralOperator", input_db, stransportModel ) );

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
        std::make_shared<AMP::Solver::PetscSNESSolverParameters>( nonlinearSolver_db1 );

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
    std::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver1 =
        nonlinearSolver1->getKrylovSolver();
    linearSolver1->setPreconditioner( linearThermalPreconditioner1 );
    nonlinearThermalOperator1->residual( RightHandSideVec1, TemperatureInKelvinVec1, ResidualVec1 );

    // CREATE THE CONTACT GAP OPERATOR
    AMP_INSIST( input_db->keyExists( "GapOperator" ), "Key ''GapOperator'' is missing!" );
    auto gapDatabase =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "GapOperator" ) );

    double heff      = ( gapDatabase )->getScalar<double>( "Convective_Coefficient" );
    auto gapVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "Gap" );

    // CREATE THE LINEAR THERMAL OPERATOR 2
    AMP_INSIST( input_db->keyExists( "LinearThermalOperator2" ), "key missing!" );

    std::shared_ptr<AMP::Operator::ElementPhysicsModel> thermalTransportModel2;
    auto linearThermalOperator2 = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter2, "LinearThermalOperator2", input_db, thermalTransportModel2 ) );

    auto thermalVolumeOperator2 =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>(
            linearThermalOperator2->getVolumeOperator() );

    // initialize the output variable
    auto outputVariable2 = thermalVolumeOperator2->getOutputVariable();

    auto TemperatureInKelvinVec2 =
        TemperatureInKelvin->select( vectorSelector2, TemperatureVar->getName() );
    auto RightHandSideVec2 = AMP::LinearAlgebra::createVector( nodalDofMap2, outputVariable2 );
    auto ResidualVec2      = AMP::LinearAlgebra::createVector( nodalDofMap2, outputVariable2 );

    auto mlSolverParams2 =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db1 );
    mlSolverParams2->d_pOperator = linearThermalOperator2;
    std::shared_ptr<AMP::Solver::TrilinosMLSolver> mlSolver2 =
        std::make_shared<AMP::Solver::TrilinosMLSolver>( mlSolverParams2 );
    mlSolver2->setZeroInitialGuess( true );

    //-------------------------------------
    auto variableFluxVec1 = AMP::LinearAlgebra::createVector( nodalDofMap1, TemperatureVar );
    auto scratchTempVec1  = AMP::LinearAlgebra::createVector( nodalDofMap1, TemperatureVar );
    variableFluxVec1->setToScalar( 0.0 );

    auto variableFluxVec2 = AMP::LinearAlgebra::createVector( nodalDofMap2, TemperatureVar );
    auto scratchTempVec2  = AMP::LinearAlgebra::createVector( nodalDofMap2, TemperatureVar );
    variableFluxVec2->setToScalar( 0.0 );

    //-------------------------------------

    auto map3dto1d_db1 =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "MapPelletto1D" ) );
    auto map3dto1dParams1 = std::make_shared<AMP::Operator::MapOperatorParameters>( map3dto1d_db1 );
    map3dto1dParams1->d_Mesh = meshAdapter1;
    auto map1ToLowDim        = std::make_shared<AMP::Operator::Map3Dto1D>( map3dto1dParams1 );

    auto map1dto3d_db1 =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "Map1DtoClad" ) );
    auto map1dto3dParams1 = std::make_shared<AMP::Operator::MapOperatorParameters>( map1dto3d_db1 );
    map1dto3dParams1->d_Mesh = meshAdapter2;
    //-------------------------------------
    // This is related to But # 1219 and 1210.
    //  -- It dies in compute_Z_locations of the constructor for mat1dto3d.
    ut->passes( "Everything up till constructing 1Dto3D passes." );
    //-------------------------------------
    auto map1ToHighDim = std::make_shared<AMP::Operator::Map1Dto3D>( map1dto3dParams1 );

    map1ToLowDim->setZLocations( map1ToHighDim->getZLocations() );

    auto map3dto1d_db2 =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "MapCladto1D" ) );
    auto map3dto1dParams2 = std::make_shared<AMP::Operator::MapOperatorParameters>( map3dto1d_db2 );
    map3dto1dParams2->d_Mesh = meshAdapter2;
    auto map2ToLowDim        = std::make_shared<AMP::Operator::Map3Dto1D>( map3dto1dParams2 );

    auto map1dto3d_db2 =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "Map1DtoPellet" ) );
    auto map1dto3dParams2 = std::make_shared<AMP::Operator::MapOperatorParameters>( map1dto3d_db2 );
    map1dto3dParams2->d_Mesh = meshAdapter1;
    auto map2ToHighDim       = std::make_shared<AMP::Operator::Map1Dto3D>( map1dto3dParams2 );

    map2ToLowDim->setZLocations( map2ToHighDim->getZLocations() );

    //------------------------------------------

    auto boundaryOp1 = nonlinearThermalOperator1->getBoundaryOperator();
    auto robinBoundaryOp1 =
        ( std::dynamic_pointer_cast<AMP::Operator::BoundaryOperator>( boundaryOp1 ) );

    auto boundaryDatabase1 = std::dynamic_pointer_cast<AMP::Database>(
        input_db->getDatabase( nonlinearThermalDatabase1->getString( "BoundaryOperator" ) ) );
    //  std::shared_ptr<AMP::Database> robinboundaryDatabase1 =
    //  std::dynamic_pointer_cast<AMP::Database>(
    //  boundaryDatabase1->getDatabase("RobinVectorCorrection"));
    auto robinboundaryDatabase1 = std::dynamic_pointer_cast<AMP::Database>( boundaryDatabase1 );

    robinboundaryDatabase1->putScalar( "constant_flux", false );
    robinboundaryDatabase1->putScalar( "skip_matrix_correction", true );
    auto correctionParameters1 = std::make_shared<AMP::Operator::NeumannVectorCorrectionParameters>(
        robinboundaryDatabase1 );

    //------------------------------------------

    auto boundaryOp2 = linearThermalOperator2->getBoundaryOperator();
    auto robinBoundaryOp2 =
        ( std::dynamic_pointer_cast<AMP::Operator::ColumnBoundaryOperator>( boundaryOp2 ) )
            ->getBoundaryOperator( 0 );

    auto robinboundaryDatabase2 = std::dynamic_pointer_cast<AMP::Database>(
        input_db->getDatabase( "RobinMatrixCorrection" ) );

    robinboundaryDatabase2->putScalar( "constant_flux", false );
    robinboundaryDatabase2->putScalar( "skip_matrix_correction", true );
    auto correctionParameters2 =
        std::make_shared<AMP::Operator::RobinMatrixCorrectionParameters>( robinboundaryDatabase2 );


    //-------------------------------------

    size_t gapVecCladSize = map1ToHighDim->getNumZlocations();
    auto gapVecClad = AMP::LinearAlgebra::createSimpleVector<double>( gapVecCladSize, gapVariable );

    size_t gapVecPelletSize = map2ToHighDim->getNumZlocations();
    auto gapVecPellet =
        AMP::LinearAlgebra::createSimpleVector<double>( gapVecPelletSize, gapVariable );

    map2ToHighDim->setVector( scratchTempVec1 );
    map2ToLowDim->setVector( gapVecPellet );
    map1ToHighDim->setVector( scratchTempVec2 );
    map1ToLowDim->setVector( gapVecClad );

    int cnt      = 0;
    auto vecLag1 = AMP::LinearAlgebra::createVector( nodalDofMap1, outputVariable1 );
    vecLag1->copyVector( TemperatureInKelvinVec1 );
    auto vecLag2 = AMP::LinearAlgebra::createVector( nodalDofMap2, outputVariable2 );
    vecLag2->copyVector( TemperatureInKelvinVec2 );

    bool testPassed = false;

    int maxIt = input_db->getWithDefault( "max_iterations", 100 );

    while ( cnt < maxIt ) {
        cnt++;

        RightHandSideVec1->zero();
        RightHandSideVec2->zero();

        RightHandSideVec1->copyVector( PowerInWattsVec );
        std::cout << "PowerInWattsVec norm  inside loop = " << RightHandSideVec1->L2Norm() << "\n";

        map2ToLowDim->apply( TemperatureInKelvinVec2, gapVecPellet );
        map2ToHighDim->apply( gapVecPellet, scratchTempVec1 );

        scratchTempVec1->scale( heff );
        variableFluxVec1->copyVector( scratchTempVec1 );

        correctionParameters1->d_variableFlux = variableFluxVec1;
        robinBoundaryOp1->reset( correctionParameters1 );

        std::cout << "Variable flux1 norm inside loop : " << variableFluxVec1->L2Norm()
                  << std::endl;

        nonlinearThermalOperator1->modifyRHSvector( RightHandSideVec1 );
        nonlinearThermalOperator1->modifyInitialSolutionVector( TemperatureInKelvinVec1 );
        nonlinearSolver1->solve( RightHandSideVec1, TemperatureInKelvinVec1 );
        nonlinearThermalOperator1->residual(
            RightHandSideVec1, TemperatureInKelvinVec1, ResidualVec1 );

        std::cout << "Norm of TemperatureInKelvinVec1: " << TemperatureInKelvinVec1->L2Norm()
                  << std::endl;

        map1ToLowDim->apply( TemperatureInKelvinVec1, gapVecClad );

        std::cout << "Norm of solVec after map1toLowDim: " << gapVecClad->L2Norm() << std::endl;

        map1ToHighDim->apply( gapVecClad, scratchTempVec2 );

        std::cout << "Norm of scratch2: " << scratchTempVec2->L2Norm() << std::endl;

        scratchTempVec2->scale( heff );
        variableFluxVec2->copyVector( scratchTempVec2 );

        correctionParameters2->d_variableFlux = variableFluxVec2;
        robinBoundaryOp2->reset( correctionParameters2 );

        std::cout << "Variable flux2 norm inside loop : " << variableFluxVec2->L2Norm()
                  << std::endl;

        linearThermalOperator2->modifyRHSvector( RightHandSideVec2 );
        linearThermalOperator2->residual(
            RightHandSideVec2, TemperatureInKelvinVec2, ResidualVec2 );
        mlSolver2->solve( RightHandSideVec2, TemperatureInKelvinVec2 );

        std::cout << "Residual Norm on Pellet after " << cnt
                  << " iteration is : " << ResidualVec1->L2Norm() << std::endl;
        std::cout << "Residual Norm on Clad after " << cnt
                  << " iteration is : " << ResidualVec2->L2Norm() << std::endl;

        vecLag2->subtract( *TemperatureInKelvinVec2, *vecLag2 );

//          if( nodes == 2 ) {
#ifdef USE_EXT_SILO
        std::shared_ptr<AMP::Utilities::Writer> siloWriter =
            AMP::Utilities::Writer::buildWriter( "Silo" );

        siloWriter->registerVector(
            TemperatureInKelvin, manager, AMP::Mesh::GeomType::Vertex, "TemperatureInKelvin" );

        siloWriter->writeFile( input_file, 0 );
#endif
        //          }
        if ( vecLag2->L2Norm() < 1.e-6 ) {
            testPassed = true;
            break;
        } else {
            std::cout << "for iteration cnt = " << cnt << " --> " << vecLag1->L2Norm() << " "
                      << vecLag2->L2Norm() << std::endl;
        }
        std::cout << std::endl;

        vecLag1->copyVector( TemperatureInKelvinVec1 );
        vecLag2->copyVector( TemperatureInKelvinVec2 );
    }

    //-------------------------------------

    if ( testPassed ) {
        ut->passes( "Seggregated solve of Composite Operator using control loop of Nonlinear "
                    "Thermal+Robin->Map->Gap->Map->Ninlinear Thermal+Robin ." );
    } else {
        ut->failure( "Seggregated solve of Composite Operator using control loop of Nonlinear "
                     "Thermal+Robin->Map->Gap->Map->Ninlinear Thermal+Robin ." );
    }

    //} else {
    //  ut.expected_failure("parallel map3D-1D and map1D-3D fail in parallel, see bug #1219.");
    //}
    input_db.reset();

    ut->passes( exeName );
}

int testNonlinearThermalContact_1( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    thermalContactTest( &ut, "testNonlinearThermalContactPicard_HALDEN" );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
