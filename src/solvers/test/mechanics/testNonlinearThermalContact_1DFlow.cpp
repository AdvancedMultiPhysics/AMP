#include "AMP/ampmesh/Mesh.h"
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
#include "AMP/operators/subchannel/FlowFrapconOperator.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolverParameters.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolverParameters.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/Writer.h"
#include "AMP/vectors/SimpleVector.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include <memory>

#include <string>


static void thermalContactTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    //  AMP::AMPManager::startup();
    //  AMP::Materials::initialize();


    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP::PIO::logAllNodes( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    //--------------------------------------------------
    //   Create the Mesh.
    //--------------------------------------------------
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    std::shared_ptr<AMP::Database> mesh_db = input_db->getDatabase( "Mesh" );
    std::shared_ptr<AMP::Mesh::MeshParameters> mgrParams(
        new AMP::Mesh::MeshParameters( mesh_db ) );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    std::shared_ptr<AMP::Mesh::Mesh> manager = AMP::Mesh::Mesh::buildMesh( mgrParams );
    //--------------------------------------------------

    //--------------------------------------------------
    // Create a DOF manager for a nodal vector
    //--------------------------------------------------
    int DOFsPerNode          = 1;
    int DOFsPerElement       = 8;
    int nodalGhostWidth      = 1;
    int gaussPointGhostWidth = 1;
    bool split               = true;
    AMP::Discretization::DOFManager::shared_ptr nodalDofMap =
        AMP::Discretization::simpleDOFManager::create(
            manager, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    //--------------------------------------------------

    AMP::Mesh::Mesh::shared_ptr meshAdapter1 = manager->Subset( "pellet" );
    AMP::Mesh::Mesh::shared_ptr meshAdapter2 = manager->Subset( "clad" );
    AMP::Discretization::DOFManager::shared_ptr nodalDofMap1 =
        AMP::Discretization::simpleDOFManager::create(
            meshAdapter1, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    AMP::Discretization::DOFManager::shared_ptr nodalDofMap2 =
        AMP::Discretization::simpleDOFManager::create(
            meshAdapter2, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    AMP::Discretization::DOFManager::shared_ptr gaussPointDofMap1 =
        AMP::Discretization::simpleDOFManager::create( meshAdapter1,
                                                       AMP::Mesh::GeomType::Volume,
                                                       gaussPointGhostWidth,
                                                       DOFsPerElement,
                                                       split );
    AMP::LinearAlgebra::VS_Mesh vectorSelector1( meshAdapter1 );
    AMP::LinearAlgebra::VS_Mesh vectorSelector2( meshAdapter2 );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    AMP::LinearAlgebra::Variable::shared_ptr TemperatureVar(
        new AMP::LinearAlgebra::Variable( "Temperature" ) );

    double intguess = input_db->getWithDefault<double>( "InitialGuess", 400.0 );

    AMP::LinearAlgebra::Vector::shared_ptr TemperatureInKelvin =
        AMP::LinearAlgebra::createVector( nodalDofMap, TemperatureVar );
    TemperatureInKelvin->setToScalar( intguess );

    //-----------------------------------------------
    //   CREATE THE NONLINEAR THERMAL OPERATOR 1 ----
    //-----------------------------------------------

    AMP_INSIST( input_db->keyExists( "NonlinearThermalOperator1" ), "key missing!" );

    std::shared_ptr<AMP::Operator::ElementPhysicsModel> thermalTransportModel1;
    std::shared_ptr<AMP::Database> nonlinearThermalDatabase1 =
        input_db->getDatabase( "NonlinearThermalOperator1" );
    std::shared_ptr<AMP::Operator::NonlinearBVPOperator> nonlinearThermalOperator1 =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                meshAdapter1, "NonlinearThermalOperator1", input_db, thermalTransportModel1 ) );

    // initialize the input variable
    std::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperator> thermalVolumeOperator1 =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nonlinearThermalOperator1->getVolumeOperator() );

    // initialize the output variable
    AMP::LinearAlgebra::Variable::shared_ptr outputVariable1 =
        thermalVolumeOperator1->getOutputVariable();

    AMP::LinearAlgebra::Vector::shared_ptr TemperatureInKelvinVec1 =
        TemperatureInKelvin->select( vectorSelector1, TemperatureVar->getName() );
    AMP::LinearAlgebra::Vector::shared_ptr RightHandSideVec1 =
        AMP::LinearAlgebra::createVector( nodalDofMap1, outputVariable1 );
    AMP::LinearAlgebra::Vector::shared_ptr ResidualVec1 =
        AMP::LinearAlgebra::createVector( nodalDofMap1, outputVariable1 );

    //-------------------------------------
    //   CREATE THE LINEAR THERMAL OPERATOR 1 ----
    //-------------------------------------

    std::shared_ptr<AMP::Operator::ElementPhysicsModel> transportModel1;
    std::shared_ptr<AMP::Operator::LinearBVPOperator> linearThermalOperator1 =
        std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                meshAdapter1, "LinearThermalOperator1", input_db, transportModel1 ) );

    //-------------------------------------
    //  CREATE THE NEUTRONICS SOURCE  //
    //-------------------------------------
    AMP_INSIST( input_db->keyExists( "NeutronicsOperator" ),
                "Key ''NeutronicsOperator'' is missing!" );
    std::shared_ptr<AMP::Database> neutronicsOp_db = input_db->getDatabase( "NeutronicsOperator" );
    std::shared_ptr<AMP::Operator::NeutronicsRhsParameters> neutronicsParams(
        new AMP::Operator::NeutronicsRhsParameters( neutronicsOp_db ) );
    neutronicsParams->d_Mesh = meshAdapter1;
    std::shared_ptr<AMP::Operator::NeutronicsRhs> neutronicsOperator(
        new AMP::Operator::NeutronicsRhs( neutronicsParams ) );

    AMP::LinearAlgebra::Variable::shared_ptr SpecificPowerVar =
        neutronicsOperator->getOutputVariable();
    AMP::LinearAlgebra::Vector::shared_ptr SpecificPowerVec =
        AMP::LinearAlgebra::createVector( gaussPointDofMap1, SpecificPowerVar );

    neutronicsOperator->apply( nullVec, SpecificPowerVec );

    //----------------------------------------------------------
    //  Integrate Nuclear Rhs over Desnity * GeomType::Volume //
    //----------------------------------------------------------

    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );

    std::shared_ptr<AMP::Operator::ElementPhysicsModel> stransportModel;
    std::shared_ptr<AMP::Operator::VolumeIntegralOperator> sourceOperator =
        std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                meshAdapter1, "VolumeIntegralOperator", input_db, stransportModel ) );

    // Create the power (heat source) vector.
    AMP::LinearAlgebra::Variable::shared_ptr PowerInWattsVar = sourceOperator->getOutputVariable();
    AMP::LinearAlgebra::Vector::shared_ptr PowerInWattsVec =
        AMP::LinearAlgebra::createVector( nodalDofMap1, PowerInWattsVar );
    PowerInWattsVec->zero();

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply( SpecificPowerVec, PowerInWattsVec );

    //--------------------------------------

    AMP_INSIST( input_db->keyExists( "NonlinearSolver" ), "Key ''NonlinearSolver'' is missing!" );

    std::shared_ptr<AMP::Database> nonlinearSolver_db1 = input_db->getDatabase( "NonlinearSolver" );
    std::shared_ptr<AMP::Database> linearSolver_db1 =
        nonlinearSolver_db1->getDatabase( "LinearSolver" );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // initialize the nonlinear solver
    std::shared_ptr<AMP::Solver::PetscSNESSolverParameters> nonlinearSolverParams1(
        new AMP::Solver::PetscSNESSolverParameters( nonlinearSolver_db1 ) );

    // change the next line to get the correct communicator out
    nonlinearSolverParams1->d_comm          = globalComm;
    nonlinearSolverParams1->d_pOperator     = nonlinearThermalOperator1;
    nonlinearSolverParams1->d_pInitialGuess = TemperatureInKelvinVec1;

    std::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearSolver1(
        new AMP::Solver::PetscSNESSolver( nonlinearSolverParams1 ) );

    //----------------------------------------------------------------------------------------------------------------------------------------------//

    std::shared_ptr<AMP::Database> thermalPreconditioner_db1 =
        linearSolver_db1->getDatabase( "Preconditioner" );
    std::shared_ptr<AMP::Solver::SolverStrategyParameters> thermalPreconditionerParams1(
        new AMP::Solver::SolverStrategyParameters( thermalPreconditioner_db1 ) );
    thermalPreconditionerParams1->d_pOperator = linearThermalOperator1;
    std::shared_ptr<AMP::Solver::TrilinosMLSolver> linearThermalPreconditioner1(
        new AMP::Solver::TrilinosMLSolver( thermalPreconditionerParams1 ) );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // register the preconditioner with the Jacobian free Krylov solver
    std::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver1 =
        nonlinearSolver1->getKrylovSolver();
    linearSolver1->setPreconditioner( linearThermalPreconditioner1 );
    nonlinearThermalOperator1->residual( RightHandSideVec1, TemperatureInKelvinVec1, ResidualVec1 );

    //---------------------------------------------
    //     CREATE THE CONTACT GAP OPERATOR
    //---------------------------------------------

    AMP_INSIST( input_db->keyExists( "GapOperator" ), "Key ''GapOperator'' is missing!" );
    std::shared_ptr<AMP::Database> gapDatabase =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "GapOperator" ) );

    double heff = ( gapDatabase )->getScalar<double>( "Convective_Coefficient" );
    std::shared_ptr<AMP::LinearAlgebra::Variable> gapVariable(
        new AMP::LinearAlgebra::Variable( "Gap" ) );

    //--------------------------------------------
    //   CREATE THE LINEAR THERMAL OPERATOR 2 ----
    //--------------------------------------------

    AMP_INSIST( input_db->keyExists( "LinearThermalOperator2" ), "key missing!" );

    std::shared_ptr<AMP::Operator::ElementPhysicsModel> thermalTransportModel2;
    std::shared_ptr<AMP::Database> linearThermalDatabase2 =
        input_db->getDatabase( "LinearThermalOperator2" );
    std::shared_ptr<AMP::Operator::LinearBVPOperator> linearThermalOperator2 =
        std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                meshAdapter2, "LinearThermalOperator2", input_db, thermalTransportModel2 ) );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    std::shared_ptr<AMP::Operator::DiffusionLinearFEOperator> thermalVolumeOperator2 =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>(
            linearThermalOperator2->getVolumeOperator() );

    // initialize the output variable
    AMP::LinearAlgebra::Variable::shared_ptr outputVariable2 =
        thermalVolumeOperator2->getOutputVariable();

    AMP::LinearAlgebra::Vector::shared_ptr TemperatureInKelvinVec2 =
        TemperatureInKelvin->select( vectorSelector2, TemperatureVar->getName() );
    AMP::LinearAlgebra::Vector::shared_ptr RightHandSideVec2 =
        AMP::LinearAlgebra::createVector( nodalDofMap2, outputVariable2 );
    AMP::LinearAlgebra::Vector::shared_ptr ResidualVec2 =
        AMP::LinearAlgebra::createVector( nodalDofMap2, outputVariable2 );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    std::shared_ptr<AMP::Solver::SolverStrategyParameters> mlSolverParams2(
        new AMP::Solver::SolverStrategyParameters( linearSolver_db1 ) );
    mlSolverParams2->d_pOperator = linearThermalOperator2;
    std::shared_ptr<AMP::Solver::TrilinosMLSolver> mlSolver2(
        new AMP::Solver::TrilinosMLSolver( mlSolverParams2 ) );
    mlSolver2->setZeroInitialGuess( true );

    //-------------------------------------

    AMP::LinearAlgebra::Vector::shared_ptr variableFluxVec1 =
        AMP::LinearAlgebra::createVector( nodalDofMap1, TemperatureVar );
    AMP::LinearAlgebra::Vector::shared_ptr scratchTempVec1 =
        AMP::LinearAlgebra::createVector( nodalDofMap1, TemperatureVar );
    variableFluxVec1->setToScalar( 0.0 );

    AMP::LinearAlgebra::Vector::shared_ptr variableFluxVec2 =
        AMP::LinearAlgebra::createVector( nodalDofMap2, TemperatureVar );
    AMP::LinearAlgebra::Vector::shared_ptr scratchTempVec2 =
        AMP::LinearAlgebra::createVector( nodalDofMap2, TemperatureVar );
    variableFluxVec2->setToScalar( 0.0 );

    //-------------------------------------

    std::shared_ptr<AMP::Database> map3dto1d_db1 =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "MapPelletto1D" ) );
    std::shared_ptr<AMP::Operator::MapOperatorParameters> map3dto1dParams1(
        new AMP::Operator::MapOperatorParameters( map3dto1d_db1 ) );
    map3dto1dParams1->d_Mesh = meshAdapter1;
    std::shared_ptr<AMP::Operator::Map3Dto1D> map1ToLowDim(
        new AMP::Operator::Map3Dto1D( map3dto1dParams1 ) );

    std::shared_ptr<AMP::Database> map1dto3d_db1 =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "Map1DtoClad" ) );
    std::shared_ptr<AMP::Operator::MapOperatorParameters> map1dto3dParams1(
        new AMP::Operator::MapOperatorParameters( map1dto3d_db1 ) );
    map1dto3dParams1->d_Mesh = meshAdapter2;
    //-------------------------------------
    ut->passes( "Everything up till constructing 1Dto3D passes." );
    //-------------------------------------
    std::shared_ptr<AMP::Operator::Map1Dto3D> map1ToHighDim(
        new AMP::Operator::Map1Dto3D( map1dto3dParams1 ) );

    map1ToLowDim->setZLocations( map1ToHighDim->getZLocations() );

    std::shared_ptr<AMP::Database> map3dto1d_db2 =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "MapCladto1D" ) );
    std::shared_ptr<AMP::Operator::MapOperatorParameters> map3dto1dParams2(
        new AMP::Operator::MapOperatorParameters( map3dto1d_db2 ) );
    map3dto1dParams2->d_Mesh = meshAdapter2;
    std::shared_ptr<AMP::Operator::Map3Dto1D> map2ToLowDim(
        new AMP::Operator::Map3Dto1D( map3dto1dParams2 ) );

    std::shared_ptr<AMP::Database> map1dto3d_db2 =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "Map1DtoPellet" ) );
    std::shared_ptr<AMP::Operator::MapOperatorParameters> map1dto3dParams2(
        new AMP::Operator::MapOperatorParameters( map1dto3d_db2 ) );
    map1dto3dParams2->d_Mesh = meshAdapter1;
    std::shared_ptr<AMP::Operator::Map1Dto3D> map2ToHighDim(
        new AMP::Operator::Map1Dto3D( map1dto3dParams2 ) );

    map2ToLowDim->setZLocations( map2ToHighDim->getZLocations() );

    ///-------------------------------------
    // From flow operator test
    //-------------------------------------

    std::shared_ptr<AMP::Database> mapcladflow_db =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "MapCladtoFlow" ) );
    std::shared_ptr<AMP::Operator::MapOperatorParameters> mapcladflowParams(
        new AMP::Operator::MapOperatorParameters( mapcladflow_db ) );
    mapcladflowParams->d_Mesh = meshAdapter2;
    std::shared_ptr<AMP::Operator::Map3Dto1D> mapCladToFlow(
        new AMP::Operator::Map3Dto1D( mapcladflowParams ) );

    std::shared_ptr<AMP::Database> mapflowclad_db =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "MapFlowtoClad" ) );
    std::shared_ptr<AMP::Operator::MapOperatorParameters> mapflowcladParams(
        new AMP::Operator::MapOperatorParameters( mapflowclad_db ) );
    mapflowcladParams->d_Mesh = meshAdapter2;
    std::shared_ptr<AMP::Operator::Map1Dto3D> mapFlowToClad(
        new AMP::Operator::Map1Dto3D( mapflowcladParams ) );

    mapCladToFlow->setZLocations( mapFlowToClad->getZLocations() );

    //-------------------------------------
    unsigned int flowVecSize = mapFlowToClad->getNumZlocations();

    //--------------------------------------
    //     CREATE THE FLOW OPERATOR   ------
    //--------------------------------------

    AMP_INSIST( input_db->keyExists( "FlowFrapconOperator" ),
                "Key ''FlowFrapconOperator'' is missing!" );

    std::shared_ptr<AMP::Operator::ElementPhysicsModel> flowtransportModel;
    std::shared_ptr<AMP::Database> flowDatabase =
        std::dynamic_pointer_cast<AMP::Database>( input_db->getDatabase( "FlowFrapconOperator" ) );
    flowDatabase->putScalar( "numPoints", flowVecSize );
    std::shared_ptr<AMP::Operator::FlowFrapconOperator> flowOperator =
        std::dynamic_pointer_cast<AMP::Operator::FlowFrapconOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                meshAdapter2, "FlowFrapconOperator", input_db, flowtransportModel ) );

    flowOperator->setZLocations( mapFlowToClad->getZLocations() );

    AMP::LinearAlgebra::Variable::shared_ptr inputVariable  = flowOperator->getInputVariable();
    AMP::LinearAlgebra::Variable::shared_ptr outputVariable = flowOperator->getOutputVariable();

    // double Cp, De, G, K, Re, Pr, hclad, dz, Tc, Tin;
    double De, K, Re, Pr, hclad;

    // Cp   = (flowDatabase)->getScalar<double>("Heat_Capacity");
    De = ( flowDatabase )->getScalar<double>( "Channel_Diameter" );
    // G    = (flowDatabase)->getScalar<double>("Mass_Flux");
    K  = ( flowDatabase )->getScalar<double>( "Conductivity" );
    Re = ( flowDatabase )->getScalar<double>( "Reynolds" );
    Pr = ( flowDatabase )->getScalar<double>( "Prandtl" );
    // Tin  = (flowDatabase)->getScalar<double>("Temp_Inlet");

    hclad = ( 0.023 * K / De ) * pow( Re, 0.8 ) * pow( Pr, 0.4 );

    // dz = 0.0127/flowVecSize ;

    AMP::LinearAlgebra::Vector::shared_ptr solVec =
        AMP::LinearAlgebra::SimpleVector<double>::create( flowVecSize, inputVariable );
    AMP::LinearAlgebra::Vector::shared_ptr rhsVec =
        AMP::LinearAlgebra::SimpleVector<double>::create( flowVecSize, outputVariable );
    AMP::LinearAlgebra::Vector::shared_ptr resVec =
        AMP::LinearAlgebra::SimpleVector<double>::create( flowVecSize, outputVariable );
    // AMP::LinearAlgebra::Vector::shared_ptr workVec =
    // AMP::LinearAlgebra::SimpleVector<double>::create( flowVecSize ,
    // inputVariable  );

    AMP::LinearAlgebra::Vector::shared_ptr vecLag =
        AMP::LinearAlgebra::SimpleVector<double>::create( flowVecSize, outputVariable );

    resVec->setToScalar( 350 );
    //-------------------------------------

    AMP::LinearAlgebra::Vector::shared_ptr robinRHSVec = AMP::LinearAlgebra::createVector(
        nodalDofMap2, thermalVolumeOperator2->getInputVariable() );

    //------------------------------------------

    AMP::Operator::Operator::shared_ptr boundaryOp1;
    boundaryOp1 = nonlinearThermalOperator1->getBoundaryOperator();

    AMP::Operator::Operator::shared_ptr robinBoundaryOp1;
    //  robinBoundaryOp1 =
    //  (std::dynamic_pointer_cast<AMP::Operator::ColumnBoundaryOperator>(boundaryOp1)
    //  )->getBoundaryOperator(0);
    robinBoundaryOp1 =
        ( std::dynamic_pointer_cast<AMP::Operator::BoundaryOperator>( boundaryOp1 ) );

    std::shared_ptr<AMP::Database> boundaryDatabase1 = std::dynamic_pointer_cast<AMP::Database>(
        nonlinearThermalDatabase1->getDatabase( "BoundaryOperator" ) );
    //  std::shared_ptr<AMP::Database> robinboundaryDatabase1 =
    //  std::dynamic_pointer_cast<AMP::Database>(
    //  boundaryDatabase1->getDatabase("RobinVectorCorrection"));
    std::shared_ptr<AMP::Database> robinboundaryDatabase1 =
        std::dynamic_pointer_cast<AMP::Database>( boundaryDatabase1 );

    robinboundaryDatabase1->putScalar( "constant_flux", false );
    robinboundaryDatabase1->putScalar( "skip_matrix_correction", true );
    std::shared_ptr<AMP::Operator::NeumannVectorCorrectionParameters> correctionParameters1(
        new AMP::Operator::NeumannVectorCorrectionParameters( robinboundaryDatabase1 ) );

    //------------------------------------------

    AMP::Operator::Operator::shared_ptr boundaryOp2;
    boundaryOp2 = linearThermalOperator2->getBoundaryOperator();
    AMP::Operator::Operator::shared_ptr robinBoundaryOp2, robinBoundaryOp3;
    robinBoundaryOp2 =
        ( std::dynamic_pointer_cast<AMP::Operator::ColumnBoundaryOperator>( boundaryOp2 ) )
            ->getBoundaryOperator( 0 );
    robinBoundaryOp3 =
        ( std::dynamic_pointer_cast<AMP::Operator::ColumnBoundaryOperator>( boundaryOp2 ) )
            ->getBoundaryOperator( 1 );

    std::shared_ptr<AMP::Database> boundaryDatabase2 = std::dynamic_pointer_cast<AMP::Database>(
        linearThermalDatabase2->getDatabase( "BoundaryOperator" ) );
    std::shared_ptr<AMP::Database> robinboundaryDatabase2 =
        std::dynamic_pointer_cast<AMP::Database>(
            boundaryDatabase2->getDatabase( "RobinMatrixCorrection1" ) );
    std::shared_ptr<AMP::Database> robinboundaryDatabase3 =
        std::dynamic_pointer_cast<AMP::Database>(
            boundaryDatabase2->getDatabase( "RobinMatrixCorrection2" ) );

    robinboundaryDatabase2->putScalar( "constant_flux", false );
    robinboundaryDatabase2->putScalar( "skip_matrix_correction", true );
    std::shared_ptr<AMP::Operator::RobinMatrixCorrectionParameters> correctionParameters2(
        new AMP::Operator::RobinMatrixCorrectionParameters( robinboundaryDatabase2 ) );

    robinboundaryDatabase3->putScalar( "constant_flux", false );
    robinboundaryDatabase3->putScalar( "skip_matrix_correction", true );
    std::shared_ptr<AMP::Operator::RobinMatrixCorrectionParameters> correctionParameters3(
        new AMP::Operator::RobinMatrixCorrectionParameters( robinboundaryDatabase3 ) );


    //-------------------------------------

    size_t gapVecCladSize = map1ToHighDim->getNumZlocations();
    AMP::LinearAlgebra::Vector::shared_ptr gapVecClad =
        AMP::LinearAlgebra::SimpleVector<double>::create( gapVecCladSize, gapVariable );

    size_t gapVecPelletSize = map2ToHighDim->getNumZlocations();
    AMP::LinearAlgebra::Vector::shared_ptr gapVecPellet =
        AMP::LinearAlgebra::SimpleVector<double>::create( gapVecPelletSize, gapVariable );

    int cnt = 0;
    AMP::LinearAlgebra::Vector::shared_ptr vecLag1 =
        AMP::LinearAlgebra::createVector( nodalDofMap1, outputVariable1 );
    vecLag1->copyVector( TemperatureInKelvinVec1 );
    AMP::LinearAlgebra::Vector::shared_ptr vecLag2 =
        AMP::LinearAlgebra::createVector( nodalDofMap2, outputVariable2 );
    vecLag2->copyVector( TemperatureInKelvinVec2 );

    bool testPassed = false;

    int maxIt = input_db->getWithDefault( "max_iterations", 100 );

    while ( cnt < maxIt ) {
        cnt++;

        RightHandSideVec1->zero();
        RightHandSideVec2->zero();

        RightHandSideVec1->copyVector( PowerInWattsVec );
        std::cout << "PowerInWattsVec norm  inside loop = " << RightHandSideVec1->L2Norm() << "\n";

        //------------------------------------------------------------
        /*
                  solVec->setValueByLocalID(0, Tin);
                  for( int i=1; i<flowVecSize ; i++) {
                    Tc = resVec->getValueByLocalID(i) + (resVec->getValueByLocalID(i)-
           resVec->getValueByLocalID(i-1))*((Cp*G*De)/(4.*hclad*dz));
                    solVec->setValueByLocalID(i, Tc);
                  }

                  std::cout<<"Converged Flow Solution " <<std::endl;
                  for( int i=0; i<flowVecSize ; i++) {
                    std::cout<<" @i : "<< i<<" is "<<solVec->getValueByLocalID(i) ;
                  }
                  std::cout<<std::endl;
        */

        //------------------------------------------------------------

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

        //------------------------------------------------------------

        mapCladToFlow->residual( nullVec, TemperatureInKelvinVec2, solVec );
        while ( true ) {
            flowOperator->residual( rhsVec, solVec, resVec );
            if ( abs( resVec->L2Norm() - vecLag->L2Norm() ) < .000005 * vecLag->L2Norm() )
                break;
            else
                std::cout << "for iteration cnt = " << cnt << " --> " << vecLag->L2Norm() << " "
                          << resVec->L2Norm() << std::endl;

            std::cout << "Intermediate Flow Solution " << std::endl;
            for ( unsigned int i = 0; i < flowVecSize; i++ ) {
                std::cout << " @i : " << i << " is " << resVec->getValueByLocalID( i );
            }
            std::cout << std::endl;
            vecLag->copyVector( resVec );
        }

        mapFlowToClad->residual( nullVec, resVec, robinRHSVec );

        robinRHSVec->scale( hclad );
        correctionParameters3->d_variableFlux = robinRHSVec;
        robinBoundaryOp3->reset( correctionParameters3 );


        //-----------------------------------------------
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

        //------------------------------------------------------------

        std::cout << "Residual Norm on Pellet after " << cnt
                  << " iteration is : " << ResidualVec1->L2Norm() << std::endl;
        std::cout << "Residual Norm on Clad after " << cnt
                  << " iteration is : " << ResidualVec2->L2Norm() << std::endl;

        vecLag2->subtract( TemperatureInKelvinVec2, vecLag2 );

//          if( nodes == 2 ) {
#ifdef USE_EXT_SILO
        AMP::Utilities::Writer::shared_ptr siloWriter =
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
                    "Thermal+Robin->Map->Gap->Map->Nonlinear Thermal+Robin ." );
    } else {
        ut->failure( "Seggregated solve of Composite Operator using control loop of Nonlinear "
                     "Thermal+Robin->Map->Gap->Map->Nonlinear Thermal+Robin ." );
    }

    //} else {
    //  ut.expected_failure("parallel map3D-1D and map1D-3D fail in parallel, see bug #1219.");
    //}
    input_db.reset();

    ut->passes( exeName );

    //  AMP::AMPManager::shutdown();
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