#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/discretization/structuredFaceDOFManager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/CoupledOperator.h"
#include "AMP/operators/CoupledOperatorParameters.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/VectorCopyOperator.h"
#include "AMP/operators/boundary/ColumnBoundaryOperator.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrection.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrectionParameters.h"
#include "AMP/operators/boundary/libmesh/RobinVectorCorrection.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/operators/map/AsyncMapColumnOperator.h"
#include "AMP/operators/map/CladToSubchannelMap.h"
#include "AMP/operators/map/NodeToNodeMap.h"
#include "AMP/operators/map/ScalarZAxisMap.h"
#include "AMP/operators/map/SubchannelToCladMap.h"
#include "AMP/operators/subchannel/CoupledChannelToCladMapOperator.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/operators/subchannel/SubchannelHelpers.h"
#include "AMP/operators/subchannel/SubchannelOperatorParameters.h"
#include "AMP/operators/subchannel/SubchannelPhysicsModel.h"
#include "AMP/operators/subchannel/SubchannelToPointMap.h"
#include "AMP/operators/subchannel/SubchannelTwoEqLinearOperator.h"
#include "AMP/operators/subchannel/SubchannelTwoEqNonlinearOperator.h"
#include "AMP/solvers/BandedSolver.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include "ProfilerApp.h"

#include <memory>
#include <string>


// Function to get an arbitrary power profile (W/kg) assuming a density of 1 kg/m^3 for the volume
// integral
// P is the total power of the pin, V is the volume of the pin
static double
getPower( const std::vector<double> &range, double P, double V, const AMP::Mesh::Point &pos )
{
    const double pi = 3.1415926535897932;
    double x        = ( pos[0] - range[0] ) / ( range[1] - range[0] );
    double y        = ( pos[1] - range[2] ) / ( range[3] - range[2] );
    double z        = ( pos[2] - range[4] ) / ( range[5] - range[4] );
    return P / V * ( 0.8 + 0.2 * x + 0.2 * y ) * pi / 2 * sin( pi * z );
}


// Function to create the solution vectors
static void createVectors( std::shared_ptr<AMP::Mesh::Mesh> pinMesh,
                           std::shared_ptr<AMP::Mesh::Mesh> subchannelMesh,
                           AMP::LinearAlgebra::Vector::shared_ptr &globalMultiVector,
                           AMP::LinearAlgebra::Vector::shared_ptr &specificPowerGpVec )
{
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto multivec     = AMP::LinearAlgebra::MultiVector::create( "multivector", globalComm );
    globalMultiVector = multivec;

    auto thermalVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "Temperature" );
    auto flowVariable    = std::make_shared<AMP::LinearAlgebra::Variable>( "Flow" );
    auto powerVariable =
        std::make_shared<AMP::LinearAlgebra::Variable>( "SpecificPowerInWattsPerGram" );

    AMP::LinearAlgebra::Vector::shared_ptr thermalVec;
    if ( pinMesh ) {
        auto nodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
            pinMesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
        thermalVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, thermalVariable );
    }
    multivec->addVector( thermalVec );

    AMP::LinearAlgebra::Vector::shared_ptr flowVec;
    if ( subchannelMesh ) {
        int DOFsPerFace[3]  = { 0, 0, 2 };
        auto faceDOFManager = std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
            subchannelMesh, DOFsPerFace, 0 );
        // create solution, rhs, and residual vectors
        flowVec = AMP::LinearAlgebra::createVector( faceDOFManager, flowVariable, true );
    }
    multivec->addVector( flowVec );

    if ( pinMesh ) {
        auto gaussPtDOFManager = AMP::Discretization::simpleDOFManager::create(
            pinMesh, AMP::Mesh::GeomType::Cell, 1, 8 );
        specificPowerGpVec = AMP::LinearAlgebra::createVector( gaussPtDOFManager, powerVariable );
        specificPowerGpVec->setToScalar( 0.0 );
    }
}


static void SubchannelSolve( AMP::UnitTest *ut, const std::string &exeName )
{
    PROFILE( "Main" );
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    AMP::logAllNodes( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    globalComm.barrier();

    // Read the input file
    auto db = AMP::Database::parseInputFile( input_file );
    db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database   = db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( database );
    meshParams->setComm( globalComm );

    // Get the meshes
    auto manager = AMP::Mesh::MeshFactory::create( meshParams );
    auto pinMesh = manager->Subset( "MultiPin" );
    std::shared_ptr<AMP::Mesh::Mesh> cladMesh;
    if ( pinMesh ) {
        pinMesh->setName( "MultiPin" );
        cladMesh = pinMesh->Subset( "clad" );
    }
    auto subchannelMesh = manager->Subset( "subchannel" );
    std::shared_ptr<AMP::Mesh::Mesh> xyFaceMesh;
    if ( subchannelMesh ) {
        auto face  = AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( subchannelMesh, 0 );
        xyFaceMesh = subchannelMesh->Subset( face );
    }

    // Variables
    auto thermalVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "Temperature" );
    auto flowVariable    = std::make_shared<AMP::LinearAlgebra::Variable>( "Flow" );
    auto powerVariable =
        std::make_shared<AMP::LinearAlgebra::Variable>( "SpecificPowerInWattsPerGram" );

    auto nonlinearColumnOperator      = std::make_shared<AMP::Operator::ColumnOperator>();
    auto volumeIntegralColumnOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    auto mapsColumn                   = std::make_shared<AMP::Operator::ColumnOperator>();
    auto n2nColumn                    = std::make_shared<AMP::Operator::AsyncMapColumnOperator>();
    auto szaColumn                    = std::make_shared<AMP::Operator::AsyncMapColumnOperator>();

    std::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearCoupledSolver;
    std::shared_ptr<AMP::Solver::PetscKrylovSolver> linearColumnSolver;
    std::shared_ptr<AMP::Solver::ColumnSolver> columnPreconditioner;

    if ( pinMesh ) {
        auto pinMeshIDs = pinMesh->getBaseMeshIDs();

        // CREATE OPERATORS
        for ( auto pinMeshID : pinMeshIDs ) {
            auto mesh = manager->Subset( pinMeshID );
            if ( !mesh )
                continue;

            std::string meshName = mesh->getName();
            std::string prefix, prefixPower;

            if ( meshName == "clad" ) {
                prefix      = "Clad";
                prefixPower = "Clad";
            } else if ( meshName == "pellet_1" ) {
                prefix      = "BottomPellet";
                prefixPower = "Pellet";
            } else if ( meshName == "pellet_3" ) {
                prefix      = "TopPellet";
                prefixPower = "Pellet";
            } else if ( meshName.compare( 0, 7, "pellet_" ) == 0 ) {
                prefix      = "MiddlePellet";
                prefixPower = "Pellet";
            } else {
                AMP_ERROR( "Unknown Mesh" );
            }

            // CREATE THE NONLINEAR THERMAL OPERATOR 1
            AMP_INSIST( db->keyExists( prefix + "NonlinearThermalOperator" ), "key missing!" );
            auto thermalNonlinearOperator =
                std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
                    AMP::Operator::OperatorBuilder::createOperator(
                        mesh, prefix + "NonlinearThermalOperator", db ) );
            nonlinearColumnOperator->append( thermalNonlinearOperator );

            AMP_INSIST( db->keyExists( prefixPower + "VolumeIntegralOperator" ), "key missing!" );
            auto specificPowerGpVecToPowerDensityNodalVecOperator =
                std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
                    AMP::Operator::OperatorBuilder::createOperator(
                        mesh, prefixPower + "VolumeIntegralOperator", db ) );
            volumeIntegralColumnOperator->append(
                specificPowerGpVecToPowerDensityNodalVecOperator );
        }
    }

    // Get the subchannel hydraulic diameter for the temperature boundary condition
    auto ChannelDiameterVec =
        AMP::Operator::Subchannel::getCladHydraulicDiameter( cladMesh, subchannelMesh, globalComm );


    AMP::LinearAlgebra::Vector::shared_ptr subchannelFuelTemp;
    AMP::LinearAlgebra::Vector::shared_ptr subchannelFlowTemp;
    if ( subchannelMesh ) {
        int DOFsPerFace[3]        = { 0, 0, 1 };
        auto scalarFaceDOFManager = std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
            subchannelMesh, DOFsPerFace, 0 );
        subchannelFuelTemp =
            AMP::LinearAlgebra::createVector( scalarFaceDOFManager, thermalVariable );
        subchannelFlowTemp =
            AMP::LinearAlgebra::createVector( scalarFaceDOFManager, thermalVariable );
    }

    // get subchannel physics model
    // for post processing - need water library to convert enthalpy to temperature...
    auto subchannelPhysics_db = db->getDatabase( "SubchannelPhysicsModel" );
    auto params =
        std::make_shared<AMP::Operator::ElementPhysicsModelParameters>( subchannelPhysics_db );
    auto subchannelPhysicsModel = std::make_shared<AMP::Operator::SubchannelPhysicsModel>( params );
    std::shared_ptr<AMP::Operator::SubchannelTwoEqNonlinearOperator> subchannelNonlinearOperator;
    std::shared_ptr<AMP::Operator::LinearOperator> subchannelLinearOperator;

    // Get the subchannel operators
    std::vector<double> clad_x, clad_y, clad_d;
    AMP::Operator::Subchannel::getCladProperties( globalComm, cladMesh, clad_x, clad_y, clad_d );
    if ( subchannelMesh ) {
        auto subChannelMeshIDs = subchannelMesh->getBaseMeshIDs();

        for ( auto subChannelMeshID : subChannelMeshIDs ) {
            auto mesh = manager->Subset( subChannelMeshID );
            if ( !mesh )
                continue;

            auto meshName = mesh->getName();
            if ( meshName == "subchannel" ) {
                // create the non-linear operator
                subchannelNonlinearOperator =
                    std::dynamic_pointer_cast<AMP::Operator::SubchannelTwoEqNonlinearOperator>(
                        AMP::Operator::OperatorBuilder::createOperator(
                            mesh, "SubchannelTwoEqNonlinearOperator", db ) );
                auto nonlinearOpParams =
                    std::const_pointer_cast<AMP::Operator::SubchannelOperatorParameters>(
                        subchannelNonlinearOperator->getParams() );
                nonlinearOpParams->clad_x = clad_x;
                nonlinearOpParams->clad_y = clad_y;
                nonlinearOpParams->clad_d = clad_d;
                subchannelNonlinearOperator->reset( nonlinearOpParams );

                int DOFsPerFace[3] = { 0, 0, 2 };
                auto flowDOFManager =
                    std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
                        subchannelMesh, DOFsPerFace, 0 );
                auto subchannelFlow =
                    AMP::LinearAlgebra::createVector( flowDOFManager, flowVariable );
                subchannelNonlinearOperator->setVector( subchannelFuelTemp );

                // pass creation test
                ut->passes( exeName + ": creation" );
                std::cout.flush();
                nonlinearColumnOperator->append( subchannelNonlinearOperator );
                // Do not add the subchannel to the linear operator (we will add it later)
            }
        }
    }


    // CREATE MAPS
    AMP::LinearAlgebra::Vector::shared_ptr thermalMapVec;
    AMP::LinearAlgebra::Vector::shared_ptr density_map_vec;
    if ( cladMesh ) {
        auto nodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
            cladMesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
        auto densityVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "Density" );
        density_map_vec      = AMP::LinearAlgebra::createVector( nodalScalarDOF, densityVariable );
        density_map_vec->zero();
    }
    if ( pinMesh ) {
        // flow temperature on clad outer surfaces and pellet temperature on clad innner surfaces,
        // and clad inner
        // surface temp on pellet outer surfaces
        auto nodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
            pinMesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
        thermalMapVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, thermalVariable, true );

        auto pinMeshIDs = pinMesh->getBaseMeshIDs();

        auto pins = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( pinMesh )->getMeshes();

        for ( const auto &pin : pins ) {
            if ( db->keyExists( "ThermalNodeToNodeMaps" ) ) {
                auto map =
                    AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::NodeToNodeMap>(
                        pin, db->getDatabase( "ThermalNodeToNodeMaps" ) );
                for ( size_t j = 0; j < map->getNumberOfOperators(); j++ )
                    n2nColumn->append( map->getOperator( j ) );
            }
            if ( db->keyExists( "ThermalScalarZAxisMaps" ) ) {
                auto sza =
                    AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::ScalarZAxisMap>(
                        pin, db->getDatabase( "ThermalScalarZAxisMaps" ) );
                for ( size_t j = 0; j < sza->getNumberOfOperators(); j++ )
                    szaColumn->append( sza->getOperator( j ) );
            }
        }
        if ( n2nColumn->getNumberOfOperators() > 0 )
            mapsColumn->append( n2nColumn );
        if ( szaColumn->getNumberOfOperators() > 0 )
            mapsColumn->append( szaColumn );

        n2nColumn->setVector( thermalMapVec );
        szaColumn->setVector( thermalMapVec );

        int curOperator = 0;
        for ( auto pinMeshID : pinMeshIDs ) {
            auto mesh = manager->Subset( pinMeshID );
            if ( !mesh )
                continue;

            std::string meshName = mesh->getName();
            std::string prefix;

            if ( meshName == "clad" ) {
                prefix = "Clad";
            } else if ( meshName == "pellet_1" ) {
                prefix = "BottomPellet";
            } else if ( meshName == "pellet_3" ) {
                prefix = "TopPellet";
            } else if ( meshName.compare( 0, 7, "pellet_" ) == 0 ) {
                prefix = "MiddlePellet";
            } else {
                AMP_ERROR( "Unknown Mesh" );
            }

            auto curBVPop = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
                nonlinearColumnOperator->getOperator( curOperator ) );
            auto curBCcol = std::dynamic_pointer_cast<AMP::Operator::ColumnBoundaryOperator>(
                curBVPop->getBoundaryOperator() );
            auto operator_db = db->getDatabase( prefix + "NonlinearThermalOperator" );
            auto curBCdb     = db->getDatabase( operator_db->getString( "BoundaryOperator" ) );
            auto opNames     = curBCdb->getVector<std::string>( "boundaryOperators" );
            int opNumber     = curBCdb->getScalar<int>( "numberOfBoundaryOperators" );
            for ( int curBCentry = 0; curBCentry != opNumber; curBCentry++ ) {
                if ( opNames[curBCentry] == "P2CRobinVectorCorrection" ) {
                    auto gapBC = std::dynamic_pointer_cast<AMP::Operator::RobinVectorCorrection>(
                        curBCcol->getBoundaryOperator( curBCentry ) );
                    AMP_ASSERT( thermalMapVec );
                    gapBC->setVariableFlux( thermalMapVec );
                    gapBC->reset( gapBC->getOperatorParameters() );
                } else if ( ( opNames[curBCentry] == "BottomP2PNonlinearRobinVectorCorrection" ) ||
                            ( opNames[curBCentry] == "MiddleP2PNonlinearRobinBoundaryCondition" ) ||
                            ( opNames[curBCentry] == "TopP2PNonlinearRobinBoundaryCondition" ) ) {
                    auto p2pBC = std::dynamic_pointer_cast<AMP::Operator::RobinVectorCorrection>(
                        curBCcol->getBoundaryOperator( curBCentry ) );
                    AMP_ASSERT( thermalMapVec );
                    p2pBC->setVariableFlux( thermalMapVec );
                    p2pBC->reset( p2pBC->getOperatorParameters() );
                } else if ( opNames[curBCentry] == "C2WBoundaryVectorCorrection" ) {
                    auto thisDb    = db->getDatabase( opNames[curBCentry] );
                    bool isCoupled = thisDb->getWithDefault<bool>( "IsCoupledBoundary_0", false );
                    if ( isCoupled ) {
                        auto c2wBC =
                            std::dynamic_pointer_cast<AMP::Operator::RobinVectorCorrection>(
                                curBCcol->getBoundaryOperator( curBCentry ) );
                        AMP_ASSERT( thermalMapVec );
                        c2wBC->setVariableFlux( thermalMapVec );
                        c2wBC->setFrozenVector( density_map_vec );
                        c2wBC->setFrozenVector( ChannelDiameterVec );
                        c2wBC->reset( c2wBC->getOperatorParameters() );
                    }
                } else if ( opNames[curBCentry] == "C2PRobinVectorCorrection" ) {
                    auto gapBC = std::dynamic_pointer_cast<AMP::Operator::RobinVectorCorrection>(
                        curBCcol->getBoundaryOperator( curBCentry ) );
                    AMP_ASSERT( thermalMapVec );
                    gapBC->setVariableFlux( thermalMapVec );
                    gapBC->reset( gapBC->getOperatorParameters() );
                } else {
                    AMP_ERROR( "Unknown boundary operator" );
                }
            }
            curOperator++;
        }
    }

    // Create the maps from the clad temperature to the subchannel temperature
    auto cladToSubchannelDb = db->getDatabase( "CladToSubchannelMaps" );
    auto cladToSubchannelMap =
        AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::CladToSubchannelMap>(
            manager, cladToSubchannelDb );
    cladToSubchannelMap->setVector( subchannelFuelTemp );
    mapsColumn->append( cladToSubchannelMap );

    // Create the maps from the flow variable (enthalpy and pressure) on subchannel mesh and convert
    // to temperature and
    // density then map to clad surface
    auto thermalCladToSubchannelDb = db->getDatabase( "ThermalSubchannelToCladMaps" );
    auto thermalSubchannelToCladMap =
        AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::SubchannelToCladMap>(
            manager, thermalCladToSubchannelDb );
    auto densityCladToSubchannelDb = db->getDatabase( "DensitySubchannelToCladMaps" );
    auto densitySubchannelToCladMap =
        AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::SubchannelToCladMap>(
            manager, densityCladToSubchannelDb );
    if ( cladMesh ) {
        AMP::LinearAlgebra::VS_Comm commSelector( cladMesh->getComm() );
        auto subsetTheramlVec = thermalMapVec->select( commSelector );
        thermalSubchannelToCladMap->setVector( subsetTheramlVec );
        densitySubchannelToCladMap->setVector( density_map_vec );
    }
    auto emptyDb = std::make_shared<AMP::Database>( "empty" );
    emptyDb->putScalar( "print_info_level", 0 );
    auto coupledChannelMapOperatorParams =
        std::make_shared<AMP::Operator::CoupledChannelToCladMapOperatorParameters>( emptyDb );
    coupledChannelMapOperatorParams->d_variable               = flowVariable;
    coupledChannelMapOperatorParams->d_vector                 = subchannelFlowTemp;
    coupledChannelMapOperatorParams->d_Mesh                   = subchannelMesh;
    coupledChannelMapOperatorParams->d_thermalMapOperator     = thermalSubchannelToCladMap;
    coupledChannelMapOperatorParams->d_densityMapOperator     = densitySubchannelToCladMap;
    coupledChannelMapOperatorParams->d_subchannelMesh         = subchannelMesh;
    coupledChannelMapOperatorParams->d_subchannelPhysicsModel = subchannelPhysicsModel;
    auto coupledChannelMapOperator =
        std::make_shared<AMP::Operator::CoupledChannelToCladMapOperator>(
            coupledChannelMapOperatorParams );
    mapsColumn->append( coupledChannelMapOperator );

    std::shared_ptr<AMP::Operator::Operator> thermalCopyOperator;
    if ( pinMesh ) {
        auto copyOp_db = db->getDatabase( "CopyOperator" );
        auto vecCopyOperatorParams =
            std::make_shared<AMP::Operator::VectorCopyOperatorParameters>( copyOp_db );
        vecCopyOperatorParams->d_copyVariable = thermalVariable;
        vecCopyOperatorParams->d_copyVector   = thermalMapVec;
        vecCopyOperatorParams->d_Mesh         = pinMesh;
        thermalCopyOperator.reset( new AMP::Operator::VectorCopyOperator( vecCopyOperatorParams ) );
        thermalMapVec->zero();
    }

    auto CoupledOpParams = std::make_shared<AMP::Operator::CoupledOperatorParameters>( emptyDb );
    CoupledOpParams->d_CopyOperator = thermalCopyOperator;
    CoupledOpParams->d_MapOperator  = mapsColumn;
    CoupledOpParams->d_BVPOperator  = nonlinearColumnOperator;
    auto nonlinearCoupledOperator =
        std::make_shared<AMP::Operator::CoupledOperator>( CoupledOpParams );


    // Create the solution vector
    AMP::LinearAlgebra::Vector::shared_ptr globalSolMultiVector;
    AMP::LinearAlgebra::Vector::shared_ptr specificPowerGpVec;
    createVectors( pinMesh, subchannelMesh, globalSolMultiVector, specificPowerGpVec );

    // Create the rhs and res vectors
    auto globalRhsMultiVector = globalSolMultiVector->clone();
    auto globalResMultiVector = globalSolMultiVector->clone();
    auto flowSolVec           = globalSolMultiVector->subsetVectorForVariable( flowVariable );
    auto flowRhsVec           = globalRhsMultiVector->subsetVectorForVariable( flowVariable );
    auto flowResVec           = globalResMultiVector->subsetVectorForVariable( flowVariable );
    auto globalThermalSolVec  = globalSolMultiVector->subsetVectorForVariable( thermalVariable );
    auto globalThermalRhsVec  = globalRhsMultiVector->subsetVectorForVariable( thermalVariable );
    auto globalThermalResVec  = globalResMultiVector->subsetVectorForVariable( thermalVariable );

    // Initialize the pin temperatures
    int root_subchannel = -1;
    std::vector<double> range( 6 );
    if ( subchannelMesh ) {
        range = subchannelMesh->getBoundingBox();
        AMP_ASSERT( range.size() == 6 );
        if ( subchannelMesh->getComm().getRank() == 0 )
            root_subchannel = globalComm.getRank();
    }
    root_subchannel = globalComm.maxReduce( root_subchannel );
    globalComm.bcast( &range[0], 6, root_subchannel );
    // Desired power of the fuel pin (W)
    auto P =
        db->getDatabase( "SubchannelTwoEqNonlinearOperator" )->getScalar<double>( "Rod_Power" );
    // GeomType::Cell of fuel in a 3.81m pin
    if ( pinMesh ) {
        const double V = 1.939e-4;
        globalThermalSolVec->setToScalar( 600 );
        auto gaussPtDOFManager = AMP::Discretization::simpleDOFManager::create(
            pinMesh, AMP::Mesh::GeomType::Cell, 1, 8 );
        auto it = pinMesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
        std::vector<size_t> dofs;
        for ( size_t i = 0; i < it.size(); i++ ) {
            gaussPtDOFManager->getDOFs( it->globalID(), dofs );
            for ( size_t dof : dofs ) {
                double val = getPower( range, P, V, it->centroid() );
                specificPowerGpVec->setValuesByGlobalID( 1, &dof, &val );
            }
            ++it;
        }
        if ( cladMesh ) {
            AMP::LinearAlgebra::VS_Mesh meshSelector( cladMesh );
            auto cladPower = specificPowerGpVec->select( meshSelector );
            cladPower->zero();
        }
        specificPowerGpVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        volumeIntegralColumnOperator->apply( specificPowerGpVec, globalThermalRhsVec );
    }

    if ( subchannelMesh ) {
        // get exit pressure
        auto Pout = db->getDatabase( "SubchannelTwoEqNonlinearOperator" )
                        ->getScalar<double>( "Exit_Pressure" );
        // get inlet temperature
        auto Tin = db->getDatabase( "SubchannelTwoEqNonlinearOperator" )
                       ->getScalar<double>( "Inlet_Temperature" );
        // compute inlet enthalpy
        std::map<std::string, std::shared_ptr<std::vector<double>>> enthalpyArgMap;
        enthalpyArgMap.insert(
            std::make_pair( "temperature", std::make_shared<std::vector<double>>( 1, Tin ) ) );
        enthalpyArgMap.insert(
            std::make_pair( "pressure", std::make_shared<std::vector<double>>( 1, Pout ) ) );
        std::vector<double> enthalpyResult( 1 );
        subchannelPhysicsModel->getProperty( "Enthalpy", enthalpyResult, enthalpyArgMap );
        double hin = enthalpyResult[0];
        std::cout << "Enthalpy Solution:" << hin << std::endl;
        std::cout << "Outlet pressure:" << Pout << std::endl;

        AMP::LinearAlgebra::VS_Mesh meshSelector( subchannelMesh );
        auto tmpVec             = flowSolVec->selectInto( meshSelector );
        auto subchannelEnthalpy = tmpVec->select( AMP::LinearAlgebra::VS_Stride( 0, 2 ) );
        auto subchannelPressure = tmpVec->select( AMP::LinearAlgebra::VS_Stride( 1, 2 ) );

        subchannelEnthalpy->setToScalar( AMP::Operator::Subchannel::scaleEnthalpy * hin );
        subchannelPressure->setToScalar( AMP::Operator::Subchannel::scalePressure * Pout );
    }

    nonlinearCoupledOperator->residual(
        globalRhsMultiVector, globalSolMultiVector, globalResMultiVector );

    size_t totalOp;
    if ( subchannelMesh ) {
        totalOp = nonlinearColumnOperator->getNumberOfOperators() - 1;
    } else {
        totalOp = nonlinearColumnOperator->getNumberOfOperators();
    }
    for ( size_t id = 0; id != totalOp; id++ ) {
        auto nonlinearThermalOperator =
            std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
                nonlinearColumnOperator->getOperator( id ) );
        nonlinearThermalOperator->modifyInitialSolutionVector( globalThermalSolVec );
        nonlinearThermalOperator->modifyRHSvector( globalThermalRhsVec );
    }
    globalThermalRhsVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    auto linearColParams =
        nonlinearColumnOperator->getParameters( "Jacobian", globalSolMultiVector );

    auto linearColumnOperator = std::make_shared<AMP::Operator::ColumnOperator>( linearColParams );

    // create nonlinear solver
    std::shared_ptr<AMP::Solver::SolverStrategy> nonlinearSolver;
    { // Limit the scope so we can add an if else statement for Petsc vs NOX

        // get the solver databases
        auto nonlinearSolver_db = db->getDatabase( "NonlinearSolver" );
        // create preconditioner (thermal domains)
        auto columnPreconditioner_db = db->getDatabase( "Preconditioner" );
        auto columnPreconditionerParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( columnPreconditioner_db );
        columnPreconditionerParams->d_pOperator = linearColumnOperator;
        columnPreconditioner.reset( new AMP::Solver::ColumnSolver( columnPreconditionerParams ) );

        auto trilinosPreconditioner_db =
            columnPreconditioner_db->getDatabase( "TrilinosPreconditioner" );
        for ( size_t id = 0; id < totalOp; ++id ) {
            auto trilinosPreconditionerParams =
                std::make_shared<AMP::Solver::SolverStrategyParameters>(
                    trilinosPreconditioner_db );
            trilinosPreconditionerParams->d_pOperator = linearColumnOperator->getOperator( id );
            auto trilinosPreconditioner =
                std::make_shared<AMP::Solver::TrilinosMLSolver>( trilinosPreconditionerParams );
            columnPreconditioner->append( trilinosPreconditioner );
        }

        // Create the subchannel preconditioner
        if ( subchannelMesh ) {
            auto subchannelPreconditioner_db =
                columnPreconditioner_db->getDatabase( "SubchannelPreconditioner" );
            AMP_ASSERT( subchannelPreconditioner_db );
            auto subchannelPreconditionerParams =
                std::make_shared<AMP::Solver::SolverStrategyParameters>(
                    subchannelPreconditioner_db );
            subchannelPreconditionerParams->d_pOperator =
                linearColumnOperator->getOperator( totalOp );
            auto preconditioner = subchannelPreconditioner_db->getString( "Type" );
            if ( preconditioner == "ML" ) {
                auto subchannelPreconditioner = std::make_shared<AMP::Solver::TrilinosMLSolver>(
                    subchannelPreconditionerParams );
                columnPreconditioner->append( subchannelPreconditioner );
            } else if ( preconditioner == "Banded" ) {
                subchannelPreconditioner_db->putScalar( "KL", 3 );
                subchannelPreconditioner_db->putScalar( "KU", 3 );
                auto subchannelPreconditioner =
                    std::make_shared<AMP::Solver::BandedSolver>( subchannelPreconditionerParams );
                columnPreconditioner->append( subchannelPreconditioner );
            } else if ( preconditioner == "None" ) {
            } else {
                AMP_ERROR( "Invalid preconditioner type" );
            }
        }

        // create nonlinear solver parameters
        auto nonlinearSolverParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );
        nonlinearSolverParams->d_comm          = globalComm;
        nonlinearSolverParams->d_pOperator     = nonlinearCoupledOperator;
        nonlinearSolverParams->d_pInitialGuess = globalSolMultiVector;
        nonlinearSolver = std::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams );

        auto linearSolver =
            std::dynamic_pointer_cast<AMP::Solver::PetscSNESSolver>( nonlinearSolver )
                ->getKrylovSolver();
        // set preconditioner
        linearSolver->setNestedSolver( columnPreconditioner );
    }

    // don't use zero initial guess
    nonlinearSolver->setZeroInitialGuess( false );


    double tempResNorm = 0.0;
    double flowResNorm = 0.0;

    // Solve
    {
        PROFILE( "Solve" );
        AMP::pout << "Rhs norm: " << std::setprecision( 13 ) << globalRhsMultiVector->L2Norm()
                  << std::endl;
        AMP::pout << "Initial solution norm: " << std::setprecision( 13 )
                  << globalSolMultiVector->L2Norm() << std::endl;
        nonlinearCoupledOperator->residual(
            globalRhsMultiVector, globalSolMultiVector, globalResMultiVector );

        if ( pinMesh )
            tempResNorm = static_cast<double>( globalThermalResVec->L2Norm() );
        if ( subchannelMesh )
            flowResNorm = static_cast<double>( flowResVec->L2Norm() );

        AMP::pout << "Initial residual norm: " << std::setprecision( 13 )
                  << static_cast<double>( globalResMultiVector->L2Norm() ) << std::endl;
        AMP::pout << "Initial temp residual norm: " << std::setprecision( 13 ) << tempResNorm
                  << std::endl;
        AMP::pout << "Initial flow residual norm: " << std::setprecision( 13 ) << flowResNorm
                  << std::endl;
        nonlinearSolver->apply( globalRhsMultiVector, globalSolMultiVector );
        nonlinearCoupledOperator->residual(
            globalRhsMultiVector, globalSolMultiVector, globalResMultiVector );
        AMP::pout << "Final residual norm: " << std::setprecision( 13 )
                  << static_cast<double>( globalResMultiVector->L2Norm() ) << std::endl;
    }

    // Compute the flow temperature and density
    AMP::LinearAlgebra::Vector::shared_ptr flowTempVec;
    AMP::LinearAlgebra::Vector::shared_ptr deltaFlowTempVec;
    AMP::LinearAlgebra::Vector::shared_ptr flowDensityVec;
    if ( subchannelMesh ) {
        flowTempVec         = subchannelFuelTemp->clone();
        flowDensityVec      = subchannelFuelTemp->clone();
        int DOFsPerFace[3]  = { 0, 0, 2 };
        auto faceDOFManager = std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
            subchannelMesh, DOFsPerFace, 0 );
        DOFsPerFace[2]            = 1;
        auto scalarFaceDOFManager = std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
            subchannelMesh, DOFsPerFace, 0 );
        auto face = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
        std::vector<size_t> dofs;
        std::vector<size_t> scalarDofs;
        // Scale factors to get correct units
        const double h_scale = 1.0 / AMP::Operator::Subchannel::scaleEnthalpy;
        const double P_scale = 1.0 / AMP::Operator::Subchannel::scalePressure;
        for ( size_t i = 0; i < face.size(); i++ ) {
            faceDOFManager->getDOFs( face->globalID(), dofs );
            scalarFaceDOFManager->getDOFs( face->globalID(), scalarDofs );
            std::map<std::string, std::shared_ptr<std::vector<double>>> subchannelArgMap;
            auto vec1 = std::make_shared<std::vector<double>>(
                1, h_scale * flowSolVec->getValueByGlobalID( dofs[0] ) );
            auto vec2 = std::make_shared<std::vector<double>>(
                1, P_scale * flowSolVec->getValueByGlobalID( dofs[1] ) );
            subchannelArgMap.insert( std::make_pair( "enthalpy", vec1 ) );
            subchannelArgMap.insert( std::make_pair( "pressure", vec2 ) );
            std::vector<double> outTemperatureResult( 1 );
            subchannelPhysicsModel->getProperty(
                "Temperature", outTemperatureResult, subchannelArgMap );
            std::vector<double> specificVolume( 1 );
            subchannelPhysicsModel->getProperty(
                "SpecificVolume", specificVolume, subchannelArgMap );
            flowTempVec->setValueByGlobalID( scalarDofs[0], outTemperatureResult[0] );
            flowDensityVec->setValueByGlobalID( scalarDofs[0], 1.0 / specificVolume[0] );
            ++face;
        }
        flowTempVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        double Tin = db->getDatabase( "SubchannelTwoEqNonlinearOperator" )
                         ->getScalar<double>( "Inlet_Temperature" );
        deltaFlowTempVec = flowTempVec->clone();
        deltaFlowTempVec->copyVector( flowTempVec );
        deltaFlowTempVec->addScalar( *deltaFlowTempVec, -Tin );
    }
    double flowTempMin = 1e100;
    double flowTempMax = -1e100;
    if ( flowTempVec ) {
        flowTempMin = static_cast<double>( flowTempVec->min() );
        flowTempMax = static_cast<double>( flowTempVec->max() );
    }

    AMP::pout << "Subchannel Flow Temp Max : " << flowTempMax << " Min : " << flowTempMin
              << std::endl;

    // Test the subchannel to point map
    auto mapParams    = std::make_shared<AMP::Operator::SubchannelToPointMapParameters>();
    mapParams->d_Mesh = subchannelMesh;
    mapParams->d_comm = globalComm;
    mapParams->d_subchannelPhysicsModel = subchannelPhysicsModel;
    mapParams->d_outputVar = std::make_shared<AMP::LinearAlgebra::Variable>( "Density" );
    if ( subchannelMesh ) {
        auto face = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
        for ( size_t i = 0; i < face.size(); i++ ) {
            auto pos = face->centroid();
            mapParams->x.push_back( pos[0] );
            mapParams->y.push_back( pos[1] );
            mapParams->z.push_back( pos[2] );
            ++face;
        }
        AMP_ASSERT( mapParams->x.size() == flowDensityVec->getLocalSize() );
    }
    AMP::Operator::SubchannelToPointMap densityMap( mapParams );
    mapParams->d_outputVar = std::make_shared<AMP::LinearAlgebra::Variable>( "Temperature" );
    AMP::Operator::SubchannelToPointMap temperatureMap( mapParams );
    auto densityMapVec = AMP::LinearAlgebra::createSimpleVector<double>(
        mapParams->x.size(), densityMap.getOutputVariable() );
    auto temperatureMapVec = AMP::LinearAlgebra::createSimpleVector<double>(
        mapParams->x.size(), temperatureMap.getOutputVariable() );
    densityMap.apply( flowSolVec, densityMapVec );
    temperatureMap.apply( flowSolVec, temperatureMapVec );
    if ( subchannelMesh ) {
        auto face = xyFaceMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
        std::vector<size_t> dofs;
        bool pass_density     = true;
        bool pass_temperature = true;
        for ( size_t i = 0; i < face.size(); i++ ) {
            flowDensityVec->getDOFManager()->getDOFs( face->globalID(), dofs );
            double density1 = flowDensityVec->getValueByGlobalID( dofs[0] );
            double density2 = densityMapVec->getValueByLocalID( i );
            if ( !AMP::Utilities::approx_equal( density1, density2 ) )
                pass_density = false;
            double temp1 = flowTempVec->getValueByGlobalID( dofs[0] );
            double temp2 = temperatureMapVec->getValueByLocalID( i );
            if ( !AMP::Utilities::approx_equal( temp1, temp2 ) )
                pass_temperature = false;
            ++face;
        }
        if ( pass_density )
            ut->passes( "Subchannel density to point map" );
        else
            ut->failure( "Subchannel density to point map" );
        if ( pass_temperature )
            ut->passes( "Subchannel temperature to point map" );
        else
            ut->failure( "Subchannel temperature to point map" );
    }

    ut->passes( "test runs to completion" );

    globalComm.barrier();
    PROFILE_SAVE( "exeName" );
}


int testSubchannelSolve( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    PROFILE_ENABLE( 0 );

    std::string exeName = "testSubchannelSolve-1";
    if ( argc == 2 )
        exeName = argv[1];

    SubchannelSolve( &ut, exeName );

    ut.report();
    PROFILE_SAVE( exeName );

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
