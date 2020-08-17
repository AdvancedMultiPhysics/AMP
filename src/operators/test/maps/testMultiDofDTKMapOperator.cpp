#include "AMP/ampmesh/Mesh.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/map/dtk/DTKMapOperator.h"
#include "AMP/operators/map/dtk/MultiDofDTKMapOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/Writer.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"
#include <discretization/simpleDOF_Manager.h>

#define __INIT_FN1__( x, y, z ) ( x + y + z )
#define __INIT_FN2__( x, y, z ) ( 2 * x + y + z )
#define __INIT_FN3__( x, y, z ) ( 4 * x + y + z )

AMP::LinearAlgebra::VS_Comm createCommSelect( AMP::AMP_MPI globalComm, bool createOnThisRank )
{
    int inComm = createOnThisRank ? 1 : 0;
    // Create a comm spanning the meshes
    AMP::LinearAlgebra::VS_Comm commSelect( AMP::AMP_MPI( globalComm.split( inComm ) ) );
    return commSelect;
}

int runTest( std::string exeName, AMP::UnitTest *ut )
{
    std::string const inputFile = "input_" + exeName;
    std::string const logFile   = "output_" + exeName;

    AMP::PIO::logAllNodes( logFile );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Parse input file
    std::shared_ptr<AMP::Database> inputDatabase( new AMP::Database( "input_db" ) );
    AMP::Database::parseInputFile( inputFile, inputDatabase );

    // Read the mesh
    AMP::pout << "--------------------\n";
    AMP::pout << "    LOADING MESH    \n";
    AMP::pout << "--------------------\n";
    AMP::Database::shared_ptr meshDatabase = inputDatabase->getDatabase( "Mesh" );
    AMP::Mesh::MeshParameters::shared_ptr meshParams(
        new AMP::Mesh::MeshParameters( meshDatabase ) );
    meshParams->setComm( globalComm );
    AMP::Mesh::Mesh::shared_ptr mesh = AMP::Mesh::Mesh::buildMesh( meshParams );

    // Subset the mesh
    AMP::Mesh::Mesh::shared_ptr cellSandwichMesh = mesh->Subset( "CellSandwich_2_1" );
    AMP::Mesh::Mesh::shared_ptr anodeCCMesh      = mesh->Subset( "AnodeCC_1_1" );
    AMP::Mesh::Mesh::shared_ptr cathodeCCMesh    = mesh->Subset( "CathodeCC_3_1" );

    // Distribute degrees of freedom
    int const ghostWidth = 1;
    bool const split     = true;
    AMP::Discretization::DOFManager::shared_ptr phiDofMap =
        AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, ghostWidth, 1, split );
    AMP::Discretization::DOFManager::shared_ptr eectDofMap =
        AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, ghostWidth, 5, split );

    // Construct Vectors
    AMP::pout << "------------------------------------\n";
    AMP::pout << "     BUILD VECTORS AND FILL THEM    \n";
    AMP::pout << "------------------------------------\n";
    AMP::LinearAlgebra::Variable::shared_ptr potentialVariable(
        new AMP::LinearAlgebra::Variable( "Potential" ) );
    AMP::LinearAlgebra::Vector::shared_ptr potentialMapVec =
        AMP::LinearAlgebra::createVector( phiDofMap, potentialVariable, split );
    AMP::LinearAlgebra::Vector::shared_ptr potentialSolVec =
        AMP::LinearAlgebra::createVector( phiDofMap, potentialVariable, split );
    AMP::LinearAlgebra::Vector::shared_ptr potentialResVec =
        AMP::LinearAlgebra::createVector( phiDofMap, potentialVariable, split );
    AMP::LinearAlgebra::Vector::shared_ptr potentialRhsVec =
        AMP::LinearAlgebra::createVector( phiDofMap, potentialVariable, split );

    AMP::LinearAlgebra::Variable::shared_ptr batteryVariables(
        new AMP::LinearAlgebra::Variable( "Battery" ) );
    AMP::LinearAlgebra::Vector::shared_ptr BatterySolVec =
        AMP::LinearAlgebra::createVector( eectDofMap, batteryVariables, split );
    AMP::LinearAlgebra::Vector::shared_ptr BatteryResVec =
        AMP::LinearAlgebra::createVector( eectDofMap, batteryVariables, split );
    AMP::LinearAlgebra::Vector::shared_ptr BatteryMapVec =
        AMP::LinearAlgebra::createVector( eectDofMap, batteryVariables, split );
    AMP::LinearAlgebra::Vector::shared_ptr BatteryRhsVec =
        AMP::LinearAlgebra::createVector( eectDofMap, batteryVariables, split );

    AMP::LinearAlgebra::Vector::shared_ptr ElectrodeSolVec =
        BatterySolVec->select( AMP::LinearAlgebra::VS_Stride( 3, 5 ), "V4" );
    AMP::LinearAlgebra::Vector::shared_ptr ElectrodeMapVec =
        BatteryMapVec->select( AMP::LinearAlgebra::VS_Stride( 3, 5 ), "V4" );
    //---------------------------------------------------

    AMP::Utilities::Writer::shared_ptr siloWriter = AMP::Utilities::Writer::buildWriter( "Silo" );
    siloWriter->registerMesh( mesh );
    siloWriter->setDecomposition( 1 );
    siloWriter->registerVector(
        potentialMapVec, mesh, AMP::Mesh::GeomType::Vertex, "potentialMapVec" );
    siloWriter->registerVector(
        potentialSolVec, mesh, AMP::Mesh::GeomType::Vertex, "potentialSolVec" );
    siloWriter->registerVector(
        ElectrodeMapVec, mesh, AMP::Mesh::GeomType::Vertex, "batteryMapVec" );
    siloWriter->registerVector(
        ElectrodeSolVec, mesh, AMP::Mesh::GeomType::Vertex, "batterySolVec" );

    //---------------------------------------------------

    std::shared_ptr<AMP::LinearAlgebra::MultiVector> multiSolVec =
        AMP::LinearAlgebra::MultiVector::create( "MultiSolVec", globalComm );
    multiSolVec->addVector( BatterySolVec );
    multiSolVec->addVector( potentialSolVec );

    std::shared_ptr<AMP::LinearAlgebra::MultiVector> multiResVec =
        AMP::LinearAlgebra::MultiVector::create( "MultiResVec", globalComm );
    multiResVec->addVector( BatteryResVec );
    multiResVec->addVector( potentialResVec );

    std::shared_ptr<AMP::LinearAlgebra::MultiVector> multiRhsVec =
        AMP::LinearAlgebra::MultiVector::create( "MultiRhsVec", globalComm );
    multiRhsVec->addVector( BatteryRhsVec );
    multiRhsVec->addVector( potentialRhsVec );

    // Make new vectors
    std::shared_ptr<AMP::LinearAlgebra::MultiVector> multiMapVec =
        AMP::LinearAlgebra::MultiVector::create( "MultiMapVec", globalComm );
    multiMapVec->addVector( BatteryMapVec );
    multiMapVec->addVector( potentialMapVec );

    multiSolVec->setToScalar( -1.0 );
    multiMapVec->setToScalar( -2.0 ); // TODO: possible issue here...


    // Filling the vectors
    AMP::Mesh::MeshIterator node;
    AMP::Mesh::MeshIterator end_node;
    if ( anodeCCMesh ) {
        node     = anodeCCMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, 5, 0 );
        end_node = node.end();
        for ( ; node != end_node; ++node ) {
            std::vector<size_t> bndGlobalIds;
            phiDofMap->getDOFs( node->globalID(), bndGlobalIds );

            std::vector<double> pt = node->coord();
            double val             = __INIT_FN1__( pt[0], pt[1], pt[2] );
            AMP_ASSERT( bndGlobalIds.size() == 1 );

            potentialSolVec->setValueByGlobalID( bndGlobalIds[0], val );
        } // end for node
    }

    if ( cathodeCCMesh ) {
        node     = cathodeCCMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, 3, 0 );
        end_node = node.end();
        for ( ; node != end_node; ++node ) {
            std::vector<size_t> bndGlobalIds;
            phiDofMap->getDOFs( node->globalID(), bndGlobalIds );

            std::vector<double> pt = node->coord();
            double val             = __INIT_FN3__( pt[0], pt[1], pt[2] );
            AMP_ASSERT( bndGlobalIds.size() == 1 );

            potentialSolVec->setValueByGlobalID( bndGlobalIds[0], val );
        } // end for node
    }

    if ( cellSandwichMesh ) {
        node     = cellSandwichMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, 1, 0 );
        end_node = node.end();
        for ( ; node != end_node; ++node ) {
            std::vector<size_t> bndGlobalIds;
            eectDofMap->getDOFs( node->globalID(), bndGlobalIds );

            std::vector<double> pt = node->coord();
            double val             = __INIT_FN2__( pt[0], pt[1], pt[2] );
            AMP_ASSERT( bndGlobalIds.size() == 5 );

            BatterySolVec->setValueByGlobalID( bndGlobalIds[3], val );
        } // end for node

        node     = cellSandwichMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, 2, 0 );
        end_node = node.end();
        for ( ; node != end_node; ++node ) {
            std::vector<size_t> bndGlobalIds;
            eectDofMap->getDOFs( node->globalID(), bndGlobalIds );

            std::vector<double> pt = node->coord();
            double val             = __INIT_FN2__( pt[0], pt[1], pt[2] );
            AMP_ASSERT( bndGlobalIds.size() == 5 );

            BatterySolVec->setValueByGlobalID( bndGlobalIds[3], val );
        } // end for node
    }
    multiSolVec->makeConsistent( AMP::LinearAlgebra::Vector::ScatterType::CONSISTENT_SET );
    siloWriter->writeFile( logFile, 0 );

    // create dtk map operator.
    AMP::pout << "----------------------------\n";
    AMP::pout << "     CREATE MAP OPERATOR    \n";
    AMP::pout << "----------------------------\n";
    std::shared_ptr<AMP::Database> nullDatabase;
    // INTERFACE WITH ANODE
    AMP::pout << "interface anodeCC cellSandwich\n";
    std::shared_ptr<AMP::Operator::MultiDofDTKMapOperatorParameters>
        anodeCCCellSandwichMapOperatorParams(
            new AMP::Operator::MultiDofDTKMapOperatorParameters( nullDatabase ) );
    anodeCCCellSandwichMapOperatorParams->d_globalComm    = globalComm;
    anodeCCCellSandwichMapOperatorParams->d_Mesh1         = anodeCCMesh;
    anodeCCCellSandwichMapOperatorParams->d_BoundaryID1   = 5;
    anodeCCCellSandwichMapOperatorParams->d_Variable1     = potentialVariable->getName();
    anodeCCCellSandwichMapOperatorParams->d_StrideOffset1 = 0;
    anodeCCCellSandwichMapOperatorParams->d_StrideLength1 = 1;
    anodeCCCellSandwichMapOperatorParams->d_Mesh2         = cellSandwichMesh;
    anodeCCCellSandwichMapOperatorParams->d_BoundaryID2   = 1;
    anodeCCCellSandwichMapOperatorParams->d_Variable2     = batteryVariables->getName();
    anodeCCCellSandwichMapOperatorParams->d_StrideOffset2 = 3;
    anodeCCCellSandwichMapOperatorParams->d_StrideLength2 = 5;
    anodeCCCellSandwichMapOperatorParams->d_SourceVector  = multiSolVec;
    anodeCCCellSandwichMapOperatorParams->d_TargetVector  = multiMapVec;
    std::shared_ptr<AMP::Operator::Operator> anodeCCCellSandwichMapOperator(
        new AMP::Operator::MultiDofDTKMapOperator( anodeCCCellSandwichMapOperatorParams ) );

    // INTERFACE WITH CATHODE
    AMP::pout << "interface cellSandwich cathodeCC\n";
    std::shared_ptr<AMP::Operator::MultiDofDTKMapOperatorParameters>
        cellSandwichCathodeCCMapOperatorParams(
            new AMP::Operator::MultiDofDTKMapOperatorParameters( nullDatabase ) );
    cellSandwichCathodeCCMapOperatorParams->d_globalComm    = globalComm;
    cellSandwichCathodeCCMapOperatorParams->d_Mesh1         = cellSandwichMesh;
    cellSandwichCathodeCCMapOperatorParams->d_BoundaryID1   = 2;
    cellSandwichCathodeCCMapOperatorParams->d_Variable1     = batteryVariables->getName();
    cellSandwichCathodeCCMapOperatorParams->d_StrideOffset1 = 3;
    cellSandwichCathodeCCMapOperatorParams->d_StrideLength1 = 5;
    cellSandwichCathodeCCMapOperatorParams->d_Mesh2         = cathodeCCMesh;
    cellSandwichCathodeCCMapOperatorParams->d_BoundaryID2   = 3;
    cellSandwichCathodeCCMapOperatorParams->d_Variable2     = potentialVariable->getName();
    cellSandwichCathodeCCMapOperatorParams->d_StrideOffset2 = 0;
    cellSandwichCathodeCCMapOperatorParams->d_StrideLength2 = 1;
    cellSandwichCathodeCCMapOperatorParams->d_SourceVector  = multiSolVec;
    cellSandwichCathodeCCMapOperatorParams->d_TargetVector  = multiMapVec;
    std::shared_ptr<AMP::Operator::Operator> cellSandwichCathodeCCMapOperator(
        new AMP::Operator::MultiDofDTKMapOperator( cellSandwichCathodeCCMapOperatorParams ) );

    // apply the map.
    AMP::pout << "----------------------\n";
    AMP::pout << "     APPLY THE MAP    \n";
    AMP::pout << "----------------------\n";
    siloWriter->writeFile( logFile, 1 );
    AMP::pout << "interface cellSandwich cathodeCC\n";
    cellSandwichCathodeCCMapOperator->apply( multiSolVec, multiResVec );
    siloWriter->writeFile( logFile, 2 );
    AMP::pout << "interface anodeCC cellSandwich\n";
    anodeCCCellSandwichMapOperator->apply( multiSolVec,
                                           multiResVec ); // this map doesn't seem to work properly
    siloWriter->writeFile( logFile, 3 );


    // check the answer
    AMP::pout << "----------------------\n";
    AMP::pout << "     COMPUTE ERROR    \n";
    AMP::pout << "----------------------\n";
    double const absoluteTolerance = 1.0e-14;
    double const relativeTolerance = 1.0e-14;
    AMP::LinearAlgebra::Vector::const_shared_ptr errorVec;
    double tolerance;
    double errorNorm;
    std::string whatAmIChecking;


    std::vector<AMP::Mesh::MeshID> meshIDs = mesh->getBaseMeshIDs();
    for ( size_t meshIndex = 0; meshIndex < meshIDs.size(); meshIndex++ ) {
        AMP::Mesh::Mesh::shared_ptr adapter = mesh->Subset( meshIDs[meshIndex] );
        if ( adapter.get() == NULL )
            continue;

        std::string meshName = adapter->getName();
        AMP::LinearAlgebra::VS_Mesh meshSelector( adapter );

        AMP::LinearAlgebra::Vector::const_shared_ptr commSubsetPVec =
            potentialMapVec->constSelect( meshSelector, "Potential" );
        AMP::LinearAlgebra::Vector::const_shared_ptr commSubsetEVec =
            ElectrodeMapVec->constSelect( meshSelector, "V4" );

        if ( meshName.compare( "CathodeCC_3_1" ) == 0 ) {
            node     = adapter->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, 3, 0 );
            end_node = node.end();
            errorVec = commSubsetPVec->constSelect(
                AMP::LinearAlgebra::VS_MeshIterator( node.begin(), adapter->getComm() ), "error" );
            tolerance       = absoluteTolerance + relativeTolerance * errorVec->L2Norm();
            whatAmIChecking = "interface between cellSandwich and cathodeCC - cathodeCC side";

            for ( ; node != end_node; ++node ) {
                std::vector<size_t> bndGlobalIds;
                phiDofMap->getDOFs( node->globalID(), bndGlobalIds );

                std::vector<double> pt = node->coord();
                double val             = __INIT_FN2__( pt[0], pt[1], pt[2] );
                potentialMapVec->addValueByGlobalID( bndGlobalIds[0], -val );
            } // end for node
            errorNorm = errorVec->L2Norm();
            std::cout << whatAmIChecking << " error = " << errorNorm << "\n";
            if ( errorNorm > tolerance )
                ut->failure( whatAmIChecking );

        } else if ( meshName.compare( "AnodeCC_1_1" ) == 0 ) {
            node     = adapter->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, 5, 0 );
            end_node = node.end();
            errorVec = commSubsetPVec->constSelect(
                AMP::LinearAlgebra::VS_MeshIterator( node.begin(), adapter->getComm() ), "error" );
            tolerance       = absoluteTolerance + relativeTolerance * errorVec->L2Norm();
            whatAmIChecking = "interface between cellSandwich and anodeCC - anodeCC side";

            for ( ; node != end_node; ++node ) {
                std::vector<size_t> bndGlobalIds;
                phiDofMap->getDOFs( node->globalID(), bndGlobalIds );

                std::vector<double> pt = node->coord();
                double val             = __INIT_FN2__( pt[0], pt[1], pt[2] );
                potentialMapVec->addValueByGlobalID( bndGlobalIds[0], -val );
            } // end for node
            errorNorm = errorVec->L2Norm();
            std::cout << whatAmIChecking << " error = " << errorNorm << "\n";
            if ( errorNorm > tolerance )
                ut->failure( whatAmIChecking );

        } else if ( meshName.compare( "CellSandwich_2_1" ) == 0 ) {
            node     = adapter->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, 2, 0 );
            end_node = node.end();
            errorVec = commSubsetEVec->constSelect(
                AMP::LinearAlgebra::VS_MeshIterator( node.begin(), adapter->getComm() ), "error" );
            tolerance       = absoluteTolerance + relativeTolerance * errorVec->L2Norm();
            whatAmIChecking = "interface between cellSandwich and cathodeCC - cellSandwich side";

            for ( ; node != end_node; ++node ) {
                std::vector<size_t> bndGlobalIds;
                eectDofMap->getDOFs( node->globalID(), bndGlobalIds );

                std::vector<double> pt = node->coord();
                double val             = __INIT_FN3__( pt[0], pt[1], pt[2] );
                BatteryMapVec->addValueByGlobalID( bndGlobalIds[3], -val );
            } // end for node
            errorNorm = errorVec->L2Norm();
            std::cout << whatAmIChecking << " error = " << errorNorm << "\n";
            if ( errorNorm > tolerance )
                ut->failure( whatAmIChecking );
            ////########################################
            node     = adapter->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, 1, 0 );
            end_node = node.end();
            errorVec = commSubsetEVec->constSelect(
                AMP::LinearAlgebra::VS_MeshIterator( node.begin(), adapter->getComm() ), "error" );
            tolerance       = absoluteTolerance + relativeTolerance * errorVec->L2Norm();
            whatAmIChecking = "interface between cellSandwich and anodeCC - cellSandwich side";

            for ( ; node != end_node; ++node ) {
                std::vector<size_t> bndGlobalIds;
                eectDofMap->getDOFs( node->globalID(), bndGlobalIds );

                std::vector<double> pt = node->coord();
                double val             = __INIT_FN1__( pt[0], pt[1], pt[2] );
                BatteryMapVec->addValueByGlobalID( bndGlobalIds[3], -val );
            } // end for node
            errorNorm = errorVec->L2Norm();
            std::cout << whatAmIChecking << " error = " << errorNorm << "\n";
            if ( errorNorm > tolerance )
                ut->failure( whatAmIChecking );
        }
    }

    siloWriter->writeFile( logFile, 4 );

    return 1;
}

int testMultiDofDTKMapOperator( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );

    AMP::UnitTest ut;

    runTest( "testMultiDofDTKMapOperator-1", &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}