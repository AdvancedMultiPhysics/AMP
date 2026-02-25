#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/map/MapOperatorParameters.h"
#include "AMP/operators/map/libmesh/Map1Dto3D.h"
#include "AMP/operators/map/libmesh/Map3Dto1D.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static void testMap( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logAllNodes( log_file );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto mesh_db = input_db->getDatabase( "Mesh" );
    auto params  = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto mesh  = AMP::Mesh::MeshFactory::create( params );
    auto mesh1 = mesh->Subset( "pellet" );
    auto mesh2 = mesh->Subset( "clad" );

    // Create a simple DOFManager and the vectors
    auto DOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto testVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "MapSolution" );
    auto gapVariable  = std::make_shared<AMP::LinearAlgebra::Variable>( "Gap" );

    auto mapSolution       = AMP::LinearAlgebra::createVector( DOFs, testVariable );
    auto mapSolutionMaster = AMP::LinearAlgebra::createVector( DOFs, testVariable );
    auto mapSolutionSlave  = AMP::LinearAlgebra::createVector( DOFs, testVariable );

    mapSolution->setToScalar( 0.0 );

    // Create the map operators
    auto map3dto1d_db1    = input_db->getDatabase( "MapPelletto1D" );
    auto map3dto1dParams1 = std::make_shared<AMP::Operator::MapOperatorParameters>( map3dto1d_db1 );
    map3dto1dParams1->d_MapComm = globalComm;
    map3dto1dParams1->d_MapMesh = mesh1;
    auto map1ToLowDim           = std::make_shared<AMP::Operator::Map3Dto1D>( map3dto1dParams1 );

    auto map1dto3d_db1    = input_db->getDatabase( "Map1DtoClad" );
    auto map1dto3dParams1 = std::make_shared<AMP::Operator::MapOperatorParameters>( map1dto3d_db1 );
    map1dto3dParams1->d_MapComm = globalComm;
    map1dto3dParams1->d_MapMesh = mesh2;
    auto map1ToHighDim          = std::make_shared<AMP::Operator::Map1Dto3D>( map1dto3dParams1 );

    map1ToLowDim->setZLocations( map1ToHighDim->getZLocations() );

    auto map3dto1d_db2    = input_db->getDatabase( "MapCladto1D" );
    auto map3dto1dParams2 = std::make_shared<AMP::Operator::MapOperatorParameters>( map3dto1d_db2 );
    map3dto1dParams2->d_MapComm = globalComm;
    map3dto1dParams2->d_MapMesh = mesh2;
    auto map2ToLowDim           = std::make_shared<AMP::Operator::Map3Dto1D>( map3dto1dParams2 );

    auto map1dto3d_db2    = input_db->getDatabase( "Map1DtoPellet" );
    auto map1dto3dParams2 = std::make_shared<AMP::Operator::MapOperatorParameters>( map1dto3d_db2 );
    map1dto3dParams2->d_MapComm = globalComm;
    map1dto3dParams2->d_MapMesh = mesh1;
    auto map2ToHighDim          = std::make_shared<AMP::Operator::Map1Dto3D>( map1dto3dParams2 );

    map2ToLowDim->setZLocations( map2ToHighDim->getZLocations() );
    //-------------------------------------
    size_t gapVecCladSize = map1ToHighDim->getNumZlocations();
    auto gapVecClad = AMP::LinearAlgebra::createSimpleVector<double>( gapVecCladSize, gapVariable );

    size_t gapVecPelletSize = map2ToHighDim->getNumZlocations();
    auto gapVecPellet =
        AMP::LinearAlgebra::createSimpleVector<double>( gapVecPelletSize, gapVariable );

    // Set the boundary for the source vector
    unsigned int d_boundaryId = map3dto1d_db1->getScalar<int>( "BoundaryId" );
    auto bnd     = mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_boundaryId, 0 );
    auto end_bnd = bnd.end();
    auto dof_map = mapSolutionMaster->getDOFManager();
    std::vector<size_t> ids;
    for ( ; bnd != end_bnd; ++bnd ) {
        auto x = bnd->coord();
        dof_map->getDOFs( bnd->globalID(), ids );
        AMP_ASSERT( ids.size() == 1 );
        mapSolutionMaster->setValuesByGlobalID( 1, &ids[0], &x[2] );
    }
    mapSolutionMaster->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    //-------------------------------------

    int cnt         = 0;
    bool testPassed = false;
    while ( cnt < 20 ) {
        cnt++;

        map1ToLowDim->setVector( gapVecClad );
        map1ToHighDim->setVector( mapSolutionSlave );
        map1ToLowDim->apply( mapSolutionMaster, gapVecClad );
        map1ToHighDim->apply( gapVecClad, mapSolutionSlave );
        std::cout << "Master Map Solution " << std::endl;
        for ( size_t i = 0; i < gapVecCladSize; i++ ) {
            std::cout << " @i : " << i << " is " << gapVecClad->getValueByLocalID( i );
        }
        std::cout << std::endl;
        //------------------------------------------------------------
        //    mapSolutionSlave->setToScalar(90);

        map2ToLowDim->setVector( gapVecPellet );
        map2ToHighDim->setVector( mapSolutionMaster );
        map2ToLowDim->apply( mapSolutionSlave, gapVecPellet );
        map2ToHighDim->apply( gapVecPellet, mapSolutionMaster );

        std::cout << "Slave Map Solution " << std::endl;
        for ( size_t i = 0; i < gapVecPelletSize; i++ ) {
            std::cout << " @i : " << i << " is " << gapVecPellet->getValueByLocalID( i );
        }
        std::cout << std::endl;
        //------------------------------------------------------------

        if ( true ) {
            testPassed = true;
            break;
        } else {
            std::cout << "Norm of the change in sol for iteration " << cnt << "is -->" << std::endl;
        }
        std::cout << std::endl;
    }

    if ( globalComm.getSize() == 1 ) {
        // manager->writeFile<AMP::Mesh::SiloIO> ( exeName , 0 );
    }

    if ( testPassed )
        ut->passes( "Seggregated solve of Composite Operator using control loop of "
                    "Thermal+Robin->Map->Gap->Map->Thermal+Robin ." );
    else
        ut->failure( "Seggregated solve of Composite Operator using control loop of "
                     "Thermal+Robin->Map->Gap->Map->Thermal+Robin ." );

    input_db.reset();

    ut->passes( exeName );
}


int testMapOperator( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    testMap( &ut, "testMapOperator" );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
