#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshListIterator.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/map/AsyncMapColumnOperator.h"
#include "AMP/operators/map/SubchannelToCladMap.h"
#include "AMP/operators/map/libmesh/SubchannelToCladGPMap.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.h"


static double getTemp( const AMP::Mesh::Point &x ) { return 500 + x[2] * 100; }


static AMP::Mesh::MeshIterator getZFaceIterator( std::shared_ptr<AMP::Mesh::Mesh> subChannel,
                                                 int ghostWidth )
{
    std::multimap<double, std::unique_ptr<AMP::Mesh::MeshElement>> xyFace;
    auto iterator = subChannel->getIterator( AMP::Mesh::GeomType::Face, ghostWidth );
    for ( size_t i = 0; i < iterator.size(); ++i ) {
        auto nodes    = iterator->getElements( AMP::Mesh::GeomType::Vertex );
        auto center   = iterator->centroid();
        bool is_valid = true;
        for ( auto &node : nodes ) {
            auto coord = node.coord();
            if ( !AMP::Utilities::approx_equal( coord[2], center[2], 1e-6 ) )
                is_valid = false;
        }
        if ( is_valid )
            xyFace.insert( decltype( xyFace )::value_type( center[2], iterator->clone() ) );
        ++iterator;
    }
    auto elements = std::make_shared<std::vector<std::unique_ptr<AMP::Mesh::MeshElement>>>();
    elements->reserve( xyFace.size() );
    for ( auto &elem : xyFace )
        elements->push_back( elem.second->clone() );
    return AMP::Mesh::MeshListIterator( elements );
}


static void runTest( const std::string &fname, AMP::UnitTest *ut )
{
    // Read the input file
    auto input_db = AMP::Database::parseInputFile( fname );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto mesh_db = input_db->getDatabase( "Mesh" );
    auto params  = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto manager  = AMP::Mesh::MeshFactory::create( params );
    auto pin_mesh = manager->Subset( "MultiPin" );
    std::shared_ptr<AMP::Mesh::Mesh> clad_mesh;
    if ( pin_mesh ) {
        pin_mesh->setName( "MultiPin" );
        clad_mesh = pin_mesh->Subset( "clad" );
    }
    auto subchannel_mesh = manager->Subset( "subchannel" );
    std::shared_ptr<AMP::Mesh::Mesh> subchannel_face;
    if ( subchannel_mesh ) {
        subchannel_mesh->setName( "subchannel" );
        subchannel_face = subchannel_mesh->Subset( getZFaceIterator( subchannel_mesh, 1 ) );
    }

    // Get the database for the map
    auto nodal_map_db = input_db->getDatabase( "SubchannelToNodeMap" );
    auto gauss_map_db = input_db->getDatabase( "SubchannelToGPMap" );

    // Create the DOFManagers and the vectors
    // int DOFsPerNode = map_db->getScalar<int>("DOFsPerObject");
    // std::string varName = map_db->getString("VariableName");
    int DOFsPerNode     = 1;
    std::string varName = "Temperature";
    auto temperature    = std::make_shared<AMP::LinearAlgebra::Variable>( varName );
    std::shared_ptr<AMP::Discretization::DOFManager> pin_DOFs;
    std::shared_ptr<AMP::Discretization::DOFManager> subchannel_DOFs;
    AMP::LinearAlgebra::Vector::shared_ptr T_clad;
    AMP::LinearAlgebra::Vector::shared_ptr T_subchannel;
    AMP::LinearAlgebra::Vector::shared_ptr dummy;
    if ( pin_mesh ) {
        pin_DOFs = AMP::Discretization::simpleDOFManager::create(
            pin_mesh, AMP::Mesh::GeomType::Vertex, 1, DOFsPerNode );
        T_clad = AMP::LinearAlgebra::createVector( pin_DOFs, temperature );
        T_clad->setToScalar( 500 );
    }
    if ( subchannel_face ) {
        subchannel_DOFs = AMP::Discretization::simpleDOFManager::create(
            subchannel_face, AMP::Mesh::GeomType::Face, 1, DOFsPerNode );
        T_subchannel = AMP::LinearAlgebra::createVector( subchannel_DOFs, temperature );
        T_subchannel->setToScalar( 0.0 );
    }

    // Initialize the subchannel temperatures
    if ( subchannel_face ) {
        auto it = subchannel_face->getIterator( AMP::Mesh::GeomType::Face, 0 );
        std::vector<size_t> dofs;
        for ( size_t i = 0; i < it.size(); i++ ) {
            subchannel_DOFs->getDOFs( it->globalID(), dofs );
            double val = getTemp( it->centroid() );
            T_subchannel->setValuesByGlobalID( 1, &dofs[0], &val );
            ++it;
        }
    }


    // Test the creation/destruction of SubchannelToCladMap (no apply call)
    try {
        auto map = AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::SubchannelToCladMap>(
            manager, nodal_map_db );
        map.reset();
        ut->passes( "Created / Destroyed SubchannelToCladMap" );
    } catch ( ... ) {
        ut->failure( "Created / Destroyed SubchannelToCladMap" );
    }
    try {
        auto map =
            AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::SubchannelToCladGPMap>(
                manager, gauss_map_db );
        map.reset();
        ut->passes( "Created / Destroyed SubchannelToCladGPMap" );
    } catch ( ... ) {
        ut->failure( "Created / Destroyed SubchannelToCladGPMap" );
    }


    // Perform a complete test of SubchannelToCladMap
    auto map = AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::SubchannelToCladMap>(
        manager, nodal_map_db );
    map->setVector( T_clad );

    // Apply the map
    globalComm.barrier();
    map->apply( T_subchannel, dummy );

    // Check the results
    if ( pin_mesh ) {
        bool passes = true;
        auto it     = pin_mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, 4, 1 );
        std::vector<size_t> dofs;
        for ( size_t i = 0; i < it.size(); i++ ) {
            pin_DOFs->getDOFs( it->globalID(), dofs );
            AMP_ASSERT( dofs.size() == 1 );
            auto pos  = it->centroid();
            double v1 = T_clad->getValueByGlobalID( dofs[0] );
            double v2 = getTemp( pos );
            if ( !AMP::Utilities::approx_equal( v1, v2 ) )
                passes = false;
        }
        if ( passes )
            ut->passes( "correctly mapped temperature" );
        else
            ut->failure( "correctly mapped temperature" );
    }


    // Perform a complete test of SubchannelToCladGPMap
    std::shared_ptr<AMP::Discretization::DOFManager> gauss_DOFs;
    AMP::LinearAlgebra::Vector::shared_ptr T_gauss;
    if ( pin_mesh ) {
        gauss_DOFs = AMP::Discretization::simpleDOFManager::create(
            pin_mesh, AMP::Mesh::GeomType::Face, 1, 4 );
        T_gauss = AMP::LinearAlgebra::createVector( gauss_DOFs, temperature );
        T_gauss->zero();
    }
    map = AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::SubchannelToCladGPMap>(
        manager, gauss_map_db );
    map->setVector( T_gauss );

    // Apply the map
    globalComm.barrier();
    map->apply( T_subchannel, dummy );

    // Check the results
    if ( clad_mesh ) {
        bool passes = true;
        auto it     = clad_mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, 4, 1 );
        std::vector<size_t> dofs( 4 );
        std::vector<double> vals( 4 );
        for ( size_t i = 0; i < it.size(); i++ ) {
            gauss_DOFs->getDOFs( it->globalID(), dofs );
            AMP_ASSERT( dofs.size() == 4 );
            auto pos = it->centroid();
            vals.resize( dofs.size() );
            T_gauss->getValuesByGlobalID( dofs.size(), &dofs[0], &vals[0] );
            double v1 = ( vals[0] + vals[1] + vals[2] + vals[3] ) / 4;
            double v2 = getTemp( pos );
            if ( !AMP::Utilities::approx_equal( v1, v2 ) )
                passes = false;
        }
        if ( passes )
            ut->passes( "correctly mapped temperature (gauss points)" );
        else
            ut->failure( "correctly mapped temperature (gauss points)" );
    }
}


int testSubchannelToCladMap( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    runTest( "inputSubchannelToCladMap-1", &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
