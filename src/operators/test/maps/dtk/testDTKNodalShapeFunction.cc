
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>

#include "utils/shared_ptr.h"

#include "utils/AMPManager.h"
#include "utils/AMP_MPI.h"
#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/PIO.h"

#include "vectors/VectorBuilder.h"

#include "discretization/simpleDOF_Manager.h"

#include "ampmesh/Mesh.h"

#include "operators/map/dtk/DTKAMPMeshEntityIterator.h"
#include "operators/map/dtk/DTKAMPMeshNodalShapeFunction.h"

bool selectAll( DataTransferKit::Entity entity ) { return true; }

void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testDTKNodalShapeFunction" );
    std::string log_file = "output_" + exeName;
    std::string msgPrefix;
    AMP::PIO::logOnlyNodeZero( log_file );

    AMP::pout << "Loading the  mesh" << std::endl;
    AMP::shared_ptr<AMP::InputDatabase> input_db( new AMP::InputDatabase( "input_db" ) );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    std::string input_file = "input_" + exeName;
    AMP::InputManager::getManager()->parseInputFile( input_file, input_db );
    input_db->printClassData( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    AMP::shared_ptr<AMP::Database> meshDatabase = input_db->getDatabase( "Mesh" );

    AMP::shared_ptr<AMP::Mesh::MeshParameters> meshParams(
        new AMP::Mesh::MeshParameters( meshDatabase ) );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    AMP::Mesh::Mesh::shared_ptr mesh = AMP::Mesh::Mesh::buildMesh( meshParams );

    bool const split      = true;
    int const ghostWidth  = 1;
    int const dofsPerNode = 1;
    AMP::Discretization::DOFManager::shared_ptr dofManager =
        AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, ghostWidth, dofsPerNode );
    AMP::LinearAlgebra::Variable::shared_ptr variable( new AMP::LinearAlgebra::Variable( "var" ) );
    AMP::LinearAlgebra::Vector::shared_ptr vector =
        AMP::LinearAlgebra::createVector( dofManager, variable, split );

    // get the volume iterator
    AMP::Mesh::MeshIterator vol_iterator = mesh->getIterator( AMP::Mesh::GeomType::Volume );

    // get the vertex iterator
    AMP::Mesh::MeshIterator vert_iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex );

    // map the volume ids to dtk ids
    AMP::shared_ptr<std::map<AMP::Mesh::MeshElementID, DataTransferKit::EntityId>> vol_id_map =
        AMP::make_shared<std::map<AMP::Mesh::MeshElementID, DataTransferKit::EntityId>>();
    {
        int counter = 0;
        for ( vol_iterator = vol_iterator.begin(); vol_iterator != vol_iterator.end();
              ++vol_iterator, ++counter ) {
            vol_id_map->emplace( vol_iterator->globalID(), counter );
        }
        int comm_rank = globalComm.getRank();
        int comm_size = globalComm.getSize();
        std::vector<std::size_t> offsets( comm_size, 0 );
        globalComm.allGather( vol_id_map->size(), offsets.data() );
        for ( int n = 1; n < comm_size; ++n ) {
            offsets[n] += offsets[n - 1];
        }
        if ( comm_rank > 0 ) {
            for ( auto &i : *vol_id_map )
                i.second += offsets[comm_rank - 1];
        }
    }

    // map the vertex ids to dtk ids
    AMP::shared_ptr<std::map<AMP::Mesh::MeshElementID, DataTransferKit::EntityId>> vert_id_map =
        AMP::make_shared<std::map<AMP::Mesh::MeshElementID, DataTransferKit::EntityId>>();
    {
        int counter = 0;
        for ( vert_iterator = vert_iterator.begin(); vert_iterator != vert_iterator.end();
              ++vert_iterator, ++counter ) {
            vert_id_map->emplace( vert_iterator->globalID(), counter );
        }
        int comm_rank = globalComm.getRank();
        int comm_size = globalComm.getSize();
        std::vector<std::size_t> offsets( comm_size, 0 );
        globalComm.allGather( vert_id_map->size(), offsets.data() );
        for ( int n = 1; n < comm_size; ++n ) {
            offsets[n] += offsets[n - 1];
        }
        if ( comm_rank > 0 ) {
            for ( auto &i : *vert_id_map )
                i.second += offsets[comm_rank - 1];
        }
    }

    // make the rank map.
    AMP::shared_ptr<std::unordered_map<int, int>> rank_map =
        AMP::make_shared<std::unordered_map<int, int>>();
    auto global_ranks = mesh->getComm().globalRanks();
    int size          = mesh->getComm().getSize();
    for ( int n = 0; n < size; ++n ) {
        rank_map->emplace( global_ranks[n], n );
    }

    // Create and test a nodal shape function.
    AMP::shared_ptr<DataTransferKit::EntityShapeFunction> dtk_shape_function(
        new AMP::Operator::AMPMeshNodalShapeFunction( dofManager ) );

    Teuchos::Array<double> ref_center( 3 );
    ref_center[0] = 0.0;
    ref_center[1] = 0.0;
    ref_center[2] = 0.0;

    Teuchos::Array<double> ref_node_1( 3 );
    ref_node_1[0] = -1.0;
    ref_node_1[1] = -1.0;
    ref_node_1[2] = -1.0;

    Teuchos::Array<double> ref_node_2( 3 );
    ref_node_2[0] = 1.0;
    ref_node_2[1] = -1.0;
    ref_node_2[2] = -1.0;

    Teuchos::Array<double> ref_node_3( 3 );
    ref_node_3[0] = 1.0;
    ref_node_3[1] = 1.0;
    ref_node_3[2] = -1.0;

    Teuchos::Array<double> ref_node_4( 3 );
    ref_node_4[0] = -1.0;
    ref_node_4[1] = 1.0;
    ref_node_4[2] = -1.0;

    Teuchos::Array<double> ref_node_5( 3 );
    ref_node_5[0] = -1.0;
    ref_node_5[1] = -1.0;
    ref_node_5[2] = 1.0;

    Teuchos::Array<double> ref_node_6( 3 );
    ref_node_6[0] = 1.0;
    ref_node_6[1] = -1.0;
    ref_node_6[2] = 1.0;

    Teuchos::Array<double> ref_node_7( 3 );
    ref_node_7[0] = 1.0;
    ref_node_7[1] = 1.0;
    ref_node_7[2] = 1.0;

    Teuchos::Array<double> ref_node_8( 3 );
    ref_node_8[0] = -1.0;
    ref_node_8[1] = 1.0;
    ref_node_8[2] = 1.0;

    AMP::Mesh::MeshIterator elem_iterator = mesh->getIterator( AMP::Mesh::GeomType::Volume );
    DataTransferKit::EntityIterator dtk_elem_iterator =
        AMP::Operator::AMPMeshEntityIterator( rank_map, vol_id_map, elem_iterator, selectAll );

    Teuchos::Array<std::size_t> dof_ids;
    Teuchos::Array<double> values;
    Teuchos::Array<Teuchos::Array<double>> gradients;

    // Test the shape function for the hex elements.
    for ( dtk_elem_iterator = dtk_elem_iterator.begin();
          dtk_elem_iterator != dtk_elem_iterator.end();
          ++dtk_elem_iterator ) {
        // Check the DOF ids.
        dtk_shape_function->entitySupportIds( *dtk_elem_iterator, dof_ids );
        AMP_ASSERT( 8 == dof_ids.size() );

        // Evaluate the value at the centroid.
        dtk_shape_function->evaluateValue( *dtk_elem_iterator, ref_center(), values );
        AMP_ASSERT( 8 == values.size() );
        AMP_ASSERT( 0.125 == values[0] );
        AMP_ASSERT( 0.125 == values[1] );
        AMP_ASSERT( 0.125 == values[2] );
        AMP_ASSERT( 0.125 == values[3] );
        AMP_ASSERT( 0.125 == values[4] );
        AMP_ASSERT( 0.125 == values[5] );
        AMP_ASSERT( 0.125 == values[6] );
        AMP_ASSERT( 0.125 == values[7] );

        // Evaluate the value at node 1.
        dtk_shape_function->evaluateValue( *dtk_elem_iterator, ref_node_1(), values );
        AMP_ASSERT( 8 == values.size() );
        AMP_ASSERT( 1.0 == values[0] );
        AMP_ASSERT( 0.0 == values[1] );
        AMP_ASSERT( 0.0 == values[2] );
        AMP_ASSERT( 0.0 == values[3] );
        AMP_ASSERT( 0.0 == values[4] );
        AMP_ASSERT( 0.0 == values[5] );
        AMP_ASSERT( 0.0 == values[6] );
        AMP_ASSERT( 0.0 == values[7] );

        // Evaluate the value at node 2.
        dtk_shape_function->evaluateValue( *dtk_elem_iterator, ref_node_2(), values );
        AMP_ASSERT( 8 == values.size() );
        AMP_ASSERT( 0.0 == values[0] );
        AMP_ASSERT( 1.0 == values[1] );
        AMP_ASSERT( 0.0 == values[2] );
        AMP_ASSERT( 0.0 == values[3] );
        AMP_ASSERT( 0.0 == values[4] );
        AMP_ASSERT( 0.0 == values[5] );
        AMP_ASSERT( 0.0 == values[6] );
        AMP_ASSERT( 0.0 == values[7] );

        // Evaluate the value at node 3.
        dtk_shape_function->evaluateValue( *dtk_elem_iterator, ref_node_3(), values );
        AMP_ASSERT( 8 == values.size() );
        AMP_ASSERT( 0.0 == values[0] );
        AMP_ASSERT( 0.0 == values[1] );
        AMP_ASSERT( 1.0 == values[2] );
        AMP_ASSERT( 0.0 == values[3] );
        AMP_ASSERT( 0.0 == values[4] );
        AMP_ASSERT( 0.0 == values[5] );
        AMP_ASSERT( 0.0 == values[6] );
        AMP_ASSERT( 0.0 == values[7] );

        // Evaluate the value at node 4.
        dtk_shape_function->evaluateValue( *dtk_elem_iterator, ref_node_4(), values );
        AMP_ASSERT( 8 == values.size() );
        AMP_ASSERT( 0.0 == values[0] );
        AMP_ASSERT( 0.0 == values[1] );
        AMP_ASSERT( 0.0 == values[2] );
        AMP_ASSERT( 1.0 == values[3] );
        AMP_ASSERT( 0.0 == values[4] );
        AMP_ASSERT( 0.0 == values[5] );
        AMP_ASSERT( 0.0 == values[6] );
        AMP_ASSERT( 0.0 == values[7] );

        // Evaluate the value at node 5.
        dtk_shape_function->evaluateValue( *dtk_elem_iterator, ref_node_5(), values );
        AMP_ASSERT( 8 == values.size() );
        AMP_ASSERT( 0.0 == values[0] );
        AMP_ASSERT( 0.0 == values[1] );
        AMP_ASSERT( 0.0 == values[2] );
        AMP_ASSERT( 0.0 == values[3] );
        AMP_ASSERT( 1.0 == values[4] );
        AMP_ASSERT( 0.0 == values[5] );
        AMP_ASSERT( 0.0 == values[6] );
        AMP_ASSERT( 0.0 == values[7] );

        // Evaluate the value at node 6.
        dtk_shape_function->evaluateValue( *dtk_elem_iterator, ref_node_6(), values );
        AMP_ASSERT( 8 == values.size() );
        AMP_ASSERT( 0.0 == values[0] );
        AMP_ASSERT( 0.0 == values[1] );
        AMP_ASSERT( 0.0 == values[2] );
        AMP_ASSERT( 0.0 == values[3] );
        AMP_ASSERT( 0.0 == values[4] );
        AMP_ASSERT( 1.0 == values[5] );
        AMP_ASSERT( 0.0 == values[6] );
        AMP_ASSERT( 0.0 == values[7] );

        // Evaluate the value at node 7.
        dtk_shape_function->evaluateValue( *dtk_elem_iterator, ref_node_7(), values );
        AMP_ASSERT( 8 == values.size() );
        AMP_ASSERT( 0.0 == values[0] );
        AMP_ASSERT( 0.0 == values[1] );
        AMP_ASSERT( 0.0 == values[2] );
        AMP_ASSERT( 0.0 == values[3] );
        AMP_ASSERT( 0.0 == values[4] );
        AMP_ASSERT( 0.0 == values[5] );
        AMP_ASSERT( 1.0 == values[6] );
        AMP_ASSERT( 0.0 == values[7] );

        // Evaluate the value at node 8.
        dtk_shape_function->evaluateValue( *dtk_elem_iterator, ref_node_8(), values );
        AMP_ASSERT( 8 == values.size() );
        AMP_ASSERT( 0.0 == values[0] );
        AMP_ASSERT( 0.0 == values[1] );
        AMP_ASSERT( 0.0 == values[2] );
        AMP_ASSERT( 0.0 == values[3] );
        AMP_ASSERT( 0.0 == values[4] );
        AMP_ASSERT( 0.0 == values[5] );
        AMP_ASSERT( 0.0 == values[6] );
        AMP_ASSERT( 1.0 == values[7] );

        // Evaluate the gradient at the centroid.
        dtk_shape_function->evaluateGradient( *dtk_elem_iterator, ref_center(), gradients );
        AMP_ASSERT( 8 == gradients.size() );
        AMP_ASSERT( 3 == gradients[0].size() );
        AMP_ASSERT( -0.125 == gradients[0][0] );
        AMP_ASSERT( -0.125 == gradients[0][1] );
        AMP_ASSERT( -0.125 == gradients[0][2] );
        AMP_ASSERT( 3 == gradients[1].size() );
        AMP_ASSERT( 0.125 == gradients[1][0] );
        AMP_ASSERT( -0.125 == gradients[1][1] );
        AMP_ASSERT( -0.125 == gradients[1][2] );
        AMP_ASSERT( 3 == gradients[2].size() );
        AMP_ASSERT( 0.125 == gradients[2][0] );
        AMP_ASSERT( 0.125 == gradients[2][1] );
        AMP_ASSERT( -0.125 == gradients[2][2] );
        AMP_ASSERT( 3 == gradients[3].size() );
        AMP_ASSERT( -0.125 == gradients[3][0] );
        AMP_ASSERT( 0.125 == gradients[3][1] );
        AMP_ASSERT( -0.125 == gradients[3][2] );
        AMP_ASSERT( 3 == gradients[4].size() );
        AMP_ASSERT( -0.125 == gradients[4][0] );
        AMP_ASSERT( -0.125 == gradients[4][1] );
        AMP_ASSERT( 0.125 == gradients[4][2] );
        AMP_ASSERT( 3 == gradients[5].size() );
        AMP_ASSERT( 0.125 == gradients[5][0] );
        AMP_ASSERT( -0.125 == gradients[5][1] );
        AMP_ASSERT( 0.125 == gradients[5][2] );
        AMP_ASSERT( 3 == gradients[6].size() );
        AMP_ASSERT( 0.125 == gradients[6][0] );
        AMP_ASSERT( 0.125 == gradients[6][1] );
        AMP_ASSERT( 0.125 == gradients[6][2] );
        AMP_ASSERT( 3 == gradients[7].size() );
        AMP_ASSERT( -0.125 == gradients[7][0] );
        AMP_ASSERT( 0.125 == gradients[7][1] );
        AMP_ASSERT( 0.125 == gradients[7][2] );
    }

    // Test the shape function with the nodes.
    AMP::Mesh::MeshIterator node_iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex );
    DataTransferKit::EntityIterator dtk_node_iterator =
        AMP::Operator::AMPMeshEntityIterator( rank_map, vert_id_map, node_iterator, selectAll );
    std::vector<std::size_t> node_dofs;
    for ( dtk_node_iterator = dtk_node_iterator.begin(), node_iterator = node_iterator.begin();
          dtk_node_iterator != dtk_node_iterator.end();
          ++dtk_node_iterator, ++node_iterator ) {
        dtk_shape_function->entitySupportIds( *dtk_node_iterator, dof_ids );
        dofManager->getDOFs( node_iterator->globalID(), node_dofs );
        AMP_ASSERT( 1 == dof_ids.size() );
        AMP_ASSERT( 1 == node_dofs.size() );
        AMP_ASSERT( dof_ids[0] == node_dofs[0] );
    }

    ut->passes( exeName );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

    myTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
