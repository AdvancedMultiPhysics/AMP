
#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include <iostream>
#include <string>
#include <cstdlib>

#include "utils/shared_ptr.h"

#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/AMPManager.h"
#include "utils/PIO.h"

#include "ampmesh/Mesh.h"

#include "operators/map/dtk/DTKAMPMeshEntityIterator.h"
#include "operators/map/dtk/DTKAMPMeshEntityLocalMap.h"

bool selectAll( DataTransferKit::Entity entity )
{
    return true;
}

void myTest(AMP::UnitTest *ut)
{
    std::string exeName("testDTKEntityLocalMap");
    std::string log_file = "output_" + exeName;
    std::string msgPrefix;
    AMP::PIO::logOnlyNodeZero(log_file);

    AMP::pout<<"Loading the  mesh"<<std::endl;
    boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
    AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

    std::string input_file = "input_" + exeName;
    AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
    input_db->printClassData(AMP::plog);

    AMP_INSIST(input_db->keyExists("Mesh"), "Key ''Mesh'' is missing!");
    boost::shared_ptr<AMP::Database> meshDatabase = input_db->getDatabase("Mesh");

    boost::shared_ptr<AMP::Mesh::MeshParameters> meshParams(new AMP::Mesh::MeshParameters(meshDatabase));
    meshParams->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));
    AMP::Mesh::Mesh::shared_ptr mesh = AMP::Mesh::Mesh::buildMesh(meshParams);

    AMP::Mesh::MeshIterator elem_iterator = mesh->getIterator( AMP::Mesh::Volume );

    // Create and test a local map.
    AMP::shared_ptr<DataTransferKit::EntityLocalMap> dtk_local_map(
	new AMP::Operator::AMPMeshEntityLocalMap() );

    DataTransferKit::EntityIterator dtk_iterator =
	AMP::Operator::AMPMeshEntityIterator( elem_iterator, selectAll );

    int num_points = 10;

    double epsilon = 1.0e-12;

    // Test the local map with elements.
    for ( dtk_iterator = dtk_iterator.begin();
    	  dtk_iterator != dtk_iterator.end();
    	  ++dtk_iterator )
    {
   	// Get the bounding box.
    	Teuchos::Tuple<double,6> element_box;
    	dtk_iterator->boundingBox( element_box );

	// Check the measure.
	double measure = (element_box[3]-element_box[0])*
			 (element_box[4]-element_box[1])*
			 (element_box[5]-element_box[2]);
	AMP_ASSERT( std::abs(measure-dtk_local_map->measure(*dtk_iterator)) < epsilon );

	// Check the centroid.
	std::vector<double> gold_centroid(3);
	gold_centroid[0] = (element_box[3]-element_box[0]) / 2.0 + element_box[0];
	gold_centroid[1] = (element_box[4]-element_box[1]) / 2.0 + element_box[1];
	gold_centroid[2] = (element_box[5]-element_box[2]) / 2.0 + element_box[2];
	Teuchos::Array<double> centroid(3);
	dtk_local_map->centroid( *dtk_iterator, centroid() );
	AMP_ASSERT( std::abs(gold_centroid[0]-centroid[0]) < epsilon );
	AMP_ASSERT( std::abs(gold_centroid[1]-centroid[1]) < epsilon );
	AMP_ASSERT( std::abs(gold_centroid[2]-centroid[2]) < epsilon );

	Teuchos::Array<double> point(3);
	Teuchos::Array<double> ref_point(3);
	Teuchos::Array<double> phys_point(3);
	for ( int n = 0; n < num_points; ++n )
	{
	    // Create a random point in the neighborhood of the box.
	    point[0] = centroid[0] + 
		       (element_box[3]-element_box[0])*
		       ( 2.0 * std::rand() / RAND_MAX - 1.0 );
	    point[1] = centroid[1] + 
		       (element_box[4]-element_box[1])*
		       ( 2.0 * std::rand() / RAND_MAX - 1.0 );
	    point[2] = centroid[2] + 
		       (element_box[5]-element_box[2])*
		       ( 2.0 * std::rand() / RAND_MAX - 1.0 );

	    // Determine if it is in the box.
	    bool gold_inclusion = ( point[0] >= element_box[0] ) &&
				  ( point[0] <= element_box[3] ) &&
				  ( point[1] >= element_box[1] ) &&
				  ( point[1] <= element_box[4] ) &&
				  ( point[2] >= element_box[2] ) &&
				  ( point[2] <= element_box[5] );

	    // Check safety of mapping to the reference frame.
	    bool is_safe = dtk_local_map->isSafeToMapToReferenceFrame(
		*dtk_iterator, point() );
	    AMP_ASSERT( is_safe == gold_inclusion );

	    // Map to the reference frame.
	    bool map_success = dtk_local_map->mapToReferenceFrame(
		*dtk_iterator, point(), ref_point() );
	    AMP_ASSERT( map_success );
	    	
	    // Check point inclusion.
	    bool point_inclusion = dtk_local_map->checkPointInclusion(
		*dtk_iterator, ref_point() );
	    AMP_ASSERT( point_inclusion == gold_inclusion );

	    // Check mapping to the physical frame.
	    dtk_local_map->mapToPhysicalFrame(
		*dtk_iterator, ref_point(), phys_point() );
	    AMP_ASSERT( std::abs(point[0]-phys_point[0]) < epsilon );
	    AMP_ASSERT( std::abs(point[1]-phys_point[1]) < epsilon );
	    AMP_ASSERT( std::abs(point[2]-phys_point[2]) < epsilon );
	}	
    }

    // Test the local map with nodes.
    AMP::Mesh::MeshIterator node_iterator = mesh->getIterator( AMP::Mesh::Vertex );
    for ( node_iterator = node_iterator.begin();
	  node_iterator != node_iterator.end();
	  ++node_iterator )
    {
	DataTransferKit::Entity dtk_node = AMP::Operator::AMPMeshEntity( *node_iterator );
	Teuchos::Array<double> centroid( 3 );    
	dtk_local_map->centroid( dtk_node, centroid() );
	std::vector<double> coords = node_iterator->coord();
	AMP_ASSERT( coords[0] == centroid[0] );
	AMP_ASSERT( coords[1] == centroid[1] );
	AMP_ASSERT( coords[2] == centroid[2] );
    }

    ut->passes( exeName );

}


int main(int argc, char *argv[])
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup(argc,argv,startup_properties);
    AMP::UnitTest ut;

    myTest(&ut);

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}   


