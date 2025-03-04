//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   operators/test/testMoabInterface.cc
 * \brief  This tests the Moab iMesh interface
 */
//---------------------------------------------------------------------------//

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "AMP/IO/PIO.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"

// MOAB Includes
#include "Coupler.hpp"
#include "MBParallelComm.hpp"
#include "MBRange.hpp"
#include "MBiMesh.hpp"
#include "iMesh.h"
#include "moab/Interface.hpp"


extern "C" {
static void getmoabmeshdata_( void **, void ** );
}
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

static void moabInterface( AMP::UnitTest *ut )
{

#ifdef AMP_USE_MOAB

    // Create new iMesh instance
    AMP::pout << "Creating iMesh instance" << std::endl;
    iMesh_Instance mesh;
    int ierr;
    std::string options;
    iMesh_newMesh( options.c_str(), &mesh, &ierr, 0 );

    AMP::pout << "iMesh_newMesh error code: " << ierr << std::endl;

    // Load mesh from file
    AMP::pout << "Loading mesh file" << std::endl;
    std::string meshFile = "pellet_1x.e";
    iMesh_load(
        mesh, 0, meshFile.c_str(), options.c_str(), &ierr, meshFile.length(), options.length() );

    AMP::pout << "iMesh_load error code: " << ierr << std::endl;

    // Get list of regions
    iBase_EntityHandle *regions;
    int regions_alloc = 0, regions_size;
    iMesh_getEntities( mesh,
                       0,
                       iBase_REGION,
                       iMesh_ALL_TOPOLOGIES,
                       &regions,
                       &regions_alloc,
                       &regions_size,
                       &ierr );

    AMP::pout << "getEntities error code: " << ierr << std::endl;

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    AMP::pout << "Mesh has " << regions_size << " elements" << std::endl;

    // Right now each processor is reading the mesh independently
    //  so the mesh size is the same as the global size on all procs
    if ( regions_size == 3705 )
        ut->passes( "Mesh is the right size" );
    else
        ut->failure( "Mesh is not the right size" );

    // Get list of nodes
    iBase_EntityHandle *nodes;
    int nodes_alloc = 0, nodes_size;
    iMesh_getEntities(
        mesh, 0, iBase_VERTEX, iMesh_ALL_TOPOLOGIES, &nodes, &nodes_alloc, &nodes_size, &ierr );

    AMP::pout << "getEntities error code: " << ierr << std::endl;

    AMP::pout << "Mesh has " << nodes_size << " vertices" << std::endl;

    // Create vector with "temperature" data
    std::vector<double> myTemps( nodes_size, -1.0 );
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dist( 800, 850 );
    for ( int i = 0; i < nodes_size; ++i )
        myTemps[i] = dist( gen );

    // Add a temperature tag
    iBase_TagHandle tempTagHandle;
    std::string tempTagName = "Temperature";
    iMesh_createTag(
        mesh, tempTagName.c_str(), 1, MB_TYPE_DOUBLE, &tempTagHandle, &ierr, tempTagName.length() );

    AMP::pout << "createTag error code: " << ierr << std::endl;

    // Assign data to this tag
    iMesh_setDblArrData(
        mesh, nodes, nodes_size, tempTagHandle, &myTemps[0], myTemps.size(), &ierr );

    AMP::pout << "setDblArrData error code: " << ierr << std::endl;

    // accessing MBInterface
    AMP::pout << "Casting to MBInterface" << std::endl;
    MBiMesh *mbimesh               = reinterpret_cast<MBiMesh *>( mesh );
    moab::Interface *moabInterface = mbimesh->mbImpl;

    AMP::pout << "Getting dimension" << std::endl;
    int moabDim;
    MBErrorCode result = moabInterface->get_dimension( moabDim );
    AMP::pout << "Dimension is " << moabDim << std::endl;

    AMP::pout << "Getting vertex coordinates" << std::endl;
    std::vector<double> moabMeshCoords;
    result = moabInterface->get_vertex_coordinates( moabMeshCoords );

    // for( int i=0; i<moabMeshCoords.size()/3; ++i )
    //    AMP::pout << "Coord " << i << ": " << moabMeshCoords[3*i]   << " "
    //                                       << moabMeshCoords[3*i+1] << " "
    //                                       << moabMeshCoords[3*i+2] << std::endl;

    if ( moabMeshCoords.size() == 3 * nodes_size )
        ut->passes( "Retrieved correct number of vertices" );
    else
        ut->failure( "Did not retrieve correct number of vertices" );

    AMP::pout << "Retrieved " << moabMeshCoords.size() << " coordinates" << std::endl;


    // create MBParallelComm
    AMP::pout << "Creating MBParallelComm" << std::endl;
    int moabCommOut = 0;
    MBParallelComm *moabCommunicator =
        new MBParallelComm( moabInterface, globalComm.getCommunicator(), &moabCommOut );
    // moab::ParallelComm *moabCommunicator;

    // Access the Range on the source mesh.
    AMP::pout << "Getting MGRange object" << std::endl;
    moab::Range moabRange;
    int problemDimension      = 3;
    moab::ErrorCode moabError = moabCommunicator->get_part_entities( moabRange, problemDimension );

    AMP::pout << "get_part_entities error code: " << moabError << std::endl;

    // create MBCoupler
    AMP::pout << "Creating MBCoupler" << std::endl;
    int moabCouplerID = 0;
    moab::Coupler moabCoupler( moabInterface, moabCommunicator, moabRange, moabCouplerID );

    // Create list of points
    int numCoords = 3;
    std::vector<double> myCoords( 3 * numCoords, 0.0 );

    // First point
    myCoords[0] = 0.0;
    myCoords[1] = 0.0;
    myCoords[2] = 0.001;

    // Second point
    myCoords[3] = 0.001;
    myCoords[4] = 0.002;
    myCoords[5] = 0.002;

    // Third point
    myCoords[6] = -0.001;
    myCoords[7] = 0.002;
    myCoords[8] = -0.003;

    // Input coords to coupler
    AMP::pout << "Providing points to coupler" << std::endl;
    moabError = moabCoupler.locate_points( &myCoords[0], numCoords );

    AMP::pout << "locate_points error code: " << moabError << std::endl;

    // Perform interpolation
    AMP::pout << "Interpolating" << std::endl;
    std::vector<double> interpTemps( numCoords, 0.0 );
    moabError = moabCoupler.interpolate( moab::Coupler::LINEAR_FE, tempTagName, &interpTemps[0] );

    AMP::pout << "interpolate error code: " << moabError << std::endl;

#endif

    if ( ut->NumPassGlobal() == 0 )
        ut->failure( "if it doesn't pass, it must have failed." );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    moabInterface( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
#ifdef AMP_USE_NEK
// NEK_END();
#endif
    return num_failed;
}

//---------------------------------------------------------------------------//
//                 end of test_ORIGEN_utils.cc
//---------------------------------------------------------------------------//
