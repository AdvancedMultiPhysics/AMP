//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   operators/test/testNekPipe.cc
 * \brief  This tests the header file that accesses Nek-5000; runs Nek exmaple moab/pipe.
 *
 * This test is intended to test our ability to use a Nek-generated Moab 
 * instance.  It runs the Nek pipe problem, extracts the Moab instance
 * generated by Nek, builds a Moab Coupler and interpolates a variable onto
 * a set of points.  The Nek pipe problem does not do heat transfer so we
 * extract the pressure instead just to make sure we have some valid data.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>

#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/PIO.h"

// Nek includes
#include "nek/Nek5000_API2.h"

// MOAB Includes
#include "moab/Interface.hpp"
#include "moab/ParallelComm.hpp"
#include "moab/Range.hpp"
#include "Coupler.hpp"
#include "iMesh.h"
#include "MBiMesh.hpp"

extern "C" {
    void getmoabmeshdata_( void **, void ** );
}
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

void nekPipe(AMP::UnitTest *ut)
{
    // this test is based on testSNES-B-TM-4

    // Print Banner
    AMP::Utilities::printBanner();

    // Log all nodes
    AMP::PIO::logAllNodes( "output_testNekPipe" );


    AMP::pout << "Starting to run Nek-500 for the pipe problem"<< std::endl;

#ifdef USE_NEK     
	  std::cout<<"Preparing NEK Init"<< std::endl;
    // must agree with first line of SESSION.NAME
    std::string nekName = "pipe";
    NEK_PREPARE( nekName );
	  std::cout<<"Calling NEK Init"<< std::endl;
    int myMpiComm;
    NEK_INIT( &myMpiComm );
	  std::cout<<"NEK Init succeeded"<<std::endl;
    NEK_SOLVE();
    ut->passes("Nek has solved the pipe problem.");
    
    void *mesh_ptr;
    void *tag;
    getmoabmeshdata_( &mesh_ptr, &tag );

    iMesh_Instance mesh = (iMesh_Instance) mesh_ptr;

    AMP::pout << "Tag value is " << tag << std::endl;

    iBase_EntityHandle *ents;
    int ents_alloc = 0, ents_size;
    int ierr;
    iMesh_getEntities(mesh, 0, iBase_REGION,
                      iMesh_ALL_TOPOLOGIES,
                      &ents, &ents_alloc, 
                      &ents_size, &ierr);

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    int totalSize;
    totalSize = globalComm.sumReduce( ents_size );
    AMP::pout << "Mesh size is " << totalSize << std::endl;

    if( totalSize == 5496 )
        ut->passes("Mesh is the right size");
    else
        ut->failure("Mesh is not the right size");

    // accessing MBInterface
    AMP::pout << "Casting to MBInterface" << std::endl;
    MBiMesh *mbimesh = reinterpret_cast<MBiMesh*>(mesh);
    moab::Interface *moabInterface = mbimesh->mbImpl;

    AMP::pout << "Getting dimension" << std::endl;
    int moabDim;
    moab::ErrorCode result = moabInterface->get_dimension( moabDim );
    AMP::pout << "Dimension is " << moabDim << std::endl;

    AMP::pout << "Getting vertex coordinates" << std::endl;
    std::vector<double> nekMeshCoords;
    result = moabInterface->get_vertex_coordinates( nekMeshCoords );

    AMP::pout << "Retrieved " << nekMeshCoords.size() << " coordinates" << std::endl;
    
    // Retrieve ParallelComm from Interface
    // This step fails for a single processor (even using MPI) because Nek
    //  doesn't create a ParallelComm object for a single processor 
    // We don't currently have a solution for this, so for now we can only run
    //  with multiple processors
    AMP::pout << "Retrieving ParallelComm" << std::endl;
    std::vector< moab::ParallelComm* > pcomm_vec;
    moab::ParallelComm::get_all_pcomm( moabInterface, pcomm_vec );
    AMP::pout << "Retrieved " << pcomm_vec.size() << " pcomms" << std::endl;
    moab::ParallelComm *moabCommunicator = pcomm_vec[0];

    // Access the Range on the source mesh.
    AMP::pout << "Getting Range object" << std::endl;
    moab::Range moabRange;
    int problemDimension = 3;
    moab::ErrorCode moabError = moabCommunicator->get_part_entities( moabRange, problemDimension);

    // create MBCoupler 
    AMP::pout << "Creating Coupler" << std::endl;
    int moabCouplerID = 0;
    moab::Coupler moabCoupler( moabInterface, 
                               moabCommunicator, 
                               moabRange,
                               moabCouplerID  );

    // Create list of points
    int numCoords = 4;
    std::vector<double> myCoords(3*numCoords);

    // First point
    myCoords[0] = 0.0;
    myCoords[1] = 0.0;
    myCoords[2] = -2.0;

    // Second point
    myCoords[3] = 0.0;
    myCoords[4] = 0.0;
    myCoords[5] = -1.0;
    
    // Third point
    myCoords[6] =  0.0;
    myCoords[7] =  0.0;
    myCoords[8] =  1.0;

    // Third point
    myCoords[6] =  0.0;
    myCoords[7] =  0.0;
    myCoords[8] =  2.0;

    // Input coords to coupler
    AMP::pout << "Locating points" << std::endl;
    moabError = moabCoupler.locate_points( &myCoords[0], numCoords );

    AMP::pout << "We've located the points, now interpolating" << std::endl;
    // This Nek problem doesn't actually do heat transfer so we look for 
    //  the pressure instead just to make sure we can get data off of Moab
    std::vector<double> interpPress(numCoords,0.0);
    std::string pressName = "VPRESS";
    moabError = moabCoupler.interpolate( moab::Coupler::LINEAR_FE, pressName, &interpPress[0] );

    bool nonZero = false;
    for( int i=0; i<numCoords; ++i )
    {
        AMP::plog<< "Pressure " << i << " is " << interpPress[i] << std::endl;
        if( interpPress[i] != 0.0 )
            nonZero = true;
    }

    // Make sure something is non-zero
    if( nonZero )
        ut->passes("Some non-zero pressure data retrieved");
    else
        ut->failure("Pressure data is all zero");

    // Let's see if we got the same values on different processors
    if( globalComm.minReduce(interpPress[0]) == globalComm.maxReduce(interpPress[0]) )
        ut->passes("First pressure value is consistent");
    else
        ut->failure("First pressure value is not consistent across processors");
    
    if( globalComm.minReduce(interpPress[1]) == globalComm.maxReduce(interpPress[1]) )
        ut->passes("Second pressure value is consistent");
    else
        ut->failure("Second pressure value is not consistent across processors");

    if( globalComm.minReduce(interpPress[2]) == globalComm.maxReduce(interpPress[2]) )
        ut->passes("Third pressure value is consistent");
    else
        ut->failure("Third pressure value is not consistent across processors");

    // This last value is different on different processors
    // Report this as an expected failure and be aware if it changes
    if( globalComm.minReduce(interpPress[3]) == globalComm.maxReduce(interpPress[3]) )
        ut->failure("Fourth pressure value is consistent and should be, but it didn't used to be.  Something has changed.");
        //ut->passes("Fourth pressure value is consistent");
    else
        ut->expected_failure("Fourth pressure value is not consistent across processors");




    // We're just making sure these values don't change,
    //  we have no idea if they are correct in any sense.
    /*
    if( AMP::Utilities::approx_equal( interpPress[0], 0.00872468, 1.0e-6 ) )
        ut->passes("First pressure value is correct.");
    else
        ut->failure("First pressure value is not correct.");

    if( AMP::Utilities::approx_equal( interpPress[1], 0.00591948, 1.0e-6 ) )
        ut->passes("Second pressure value is correct.");
    else
        ut->failure("Second pressure value is not correct.");

    if( AMP::Utilities::approx_equal( interpPress[2], 0.0477284, 1.0e-6 ) )
        ut->passes("Third pressure value is correct.");
    else
        ut->failure("Third pressure value is not correct.");
        */

    // We are done.
    NEK_END();
    ut->passes("Nek has cleaned itself up.");
#else
    ut->passes("Nek was not used.");
#endif

    if (ut->NumPassGlobal() == 0) ut->failure("if it doesn't pass, it must have failed.");
} 


int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;
    
    try {
        nekPipe(&ut);
	      ut.passes("Nek ran pipe to completion.");
    } catch (std::exception &err) {
        std::cout << "ERROR: While testing "<<argv[0] << err.what() << std::endl;
        ut.failure("ERROR: While testing");
    } catch( ... ) {
        std::cout << "ERROR: While testing "<<argv[0] << "An unknown exception was thrown." << std::endl;
        ut.failure("ERROR: While testing");
    }
   
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}

//---------------------------------------------------------------------------//
//                 end of test_ORIGEN_utils.cc
//---------------------------------------------------------------------------//
//
