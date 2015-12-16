//----------------------------------*-C++-*----------------------------------//
/*!
 * \file   operators/test/testMoabBasedOperator.cc
 * \brief  This tests the Moab iMesh interface with MoabBasedOperator
 */
//---------------------------------------------------------------------------//

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iostream>
#include <string>
#include <vector>

#include "utils/AMPManager.h"
#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/PIO.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"

#include "discretization/simpleDOF_Manager.h"
#include "utils/Writer.h"
#include "vectors/VectorBuilder.h"

// AMP Moab Includes
#include "operators/moab/MoabBasedOperator.h"
#include "operators/moab/MoabMapOperator.h"

// MOAB Includes
#include "Coupler.hpp"
#include "MBParallelComm.hpp"
#include "MBRange.hpp"
#include "MBiMesh.hpp"
#include "iMesh.h"
#include "moab/Interface.hpp"

//---------------------------------------------------------------------------//
// Helper Class
//---------------------------------------------------------------------------//

typedef AMP::Operator::MoabBasedOperator MoabBasedOp;
typedef AMP::shared_ptr<MoabBasedOp> SP_MoabBasedOp;

typedef AMP::Operator::MoabBasedOperatorParameters MoabOpParams;
typedef AMP::shared_ptr<MoabOpParams> SP_MoabOpParams;

typedef AMP::LinearAlgebra::Vector AMPVec;
typedef AMP::LinearAlgebra::Vector::shared_ptr SP_AMPVec;


class MoabDummyOperator : public MoabBasedOp {
public:
    explicit MoabDummyOperator( SP_MoabOpParams &moabParams ) : MoabBasedOp( moabParams )
    {
        // Create iMesh instance
        iMesh_Instance mbMesh;
        std::string options;
        int ierr;
        iMesh_newMesh( options.c_str(), &mbMesh, &ierr, options.length() );

        // Convert iMesh to MBiMesh
        MBiMesh *mbimesh = reinterpret_cast<MBiMesh *>( mbMesh );

        // Get Moab interface from MBiMesh
        d_moabInterface = mbimesh->mbImpl;

        // Get root set
        iBase_EntitySetHandle root;
        iMesh_createEntSet( mbMesh, 0, &root, &ierr );

        // Build ParallelComm and get index
        moab::ParallelComm *mbComm = new moab::ParallelComm( d_moabInterface );
        int index                  = mbComm->get_id();

        // Set read options and load file
        std::string newReadOpts;
        std::ostringstream extraOpt;
        extraOpt << ";PARALLEL_COMM=" << index;
        std::string readOpts =
            "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARTITION_DISTRIBUTE;PARALLEL_RESOLVE_"
            "SHARED_ENTS;PARALLEL_GHOSTS=3.0.1;CPUTIME";
        newReadOpts          = readOpts + extraOpt.str();
        std::string filename = moabParams->d_db->getString( "moabMeshName" );
        AMP::plog << "Moab mesh name is " << filename << std::endl;
        moab::ErrorCode result = d_moabInterface->load_file(
            filename.c_str(), (EntityHandle *) &root, newReadOpts.c_str() );

        AMP_INSIST( result == MB_SUCCESS, "File not loaded correctly" );

        // Extract nodes
        iBase_EntityHandle *nodes;
        int nodes_alloc = 0;
        int nodes_size;
        iMesh_getEntities( mbMesh,
                           root,
                           iBase_VERTEX,
                           iMesh_ALL_TOPOLOGIES,
                           &nodes,
                           &nodes_alloc,
                           &nodes_size,
                           &ierr );

        // Get list of node coordinates
        std::vector<double> myCoords;
        d_moabInterface->get_vertex_coordinates( myCoords );
        int num_nodes = myCoords.size() / 3;

        AMP_INSIST( num_nodes == nodes_size,
                    "Number of nodes must match number of vertex coordinates" );

        AMP::pout << "Found " << num_nodes << " nodes" << std::endl;

        // Put data on mesh
        std::vector<double> myTemps( num_nodes, -1.0 );
        for ( int i = 0; i < num_nodes; ++i ) {
            // Set temperature to T = x + z
            myTemps[i] = myCoords[i] + myCoords[i + 2 * num_nodes];
        }

        // Add temperature tag
        iBase_TagHandle tempTagHandle;
        std::string tempTagName = "TEMPERATURE";
        iMesh_createTag( mbMesh,
                         tempTagName.c_str(),
                         1,
                         MB_TYPE_DOUBLE,
                         &tempTagHandle,
                         &ierr,
                         tempTagName.length() );

        // Assign data to tag
        iMesh_setDblArrData(
            mbMesh, nodes, nodes_size, tempTagHandle, &myTemps[0], myTemps.size(), &ierr );
    }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr r,
                double a,
                double b )
    {
        /* Don't need an apply for this operator */
    }

    void finalize(){ /* ... */ };
};


//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//

void moabInterface( AMP::UnitTest *ut )
{
    // Print out AMP banner
    AMP::Utilities::printBanner();

    // Log all nodes
    std::string exeName     = "testMoabBasedOperator";
    std::string input_file  = "input_" + exeName;
    std::string output_file = "output_" + exeName;
    AMP::PIO::logAllNodes( output_file );

    //--------------------------------------------------
    //  Read Input File.
    //--------------------------------------------------

    AMP::shared_ptr<AMP::InputDatabase> input_db( new AMP::InputDatabase( "input_db" ) );
    AMP::InputManager::getManager()->parseInputFile( input_file, input_db );

    //--------------------------------------------------
    //   Create the Mesh.
    //--------------------------------------------------
    AMP::shared_ptr<AMP::Database> mesh_db = input_db->getDatabase( "Mesh" );
    AMP::shared_ptr<AMP::Mesh::MeshParameters> mgrParams(
        new AMP::Mesh::MeshParameters( mesh_db ) );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    AMP::shared_ptr<AMP::Mesh::Mesh> mesh = AMP::Mesh::Mesh::buildMesh( mgrParams );

    // Put moab mesh filename onto DB
    std::string moabMeshFile = "input.h5m";
    input_db->putString( "moabMeshName", moabMeshFile );

    // Build operator params
    typedef AMP::Operator::MoabBasedOperatorParameters MoabOpParams;
    typedef AMP::shared_ptr<MoabOpParams> SP_MoabOpParams;

    AMP::pout << "Building Moab Operator Parameters" << std::endl;
    SP_MoabOpParams moabParams( new MoabOpParams( input_db ) );

    // Build operator
    typedef AMP::Operator::MoabBasedOperator MoabBasedOp;
    typedef AMP::shared_ptr<MoabBasedOp> SP_MoabBasedOp;

    AMP::pout << "Building Moab Operator" << std::endl;
    SP_MoabBasedOp moabOp( new MoabDummyOperator( moabParams ) );

    // Call apply
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    moabOp->apply( nullVec, nullVec, nullVec, 0.0, 0.0 );

    // Create Parameters for Map Operator
    AMP::pout << "Creating map operator" << std::endl;
    typedef AMP::Operator::MoabMapOperatorParameters MoabMapParams;
    typedef AMP::shared_ptr<MoabMapParams> SP_MoabMapParams;

    typedef AMP::Operator::MoabMapOperator MoabMap;
    typedef AMP::shared_ptr<MoabMap> SP_MoabMap;

    input_db->putString( "MoabMapVariable", "TEMPERATURE" );
    SP_MoabMapParams mapParams( new MoabMapParams( input_db ) );
    mapParams->setMoabOperator( moabOp );
    mapParams->setMesh( mesh );

    // Create DOF manager
    size_t DOFsPerNode  = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    AMP::Discretization::DOFManager::shared_ptr nodalDofMap =
        AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::Vertex, nodalGhostWidth, DOFsPerNode, split );
    AMP::LinearAlgebra::Variable::shared_ptr nodalVar(
        new AMP::LinearAlgebra::Variable( "nodalPressure" ) );
    AMP::LinearAlgebra::Vector::shared_ptr nodalVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, nodalVar, true );

    AMP::pout << "Nodal Vector size: " << nodalVec->getGlobalSize() << std::endl;

    // Now create Moab map operator
    AMP::pout << "Creating Node-Based Moab Map Operator" << std::endl;
    input_db->putString( "InterpolateToType", "Vertex" );
    SP_MoabMap moabNodeMap( new MoabMap( mapParams ) );

    // Do interpolation
    moabNodeMap->apply( nullVec, nullVec, nodalVec, 0.0, 0.0 );

    // Check to make sure we didn't just get a vector of zeros
    AMPVec::iterator myIter;
    int ctr      = 0;
    bool nonZero = false;
    for ( myIter = nodalVec->begin(); myIter != nodalVec->end(); ++myIter ) {
        if ( *myIter != 0.0 ) nonZero = true;

        ctr++;
    }

    if ( nonZero )
        ut->passes( "Nodal vector is not identically zero" );
    else
        ut->failure( "Nodal vector is identically zero" );

    // Now let's see if the interpolated values are what we expected (should be equal to sum of x-
    // and z-coordinates)
    int offset        = 0;
    int numMismatched = 0;

    // loop over all meshes to create the preprocessor database for that mesh
    std::vector<AMP::Mesh::MeshID> meshIDs = mesh->getBaseMeshIDs();

    for ( size_t meshIndex = 0; meshIndex < meshIDs.size(); meshIndex++ ) {
        // this is an accessor to all the mesh info.
        AMP::Mesh::Mesh::shared_ptr currentMesh = mesh->Subset( meshIDs[meshIndex] );
        if ( currentMesh.get() == NULL ) continue;

        std::string meshCoords   = "Mesh_Coords";
        SP_AMPVec thisMeshCoords = currentMesh->getPositionVector( meshCoords );

        for ( unsigned int i = 0; i < currentMesh->numLocalElements( AMP::Mesh::Vertex ); ++i ) {
            double val1 = 100.0 * ( thisMeshCoords->getValueByLocalID( 3 * i ) +
                                    thisMeshCoords->getValueByLocalID(
                                        3 * i + 2 ) ); // AMP  coordinates are in meters
            double val2 = nodalVec->getValueByLocalID( offset + i ); // Moab coordinates are in cm

            // Linear interpolation should be 'exact' because we prescribed a linear function
            // Can't use approx_equal here because it fails for small values (it compares relative
            // rather than absolute
            // difference)
            if ( std::abs( val1 - val2 ) > 1.0e-10 ) {
                numMismatched++;

                // If value didn't match, print out it's index and the values
                AMP::pout << "Mismatch at index " << i << ": " << val1 << " " << val2 << std::endl;
            }
        }

        offset += currentMesh->numLocalElements( AMP::Mesh::Vertex );
    }

    if ( numMismatched == 0 )
        ut->passes( "Values interpolated correctly" );
    else
        ut->failure( "Values not interpolated correctly" );

// How about some output?
// Useful for making sure everything looks right

#ifdef USE_EXT_SILO
    AMP::Utilities::Writer::shared_ptr siloWriter = AMP::Utilities::Writer::buildWriter( "Silo" );
    siloWriter->registerMesh( mesh );
    siloWriter->registerVector( nodalVec, mesh, AMP::Mesh::Vertex, "Temperatures" );
    siloWriter->writeFile( "Moab_Temp", 0 );
#endif

    if ( ut->NumPassGlobal() == 0 ) ut->failure( "if it doesn't pass, it must have failed." );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    try {
        moabInterface( &ut );
        ut.passes( "Moab interface used correctly." );
    }
    catch ( std::exception &err ) {
        std::cout << "ERROR: While testing " << argv[0] << err.what() << std::endl;
        ut.failure( "ERROR: While testing" );
    }
    catch ( ... ) {
        std::cout << "ERROR: While testing " << argv[0] << "An unknown exception was thrown."
                  << std::endl;
        ut.failure( "ERROR: While testing" );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}

//---------------------------------------------------------------------------//
//                 end of testMoabBasedOperator.cc
//---------------------------------------------------------------------------//
