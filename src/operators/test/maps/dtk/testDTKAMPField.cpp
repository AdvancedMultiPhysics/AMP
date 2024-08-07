#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/map/dtk/DTKAMPField.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utils/Database.h>


static void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testDTKAMPMeshManager" );
    std::string log_file = "output_" + exeName;
    std::string msgPrefix;
    AMP::logOnlyNodeZero( log_file );

    // Load input and build the mesh.
    AMP::pout << "Loading the  mesh" << std::endl;

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    std::string input_file = "input_" + exeName;
    auto input_db          = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto meshDatabase = input_db->getDatabase( "Mesh" );

    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( meshDatabase );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    // Create a vector.
    bool const split      = true;
    int const ghostWidth  = 0;
    int const dofsPerNode = 1;
    auto dofManager       = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, ghostWidth, dofsPerNode );
    auto variable  = std::make_shared<AMP::LinearAlgebra::Variable>( "var" );
    auto ampVector = AMP::LinearAlgebra::createVector( dofManager, variable, split );
    std::vector<std::size_t> dofIndices;
    auto meshIterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex, ghostWidth );
    for ( meshIterator = meshIterator.begin(); meshIterator != meshIterator.end();
          ++meshIterator ) {
        dofManager->getDOFs( meshIterator->globalID(), dofIndices );
        ampVector->setLocalValueByGlobalID( dofIndices[0], static_cast<double>( dofIndices[0] ) );
    }

    // Create a dtk field around the amp vector
    AMP::Operator::DTKAMPField dtk_field( ampVector );

    // Check the dimension.
    AMP_ASSERT( 1 == dtk_field.dimension() );

    // Check the support ids.
    auto support_ids = dtk_field.getLocalSupportIds();
    int counter      = 0;
    for ( meshIterator = meshIterator.begin(); meshIterator != meshIterator.end();
          ++meshIterator, ++counter ) {
        dofManager->getDOFs( meshIterator->globalID(), dofIndices );
        AMP_ASSERT( support_ids[counter] == dofIndices[0] );
    }

    // Check reading data.
    for ( meshIterator = meshIterator.begin(); meshIterator != meshIterator.end();
          ++meshIterator, ++counter ) {
        dofManager->getDOFs( meshIterator->globalID(), dofIndices );
        AMP_ASSERT( dtk_field.readFieldData( dofIndices[0], 0 ) ==
                    ampVector->getLocalValueByGlobalID( dofIndices[0] ) );
    }

    // Check setting data.
    for ( meshIterator = meshIterator.begin(); meshIterator != meshIterator.end();
          ++meshIterator, ++counter ) {
        dofManager->getDOFs( meshIterator->globalID(), dofIndices );
        dtk_field.writeFieldData( dofIndices[0], 0, 2.0 * dofIndices[0] );
    }
    for ( meshIterator = meshIterator.begin(); meshIterator != meshIterator.end();
          ++meshIterator, ++counter ) {
        dofManager->getDOFs( meshIterator->globalID(), dofIndices );
        AMP_ASSERT( 2.0 * dofIndices[0] == ampVector->getLocalValueByGlobalID( dofIndices[0] ) );
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
