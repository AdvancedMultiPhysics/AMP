#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/testHelpers/meshGenerators.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"

#include <map>


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManagerProperties startup_properties;
    // startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

    auto mesh_generator = AMP::unit_test::AMPCubeGenerator( 10 );
    mesh_generator.build_mesh();
    auto mesh        = mesh_generator.getMesh();
    const auto &comm = mesh->getComm();

    // Create a simple DOF manager
    auto vDOF = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, false );

    std::map<size_t, size_t> vneighbors;

    auto it = mesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    for ( ; it != it.end(); ++it ) {
        const auto row = vDOF->getRowDOFs( it->globalID() );
        vneighbors[row.size()]++;
    }

    size_t i = 0u;
    std::vector<size_t> nvertices( vneighbors.size() );
    for ( auto &[key, value] : vneighbors ) {
        nvertices[i++] = value;
    }

    const auto neighborhood_sizes = ( vneighbors.size() == 4 );
    if ( comm.allReduce( neighborhood_sizes ) )
        ut.passes( "Number of types of neighbors for vertex DOFs passes" );
    else {
        ut.failure( "Number of types of neighbors for vertex DOFs fails" );
    }

    comm.sumReduce( nvertices.data(), nvertices.size() );
    i = 0u;
    for ( auto &[key, value] : vneighbors ) {
        AMP::pout << nvertices[i] << " vertices have " << key << " neighbors across all ranks"
                  << std::endl;
        i++;
    }

    if ( ( nvertices.size() == 4 ) && ( nvertices[0] == 8 ) && ( nvertices[1] == 108 ) &&
         ( nvertices[2] == 486 ) && ( nvertices[3] == 729 ) ) {
        ut.passes( "Number of vertices with different neighbors for vertex DOFs passes" );
    } else {
        ut.failure( "Number of vertices with different neighbors for vertex DOFs fails" );
    }

    // Create a simple DOF manager
    auto cDOF = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, 1, 1, false );

    std::map<size_t, size_t> cneighbors;

    auto cit = mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    for ( ; cit != cit.end(); ++cit ) {
        const auto row = cDOF->getRowDOFs( cit->globalID() );
        cneighbors[row.size()]++;
    }

    i = 0u;
    std::vector<size_t> ncells( cneighbors.size() );
    for ( auto &[key, value] : cneighbors ) {
        ncells[i++] = value;
    }

    const auto cneighborhood_sizes = ( cneighbors.size() == 4 );
    if ( comm.allReduce( cneighborhood_sizes ) )
        ut.passes( "Number of types of neighbors for cell DOFs passes" );
    else {
        ut.failure( "Number of types of neighbors for cell DOFs fails" );
    }

    comm.sumReduce( ncells.data(), ncells.size() );
    i = 0u;
    for ( auto &[key, value] : cneighbors ) {
        AMP::pout << ncells[i] << " cells have " << key << " neighbors across all ranks"
                  << std::endl;
        i++;
    }

    if ( ( ncells.size() == 4 ) && ( ncells[0] == 8 ) && ( ncells[1] == 96 ) &&
         ( ncells[2] == 384 ) && ( ncells[3] == 512 ) ) {
        ut.passes( "Number of vertices with different neighbors for cell DOFs passes" );
    } else {
        ut.failure( "Number of vertices with different neighbors for cell DOFs fails" );
    }

    // Print the results and return
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
