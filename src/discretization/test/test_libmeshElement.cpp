#include "AMP/discretization/createLibmeshElements.h"
#include "AMP/mesh/testHelpers/meshGenerators.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"


// Test the creation of libmesh elements
void testLibmeshElement( AMP::UnitTest *ut, std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    AMP::Discretization::createLibmeshElements list;
    try {
        auto iterator = mesh->getIterator( AMP::Mesh::GeomType::Cell, 1 );
        list.reinit( iterator );
        for ( size_t i = 0; i < iterator.size(); i++ ) {
            auto elem = list.getElement( iterator->globalID() );
            AMP_ASSERT( AMP::Utilities::approx_equal( elem->volume(), iterator->volume() ) );
            ++iterator;
        }
        ut->passes( "Created volume elements" );
    } catch ( ... ) {
        ut->failure( "Created volume elements" );
    }
    try {
        auto iterator = mesh->getIterator( AMP::Mesh::GeomType::Face, 1 );
        list.reinit( iterator );
        for ( size_t i = 0; i < iterator.size(); i++ ) {
            auto elem = list.getElement( iterator->globalID() );
            AMP_ASSERT( AMP::Utilities::approx_equal( elem->volume(), iterator->volume() ) );
            ++iterator;
        }
        ut->passes( "Created face elements" );
    } catch ( ... ) {
        ut->failure( "Created face elements" );
    }
}


// Main function
int main( int argc, char **argv )
{
    AMP::AMPManagerProperties startup_properties;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;
    ut.verbose( true );

    // Run the tests
    {
        std::shared_ptr<AMP::unit_test::MeshGenerator> generator;
        generator = std::make_shared<AMP::unit_test::ExodusReaderGenerator>( "pellet_1x.e" );
        generator->build_mesh();
        testLibmeshElement( &ut, generator->getMesh() );
        generator = std::make_shared<AMP::unit_test::AMPCubeGenerator>( 5 );
        generator->build_mesh();
        testLibmeshElement( &ut, generator->getMesh() );
        generator = std::make_shared<AMP::unit_test::AMPMultiMeshGenerator>();
        generator->build_mesh();
        testLibmeshElement( &ut, generator->getMesh() );
    }

    // Print the results and return
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
