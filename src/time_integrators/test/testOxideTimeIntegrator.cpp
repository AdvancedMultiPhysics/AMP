#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/time_integrators/oxide/OxideTimeIntegrator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"


static void OxideTest( AMP::UnitTest *ut, const std::string &input_file )
{

    // Load the input file
    std::string log_file = input_file + ".log";
    AMP::logOnlyNodeZero( log_file );
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto manager = AMP::Mesh::MeshFactory::create( params );
    auto mesh    = manager->Subset( "clad" );
    globalComm.barrier();

    // Create the surface mesh that we will use to create the oxide layer
    auto surface = mesh->Subset( mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, 4, 0 ) );
    surface->setName( "clad_surface" );

    // Create the temperature profile
    auto DOF = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto temp_var = std::make_shared<AMP::LinearAlgebra::Variable>( "temperature" );
    auto temp_vec = AMP::LinearAlgebra::createVector( DOF, temp_var, true );
    auto iterator = mesh->getIterator( AMP::Mesh::GeomType::Vertex );
    auto T0       = input_db->getWithDefault<double>( "T0", 650 );
    std::vector<size_t> dofs;
    for ( size_t i = 0; i < iterator.size(); i++ ) {
        auto coord = iterator->coord();
        DOF->getDOFs( iterator->globalID(), dofs );
        const double val = T0 + 100 * coord[2];
        temp_vec->setValuesByGlobalID( 1, &dofs[0], &val );
        ++iterator;
    }
    temp_vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    AMP_ASSERT( fabs( double( temp_vec->min() ) - T0 ) / T0 < 1e-9 );

    // Create the oxide time integrator
    auto parameters = std::make_shared<AMP::TimeIntegrator::OxideTimeIntegratorParameters>(
        std::shared_ptr<AMP::Database>() );
    parameters->d_mesh  = surface;
    parameters->d_temp  = temp_vec;
    parameters->depth   = 1e-3;
    auto timeIntegrator = std::make_shared<AMP::TimeIntegrator::OxideTimeIntegrator>( parameters );
    auto solution       = timeIntegrator->getSolution();
    auto oxide_var      = std::make_shared<AMP::LinearAlgebra::Variable>( "oxide" );
    auto alpha_var      = std::make_shared<AMP::LinearAlgebra::Variable>( "alpha" );
    auto oxide          = solution->subsetVectorForVariable( oxide_var );
    auto alpha          = solution->subsetVectorForVariable( alpha_var );

    // Run the time integration
    double time = 0.0;
    auto times  = input_db->getVector<double>( "Time" );
    for ( size_t i = 0; i < times.size(); i++ ) {
        auto v = timeIntegrator->getSolution();

        // Advance the solution
        double dT = times[i] - timeIntegrator->getCurrentTime();
        globalComm.barrier();
        double t0 = AMP::AMP_MPI::time();
        timeIntegrator->advanceSolution( dT, false, v, v );
        globalComm.barrier();
        time += AMP::AMP_MPI::time() - t0;

        // Check the solution
        if ( input_db->keyExists( "oxide" ) && input_db->keyExists( "alpha" ) ) {
            if ( times[i] == 0.0 )
                continue;
            auto oxide_solution = input_db->getVector<double>( "oxide" );
            auto alpha_solution = input_db->getVector<double>( "alpha" );
            double err_oxide    = double( oxide->min() ) - oxide_solution[i];
            double err_alpha    = double( alpha->min() ) - alpha_solution[i];
            if ( fabs( err_oxide / oxide_solution[i] ) < 1e-3 )
                ut->passes( "oxide solution matches" );
            else
                ut->failure( "oxide solution matches" );
            if ( fabs( err_alpha / alpha_solution[i] ) < 1e-3 )
                ut->passes( "alpha solution matches" );
            else
                ut->failure( "alpha solution matches" );
        }
    }
    AMP::pout << "Time required for integration: " << time << std::endl;

    ut->passes( "Test runs to completion" );
}


int testOxideTimeIntegrator( int argc, char *argv[] )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    OxideTest( &ut, "input_testOxideTimeIntegrator-1" );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
