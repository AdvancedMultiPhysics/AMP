#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegratorFactory.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <complex>


void record( AMP::UnitTest &ut, bool pass, const std::string &msg )
{
    if ( pass )
        ut.passes( msg );
    else
        ut.failure( msg );
}


void testRestartManager( AMP::UnitTest &ut, const std::string &input_file )
{
    // Load input file
    auto db = AMP::Database::parseInputFile( input_file );

    // Get the Mesh database and create the mesh parameters
    auto database = db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP_COMM_WORLD );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Create a vector
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "vec" );
    auto vec = AMP::LinearAlgebra::createVector( DOFs, var, true );
    vec->setRandomValues();

    bool enable_ti = false;
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegrator> timeIntegrator;
    if ( db->keyExists( "TimeIntegrator" ) ) {
        enable_ti  = true;
        auto ti_db = db->getDatabase( "TimeIntegrator" );
        ti_db->putScalar<int>( "integrator_step", 0 );
        auto ti_params = std::make_shared<AMP::TimeIntegrator::TimeIntegratorParameters>( ti_db );
        ti_params->d_global_db = db;
        ti_params->d_ic_vector = vec;
        timeIntegrator         = AMP::TimeIntegrator::TimeIntegratorFactory::create( ti_params );
    }

    // Create the restart manager and register data
    int int_v    = 13;
    double dbl_v = 15.3;
    std::string str_v( "test" );
    std::complex<double> cmplx( 15.3, 4.2 );
    AMP::IO::RestartManager writer;
    writer.registerData( int_v, "int" );
    writer.registerData( dbl_v, "double" );
    writer.registerData( cmplx, "complex" );
    writer.registerData( str_v, "string" );
    writer.registerData( db, "inputs" );
    writer.registerData( mesh, "mesh" );
    writer.registerData( vec, "vec" );

    if ( enable_ti )
        writer.registerData( timeIntegrator, "TI" );

    // Write the restart data
    writer.write( "testRestartData" );

    // Read and check the restart data
    AMP::IO::RestartManager reader( "testRestartData" );
    auto int2 = *( reader.getData<int>( "int" ) );
    auto dbl  = *( reader.getData<double>( "double" ) );
    auto cmp  = *( reader.getData<std::complex<double>>( "complex" ) );
    auto str  = *( reader.getData<std::string>( "string" ) );
    bool pass = int2 == int_v && dbl == dbl_v && cmp == cmplx && str == str_v;
    record( ut, pass, "Basic check" );
    auto db2 = reader.getData<AMP::Database>( "inputs" );
    record( ut, db2 != nullptr, "Database loaded" );
    if ( db2 )
        record( ut, *db == *db2, "Database match" );
    auto mesh2 = reader.getData<AMP::Mesh::Mesh>( "mesh" );
    record( ut, mesh2 != nullptr, "Mesh loaded" );
    if ( mesh2 )
        record( ut, *mesh == *mesh2, "Mesh match" );
    auto vec2 = reader.getData<AMP::LinearAlgebra::Vector>( "vec" );
    record( ut, vec2 != nullptr, "Load Vector" );
    if ( vec2 )
        record( ut, vec->L2Norm() == vec2->L2Norm(), "vec->L2Norm() matches" );
    if ( enable_ti ) {
        auto ti2 = reader.getData<AMP::TimeIntegrator::TimeIntegrator>( "TI" );
        if ( timeIntegrator ) {
            record( ut, ti2 != nullptr, "Load Time Integrator" );
        } else {
            record( ut, ti2 == nullptr, "Load nullptr for Time Integrator" );
        }
    }
}


int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    if ( argc != 2 )
        std::cerr << "test_RestartManager <input>\n";
    testRestartManager( ut, argv[1] );

    int N_failed = ut.NumFailGlobal();
    ut.report();
    ut.reset();
    AMP::AMPManager::shutdown();
    return N_failed;
}
