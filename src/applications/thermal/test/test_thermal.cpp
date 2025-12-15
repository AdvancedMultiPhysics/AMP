#include "AMP/IO/PIO.h"
#include "AMP/IO/Writer.h"
#include "AMP/applications/thermal/SolveThermal.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/materials/ScalarProperty.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/MathExpr.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <iostream>
#include <memory>
#include <string>


void check( const std::string &msg, bool pass, AMP::UnitTest &ut )
{
    if ( pass )
        ut.passes( msg );
    else
        ut.failure( msg );
}


// Some materials used by the test
class APG1 : public AMP::Materials::Material
{
public:
    APG1()
    {
        addScalarProperty( "Density", 2.633, "g/cm^3" );
        addScalarProperty( "Thermal Conductivity", 0.0078, "W/(cm*K)" );
        addScalarProperty( "Specific Heat", 840, "J/(kg*K)" );
        addScalarProperty( "Young’s Modulus", 70, "GPa" );
        addScalarProperty( "Poisson’s Ratio", 0.238 );
        addScalarProperty( "Thermal Expansion", 7.6e-6, "K^-1" );
    }
    std::string materialName() const override { return "APG1"; }
};
class YAG : public AMP::Materials::Material
{
public:
    YAG()
    {
        addScalarProperty( "Thermal Expansion", 6.148e-6, "K^-1" );
        addScalarProperty( "Thermal Diffusivity", 0.041, "cm^2/s^2" );
        addScalarProperty( "Thermal Conductivity", 0.112, "W/(cm*K)" );
        addScalarProperty( "Specific Heat", 590, "J/(kg*K)" );
        addScalarProperty( "Thermal Shock Resistant", 800, "W/m" );
        std::array<double, 2> range = { 20, 550 };
        std::vector<double> Tc      = { 20.,  40.,  60.,  80.,  100., 120., 150., 200.,
                                   250., 300., 350., 400., 450., 500., 550. };
        std::vector<double> Kc      = { 15.,  3.5,    1.5,   0.8,   0.5,   0.38,  0.27, 0.176,
                                   0.13, 0.1120, 0.087, 0.077, 0.069, 0.063, 0.058 };
        d_propertyMap["Thermal Conductivity"] =
            std::make_shared<AMP::Materials::InterpolatedProperty>( "YAG::Thermal Conductivity",
                                                                    "W/(cm*K)",
                                                                    "temperature",
                                                                    Tc,
                                                                    Kc,
                                                                    range,
                                                                    "K",
                                                                    300,
                                                                    "" );
    }
    std::string materialName() const override { return "YAG"; }
};


// Add a thermal source
void addSouce( std::shared_ptr<AMP::Database> db,
               std::shared_ptr<AMP::Mesh::Mesh> mesh,
               std::shared_ptr<AMP::LinearAlgebra::Vector> vec )
{
    // Get the appropriate mesh
    auto meshName = db->getString( "Mesh" );
    auto mesh2    = mesh->Subset( meshName );
    AMP::LinearAlgebra::VS_Mesh meshSelector( mesh2 );
    auto v2 = vec->select( meshSelector );
    // Apply the source
    auto expr = db->getEquation( "Power", "W/cm^3" );
    auto DOFs = v2->getDOFManager();
    std::vector<size_t> dofs;
    for ( const auto &node : mesh2->getIterator( AMP::Mesh::GeomType::Vertex, 0 ) ) {
        auto point   = node.coord();
        double power = expr->operator()( { point.x(), point.y(), point.z() } );
        DOFs->getDOFs( node.globalID(), dofs );
        double data;
        v2->getValuesByGlobalID( 1, dofs.data(), &data );
        data += power;
        v2->setValuesByGlobalID( 1, dofs.data(), &data );
    }
}


// Run the test
void myTest( const std::string &input_file, AMP::UnitTest &ut )
{
    // Load the input database
    auto db = AMP::Database::parseInputFile( input_file );
    db->print( AMP::plog );

    // Create the Mesh
    AMP_INSIST( db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    // Create a DOF manager for a nodal vector
    auto nodalDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, 1, 8, true );

    // Create the source vector
    auto powerVar = std::make_shared<AMP::LinearAlgebra::Variable>( "power" );
    auto power    = AMP::LinearAlgebra::createVector( nodalDofMap, powerVar );
    power->zero();
    for ( int i = 0; i < 100; i++ ) {
        if ( db->keyExists( "source_" + std::to_string( i ) ) ) {
            auto db2 = db->getDatabase( "source_" + std::to_string( i ) );
            addSouce( db2, mesh, power );
        }
    }
    power->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Solve for the SS temperature
    auto rhs          = AMP::applications::integrateSourceVector( mesh, power );
    auto [solVec, op] = AMP::applications::solveTemperature( mesh, rhs, db );
    double T_min      = static_cast<double>( solVec->min() );
    double T_max      = static_cast<double>( solVec->max() );
    printf( "Minimum temperature: %0.1f\n", T_min );
    printf( "Peak temperature: %0.1f\n", T_max );
    printf( "Temperature change: %0.1f\n", T_max - T_min );

    // Calculate the residual
    auto rhsVec = solVec->clone();
    auto resVec = solVec->clone();
    rhsVec->copyVector( rhs );
    op->residual( rhsVec, solVec, resVec );

    // Write the results
    auto writer = AMP::IO::Writer::buildWriter( "HDF5" );
    writer->registerMesh( mesh );
    writer->registerVector( power, mesh, AMP::Mesh::GeomType::Vertex, "power" );
    writer->registerVector( solVec, mesh, AMP::Mesh::GeomType::Vertex, "temperature" );
    writer->registerVector( resVec, mesh, AMP::Mesh::GeomType::Vertex, "residual" );
    writer->registerVector( rhsVec, mesh, AMP::Mesh::GeomType::Vertex, "rhs" );
    writer->writeFile( input_file, 0 );

    // Check the result
    if ( db->keyExists( "min_temperature" ) ) {
        auto ans = db->getScalar<double>( "min_temperature" );
        check( "min temperature matches", fabs( ans - T_min ) < 0.1, ut );
    }
    if ( db->keyExists( "max_temperature" ) ) {
        auto ans = db->getScalar<double>( "max_temperature" );
        check( "max temperature matches", fabs( ans - T_max ) < 0.1, ut );
    }
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    AMP::Materials::registerMaterial( "APG1", []() { return std::make_unique<APG1>(); } );
    AMP::Materials::registerMaterial( "YAG", []() { return std::make_unique<YAG>(); } );

    myTest( argv[1], ut );

    ut.report();
    int num_failed = ut.NumFailGlobal();
    ut.reset();
    AMP::AMPManager::shutdown();
    return num_failed;
}
