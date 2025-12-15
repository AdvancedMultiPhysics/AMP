#include "AMP/mesh/testHelpers/meshGenerators.h"


namespace AMP::unit_test {


/********************************************************
 * MeshGenerator                                         *
 ********************************************************/
std::shared_ptr<AMP::Mesh::Mesh> MeshGenerator::getMesh()
{
    if ( !mesh )
        this->build_mesh();
    return mesh;
}


/********************************************************
 * Cube generator                                        *
 ********************************************************/
void AMPCubeGenerator::build_mesh()
{
    // Create a generic MeshParameters object
    auto database = std::make_shared<AMP::Database>( "Mesh" );
    database->putScalar<int>( "dim", 3 );
    database->putScalar<std::string>( "MeshName", "AMP::cube" );
    database->putScalar<std::string>( "Generator", "cube" );
    database->putVector<int>( "Size", { SIZE_X, SIZE_Y, SIZE_Z } );
    database->putVector<double>( "Range", { 0.0, 1.0, 0.0, 1.0, 0.0, 1.0 } );
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    // Create an AMP mesh
    mesh = AMP::Mesh::BoxMesh::generate( params );
}
std::string AMPCubeGenerator::name() const
{
    char tmp[128];
    snprintf( tmp, sizeof( tmp ), "AMPCubeGenerator<%i,%i,%i>", SIZE_X, SIZE_Y, SIZE_Z );
    return std::string( tmp );
}


/********************************************************
 * Cylinder generator                                    *
 ********************************************************/
void AMPCylinderGenerator::build_mesh()
{
    // Create a generic MeshParameters object
    auto database = std::make_shared<AMP::Database>( "Mesh" );
    database->putScalar<int>( "dim", 3 );
    database->putScalar<std::string>( "MeshName", "AMP::cylinder" );
    database->putScalar<std::string>( "Generator", "cylinder" );
    database->putVector<int>( "Size", { 10, 10 } );
    database->putVector<double>( "Range", { 1.0, 0.0, 1.0 } );
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    // Create an AMP mesh
    mesh = AMP::Mesh::BoxMesh::generate( params );
}


/********************************************************
 * Tube generator                                        *
 ********************************************************/
void AMPTubeGenerator::build_mesh()
{
    // Create a generic MeshParameters object
    auto database = std::make_shared<AMP::Database>( "Mesh" );
    database->putScalar<int>( "dim", 3 );
    database->putScalar<std::string>( "MeshName", "AMP::tube" );
    database->putScalar<std::string>( "Generator", "tube" );
    database->putVector<int>( "Size", { 3, 12, 10 } );
    database->putVector<double>( "Range", { 0.7, 1.0, 0.0, 1.0 } );
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    // Create an AMP mesh
    mesh = AMP::Mesh::BoxMesh::generate( params );
}


/********************************************************
 * MulitMesh generator                                   *
 ********************************************************/
static std::unique_ptr<AMP::Database> createPelletMeshDatabase()
{
    int N_pellet = 3;
    auto db      = std::make_unique<Database>();
    // Create the multimesh database
    db->putScalar<std::string>( "MeshName", "PelletMeshes" );
    db->putScalar<std::string>( "MeshType", "Multimesh" );
    db->putScalar<std::string>( "MeshDatabasePrefix", "Mesh_" );
    db->putScalar<std::string>( "MeshArrayDatabasePrefix", "MeshArray_" );
    // Create the mesh array database (PelletMeshes)
    auto meshArrayDatabase = db->putDatabase( "MeshArray_1" );
    meshArrayDatabase->putScalar<int>( "N", N_pellet );
    meshArrayDatabase->putScalar<std::string>( "iterator", "%i" );
    std::vector<int> indexArray( N_pellet );
    for ( int i = 0; i < N_pellet; i++ )
        indexArray[i] = i + 1;
    meshArrayDatabase->putVector<int>( "indicies", indexArray );
    meshArrayDatabase->putScalar<std::string>( "MeshName", "pellet_%i" );
    meshArrayDatabase->putScalar<std::string>( "MeshType", "AMP" );
    meshArrayDatabase->putScalar<std::string>( "Generator", "cylinder" );
    meshArrayDatabase->putVector<int>( "Size", { 5, 8 } );
    meshArrayDatabase->putVector<double>( "Range", { 0.004025, 0.0, 0.0105 } );
    meshArrayDatabase->putScalar<int>( "dim", 3 );
    meshArrayDatabase->putScalar<double>( "x_offset", 0.0 );
    meshArrayDatabase->putScalar<double>( "y_offset", 0.0 );
    std::vector<double> offsetArray( N_pellet );
    for ( int i = 0; i < N_pellet; i++ )
        offsetArray[i] = ( (double) i ) * 0.0105;
    meshArrayDatabase->putVector( "z_offset", offsetArray );
    return db;
}
static std::unique_ptr<AMP::Database> createCladMeshDatabase()
{
    auto db = std::make_unique<Database>();
    // Create the multimesh database
    db->putScalar<std::string>( "MeshName", "clad" );
    db->putScalar<std::string>( "MeshType", "AMP" );
    db->putScalar<std::string>( "Generator", "tube" );
    db->putVector<int>( "Size", { 3, 36, 32 } );
    db->putVector<double>( "Range", { 0.00411, 0.00475, 0, 0.0315 } );
    db->putScalar<int>( "dim", 3 );
    return db;
}
void AMPMultiMeshGenerator::build_mesh()
{
    // Create the multimesh database
    auto meshDatabase = std::make_shared<AMP::Database>( "Mesh" );
    meshDatabase->putScalar<std::string>( "MeshName", "SinglePin" );
    meshDatabase->putScalar<std::string>( "MeshType", "Multimesh" );
    meshDatabase->putScalar<std::string>( "MeshDatabasePrefix", "Mesh_" );
    meshDatabase->putScalar<std::string>( "MeshArrayDatabasePrefix", "MeshArray_" );
    // Create the mesh array database (PelletMeshes)
    meshDatabase->putDatabase( "Mesh_1", createPelletMeshDatabase() );
    // Create the mesh database (clad)
    meshDatabase->putDatabase( "Mesh_2", createCladMeshDatabase() );
    // Create the parameter object
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( meshDatabase );
    params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    // Create the mesh
    mesh = AMP::Mesh::MeshFactory::create( params );
}


/********************************************************
 * SurfaceSubsetGenerator                                *
 ********************************************************/
void SurfaceSubsetGenerator::build_mesh()
{
    auto mesh1 = d_generator->getMesh();
    auto type  = mesh1->getGeomType();
    auto type2 = type - 1;
    auto it    = mesh1->getSurfaceIterator( type2, GCW );
    mesh       = mesh1->Subset( it );
}
SurfaceSubsetGenerator::SurfaceSubsetGenerator( std::shared_ptr<MeshGenerator> gen, int gcw )
    : GCW( gcw ), d_generator( gen )
{
}


} // namespace AMP::unit_test
