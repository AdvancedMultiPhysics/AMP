// This file contains classes for generating meshes that are used for different tests
#ifndef included_AMP_Unit_test_Mesh_Generators_h
#define included_AMP_Unit_test_Mesh_Generators_h

#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/utils/Database.h"

namespace AMP::unit_test {


// Base class for Mesh Generators
class MeshGenerator
{
public:
    // Routine to build the mesh
    virtual void build_mesh() { AMP_ERROR( "ERROR" ); }
    // Routine to get the pointer to the mesh
    virtual std::shared_ptr<AMP::Mesh::Mesh> getMesh()
    {
        if ( mesh.get() == nullptr )
            this->build_mesh();
        return mesh;
    }
    virtual ~MeshGenerator(){};
    virtual std::string name() const = 0;

protected:
    std::shared_ptr<AMP::Mesh::Mesh> mesh;
};


// Class to create a cube
template<int SIZE_X, int SIZE_Y, int SIZE_Z>
class AMPCubeGenerator3 : public MeshGenerator
{
public:
    void build_mesh() override
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
    std::string name() const override
    {
        char tmp[128];
        snprintf( tmp, sizeof( tmp ), "AMPCubeGenerator3<%i,%i,%i>", SIZE_X, SIZE_Y, SIZE_Z );
        return std::string( tmp );
    }
};
template<int SIZE>
class AMPCubeGenerator : public MeshGenerator
{
public:
    void build_mesh() override
    {
        AMPCubeGenerator3<SIZE, SIZE, SIZE> gen;
        gen.build_mesh();
        mesh = gen.getMesh();
    }
    std::string name() const override
    {
        char tmp[128];
        snprintf( tmp, sizeof( tmp ), "AMPCubeGenerator<%i>", SIZE );
        return std::string( tmp );
    }
};


// Class to create a cylinder
class AMPCylinderGenerator : public MeshGenerator
{
public:
    void build_mesh() override
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
    std::string name() const override { return "AMPCylinderGenerator"; }
};


// Class to create a tube
class AMPTubeGenerator : public MeshGenerator
{
public:
    void build_mesh() override
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
    std::string name() const override { return "AMPTubeGenerator"; }
};


// MulitMesh generator
class AMPMultiMeshGenerator : public MeshGenerator
{
public:
    void build_mesh() override
    {
        // Create the multimesh database
        auto meshDatabase = std::make_shared<AMP::Database>( "Mesh" );
        meshDatabase->putScalar<std::string>( "MeshName", "SinglePin" );
        meshDatabase->putScalar<std::string>( "MeshType", "Multimesh" );
        meshDatabase->putScalar<std::string>( "MeshDatabasePrefix", "Mesh_" );
        meshDatabase->putScalar<std::string>( "MeshArrayDatabasePrefix", "MeshArray_" );
        // Create the mesh array database (PelletMeshes)
        auto pelletMeshDatabase = meshDatabase->putDatabase( "Mesh_1" );
        createPelletMeshDatabase( pelletMeshDatabase );
        // Create the mesh database (clad)
        auto cladMeshDatabase = meshDatabase->putDatabase( "Mesh_2" );
        createCladMeshDatabase( cladMeshDatabase );
        // Create the parameter object
        auto params = std::make_shared<AMP::Mesh::MeshParameters>( meshDatabase );
        params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
        // Create the mesh
        mesh = AMP::Mesh::MeshFactory::create( params );
    }
    std::string name() const override { return "AMPMultiMeshGenerator"; }

private:
    void createPelletMeshDatabase( std::shared_ptr<Database> db )
    {
        int N_pellet = 3;
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
    }
    void createCladMeshDatabase( std::shared_ptr<Database> db )
    {
        // Create the multimesh database
        db->putScalar<std::string>( "MeshName", "clad" );
        db->putScalar<std::string>( "MeshType", "AMP" );
        db->putScalar<std::string>( "Generator", "tube" );
        db->putVector<int>( "Size", { 3, 36, 32 } );
        db->putVector<double>( "Range", { 0.00411, 0.00475, 0, 0.0315 } );
        db->putScalar<int>( "dim", 3 );
    }
};


// Surface subset generator
template<class GENERATOR, int GCW>
class SurfaceSubsetGenerator : public MeshGenerator
{
public:
    void build_mesh() override
    {
        GENERATOR generator;
        generator.build_mesh();
        auto mesh1 = generator.getMesh();
        auto type  = mesh1->getGeomType();
        auto type2 = type - 1;
        auto it    = mesh1->getSurfaceIterator( type2, GCW );
        mesh       = mesh1->Subset( it );
    }
    std::string name() const override { return "SurfaceSubsetGenerator"; }
};


} // namespace AMP::unit_test


// Include libmesh generators
#ifdef AMP_USE_LIBMESH
    #include "libmeshGenerators.h"
#endif


#endif
