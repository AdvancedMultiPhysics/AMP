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
    virtual std::shared_ptr<AMP::Mesh::Mesh> getMesh();
    virtual void build_mesh() = 0;
    virtual ~MeshGenerator(){};
    virtual std::string name() const = 0;

protected:
    std::shared_ptr<AMP::Mesh::Mesh> mesh;
};


// Class to create a cube
class AMPCubeGenerator : public MeshGenerator
{
public:
    AMPCubeGenerator() = delete;
    AMPCubeGenerator( int nx ) : SIZE_X( nx ), SIZE_Y( nx ), SIZE_Z( nx ) {}
    AMPCubeGenerator( int nx, int ny, int nz ) : SIZE_X( nx ), SIZE_Y( ny ), SIZE_Z( nz ) {}
    void build_mesh() override;
    std::string name() const override;
    const int SIZE_X, SIZE_Y, SIZE_Z;
};


// Class to create a cylinder
class AMPCylinderGenerator : public MeshGenerator
{
public:
    void build_mesh() override;
    std::string name() const override { return "AMPCylinderGenerator"; }
};


// Class to create a tube
class AMPTubeGenerator : public MeshGenerator
{
public:
    void build_mesh() override;
    std::string name() const override { return "AMPTubeGenerator"; }
};


// MulitMesh generator
class AMPMultiMeshGenerator : public MeshGenerator
{
public:
    void build_mesh() override;
    std::string name() const override { return "AMPMultiMeshGenerator"; }
};


// Surface subset generator
class SurfaceSubsetGenerator : public MeshGenerator
{
public:
    SurfaceSubsetGenerator() = delete;
    SurfaceSubsetGenerator( std::shared_ptr<MeshGenerator> gen, int gcw );
    void build_mesh() override;
    std::string name() const override { return "SurfaceSubsetGenerator"; }

private:
    const int GCW;
    std::shared_ptr<MeshGenerator> d_generator;
};


} // namespace AMP::unit_test


// Include libmesh generators
#ifdef AMP_USE_LIBMESH
    #include "libmeshGenerators.h"
#endif


#endif
