// This file contains classes for generating meshes that are based in libmesh
#ifndef included_AMP_Unit_test_Libmesh_Generators_h
#define included_AMP_Unit_test_Libmesh_Generators_h

#include "AMP/AMP_TPLs.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/testHelpers/meshGenerators.h"

#include <string>


namespace libMesh::Parallel {
class Communicator;
}
namespace AMP::Mesh {
class initializeLibMesh;
}


namespace AMP::unit_test {


// Class to create a cube in Libmesh
class LibMeshCubeGenerator : public MeshGenerator
{
public:
    LibMeshCubeGenerator() = delete;
    LibMeshCubeGenerator( int size ) : SIZE( size ) {}
    void build_mesh() override;
    std::string name() const override { return "LibMeshCubeGenerator"; }

private:
    [[maybe_unused]] int SIZE = 0;
};


// Class to read in a default exodus file
class ExodusReaderGenerator : public MeshGenerator
{
public:
    ExodusReaderGenerator() = delete;
    ExodusReaderGenerator( std::string_view file ) : d_file( file ) {}
    void build_mesh() override;
    std::string name() const override { return "ExodusReaderGenerator"; }
    [[maybe_unused]] const std::string d_file;
};


// MulitMesh generator
class MultiMeshGenerator : public MeshGenerator
{
public:
    void build_mesh() override;
    std::string name() const override { return "MultiMeshGenerator"; }
};


// libMeshThreeElement generator
class libMeshThreeElementGenerator : public MeshGenerator
{
public:
    std::string name() const override { return "libMeshThreeElementGenerator"; }
    static std::vector<unsigned int> getBndDofIndices();
    static std::vector<std::vector<unsigned int>> getElemNodeMap();
    void build_mesh() override;
    virtual ~libMeshThreeElementGenerator();

protected:
    std::shared_ptr<libMesh::Parallel::Communicator> libMeshComm;
    std::shared_ptr<AMP::Mesh::initializeLibMesh> libmeshInit;
};


} // namespace AMP::unit_test


#endif
