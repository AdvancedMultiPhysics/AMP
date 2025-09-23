#ifndef included_AMP_MeshWriters
#define included_AMP_MeshWriters

#include "AMP/mesh/Mesh.h"
#include "AMP/utils/Database.h"


namespace AMP {
class AMP_MPI;
}
namespace AMP::Mesh {
class libmeshMesh;
}


namespace AMP::Mesh::MeshWriters {


using DatabasePtr = std::shared_ptr<AMP::Database>;


// Generate mesh files
void writeDispValsForPatchTest( const std::string &filename );
DatabasePtr createBox( int nx, int ny, int nz, double Lx, double Ly, double Lz );
DatabasePtr createDistortedElement();
DatabasePtr
createPlateWithHole( int le, int me, int ne, int pe, double a, double b, double c, double r );
DatabasePtr create2elementMesh( double a, int ny, int nz, double Lx, double Ly, double Lz );
DatabasePtr create7elementMesh( int NumBoundaryNodeIds );
DatabasePtr createConstrainedMesh( int nx, int ny, int nz, double Lx, double Ly, double Lz );
DatabasePtr createCookMesh( int nx, int ny, int nz );
DatabasePtr createAMGMesh( int nx, int ny, int nz, double Lx, double Ly, double Lz );
DatabasePtr createLUML( int Nx, int Ny, int Nz, double Lx, double Ly, double Lz );
DatabasePtr createDatabase( const AMP::Mesh::Mesh &mesh );


// LibMesh generators
std::shared_ptr<libmeshMesh> readTestMeshLibMesh( std::shared_ptr<AMP::Database> db,
                                                  const AMP_MPI &,
                                                  const std::string &name = "mesh" );
std::shared_ptr<libmeshMesh> readTestMeshLibMesh( const std::string &filename,
                                                  const AMP_MPI &,
                                                  const std::string &name = "mesh" );
std::shared_ptr<libmeshMesh> readBinaryTestMeshLibMesh( const std::string &filename,
                                                        const AMP_MPI &,
                                                        const std::string &name = "mesh" );


/**
 * \brief Generate a test mesh
 * \details  This generates a known test mesh.
 * \param[in] name          Name of mesh to generate
 */
DatabasePtr generateTestMesh( const std::string &name );

/**
 * \brief Read a test mesh
 * \details  This reads (or generates) the database for a test mesh
 *     The original mesh (if read) must be stored in the ASCII format
 * \param[in] filename      File (or name) of mesh to generate
 * \param[in] useGenerator  Should we generate the mesh if known (default).
 *                          If false we will always read.
 */
DatabasePtr readTestMesh( const std::string &filename, bool useGenerator = true );

/**
 * \brief Read a test mesh
 * \details  This reads (or generates) the database for a test mesh
 *     The original mesh (if read) must be stored in the binary format
 * \param[in] filename      File (or name) of mesh to generate
 * \param[in] useGenerator  Should we generate the mesh if known (default).
 *                          If false we will always read.
 */
DatabasePtr readBinaryTestMesh( const std::string &filename, bool useGenerator = true );

/**
 * \brief Write a test mesh
 * \details  This writes the test mesh to the ASCII format
 * \param[in] db            Database containing mesh data
 * \param[in] filename      File of mesh to write
 */
void writeTestMesh( const AMP::Database &db, const std::string &filename );

/**
 * \brief Write a test mesh
 * \details  This writes the test mesh to the binary format
 * \param[in] db            Database containing mesh data
 * \param[in] filename      File of mesh to write
 */
void writeBinaryTestMesh( const AMP::Database &db, const std::string &filename );


//! Create and write all known test meshes
void generateAll();


} // namespace AMP::Mesh::MeshWriters


#endif
