#ifndef included_AMP_TriangleMeshHelpers
#define included_AMP_TriangleMeshHelpers

#include "AMP/geometry/Geometry.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/utils/AMP_MPI.h"

#include <array>
#include <memory>
#include <vector>


namespace AMP::Mesh::TriangleHelpers {


//! Count the number of unique triangles
template<size_t NG>
size_t count( const std::vector<std::array<int, NG + 1>> &tri );


//! Structure to hold an STL triangle
struct STL_Triangle {
    float n[3]     = { 0, 0, 0 };
    float v1[3]    = { 0, 0, 0 };
    float v2[3]    = { 0, 0, 0 };
    float v3[3]    = { 0, 0, 0 };
    int16_t attrib = 0;
};


//! Read an STL file
std::vector<STL_Triangle> readSTL( const std::string &filename, double scale );

//! Read the header for an STL file
size_t readSTLHeader( const std::string &filename );


//! Create triangles/vertices from a set of triangles specified by their coordinates
void createTriangles( const std::vector<STL_Triangle> &tri_list,
                      std::vector<std::array<double, 3>> &vertices,
                      std::vector<std::array<int, 3>> &triangles,
                      double tol );

//! Read an STL file and generate a mesh (triangle mesh or multi-mesh)
std::shared_ptr<AMP::Mesh::Mesh>
generateSTL( std::shared_ptr<const AMP::Mesh::MeshParameters> params );

//! Generate a triangle mesh (or multi-mesh) from a geometry
std::shared_ptr<AMP::Mesh::Mesh> generateGeom( std::shared_ptr<AMP::Geometry::Geometry> geom,
                                               const AMP_MPI &comm,
                                               double resolution );

//! Generate a triangle mesh (or multi-mesh) from parameters
std::shared_ptr<AMP::Mesh::Mesh> generate( std::shared_ptr<const MeshParameters> params );


/**
 * \brief Generate a triangle mesh from local triangle coordinates
 * \details  Create a triangle mesh from the local triangle coordinates.
 *    Note: rank 0 must contain all data, other ranks "may" contain copies
 * \param triangles     List of triangles (rank 0 should contain all triangles)
 * \param comm          Communicator to use
 * \param tol           Relative tolerance to determine if two points are the same
 * \param name          Name of mesh
 * \param splitDomain   Split multi-domain objects into seperate meshes (returning a multimesh)
 * \param loadBalanceMethod  Load balance method to use (only used if splitting multi-domain)
 */
std::shared_ptr<AMP::Mesh::Mesh> generate( const std::vector<STL_Triangle> &triangles,
                                           const AMP_MPI &comm,
                                           const std::string &name,
                                           double tol            = 1e-12,
                                           bool splitDomain      = true,
                                           int loadBalanceMethod = 1 );


//! Estimate Mesh size
size_t estimateMeshSize( std::shared_ptr<const MeshParameters> params );

//! Maximum processor size
size_t maxProcs( std::shared_ptr<const MeshParameters> params );


} // namespace AMP::Mesh::TriangleHelpers

#endif
