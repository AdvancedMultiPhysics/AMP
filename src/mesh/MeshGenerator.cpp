// This file stores the routines to generate the meshes for AMP::Mesh::Mesh
#include "AMP/AMP_TPLs.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/triangle/TriangleHelpers.h"
#include "AMP/utils/MeshPoint.h"

#ifdef AMP_USE_LIBMESH
    #include "AMP/mesh/libmesh/libmeshMesh.h"
#endif


#include <cmath>


namespace AMP::Mesh {


/********************************************************
 * Estimate the mesh size                                *
 ********************************************************/
size_t Mesh::estimateMeshSize( std::shared_ptr<const MeshParameters> params )
{
    auto db = params->d_db;
    AMP_ASSERT( db );
    size_t meshSize = 0;
    if ( db->keyExists( "NumberOfElements" ) ) {
        // User specified the number of elements, this should override everything
        meshSize = (size_t) db->getScalar<int>( "NumberOfElements" );
        // Adjust the number of elements by a weight if desired
        if ( db->keyExists( "Weight" ) ) {
            auto weight = db->getScalar<double>( "Weight" );
            meshSize    = (size_t) ceil( weight * ( (double) meshSize ) );
        }
        return meshSize;
    }
    // This is being called through the base class, call the appropriate function
    AMP_INSIST( db->keyExists( "MeshType" ), "MeshType must exist in input database" );
    auto MeshType = db->getString( "MeshType" );
    if ( db->keyExists( "NumberOfElements" ) ) {
        meshSize = db->getScalar<int>( "NumberOfElements" );
    } else if ( MeshType == "Multimesh" ) {
        // The mesh is a multimesh
        meshSize = AMP::Mesh::MultiMesh::estimateMeshSize( params );
    } else if ( MeshType == "AMP" ) {
        // The mesh is a AMP mesh
        auto filename = db->getWithDefault<std::string>( "FileName", "" );
        auto suffix   = IO::getSuffix( filename );
        if ( suffix == "stl" ) {
            // We are reading an stl file
            meshSize = AMP::Mesh::TriangleHelpers::estimateMeshSize( params );
        } else {
            meshSize = AMP::Mesh::BoxMesh::estimateMeshSize( params );
        }
    } else if ( MeshType == "TriangleGeometryMesh" ) {
        // We will build a triangle mesh from a geometry
        meshSize = AMP::Mesh::TriangleHelpers::estimateMeshSize( params );
    } else if ( MeshType == "libMesh" ) {
// The mesh is a libmesh mesh
#ifdef AMP_USE_LIBMESH
        meshSize = AMP::Mesh::libmeshMesh::estimateMeshSize( params );
#else
        AMP_ERROR( "AMP was compiled without support for libMesh" );
#endif
    } else if ( MeshType == "stk" || MeshType == "STKMesh" ) {
        AMP_ERROR( "stk mesh support was removed on 12/02/25" );
    } else if ( MeshType == "moab" || MeshType == "MOAB" ) {
        AMP_ERROR( "MOAB support was removed on 12/02/25" );
    } else {
        // Unknown mesh type
        AMP_ERROR( "Unknown mesh type " + MeshType + " and NumberOfElements is not set" );
    }
    return meshSize;
}


/********************************************************
 * Estimate the maximum number of processors             *
 ********************************************************/
size_t Mesh::maxProcs( std::shared_ptr<const MeshParameters> params )
{
    auto db = params->d_db;
    AMP_ASSERT( db );
    // Check if the user is specifying the maximum number of processors
    if ( db->keyExists( "maxProcs" ) )
        return db->getScalar<int64_t>( "maxProcs" );
    // This is being called through the base class, call the appropriate function
    AMP_INSIST( db->keyExists( "MeshType" ), "MeshType must exist in input database" );
    std::string MeshType = db->getString( "MeshType" );
    size_t maxSize       = 0;
    if ( db->keyExists( "maxProcs" ) ) {
        maxSize = db->getScalar<int>( "maxProcs" );
    } else if ( MeshType == std::string( "Multimesh" ) ) {
        // The mesh is a multimesh
        maxSize = AMP::Mesh::MultiMesh::maxProcs( params );
    } else if ( MeshType == std::string( "AMP" ) ) {
        // The mesh is a AMP mesh
        auto filename = db->getWithDefault<std::string>( "FileName", "" );
        auto suffix   = IO::getSuffix( filename );
        if ( suffix == "stl" ) {
            // We are reading an stl file
            maxSize = AMP::Mesh::TriangleHelpers::maxProcs( params );
        } else {
            maxSize = AMP::Mesh::BoxMesh::maxProcs( params );
        }
    } else if ( MeshType == "TriangleGeometryMesh" ) {
        // We will build a triangle mesh from a geometry
        maxSize = AMP::Mesh::TriangleHelpers::maxProcs( params );
    } else if ( MeshType == std::string( "libMesh" ) ) {
// The mesh is a libmesh mesh
#ifdef AMP_USE_LIBMESH
        maxSize = AMP::Mesh::libmeshMesh::maxProcs( params );
#else
        AMP_ERROR( "AMP was compiled without support for libMesh" );
#endif
    } else if ( MeshType == "stk" || MeshType == "STKMesh" ) {
        AMP_ERROR( "stk mesh support was removed on 12/02/25" );
    } else if ( MeshType == "moab" || MeshType == "MOAB" ) {
        AMP_ERROR( "MOAB support was removed on 12/02/25" );
    } else {
        // Unknown mesh type
        AMP_ERROR( "Unknown mesh type " + MeshType + " and maxProcs is not set" );
    }
    return maxSize;
}


} // namespace AMP::Mesh
