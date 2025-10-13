#include "AMP/mesh/MeshFactory.h"
#include "AMP/IO/FileSystem.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/mesh/structured/MovableBoxMesh.h"
#include "AMP/mesh/structured/PureLogicalMesh.h"
#include "AMP/mesh/structured/StructuredGeometryMesh.h"
#include "AMP/mesh/triangle/TriangleHelpers.h"
#include "AMP/utils/MeshPoint.h"

#ifdef AMP_USE_TRILINOS_STKCLASSIC
// #include "AMP/mesh/STKmesh/STKMesh.h"
#endif
#ifdef AMP_USE_LIBMESH
    #include "AMP/mesh/libmesh/libmeshMesh.h"
#endif
#ifdef AMP_USE_MOAB
    #include "AMP/mesh/moab/moabMesh.h"
#endif

#include "ProfilerApp.h"


namespace AMP::Mesh {


/********************************************************
 *  Create a mesh                                        *
 ********************************************************/
std::shared_ptr<Mesh> MeshFactory::create( std::shared_ptr<MeshParameters> params )
{
    PROFILE( "MeshFactory::create" );
    auto db = params->getDatabase();
    AMP_ASSERT( db );
    AMP_INSIST( db->keyExists( "MeshType" ), "MeshType must exist in input database" );
    AMP_INSIST( db->keyExists( "MeshName" ), "MeshName must exist in input database" );
    auto MeshType = db->getString( "MeshType" );
    auto MeshName = db->getString( "MeshName" );
    std::shared_ptr<AMP::Mesh::Mesh> mesh;
    if ( MeshType == "Multimesh" ) {
        // The mesh is a multimesh
        mesh = std::make_shared<AMP::Mesh::MultiMesh>( params );
    } else if ( MeshType == "AMP" ) {
        // The mesh is a AMP mesh
        auto filename = db->getWithDefault<std::string>( "FileName", "" );
        auto suffix   = IO::getSuffix( filename );
        if ( suffix == "stl" ) {
            // We are reading an stl file
            mesh = AMP::Mesh::TriangleHelpers::generate( params );
        } else {
            mesh = AMP::Mesh::BoxMesh::generate( params );
        }
    } else if ( MeshType == "TriangleGeometryMesh" ) {
        // We will build a triangle mesh from a geometry
        mesh = AMP::Mesh::TriangleHelpers::generate( params );
    } else if ( MeshType == "libMesh" ) {
// The mesh is a libmesh mesh
#ifdef AMP_USE_LIBMESH
        mesh = std::make_shared<AMP::Mesh::libmeshMesh>( params );
#else
        AMP_ERROR( "AMP was compiled without support for libMesh" );
#endif
    } else if ( MeshType == "STKMesh" ) {
// The mesh is a stk mesh
#ifdef AMP_USE_TRILINOS_STKClassic
        // mesh = std::make_shared<AMP::Mesh::STKMesh>( params );
        AMP_ERROR( "AMP stk mesh interface is broken" );
#else
        AMP_ERROR( "AMP was compiled without support for STKMesh" );
#endif
    } else if ( MeshType == "moab" || MeshType == "MOAB" ) {
// The mesh is a MOAB mesh
#ifdef AMP_USE_MOAB
        mesh = std::make_shared<AMP::Mesh::moabMesh>( params );
#else
        AMP_ERROR( "AMP was compiled without support for MOAB" );
#endif
    } else {
        // Search for a mesh generator
        mesh = FactoryStrategy<Mesh, std::shared_ptr<MeshParameters>>::create( MeshType, params );
    }
    mesh->setName( MeshName );
    return mesh;
}


} // namespace AMP::Mesh


/********************************************************
 *  Restart operations for Mesh                          *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::Mesh::Mesh>::DataStoreType(
    std::shared_ptr<const AMP::Mesh::Mesh> mesh, RestartManager *manager )
    : d_data( mesh )
{
    d_hash = d_data->meshID().getHash();
    // Register the comm
    manager->registerComm( mesh->getComm() );
    // Register child meshes
    mesh->registerChildObjects( manager );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::Mesh::Mesh>::write( hid_t fid,
                                                                     const std::string &name ) const
{
    hid_t gid = createGroup( fid, name );
    writeHDF5( gid, "MeshType", d_data->meshClass() );
    d_data->writeRestart( gid );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::Mesh::Mesh> AMP::IO::RestartManager::DataStoreType<AMP::Mesh::Mesh>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    hid_t gid = openGroup( fid, name );
    auto mesh = AMP::Mesh::MeshFactory::create( gid, manager );
    closeGroup( gid );
    return mesh;
}
std::shared_ptr<AMP::Mesh::Mesh> AMP::Mesh::MeshFactory::create( int64_t fid,
                                                                 AMP::IO::RestartManager *manager )
{
    std::string type;
    AMP::IO::readHDF5( fid, "MeshType", type );
    std::shared_ptr<AMP::Mesh::Mesh> mesh;
    if ( type == "MultiMesh" || type.substr( 0, 10 ) == "MultiMesh<" ) {
        mesh = std::make_shared<AMP::Mesh::MultiMesh>( fid, manager );
    } else if ( type == "StructuredGeometryMesh" ) {
        mesh = std::make_shared<AMP::Mesh::StructuredGeometryMesh>( fid, manager );
    } else if ( type == "MovableBoxMesh" ) {
        mesh = std::make_shared<AMP::Mesh::MovableBoxMesh>( fid, manager );
    } else if ( type == "PureLogicalMesh" ) {
        mesh = std::make_shared<AMP::Mesh::PureLogicalMesh>( fid, manager );
    } else {
        // Search for a mesh generator
        mesh =
            FactoryStrategy<Mesh, int64_t, AMP::IO::RestartManager *>::create( type, fid, manager );
    }
    return mesh;
}


/********************************************************
 *  Default factory functions                            *
 ********************************************************/
template<>
void AMP::FactoryStrategy<AMP::Mesh::Mesh,
                          std::shared_ptr<AMP::Mesh::MeshParameters>>::registerDefault()
{
}
template<>
void AMP::FactoryStrategy<AMP::Mesh::Mesh, int64_t, AMP::IO::RestartManager *>::registerDefault()
{
}
