#include "AMP/geometry/Geometry.h"
#include "AMP/IO/HDF.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/geometry/MeshGeometry.h"
#include "AMP/geometry/MultiGeometry.h"
#include "AMP/geometry/shapes/Box.h"
#include "AMP/geometry/shapes/Circle.h"
#include "AMP/geometry/shapes/CircleFrustum.h"
#include "AMP/geometry/shapes/CircleSurface.h"
#include "AMP/geometry/shapes/Cylinder.h"
#include "AMP/geometry/shapes/Parallelepiped.h"
#include "AMP/geometry/shapes/RegularPolygon.h"
#include "AMP/geometry/shapes/Shell.h"
#include "AMP/geometry/shapes/Sphere.h"
#include "AMP/geometry/shapes/SphereSurface.h"
#include "AMP/geometry/shapes/SquareFrustum.h"
#include "AMP/geometry/shapes/Tube.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/utils/Database.h"

#include <algorithm>

namespace AMP::Geometry {


/********************************************************
 * Create the geometry object                            *
 ********************************************************/
std::shared_ptr<AMP::Geometry::Geometry>
Geometry::buildGeometry( std::shared_ptr<const AMP::Database> db )
{
    AMP_ASSERT( db );
    auto generator = db->getString( "Generator" );
    std::for_each( generator.begin(), generator.end(), []( char &c ) { c = ::tolower( c ); } );
    std::shared_ptr<AMP::Geometry::Geometry> geom;
    if ( generator == "cube" ) {
        geom = buildBox( db );
    } else if ( generator == "tube" ) {
        geom = std::make_shared<Tube>( db );
    } else if ( generator == "circle" ) {
        geom = std::make_shared<Circle>( db );
    } else if ( generator == "circle_surface" ) {
        geom = std::make_shared<CircleSurface>( db );
    } else if ( generator == "cylinder" ) {
        geom = std::make_shared<Cylinder>( db );
    } else if ( generator == "shell" ) {
        geom = std::make_shared<Shell>( db );
    } else if ( generator == "sphere" ) {
        geom = std::make_shared<Sphere>( db );
    } else if ( generator == "sphere_surface" ) {
        geom = std::make_shared<SphereSurface>( db );
    } else if ( generator == "square_frustrum" || generator == "square_frustum" ) {
        geom = std::make_shared<SquareFrustum>( db );
    } else if ( generator == "circle_frustrum" || generator == "circle_frustum" ) {
        geom = std::make_shared<CircleFrustum>( db );
    } else if ( generator == "parallelepiped" ) {
        geom = std::make_shared<Parallelepiped>( db );
    } else if ( generator == "regular_polygon" ) {
        geom = std::make_shared<RegularPolygon>( db );
    } else if ( generator == "pentagon" ) {
        auto db2 = db->cloneDatabase();
        db2->putScalar( "N", 5 );
        geom = std::make_shared<RegularPolygon>( std::move( db2 ) );
    } else if ( generator == "mesh" ) {
        // Generate a mesh geometry
        auto mesh_db = db->getDatabase( "Mesh" );
        auto params  = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db->cloneDatabase() );
        params->setComm( AMP_COMM_SELF );
        auto mesh      = AMP::Mesh::MeshFactory::create( params );
        auto multimesh = std::dynamic_pointer_cast<AMP::Mesh::MultiMesh>( mesh );
        if ( multimesh ) {
            std::vector<std::shared_ptr<Geometry>> geoms;
            for ( const auto &mesh2 : multimesh->getMeshes() )
                geoms.push_back( std::make_shared<MeshGeometry>( mesh2 ) );
            geom = std::make_shared<MultiGeometry>( geoms );
        } else {
            geom = std::make_shared<MeshGeometry>( mesh );
        }
    } else {
        AMP_ERROR( "Unknown generator: " + generator );
    }
    AMP_INSIST( geom, "Failed to generate " + generator );
    // Displace the geometry
    double dist[3] = { db->getWithDefault<double>( "x_offset", 0.0 ),
                       db->getWithDefault<double>( "y_offset", 0.0 ),
                       db->getWithDefault<double>( "z_offset", 0.0 ) };
    geom->displace( dist );
    return geom;
}
AMP::Mesh::GeomType Geometry::getGeomType() const
{
    return static_cast<AMP::Mesh::GeomType>( d_physicalDim );
}


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
static inline uint8_t readPhysical( int64_t fid )
{
    uint8_t x = 0;
    IO::readHDF5( fid, "physicalDim", x );
    return x;
}
uint64_t Geometry::getID() const { return reinterpret_cast<uint64_t>( this ); }
void Geometry::registerChildObjects( AMP::IO::RestartManager * ) const {}
void Geometry::writeRestart( int64_t fid ) const
{
    IO::writeHDF5( fid, "physicalDim", d_physicalDim );
}
Geometry::Geometry( int64_t fid ) : d_physicalDim( readPhysical( fid ) ) {}


} // namespace AMP::Geometry


/********************************************************
 * Write/read geometry class                             *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::Geometry::Geometry>::DataStoreType(
    std::shared_ptr<const AMP::Geometry::Geometry> geom, RestartManager *manager )
    : d_data( geom )
{
    d_hash = d_data->getID();
    // Register child objects
    geom->registerChildObjects( manager );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::Geometry::Geometry>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid = createGroup( fid, name );
    writeHDF5( gid, "GeomType", d_data->getName() );
    d_data->writeRestart( gid );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::Geometry::Geometry>
AMP::IO::RestartManager::DataStoreType<AMP::Geometry::Geometry>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    using namespace AMP::Geometry;
    int64_t gid = openGroup( fid, name );
    std::string type;
    IO::readHDF5( gid, "GeomType", type );
    std::shared_ptr<AMP::Geometry::Geometry> geom;
    if ( type == "MultiGeometry" ) {
        geom = std::make_shared<MultiGeometry>( gid, manager );
    } else if ( type == "Box<1>" ) {
        geom = std::make_shared<Box<1>>( gid );
    } else if ( type == "Box<2>" ) {
        geom = std::make_shared<Box<2>>( gid );
    } else if ( type == "Box<3>" ) {
        geom = std::make_shared<Box<3>>( gid );
    } else if ( type == "Box<1>" ) {
        geom = std::make_shared<Grid<1>>( gid );
    } else if ( type == "Grid<2>" ) {
        geom = std::make_shared<Grid<2>>( gid );
    } else if ( type == "Grid<3>" ) {
        geom = std::make_shared<Grid<3>>( gid );
    } else if ( type == "Tube" ) {
        geom = std::make_shared<Tube>( gid );
    } else if ( type == "Circle" ) {
        geom = std::make_shared<Circle>( gid );
    } else if ( type == "CircleSurface" ) {
        geom = std::make_shared<CircleSurface>( gid );
    } else if ( type == "Cylinder" ) {
        geom = std::make_shared<Cylinder>( gid );
    } else if ( type == "Shell" ) {
        geom = std::make_shared<Shell>( gid );
    } else if ( type == "Sphere" ) {
        geom = std::make_shared<Sphere>( gid );
    } else if ( type == "SphereSurface" ) {
        geom = std::make_shared<SphereSurface>( gid );
    } else if ( type == "SquareFrustum" ) {
        geom = std::make_shared<SquareFrustum>( gid );
    } else if ( type == "CircleFrustum" ) {
        geom = std::make_shared<CircleFrustum>( gid );
    } else if ( type == "Parallelepiped" ) {
        geom = std::make_shared<Parallelepiped>( gid );
    } else if ( type == "RegularPolygon" ) {
        geom = std::make_shared<RegularPolygon>( gid );
    } else if ( type == "mesh" ) {
        geom = std::make_shared<MeshGeometry>( gid );
    } else {
        AMP_ERROR( "Unknown geometry " + type );
    }
    closeGroup( gid );
    return geom;
}
