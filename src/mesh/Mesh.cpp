#include "AMP/mesh/Mesh.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/geometry/Geometry.h"
#include "AMP/geometry/MeshGeometry.h"
#include "AMP/mesh/MeshElementVectorIterator.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MeshUtilities.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/SubsetMesh.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/kdtree.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <cmath>

#include "ProfilerApp.h"


namespace AMP::Mesh {


static_assert( sizeof( MeshID ) == 8, "unexpected size for MeshID" );
static_assert( sizeof( ElementID ) == 8, "unexpected size for ElementID" );
static_assert( sizeof( MeshElementID ) == 16, "unexpected size for MeshElementID" );


static unsigned int nextLocalMeshID = 1;


/********************************************************
 * Constructors                                          *
 ********************************************************/
Mesh::Mesh() : GeomDim( GeomType::Vertex ), PhysicalDim( 0 ), d_max_gcw( 0 ), d_meshID( 0 ) {}
Mesh::Mesh( std::shared_ptr<const MeshParameters> params )
{
    // Set the base properties
    AMP_ASSERT( params );
    AMP_ASSERT( sizeof( MeshElementID ) == 16 );
    GeomDim     = GeomType::Nullity;
    PhysicalDim = 0;
    d_max_gcw   = 0;
    d_comm      = params->comm;
    AMP_INSIST( !d_comm.isNull(), "Communicator in mesh params must be non NULL" );
    setMeshID();
    d_name  = "NULL";
    auto db = params->getDatabase();
    if ( db )
        d_name = db->getWithDefault<std::string>( "MeshName", d_name );
}
Mesh::Mesh( const Mesh &rhs )
    : GeomDim( rhs.GeomDim ),
      PhysicalDim( rhs.PhysicalDim ),
      d_max_gcw( rhs.d_max_gcw ),
      d_meshID( 0 ),
      d_comm( rhs.d_comm ),
      d_name( rhs.d_name ),
      d_box( rhs.d_box ),
      d_box_local( rhs.d_box_local )
{
    setMeshID();
    if ( rhs.d_geometry )
        d_geometry = rhs.d_geometry->clone();
}
Mesh::~Mesh() = default;


/********************************************************
 * Function to set the mesh ID                           *
 * This function will create a unique ID for every mesh. *
 * To accomplish this goal, the ID will consist of the   *
 * rank of the root processor (from the global comm),    *
 * and the number of meshes created by that processor.   *
 ********************************************************/
void Mesh::setMeshID()
{
    if ( d_comm.getRank() == 0 ) {
        // Root will create the meshID
        AMP_MPI globalComm( AMP_COMM_WORLD );
        d_meshID = MeshID( globalComm.getRank(), nextLocalMeshID );
        nextLocalMeshID++;
    }
    // Broadcast the meshID to all processors
    d_meshID = d_comm.bcast( d_meshID, 0 );
}


/********************************************************
 * Function to return the meshID composing the mesh      *
 ********************************************************/
bool Mesh::isBaseMesh() const { return true; }
bool Mesh::containsElement( const MeshElementID &id ) const { return id.meshID() == d_meshID; }
std::vector<MeshID> Mesh::getAllMeshIDs() const { return std::vector<MeshID>( 1, d_meshID ); }
std::vector<MeshID> Mesh::getBaseMeshIDs() const { return std::vector<MeshID>( 1, d_meshID ); }
std::vector<MeshID> Mesh::getLocalMeshIDs() const { return std::vector<MeshID>( 1, d_meshID ); }
std::vector<MeshID> Mesh::getLocalBaseMeshIDs() const { return std::vector<MeshID>( 1, d_meshID ); }


/********************************************************
 * Fill the domain box from the local box                *
 ********************************************************/
std::vector<double> Mesh::reduceBox( const std::vector<double> &x, const AMP_MPI &comm )
{
    int ndim = x.size() / 2;
    AMP_ASSERT( ndim && (int) x.size() == 2 * ndim );
    double localMin[3] = { 0 }, localMax[3] = { 0 }, globalMin[3] = { 0 }, globalMax[3] = { 0 };
    for ( int d = 0; d < ndim; d++ ) {
        localMin[d] = x[2 * d + 0];
        localMax[d] = x[2 * d + 1];
        AMP_ASSERT( fabs( localMax[d] ) < 1e100 && fabs( localMax[d] ) < 1e100 );
    }
    comm.minReduce( localMin, globalMin, ndim );
    comm.maxReduce( localMax, globalMax, ndim );
    std::vector<double> y( x.size() );
    for ( int d = 0; d < ndim; d++ ) {
        y[2 * d + 0] = globalMin[d];
        y[2 * d + 1] = globalMax[d];
    }
    return y;
}


/********************************************************
 * Function to return the mesh with the given ID         *
 ********************************************************/
std::shared_ptr<Mesh> Mesh::Subset( MeshID meshID ) const
{
    if ( d_meshID == meshID )
        return std::const_pointer_cast<Mesh>( shared_from_this() );
    else
        return std::shared_ptr<Mesh>();
}


/********************************************************
 * Function to return the mesh with the given name       *
 ********************************************************/
std::shared_ptr<Mesh> Mesh::Subset( std::string name ) const
{
    if ( d_name == name )
        return std::const_pointer_cast<Mesh>( shared_from_this() );
    else
        return std::shared_ptr<Mesh>();
}


/********************************************************
 * Function to subset a mesh using a mesh iterator       *
 ********************************************************/
std::shared_ptr<Mesh> Mesh::Subset( const MeshIterator &iterator, bool isGlobal ) const
{
    if ( isGlobal ) {
        auto N = d_comm.sumReduce( iterator.size() );
        if ( N == 0 )
            return std::shared_ptr<Mesh>();
    } else if ( iterator.size() == 0 ) {
        return std::shared_ptr<Mesh>();
    }
    return SubsetMesh::create( shared_from_this(), iterator, isGlobal );
}


/********************************************************
 * Function to return the element given an ID            *
 ********************************************************/
std::unique_ptr<MeshElement> Mesh::getElement( const MeshElementID &elem_id ) const
{
    MeshID mesh_id = elem_id.meshID();
    AMP_INSIST( mesh_id == d_meshID, "mesh id must match the mesh id of the element" );
    auto it = getIterator( elem_id.type() );
    for ( size_t i = 0; i < it.size(); i++, ++it ) {
        if ( it->globalID() == elem_id )
            return it->clone();
    }
    return std::make_unique<MeshElement>();
}


/********************************************************
 * Function to return parents of an element              *
 ********************************************************/
std::vector<std::unique_ptr<MeshElement>> Mesh::getElementParents( const MeshElement &,
                                                                   const GeomType ) const
{
    AMP_ERROR( "getElementParents is not implemented: " + meshClass() );
    return {};
}


/********************************************************
 * Return the position vector                            *
 ********************************************************/
AMP::LinearAlgebra::Vector::shared_ptr Mesh::getPositionVector( std::string name,
                                                                const int gcw ) const
{
    auto DOFs = AMP::Discretization::simpleDOFManager::create(
        std::const_pointer_cast<Mesh>( shared_from_this() ),
        AMP::Mesh::GeomType::Vertex,
        gcw,
        PhysicalDim,
        true );
    auto nodalVariable = std::make_shared<AMP::LinearAlgebra::Variable>( name );
    auto position      = AMP::LinearAlgebra::createVector( DOFs, nodalVariable, true );
    std::vector<size_t> dofs( PhysicalDim );
    for ( const auto &elem : DOFs->getIterator() ) {
        auto id    = elem.globalID();
        auto coord = elem.coord();
        DOFs->getDOFs( id, dofs );
        position->setValuesByGlobalID( dofs.size(), &dofs[0], &coord[0] );
    }
    return position;
}


/********************************************************
 * Check if the element is a member of the mesh          *
 ********************************************************/
bool Mesh::isMember( const MeshElementID &id ) const { return id.meshID() == d_meshID; }
MeshIterator Mesh::isMember( const MeshIterator &iterator ) const
{
    PROFILE( "isMember" );
    auto elements = std::make_shared<std::vector<std::unique_ptr<AMP::Mesh::MeshElement>>>();
    elements->reserve( iterator.size() );
    for ( const auto &elem : iterator ) {
        if ( isMember( elem.globalID() ) )
            elements->push_back( elem.clone() );
    }
    return AMP::Mesh::MeshElementVectorIterator( elements, 0 );
}


/********************************************************
 * Functions that aren't implemented for the base class  *
 ********************************************************/
std::shared_ptr<Mesh> Mesh::Subset( Mesh & ) const
{
    AMP_ERROR( "Subset is not implemented: " + meshClass() );
    return std::shared_ptr<Mesh>();
}
MeshIterator Mesh::getIterator( const GeomType, const int ) const
{
    AMP_ERROR( "getIterator is not implemented: " + meshClass() );
    return MeshIterator();
}
MeshIterator Mesh::getSurfaceIterator( const GeomType, const int ) const
{
    AMP_ERROR( "getSurfaceIterator is not implemented: " + meshClass() );
    return MeshIterator();
}
std::vector<int> Mesh::getBoundaryIDs() const
{
    AMP_ERROR( "getBoundaryIDs is not implemented: " + meshClass() );
    return std::vector<int>();
}
MeshIterator Mesh::getBoundaryIDIterator( const GeomType, const int, const int ) const
{
    AMP_ERROR( "getBoundaryIDIterator is not implemented: " + meshClass() );
    return MeshIterator();
}
std::vector<int> Mesh::getBlockIDs() const
{
    AMP_ERROR( "getBlockIDs is not implemented: " + meshClass() );
    return std::vector<int>();
}
MeshIterator Mesh::getBlockIDIterator( const GeomType, const int, const int ) const
{
    AMP_ERROR( "getBlockIDIterator is not implemented: " + meshClass() );
    return MeshIterator();
}
size_t Mesh::numLocalElements( const GeomType ) const
{
    AMP_ERROR( "numLocalElements is not implemented: " + meshClass() );
    return 0;
}
size_t Mesh::numGlobalElements( const GeomType ) const
{
    AMP_ERROR( "numGlobalElements is not implemented: " + meshClass() );
    return 0;
}
size_t Mesh::numGhostElements( const GeomType, int ) const
{
    AMP_ERROR( "numGhostElements is not implemented: " + meshClass() );
    return 0;
}


/********************************************************
 * Compare two meshes                                    *
 ********************************************************/
int Mesh::CompareResult::result() const
{
    if ( equal )
        return 1;
    if ( nodes && surface && block && domain )
        return 2;
    if ( surface && block && domain )
        return 3;
    return 0;
}
Mesh::CompareResult::CompareResult( int state )
    : equal( state == 1 ),
      nodes( state > 0 && state < 3 ),
      surface( state > 0 ),
      block( state > 0 ),
      domain( state > 0 ),
      geometry( state == 1 )
{
}
bool Mesh::CompareResult::operator==( const Mesh::CompareResult &rhs ) const
{
    return equal == rhs.equal && nodes == rhs.nodes && surface == rhs.surface &&
           block == rhs.block && domain == rhs.domain && geometry == rhs.geometry;
}
Mesh::CompareResult operator&&( const Mesh::CompareResult &a, const Mesh::CompareResult &b )
{
    Mesh::CompareResult c;
    c.equal    = a.equal && b.equal;
    c.nodes    = a.nodes && b.nodes;
    c.surface  = a.surface && b.surface;
    c.block    = a.block && b.block;
    c.domain   = a.domain && b.domain;
    c.geometry = a.geometry && b.geometry;
    return c;
}
static double getTol( const std::vector<double> &box, size_t N )
{
    int ndim     = box.size();
    size_t N2    = std::pow( N, 1.0 / ndim );
    double dx[3] = { 0, 0, 0 };
    for ( int d = 0; d < ndim / 2; d++ )
        dx[d] = ( box[2 * d + 1] - box[2 * d] ) / N2;
    return 0.2 * std::sqrt( dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] );
}
static inline std::vector<Point> getPoints( MeshIterator it )
{
    std::vector<Point> p( it.size() );
    for ( size_t i = 0; i < p.size(); i++, ++it )
        p[i] = it->centroid();
    return p;
}
Mesh::CompareResult Mesh::compare( const Mesh &a, const Mesh &b )
{
    // Check if the meshes are equal
    if ( a == b )
        return 1;
    // Special case for multimeshes
    auto a2 = dynamic_cast<const MultiMesh *>( &a );
    auto b2 = dynamic_cast<const MultiMesh *>( &b );
    if ( a2 || b2 ) {
        if ( !a2 || !b2 )
            return 0;
        auto list1 = a2->getMeshes();
        auto list2 = b2->getMeshes();
        if ( list1.size() != list2.size() )
            return 0;
        CompareResult result( 1 );
        for ( size_t i = 0; i < list1.size(); i++ ) {
            auto test = compare( *list1[i], *list2[i] );
            result    = result && test;
        }
        return result;
    }
    // Check dimensions and comm
    if ( a.GeomDim != b.GeomDim || a.PhysicalDim != b.PhysicalDim ||
         a.d_comm.compare( b.d_comm ) == 0 )
        return 0;
    // Default comparison
    Mesh::CompareResult result( 0 );
    // Perform simple comparisons
    result.surface = a.getBoundaryIDs() == b.getBoundaryIDs();
    result.block   = a.getBlockIDs() == b.getBlockIDs();
    // Compare domains
    size_t N1     = a.numLocalElements( a.GeomDim );
    size_t N2     = b.numLocalElements( b.GeomDim );
    auto box1     = a.getBoundingBox();
    auto box2     = b.getBoundingBox();
    double tol    = getTol( box1, std::min( N1, N2 ) );
    result.domain = true;
    for ( size_t i = 0; i < box1.size(); i++ ) {
        if ( fabs( box1[i] - box2[i] ) > tol )
            result.domain = false;
    }
    // Compare the coordinates
    if ( N1 == N2 ) {
        result.nodes = true;
        auto nodes_a = getPoints( a.getIterator( GeomType::Vertex ) );
        auto nodes_b = getPoints( b.getIterator( GeomType::Vertex ) );
        auto elems_a = getPoints( a.getIterator( a.GeomDim ) );
        auto elems_b = getPoints( b.getIterator( b.GeomDim ) );
        kdtree tree_a_node( nodes_a );
        kdtree tree_a_elem( elems_a );
        for ( const auto &p : nodes_b ) {
            auto p2      = tree_a_node.find_nearest( p );
            result.nodes = result.nodes && ( p - p2 ).norm() < tol * tol;
        }
        for ( const auto &p : elems_b ) {
            auto p2      = tree_a_elem.find_nearest( p );
            result.nodes = result.nodes && ( p - p2 ).norm() < tol * tol;
        }
    }
    // Get the geometries
    auto geom1 = a.getGeometry();
    auto geom2 = b.getGeometry();
    if ( geom1 && geom2 ) {
        result.geometry = ( *geom1 == *geom2 );
    }
    return result;
}


/********************************************************
 * MeshIterator set operations                           *
 ********************************************************/
MeshIterator Mesh::getIterator( SetOP OP, const MeshIterator &A, const MeshIterator &B )
{
    PROFILE( "getIterator" );
    // Get a list of ids to keep
    std::vector<MeshElementID> ids;
    if ( OP == SetOP::Union ) {
        // Perform a union: A U B
        PROFILE( "getIterator::Union" );
        if ( A.size() == 0 )
            return B.begin();
        if ( B.size() == 0 )
            return A.begin();
        // Get the union using the mesh IDs
        ids.reserve( A.size() + B.size() );
        for ( auto &elem : A )
            ids.push_back( elem.globalID() );
        for ( auto &elem : B )
            ids.push_back( elem.globalID() );
        Utilities::unique( ids );
    } else if ( OP == SetOP::Intersection ) {
        // Perform a intersection: A n B
        // Get the intersection using the mesh IDs
        PROFILE( "getIterator::Intersection" );
        if ( A.size() == 0 || B.size() == 0 )
            return MeshIterator();
        std::vector<MeshElementID> idA, idB;
        idA.reserve( A.size() );
        idB.reserve( B.size() );
        for ( auto &elem : A )
            idA.push_back( elem.globalID() );
        for ( auto &elem : B )
            idB.push_back( elem.globalID() );
        Utilities::quicksort( idA );
        Utilities::quicksort( idB );
        ids.resize( std::min( A.size(), B.size() ) );
        auto it =
            std::set_intersection( idA.begin(), idA.end(), idB.begin(), idB.end(), ids.begin() );
        ids.resize( it - ids.begin() );
        if ( ids.size() == A.size() )
            return A.begin();
        else if ( ids.size() == B.size() )
            return B.begin();
    } else if ( OP == SetOP::Complement ) {
        // Perform a SetOP::Complement:  A - B
        // Get the compliment using the mesh IDs
        PROFILE( "getIterator::Complement" );
        std::set<MeshElementID> compliment_set;
        for ( auto &elem : A )
            compliment_set.insert( elem.globalID() );
        for ( auto &elem : B )
            compliment_set.erase( elem.globalID() );
        ids = std::vector<MeshElementID>( compliment_set.begin(), compliment_set.end() );
    } else {
        AMP_ERROR( "Unknown set operation" );
    }
    // Create the iterator
    if ( ids.empty() )
        return MeshIterator();
    size_t N      = 0;
    auto elements = std::make_shared<std::vector<std::unique_ptr<MeshElement>>>( ids.size() );
    for ( auto &elem : A ) {
        auto idA = elem.globalID();
        size_t i = std::min( Utilities::findfirst( ids, idA ), ids.size() - 1 );
        if ( ids[i] == idA ) {
            ( *elements )[i] = elem.clone();
            N++;
        }
    }
    if ( N != elements->size() ) {
        for ( auto &elem : B ) {
            auto idB = elem.globalID();
            size_t i = std::min( Utilities::findfirst( ids, idB ), ids.size() - 1 );
            if ( ids[i] == idB ) {
                ( *elements )[i] = elem.clone();
                N++;
            }
        }
        AMP_ASSERT( N == elements->size() );
    }
    return MeshElementVectorIterator( elements, 0 );
}


/********************************************************
 * Create a view                                         *
 ********************************************************/
std::shared_ptr<Mesh> Mesh::createView( const Mesh &src, const AMP::Database &db )
{
    auto name = db.getString( "MeshName" );
    auto op   = db.getWithDefault<std::string>( "Operation", "" );
    auto list = db.getVector<std::string>( "MeshList" );
    std::vector<std::shared_ptr<Mesh>> meshes;
    for ( const auto &tmp : list ) {
        auto mesh = src.Subset( tmp );
        if ( mesh )
            meshes.push_back( mesh );
    }
    if ( src.getComm().allReduce( meshes.empty() ) )
        AMP_ERROR( "Failed to create view" );
    auto comm = src.getComm().split( meshes.empty() ? 0 : 1 );
    if ( meshes.empty() )
        return nullptr;
    auto mesh = std::make_shared<MultiMesh>( name, comm, meshes );
    if ( op == "" ) {
        return mesh;
    } else if ( op == "SurfaceIterator" ) {
        auto type = static_cast<AMP::Mesh::GeomType>( static_cast<int>( mesh->getGeomType() ) - 1 );
        auto mesh2 = mesh->Subset( mesh->getSurfaceIterator( type ) );
        mesh2->setName( name );
        return mesh2;
    } else {
        AMP_ERROR( "Unknown operation" );
    }
    return nullptr;
}


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
void Mesh::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    manager->registerComm( d_comm );
    if ( d_geometry )
        manager->registerObject( d_geometry );
}
void Mesh::writeRestart( int64_t fid ) const
{
    uint64_t geomID = d_geometry ? d_geometry->getID() : 0;
    IO::writeHDF5( fid, "Geometry", geomID );
    IO::writeHDF5( fid, "GeomDim", GeomDim );
    IO::writeHDF5( fid, "PhysicalDim", PhysicalDim );
    IO::writeHDF5( fid, "max_gcw", d_max_gcw );
    IO::writeHDF5( fid, "comm", d_comm.hash() );
    IO::writeHDF5( fid, "meshID", d_meshID );
    IO::writeHDF5( fid, "name", d_name );
    IO::writeHDF5( fid, "box", d_box );
    IO::writeHDF5( fid, "box_local", d_box_local );
}
Mesh::Mesh( int64_t fid, AMP::IO::RestartManager *manager )
{
    uint64_t commHash, geomID;
    IO::readHDF5( fid, "Geometry", geomID );
    IO::readHDF5( fid, "GeomDim", GeomDim );
    IO::readHDF5( fid, "PhysicalDim", PhysicalDim );
    IO::readHDF5( fid, "max_gcw", d_max_gcw );
    IO::readHDF5( fid, "comm", commHash );
    IO::readHDF5( fid, "meshID", d_meshID );
    IO::readHDF5( fid, "name", d_name );
    IO::readHDF5( fid, "box", d_box );
    IO::readHDF5( fid, "box_local", d_box_local );
    d_comm = manager->getComm( commHash );
    if ( geomID != 0 )
        d_geometry = manager->getData<AMP::Geometry::Geometry>( geomID );
}


/********************************************************
 * Stream operators                                      *
 ********************************************************/
std::ostream &operator<<( std::ostream &out, AMP::Mesh::GeomType x )
{
    out << static_cast<int>( x );
    return out;
}
std::ostream &operator<<( std::ostream &out, AMP::Mesh::MeshID x )
{
    out << x.getData();
    return out;
}
std::ostream &operator<<( std::ostream &out, AMP::Mesh::ElementID x )
{
    int is_local = x.is_local() ? 1 : 0;
    out << "(" << is_local << "," << x.type() << "," << x.local_id() << "," << x.owner_rank()
        << ")";
    return out;
}
std::ostream &operator<<( std::ostream &out, AMP::Mesh::MeshElementID x )
{
    int is_local = x.is_local() ? 1 : 0;
    out << "(" << is_local << "," << x.type() << "," << x.local_id() << "," << x.owner_rank() << ","
        << x.meshID() << ")";
    return out;
}
void Mesh::printMeshHierarchy( const Mesh &mesh, std::ostream &out, const std::string &prefix )
{
    out << prefix << mesh.getName() << std::endl;
    auto multimesh = dynamic_cast<const MultiMesh *>( &mesh );
    if ( multimesh ) {
        for ( auto mesh2 : multimesh->getMeshes() )
            printMeshHierarchy( *mesh2, out, prefix + "   " );
    }
}


/********************************************************
 * Arithmetic operators                                  *
 ********************************************************/
GeomType operator+( GeomType x, int y ) noexcept
{
    int z = static_cast<int>( x ) + y;
    return ( z >= 0 && z <= 5 ) ? static_cast<GeomType>( z ) : GeomType::Nullity;
}
GeomType operator-( GeomType x, int y ) noexcept
{
    int z = static_cast<int>( x ) - y;
    return ( z >= 0 && z <= 5 ) ? static_cast<GeomType>( z ) : GeomType::Nullity;
}
GeomType operator+( int x, GeomType y ) noexcept
{
    int z = x + static_cast<int>( y );
    return ( z >= 0 && z <= 5 ) ? static_cast<GeomType>( z ) : GeomType::Nullity;
}
GeomType operator-( int x, GeomType y ) noexcept
{
    int z = x - static_cast<int>( y );
    return ( z >= 0 && z <= 5 ) ? static_cast<GeomType>( z ) : GeomType::Nullity;
}


} // namespace AMP::Mesh


/********************************************************
 * Instantiate communication of MeshElementID            *
 ********************************************************/
#include "AMP/utils/AMP_MPI.I"
INSTANTIATE_MPI_BCAST( AMP::Mesh::MeshElementID );
INSTANTIATE_MPI_SENDRECV( AMP::Mesh::MeshElementID );
INSTANTIATE_MPI_GATHER( AMP::Mesh::MeshElementID );
