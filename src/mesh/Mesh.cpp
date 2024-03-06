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
#include "AMP/utils/AMP_MPI.I"
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
    : d_geometry( nullptr ),
      GeomDim( rhs.GeomDim ),
      PhysicalDim( rhs.PhysicalDim ),
      d_max_gcw( rhs.d_max_gcw ),
      d_comm( rhs.d_comm ),
      d_meshID( 0 ),
      d_name( rhs.d_name ),
      d_box( rhs.d_box ),
      d_box_local( rhs.d_box_local )
{
    setMeshID();
    if ( rhs.d_geometry )
        d_geometry = rhs.d_geometry->clone();
}


/********************************************************
 * De-constructor                                        *
 ********************************************************/
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
std::vector<MeshID> Mesh::getAllMeshIDs() const { return std::vector<MeshID>( 1, d_meshID ); }
std::vector<MeshID> Mesh::getBaseMeshIDs() const { return std::vector<MeshID>( 1, d_meshID ); }
std::vector<MeshID> Mesh::getLocalMeshIDs() const { return std::vector<MeshID>( 1, d_meshID ); }
std::vector<MeshID> Mesh::getLocalBaseMeshIDs() const { return std::vector<MeshID>( 1, d_meshID ); }


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
MeshElement Mesh::getElement( const MeshElementID &elem_id ) const
{
    MeshID mesh_id = elem_id.meshID();
    AMP_INSIST( mesh_id == d_meshID, "mesh id must match the mesh id of the element" );
    auto it = getIterator( elem_id.type() );
    for ( size_t i = 0; i < it.size(); i++, ++it ) {
        if ( it->globalID() == elem_id )
            return *it;
    }
    return MeshElement();
}


/********************************************************
 * Function to return parents of an element              *
 ********************************************************/
std::vector<MeshElement> Mesh::getElementParents( const MeshElement &, const GeomType ) const
{
    AMP_ERROR( "getElementParents is not implemented: " + meshClass() );
    return std::vector<MeshElement>();
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
    PROFILE_SCOPED( timer, "isMember" );
    auto elements = std::make_shared<std::vector<AMP::Mesh::MeshElement>>();
    elements->reserve( iterator.size() );
    for ( const auto &elem : iterator ) {
        if ( isMember( elem.globalID() ) )
            elements->push_back( elem );
    }
    return AMP::Mesh::MultiVectorIterator( elements, 0 );
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
static double getTol( const std::vector<double> &box, size_t N )
{
    int ndim     = box.size();
    size_t N2    = pow( N, 1.0 / ndim );
    double dx[3] = { 0, 0, 0 };
    for ( int d = 0; d < ndim / 2; d++ )
        dx[d] = ( box[2 * d + 1] - box[2 * d] ) / N2;
    return 0.2 * sqrt( dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2] );
}
static inline std::vector<Point> getPoints( MeshIterator it )
{
    std::vector<Point> p( it.size() );
    for ( size_t i = 0; i < p.size(); i++, ++it )
        p[i] = it->centroid();
    return p;
}
int Mesh::compare( const Mesh &a, const Mesh &b )
{
    // Check if the meshes are equal
    if ( a == b )
        return 1;
    // Special case for multimeshes
    auto a2 = dynamic_cast<const MultiMesh *>( &a );
    auto b2 = dynamic_cast<const MultiMesh *>( &b );
    if ( a2 || b2 ) {
        if ( !a2 || !b2 )
            return false;
        auto list1 = a2->getMeshes();
        auto list2 = b2->getMeshes();
        if ( list1.size() != list2.size() )
            return false;
        int result = 1;
        for ( size_t i = 0; i < list1.size(); i++ ) {
            int test = compare( *list1[i], *list2[i] );
            if ( test == 0 )
                return 0;
            result = std::max( result, test );
        }
        return result;
    }
    // Default comparison
    // Perform simple comparisons
    if ( a.GeomDim != b.GeomDim || a.PhysicalDim != b.PhysicalDim ||
         a.d_comm.compare( b.d_comm ) == 0 )
        return 0;
    if ( a.getBoundaryIDs() != b.getBoundaryIDs() || a.getBlockIDs() != b.getBlockIDs() )
        return 0;
    // Compare domains
    size_t N1  = a.numLocalElements( a.GeomDim );
    size_t N2  = b.numLocalElements( b.GeomDim );
    auto box1  = a.getBoundingBox();
    auto box2  = b.getBoundingBox();
    double tol = getTol( box1, std::min( N1, N2 ) );
    for ( size_t i = 0; i < box1.size(); i++ ) {
        if ( fabs( box1[i] - box2[i] ) > tol )
            return 0;
    }
    // Compare the coordinates
    if ( N1 == N2 ) {
        bool test    = true;
        auto nodes_a = getPoints( a.getIterator( GeomType::Vertex ) );
        auto nodes_b = getPoints( b.getIterator( GeomType::Vertex ) );
        auto elems_a = getPoints( a.getIterator( a.GeomDim ) );
        auto elems_b = getPoints( b.getIterator( b.GeomDim ) );
        kdtree tree_a_node( nodes_a );
        kdtree tree_a_elem( elems_a );
        for ( const auto &p : nodes_b ) {
            auto p2 = tree_a_node.find_nearest( p );
            test    = test && ( p - p2 ).norm() < tol * tol;
        }
        for ( const auto &p : elems_b ) {
            auto p2 = tree_a_elem.find_nearest( p );
            test    = test && ( p - p2 ).norm() < tol * tol;
        }
        if ( test )
            return 2;
    }
    // Get the geometries
    auto geom1 = a.getGeometry();
    auto geom2 = b.getGeometry();
    if ( !geom1 ) {
        auto ptr = std::const_pointer_cast<Mesh>( a.shared_from_this() );
        geom1    = std::make_shared<AMP::Geometry::MeshGeometry>( ptr );
    }
    if ( !geom2 ) {
        auto ptr = std::const_pointer_cast<Mesh>( b.shared_from_this() );
        geom2    = std::make_shared<AMP::Geometry::MeshGeometry>( ptr );
    }
    if ( *geom1 == *geom2 )
        return 3;

    AMP_WARNING( "Not finished" );
    return -1;
}


/********************************************************
 * MeshIterator set operations                           *
 ********************************************************/
MeshIterator Mesh::getIterator( SetOP OP, const MeshIterator &A, const MeshIterator &B )
{
    PROFILE_SCOPED( timer, "getIterator" );
    // Get a list of ids to keep
    std::vector<MeshElementID> ids;
    if ( OP == SetOP::Union ) {
        // Perform a union: A U B
        PROFILE_SCOPED( timer, "getIterator::Union" );
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
        PROFILE_SCOPED( timer, "getIterator::Intersection" );
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
        PROFILE_SCOPED( timer, "getIterator::Complement" );
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
    auto elements = std::make_shared<std::vector<MeshElement>>( ids.size() );
    for ( auto &elem : A ) {
        auto idA = elem.globalID();
        size_t i = std::min( Utilities::findfirst( ids, idA ), ids.size() - 1 );
        if ( ids[i] == idA ) {
            ( *elements )[i] = elem;
            N++;
        }
    }
    if ( N != elements->size() ) {
        for ( auto &elem : B ) {
            auto idB = elem.globalID();
            size_t i = std::min( Utilities::findfirst( ids, idB ), ids.size() - 1 );
            if ( ids[i] == idB ) {
                ( *elements )[i] = elem;
                N++;
            }
        }
        AMP_ASSERT( N == elements->size() );
    }
    return MultiVectorIterator( elements, 0 );
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
    writeHDF5( fid, "Geometry", geomID );
    writeHDF5( fid, "GeomDim", GeomDim );
    writeHDF5( fid, "PhysicalDim", PhysicalDim );
    writeHDF5( fid, "max_gcw", d_max_gcw );
    writeHDF5( fid, "comm", d_comm.hash() );
    writeHDF5( fid, "meshID", d_meshID );
    writeHDF5( fid, "name", d_name );
    writeHDF5( fid, "box", d_box );
    writeHDF5( fid, "box_local", d_box_local );
}
Mesh::Mesh( int64_t fid, AMP::IO::RestartManager *manager )
{
    uint64_t commHash, geomID;
    readHDF5( fid, "Geometry", geomID );
    readHDF5( fid, "GeomDim", GeomDim );
    readHDF5( fid, "PhysicalDim", PhysicalDim );
    readHDF5( fid, "max_gcw", d_max_gcw );
    readHDF5( fid, "comm", commHash );
    readHDF5( fid, "meshID", d_meshID );
    readHDF5( fid, "name", d_name );
    readHDF5( fid, "box", d_box );
    readHDF5( fid, "box_local", d_box_local );
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
INSTANTIATE_MPI_BCAST( AMP::Mesh::MeshElementID );
INSTANTIATE_MPI_SENDRECV( AMP::Mesh::MeshElementID );
INSTANTIATE_MPI_GATHER( AMP::Mesh::MeshElementID );
