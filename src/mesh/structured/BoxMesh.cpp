#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/IO/HDF.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MultiIterator.h"
#include "AMP/mesh/structured/MovableBoxMesh.h"
#include "AMP/mesh/structured/PureLogicalMesh.h"
#include "AMP/mesh/structured/StructuredGeometryMesh.h"
#include "AMP/mesh/structured/structuredMeshElement.h"
#include "AMP/mesh/structured/structuredMeshIterator.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/ArrayVectorData.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"

#include "ProfilerApp.h"

#include <algorithm>
#include <cstring>
#include <iostream>


namespace AMP::Mesh {


/****************************************************************
 * Generator                                                     *
 ****************************************************************/
std::shared_ptr<BoxMesh> BoxMesh::generate( std::shared_ptr<const MeshParameters> params )
{
    auto db        = params->getDatabase();
    auto generator = db->getWithDefault<std::string>( "Generator", "" );
    if ( generator == "logical" ) {
        return std::make_shared<PureLogicalMesh>( params );
    } else {
        std::shared_ptr<BoxMesh> mesh( new StructuredGeometryMesh( params ) );
        bool static_mesh = db->getWithDefault<bool>( "static", false );
        if ( !static_mesh )
            mesh.reset( new MovableBoxMesh( *mesh ) );
        return mesh;
    }
}


/****************************************************************
 * Estimate the mesh size                                        *
 ****************************************************************/
size_t BoxMesh::estimateMeshSize( std::shared_ptr<const MeshParameters> params )
{
    auto size = estimateLogicalMeshSize( params );
    size_t N  = 1;
    for ( auto s : size )
        N *= s;
    return N;
}
ArraySize BoxMesh::estimateLogicalMeshSize( std::shared_ptr<const MeshParameters> params )
{
    auto db    = params->getDatabase();
    auto geom  = AMP::Geometry::Geometry::buildGeometry( db );
    auto geom2 = std::dynamic_pointer_cast<AMP::Geometry::LogicalGeometry>( geom );
    return geom2->getLogicalGridSize( db->getVector<size_t>( "Size" ) );
}


/****************************************************************
 * Constructor                                                   *
 ****************************************************************/
BoxMesh::BoxMesh()
    : Mesh(),
      d_rank( -1 ),
      d_size( 0 ),
      d_blockID( 1 ),
      d_globalSize{ 1 },
      d_numBlocks{ 1 },
      d_localIndex{ 0 },
      d_indexSize{ 0 },
      d_surfaceId{ -3 }
{
}
BoxMesh::BoxMesh( std::shared_ptr<const MeshParameters> params )
    : Mesh( params ),
      d_rank( -1 ),
      d_size( 0 ),
      d_blockID( 1 ),
      d_globalSize{ 1 },
      d_numBlocks{ 1 },
      d_localIndex{ 0 },
      d_indexSize{ 0 },
      d_surfaceId{ -3 }
{
    auto db = params->getDatabase();
    if ( db ) {
        if ( db->keyExists( "blockID" ) )
            d_blockID = db->getScalar<int>( "blockID" );
    }
}
BoxMesh::BoxMesh( const BoxMesh &mesh )
    : Mesh( mesh ),
      d_rank( mesh.d_rank ),
      d_size( mesh.d_size ),
      d_blockID( mesh.d_blockID ),
      d_globalSize( mesh.d_globalSize ),
      d_numBlocks( mesh.d_numBlocks ),
      d_startIndex{ mesh.d_startIndex[0], mesh.d_startIndex[1], mesh.d_startIndex[2] },
      d_endIndex{ mesh.d_endIndex[0], mesh.d_endIndex[1], mesh.d_endIndex[2] },
      d_localIndex( mesh.d_localIndex ),
      d_indexSize( mesh.d_indexSize ),
      d_surfaceId( mesh.d_surfaceId )
{
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void BoxMesh::writeRestart( int64_t fid ) const
{
    Mesh::writeRestart( fid );
    IO::writeHDF5( fid, "rank", d_rank );
    IO::writeHDF5( fid, "size", d_size );
    IO::writeHDF5( fid, "globalSize", d_globalSize );
    IO::writeHDF5( fid, "numBlocks", d_numBlocks );
    IO::writeHDF5( fid, "startIndex[0]", d_startIndex[0] );
    IO::writeHDF5( fid, "startIndex[1]", d_startIndex[1] );
    IO::writeHDF5( fid, "startIndex[2]", d_startIndex[2] );
    IO::writeHDF5( fid, "endIndex[0]", d_endIndex[0] );
    IO::writeHDF5( fid, "endIndex[1]", d_endIndex[1] );
    IO::writeHDF5( fid, "endIndex[2]", d_endIndex[2] );
    IO::writeHDF5( fid, "localIndex", d_localIndex );
    IO::writeHDF5( fid, "indexSize", d_indexSize );
    IO::writeHDF5( fid, "surfaceId", d_surfaceId );
    IO::writeHDF5( fid, "blockID", d_blockID );
}
template<class TYPE>
static inline TYPE read( int64_t fid, const std::string &name )
{
    TYPE x;
    IO::readHDF5( fid, name, x );
    return x;
}
BoxMesh::BoxMesh( int64_t fid, AMP::IO::RestartManager *manager )
    : Mesh( fid, manager ),
      d_rank( read<int>( fid, "rank" ) ),
      d_size( read<int>( fid, "size" ) ),
      d_blockID( read<int>( fid, "blockID" ) ),
      d_globalSize( read<std::array<int, 3>>( fid, "globalSize" ) ),
      d_numBlocks( read<std::array<int, 3>>( fid, "numBlocks" ) ),
      d_localIndex( read<std::array<int, 6>>( fid, "localIndex" ) ),
      d_indexSize( read<std::array<int, 3>>( fid, "indexSize" ) ),
      d_surfaceId( read<std::array<int, 6>>( fid, "surfaceId" ) )
{
    IO::readHDF5( fid, "startIndex[0]", const_cast<std::vector<int> &>( d_startIndex[0] ) );
    IO::readHDF5( fid, "startIndex[1]", const_cast<std::vector<int> &>( d_startIndex[1] ) );
    IO::readHDF5( fid, "startIndex[2]", const_cast<std::vector<int> &>( d_startIndex[2] ) );
    IO::readHDF5( fid, "endIndex[0]", const_cast<std::vector<int> &>( d_endIndex[0] ) );
    IO::readHDF5( fid, "endIndex[1]", const_cast<std::vector<int> &>( d_endIndex[1] ) );
    IO::readHDF5( fid, "endIndex[2]", const_cast<std::vector<int> &>( d_endIndex[2] ) );
}


/****************************************************************
 * Perform the load balancing                                    *
 ****************************************************************/
void BoxMesh::loadBalance( std::array<int, 3> size,
                           int N_procs,
                           std::vector<int> *startIndex,
                           std::vector<int> minSize0 )
{
    AMP_ASSERT( size[0] > 0 && size[1] > 0 && size[2] > 0 );
    // Check if we are dealing with a serial mesh
    if ( N_procs == 1 ) {
        startIndex[0] = std::vector<int>( 1, 0 );
        startIndex[1] = std::vector<int>( 1, 0 );
        startIndex[2] = std::vector<int>( 1, 0 );
        return;
    }
    // Get the minimum size / proc
    std::array<int, 3> minSize = { 1, 1, 1 };
    if ( !minSize0.empty() ) {
        if ( minSize0.size() == 1 )
            minSize0.resize( 3, minSize0[0] );
        minSize0.resize( 3, -1 );
        minSize = { minSize0[0], minSize0[1], minSize0[2] };
        for ( int d = 0; d < 3; d++ ) {
            if ( minSize[d] == -1 )
                minSize[d] = size[d];
        }
    }
    // Get the number of processors for each dimension
    int numBlocks[3] = { 1, 1, 1 };
    auto factors     = AMP::Utilities::factor( N_procs );
    while ( !factors.empty() ) {
        int d    = -1;
        double v = -1;
        for ( int i = 0; i < 3; i++ ) {
            double tmp = (double) size[i] / (double) numBlocks[i];
            if ( tmp > v && tmp > minSize[i] && minSize[i] >= 0 ) {
                d = i;
                v = tmp;
            }
        }
        if ( d == -1 )
            break;
        numBlocks[d] *= factors.back();
        factors.pop_back();
    }
    // Calculate the starting index for each dimension
    for ( int d = 0; d < 3; d++ ) {
        double n = size[d] / static_cast<double>( numBlocks[d] ) + 1e-12;
        startIndex[d].resize( numBlocks[d] );
        for ( int i = 0; i < numBlocks[d]; i++ )
            startIndex[d][i] = static_cast<int>( i * n );
    }
}


/****************************************************************
 * Initialize the mesh                                           *
 ****************************************************************/
void BoxMesh::initialize( const std::array<int, 3> &boxSize,
                          const std::array<int, 6> &ids,
                          const std::vector<int> &minSize )
{
    PROFILE( "initialize" );
    // Perform some const casts so we can set the local variables
    int &rank        = const_cast<int &>( d_rank );
    int &size        = const_cast<int &>( d_size );
    auto &globalSize = const_cast<std::array<int, 3> &>( d_globalSize );
    auto &numBlocks  = const_cast<std::array<int, 3> &>( d_numBlocks );
    auto startIndex  = const_cast<std::vector<int> *>( d_startIndex );
    auto endIndex    = const_cast<std::vector<int> *>( d_endIndex );
    auto &localIndex = const_cast<std::array<int, 6> &>( d_localIndex );
    auto &indexSize  = const_cast<std::array<int, 3> &>( d_indexSize );
    auto &surfaceId  = const_cast<std::array<int, 6> &>( d_surfaceId );
    // Cache comm data
    if ( !d_comm.isNull() ) {
        rank = d_comm.getRank();
        size = d_comm.getSize();
    }
    // Set basic variables
    globalSize = boxSize;
    surfaceId  = ids;
    AMP_INSIST( static_cast<int>( GeomDim ) <= 3, "Geometric dimension must be <= 3" );
    for ( int i = 2 * static_cast<int>( GeomDim ); i < 6; i++ )
        surfaceId[i] = -1;
    for ( int d = 0; d < 3; d++ )
        AMP_ASSERT( ( surfaceId[2 * d + 0] == -1 ) == ( surfaceId[2 * d + 1] == -1 ) );
    for ( int d = 0; d < static_cast<int>( GeomDim ); d++ )
        AMP_ASSERT( globalSize[d] > 0 );
    // Create the load balance
    AMP_INSIST( size > 0, "Communicator must be set" );
    loadBalance( globalSize, size, startIndex, minSize );
    // Set some cached values
    for ( int d = 0; d < 3; d++ ) {
        AMP_ASSERT( !startIndex[d].empty() );
        numBlocks[d] = startIndex[d].size();
        endIndex[d].resize( numBlocks[d] );
        for ( int i = 1; i < numBlocks[d]; i++ )
            endIndex[d][i - 1] = d_startIndex[d][i];
        endIndex[d].back() = globalSize[d];
    }
    auto block = getLocalBlock( rank );
    indexSize  = { block[1] - block[0] + 3, block[3] - block[2] + 3, block[5] - block[4] + 3 };
    localIndex = block;
    for ( int d = 0; d < 3; d++ ) {
        if ( localIndex[2 * d + 1] == globalSize[d] - 1 )
            localIndex[2 * d + 1] = globalSize[d];
        localIndex[2 * d + 1]++;
    }
}
void BoxMesh::createBoundingBox()
{
    // Check some assumptions/variables
    // Fill the bounding box
    AMP_INSIST( !d_comm.isNull(), "Communicator must be set" );
    d_box_local = std::vector<double>( 2 * PhysicalDim );
    for ( int d = 0; d < PhysicalDim; d++ ) {
        d_box_local[2 * d + 0] = 1e100;
        d_box_local[2 * d + 1] = -1e100;
    }
    double x[3] = { 0, 0, 0 };
    for ( auto &node : getIterator( GeomType::Vertex, 0 ) ) {
        auto element = dynamic_cast<structuredMeshElement *>( node.getRawElement() );
        AMP_ASSERT( element != nullptr );
        coord( element->getIndex(), x );
        for ( int d = 0; d < PhysicalDim; d++ ) {
            if ( x[d] != x[d] )
                AMP_ERROR( "NaNs detected" );
            d_box_local[2 * d + 0] = std::min( d_box_local[2 * d + 0], x[d] );
            d_box_local[2 * d + 1] = std::max( d_box_local[2 * d + 1], x[d] );
        }
    }
    d_box = std::vector<double>( PhysicalDim * 2 );
    for ( int i = 0; i < PhysicalDim; i++ ) {
        d_box[2 * i + 0] = d_comm.minReduce( d_box_local[2 * i + 0] );
        d_box[2 * i + 1] = d_comm.maxReduce( d_box_local[2 * i + 1] );
    }
}
void BoxMesh::finalize( const std::string &name, const std::vector<double> &displacement )
{
    PROFILE( "finalize" );
    d_name = name;
    // Fill in the final info for the mesh
    createBoundingBox();
    // Displace the mesh
    bool test = false;
    for ( auto &elem : displacement ) {
        if ( elem != 0.0 )
            test = true;
    }
    if ( test )
        displaceMesh( displacement );
}
std::vector<double> BoxMesh::getDisplacement( std::shared_ptr<const AMP::Database> db )
{
    std::vector<double> displacement( PhysicalDim, 0.0 );
    if ( db->keyExists( "x_offset" ) && PhysicalDim >= 1 )
        displacement[0] = db->getScalar<double>( "x_offset" );
    if ( db->keyExists( "y_offset" ) && PhysicalDim >= 2 )
        displacement[1] = db->getScalar<double>( "y_offset" );
    if ( db->keyExists( "z_offset" ) && PhysicalDim >= 3 )
        displacement[2] = db->getScalar<double>( "z_offset" );
    return displacement;
}


/****************************************************************
 * Destructor                                                    *
 ****************************************************************/
BoxMesh::~BoxMesh() = default;


/****************************************************************
 * Get the surface element ranges                                *
 ****************************************************************/
int BoxMesh::getSurfaceID( int s ) const
{
    int s_max = 2 * static_cast<int>( GeomDim );
    AMP_ASSERT( s >= 0 && s < s_max );
    return d_surfaceId[s];
}
BoxMesh::ElementBlocks BoxMesh::getSurface( int s, GeomType type ) const
{
    // Check if we are keeping the given surface
    int d     = s / 2;
    int s_max = 2 * static_cast<int>( GeomDim );
    if ( d_surfaceId[s] < 0 || s > s_max )
        return {};
    // Initialize some basic info
    bool left                   = s % 2 == 0;
    std::array<int, 3> lastCell = { std::max( d_globalSize[0] - 1, 0 ),
                                    std::max( d_globalSize[1] - 1, 0 ),
                                    std::max( d_globalSize[2] - 1, 0 ) };
    auto lastNode               = lastCell;
    for ( int d = 0; d < static_cast<int>( GeomDim ); d++ ) {
        if ( d_surfaceId[2 * d + 1] != -1 )
            lastNode[d]++;
    }
    if ( d_surfaceId[0] == -2 || d_surfaceId[2] == -2 || d_surfaceId[4] == -2 )
        AMP_WARN_ONCE( "Fix last node" );
    // Create the surface list
    if ( type == GeomDim ) {
        // We are dealing with the desired geometric type (e.g. Volume)
        MeshElementIndex first( GeomDim, 0, 0, 0, 0 );
        MeshElementIndex last( GeomDim, 0, lastCell[0], lastCell[1], lastCell[2] );
        if ( left )
            last.d_index[d] = first.d_index[d];
        else
            first.d_index[d] = last.d_index[d];
        return { std::make_pair( first, last ) };
    } else if ( type == GeomType::Vertex ) {
        // We are dealing with a vertex
        MeshElementIndex first( GeomType::Vertex, 0, 0, 0, 0 );
        MeshElementIndex last( GeomType::Vertex, 0, lastNode[0], lastNode[1], lastNode[2] );
        if ( left )
            last.d_index[d] = first.d_index[d];
        else
            first.d_index[d] = last.d_index[d];
        return { std::make_pair( first, last ) };
    } else if ( type == GeomType::Edge && GeomDim == GeomType::Face ) {
        // We are dealing with the Edges of Face data
        MeshElementIndex first, last;
        if ( d == 0 && left ) {
            first = MeshElementIndex( GeomType::Edge, 0, 0, 0, 0 );
            last  = MeshElementIndex( GeomType::Edge, 0, lastCell[0], 0, 0 );
        } else if ( d == 0 ) {
            first = MeshElementIndex( GeomType::Edge, 0, 0, lastNode[1], 0 );
            last  = MeshElementIndex( GeomType::Edge, 0, lastCell[0], lastNode[1], 0 );
        } else if ( left ) {
            first = MeshElementIndex( GeomType::Edge, 1, 0, 0, 0 );
            last  = MeshElementIndex( GeomType::Edge, 1, 0, lastCell[1], 0 );
        } else {
            first = MeshElementIndex( GeomType::Edge, 1, lastNode[0], 0, 0 );
            last  = MeshElementIndex( GeomType::Edge, 1, lastNode[0], lastCell[1], 0 );
        }
        return { std::make_pair( first, last ) };
    } else if ( type == GeomType::Edge && GeomDim == GeomType::Cell ) {
        // We are dealing with the Edges of Volume data
        MeshElementIndex first( GeomType::Edge, 0, 0, 0, 0 );
        MeshElementIndex last( GeomType::Edge, 0, lastCell[0], lastCell[1], lastCell[2] );
        last.d_index[d] = lastNode[d];
        if ( left )
            last.d_index[d] = first.d_index[d];
        else
            first.d_index[d] = last.d_index[d];
        ElementBlocks list;
        if ( d == 0 ) {
            first.d_side = 1;
            last.d_side  = 1;
            list.push_back( std::make_pair( first, last ) );
            first.d_side = 2;
            last.d_side  = 2;
            list.push_back( std::make_pair( first, last ) );
        } else if ( d == 1 ) {
            first.d_side = 0;
            last.d_side  = 0;
            list.push_back( std::make_pair( first, last ) );
            first.d_side = 2;
            last.d_side  = 2;
            list.push_back( std::make_pair( first, last ) );
        } else {
            first.d_side = 0;
            last.d_side  = 0;
            list.push_back( std::make_pair( first, last ) );
            first.d_side = 1;
            last.d_side  = 1;
            list.push_back( std::make_pair( first, last ) );
        }
        return list;
    } else if ( type == GeomType::Face && GeomDim == GeomType::Cell ) {
        // We are dealing with the Faces of Volume data
        MeshElementIndex first( GeomType::Face, d, 0, 0, 0 );
        MeshElementIndex last( GeomType::Face, d, lastCell[0], lastCell[1], lastCell[2] );
        last.d_index[d] = lastNode[d];
        if ( left )
            last.d_index[d] = first.d_index[d];
        else
            first.d_index[d] = last.d_index[d];
        return { std::make_pair( first, last ) };
    } else {
        AMP_ERROR( "Unknown type" );
    }
    return {};
}


/****************************************************************
 * Estimate the maximum number of processors                     *
 ****************************************************************/
size_t BoxMesh::maxProcs( std::shared_ptr<const MeshParameters> params )
{
    // Check for valid inputs
    AMP_INSIST( params.get(), "Params must not be null" );
    auto db = params->getDatabase();
    AMP_INSIST( db.get(), "Database must exist" );
    size_t maxProcs = 1;
    if ( db->keyExists( "LoadBalanceMinSize" ) ) {
        auto minSize  = db->getVector<int>( "LoadBalanceMinSize" );
        auto meshSize = estimateLogicalMeshSize( params );
        for ( size_t i = 0; i < meshSize.size(); i++ ) {
            if ( minSize[i] == 0 )
                minSize[i] = 1;
            if ( minSize[i] != -1 )
                maxProcs *= ( meshSize[i] / minSize[i] );
        }
    } else {
        maxProcs = estimateMeshSize( params );
    }
    return maxProcs;
}


/****************************************************************
 * Function to return the element given an ID                    *
 ****************************************************************/
MeshElement BoxMesh::getElement( const MeshElementID &id ) const
{
    // Get the index of the element
    MeshElementIndex index = convert( id );
    // Create the element
    auto tmp = new structuredMeshElement( index, this );
    AMP_DEBUG_ASSERT( tmp->globalID() == id );
    return tmp;
}
MeshElement BoxMesh::getElement( const MeshElementIndex &index ) const
{
    return structuredMeshElement( index, this );
}


/****************************************************************
 * Find the mesh element index from a point                      *
 ****************************************************************/
static inline int to_nearest( double x ) { return static_cast<int>( floor( x + 0.5 ) ); }
BoxMesh::MeshElementIndex BoxMesh::getElementFromLogical( const AMP::Geometry::Point &x0,
                                                          GeomType type ) const
{
    // Correct x for periodic boundaries
    double x[3] = { x0.x(), x0.y(), x0.z() };
    for ( int d = 0; d < static_cast<int>( GeomDim ); d++ ) {
        if ( d_surfaceId[2 * d + 1] == -1 ) {
            while ( x[d] < 0 )
                x[d] += 1.0;
            while ( x[d] >= 1.0 )
                x[d] -= 1.0;
        }
    }
    // Convert x to [0,size]
    x[0] = x[0] * d_globalSize[0];
    x[1] = x[1] * d_globalSize[1];
    x[2] = x[2] * d_globalSize[2];
    // Check if element is outside domain
    for ( int d = 0; d < static_cast<int>( GeomDim ); d++ ) {
        if ( fabs( x[d] ) < 1e-6 )
            x[d] = 0;
        if ( fabs( x[d] - d_globalSize[d] ) < 1e-6 )
            x[d] = d_globalSize[d];
        if ( x[d] < 0 || x[d] > d_globalSize[d] )
            return MeshElementIndex();
    }
    // Compute the index
    MeshElementIndex index;
    if ( type == GeomDim ) {
        index = MeshElementIndex( GeomDim, 0, x[0], x[1], x[2] );
    } else if ( type == GeomType::Vertex ) {
        int i     = to_nearest( x[0] );
        int j     = to_nearest( x[1] );
        int k     = to_nearest( x[2] );
        bool keep = fabs( x[0] - i ) < 1e-6 && fabs( x[1] - j ) < 1e-6 && fabs( x[2] - k ) < 1e-6;
        keep      = keep && i >= 0 && j >= 0 && k >= 0;
        keep      = keep && i <= d_globalSize[0] && j <= d_globalSize[1] && k <= d_globalSize[2];
        if ( keep )
            index = MeshElementIndex( GeomType::Vertex, 0, i, j, k );
    } else if ( type == GeomType::Edge ) {
        AMP_ERROR( "Not finished" );
    } else if ( type == GeomType::Face ) {
        int i      = to_nearest( x[0] );
        int j      = to_nearest( x[1] );
        int k      = to_nearest( x[2] );
        int ijk    = 0;
        double min = fabs( x[0] - i );
        if ( fabs( x[1] - j ) < min && static_cast<int>( GeomDim ) >= 2 ) {
            min = fabs( x[1] - j );
            ijk = 1;
        }
        if ( fabs( x[2] - k ) < min && static_cast<int>( GeomDim ) >= 3 ) {
            min = fabs( x[2] - k );
            ijk = 2;
        }
        if ( min > 1e-6 ) {
            // Point is not on any face
        } else if ( ijk == 0 ) {
            index = MeshElementIndex( GeomType::Face, 0, i, x[1], x[2] );
        } else if ( ijk == 1 ) {
            index = MeshElementIndex( GeomType::Face, 1, x[0], j, x[2] );
        } else if ( ijk == 2 ) {
            index = MeshElementIndex( GeomType::Face, 2, x[0], x[1], k );
        }
    } else if ( type == GeomType::Cell ) {
        AMP_ERROR( "Not finished" );
    } else {
        AMP_ERROR( "Unknown mesh element type" );
    }
    return index;
}
BoxMesh::MeshElementIndex BoxMesh::getElementFromPhysical( const AMP::Geometry::Point &x,
                                                           GeomType type ) const
{
    auto logical = physicalToLogical( x );
    auto index   = getElementFromLogical( logical, type );
    return index;
}


/********************************************************
 * Function to return parents of an element              *
 ********************************************************/
std::vector<MeshElement> BoxMesh::getElementParents( const MeshElement &meshelem,
                                                     const GeomType type ) const
{
    auto id = meshelem.globalID();
    if ( type == id.type() )
        return std::vector<MeshElement>( 1, meshelem );
    AMP_INSIST( id.meshID() == d_meshID, "MeshElement is not from the given mesh" );
    // AMP_INSIST( type >= id.type() && type <= GeomDim,
    //            "Cannot get the parents of the given type for the current element" );
    // Get the element of interest
    const auto *elem = dynamic_cast<const structuredMeshElement *>( meshelem.getRawElement() );
    AMP_ASSERT( elem != nullptr );
    return elem->getParents( type );
}


/****************************************************************
 * Functions to return the number of elements                    *
 ****************************************************************/
size_t BoxMesh::numLocalElements( const GeomType type ) const
{
    if ( type > GeomDim )
        return 0;
    auto box   = getLocalBlock( d_rank );
    auto range = getIteratorRange( box, type, 0 );
    size_t N   = 0;
    for ( const auto &tmp : range )
        N += BoxMesh::MeshElementIndex::numElements( tmp.first, tmp.second );
    return N;
}
size_t BoxMesh::numGlobalElements( const GeomType type ) const
{
    if ( type > GeomDim )
        return 0;
    std::array<int, 6> box = {
        { 0, d_globalSize[0] - 1, 0, d_globalSize[1] - 1, 0, d_globalSize[2] - 1 }
    };
    auto range = getIteratorRange( box, type, 0 );
    size_t N   = 0;
    for ( auto tmp : range )
        N += MeshElementIndex::numElements( tmp.first, tmp.second );
    return N;
}
size_t BoxMesh::numGhostElements( const GeomType type, int gcw ) const
{
    if ( type > GeomDim )
        return 0;
    auto box    = getLocalBlock( d_rank );
    auto range1 = getIteratorRange( box, type, 0 );
    auto range2 = getIteratorRange( box, type, gcw );
    size_t N    = 0;
    for ( size_t i = 0; i < range1.size(); i++ ) {
        size_t N1 = BoxMesh::MeshElementIndex::numElements( range1[i].first, range1[i].second );
        size_t N2 = BoxMesh::MeshElementIndex::numElements( range2[i].first, range2[i].second );
        N += N2 - N1;
    }
    return N;
}


/****************************************************************
 * Function to get an iterator                                   *
 ****************************************************************/
BoxMesh::ElementBlocks
BoxMesh::getIteratorRange( std::array<int, 6> range, const GeomType type, const int gcw ) const
{
    if ( type > GeomDim )
        return {};
    // Get the range of cells we care about
    bool isPeriodic[3] = { d_surfaceId[1] == -1, d_surfaceId[3] == -1, d_surfaceId[5] == -1 };
    if ( std::find( d_surfaceId.begin(), d_surfaceId.end(), -2 ) != d_surfaceId.end() )
        AMP_WARN_ONCE( "boundary ids == -2 are not yet supported, mesh may be incomplete" );
    if ( gcw != 0 ) {
        for ( int d = 0; d < static_cast<int>( GeomDim ); d++ ) {
            range[2 * d + 0] -= gcw;
            range[2 * d + 1] += gcw;
            if ( !isPeriodic[d] ) {
                range[2 * d + 0] = std::max( range[2 * d + 0], 0 );
                range[2 * d + 1] = std::min( range[2 * d + 1], d_globalSize[d] - 1 );
            }
        }
    }
    // Get the element blocks we want to process
    ElementBlocks blocks;
    if ( type == GeomDim ) {
        blocks.emplace_back( MeshElementIndex( type, 0, range[0], range[2], range[4] ),
                             MeshElementIndex( type, 0, range[1], range[3], range[5] ) );
    } else if ( type == GeomType::Vertex ) {
        for ( int d = 0; d < static_cast<int>( GeomDim ); d++ ) {
            if ( gcw != 0 )
                range[2 * d + 1]++;
            else if ( !isPeriodic[d] && range[2 * d + 1] == d_globalSize[d] - 1 )
                range[2 * d + 1]++;
        }
        blocks.emplace_back( MeshElementIndex( type, 0, range[0], range[2], range[4] ),
                             MeshElementIndex( type, 0, range[1], range[3], range[5] ) );
    } else if ( type == GeomType::Edge && GeomDim == GeomType::Face ) {
        auto range1 = range;
        auto range2 = range;
        if ( gcw != 0 || ( gcw == 0 && !isPeriodic[0] && range[1] == d_globalSize[0] - 1 ) )
            range2[1]++;
        if ( gcw != 0 || ( gcw == 0 && !isPeriodic[1] && range[3] == d_globalSize[1] - 1 ) )
            range1[3]++;
        blocks.emplace_back( MeshElementIndex( type, 0, range1[0], range1[2], range1[4] ),
                             MeshElementIndex( type, 0, range1[1], range1[3], range1[5] ) );
        blocks.emplace_back( MeshElementIndex( type, 1, range2[0], range2[2], range2[4] ),
                             MeshElementIndex( type, 1, range2[1], range2[3], range2[5] ) );
    } else if ( type == GeomType::Edge && GeomDim == GeomType::Cell ) {
        auto range1 = range;
        auto range2 = range;
        auto range3 = range;
        if ( gcw != 0 || ( gcw == 0 && !isPeriodic[0] && range[1] == d_globalSize[0] - 1 ) ) {
            range2[1]++;
            range3[1]++;
        }
        if ( gcw != 0 || ( gcw == 0 && !isPeriodic[1] && range[3] == d_globalSize[1] - 1 ) ) {
            range1[3]++;
            range3[3]++;
        }
        if ( gcw != 0 || ( gcw == 0 && !isPeriodic[2] && range[5] == d_globalSize[2] - 1 ) ) {
            range1[5]++;
            range2[5]++;
        }
        blocks.emplace_back( MeshElementIndex( type, 0, range1[0], range1[2], range1[4] ),
                             MeshElementIndex( type, 0, range1[1], range1[3], range1[5] ) );
        blocks.emplace_back( MeshElementIndex( type, 1, range2[0], range2[2], range2[4] ),
                             MeshElementIndex( type, 1, range2[1], range2[3], range2[5] ) );
        blocks.emplace_back( MeshElementIndex( type, 2, range3[0], range3[2], range3[4] ),
                             MeshElementIndex( type, 2, range3[1], range3[3], range3[5] ) );
    } else if ( type == GeomType::Face && GeomDim == GeomType::Cell ) {
        auto range1 = range;
        auto range2 = range;
        auto range3 = range;
        if ( gcw != 0 || ( gcw == 0 && !isPeriodic[0] && range[1] == d_globalSize[0] - 1 ) )
            range1[1]++;
        if ( gcw != 0 || ( gcw == 0 && !isPeriodic[1] && range[3] == d_globalSize[1] - 1 ) )
            range2[3]++;
        if ( gcw != 0 || ( gcw == 0 && !isPeriodic[2] && range[5] == d_globalSize[2] - 1 ) )
            range3[5]++;
        blocks.emplace_back( MeshElementIndex( type, 0, range1[0], range1[2], range1[4] ),
                             MeshElementIndex( type, 0, range1[1], range1[3], range1[5] ) );
        blocks.emplace_back( MeshElementIndex( type, 1, range2[0], range2[2], range2[4] ),
                             MeshElementIndex( type, 1, range2[1], range2[3], range2[5] ) );
        blocks.emplace_back( MeshElementIndex( type, 2, range3[0], range3[2], range3[4] ),
                             MeshElementIndex( type, 2, range3[1], range3[3], range3[5] ) );
    } else {
        AMP_ERROR( "Unknown case" );
    }
    // Check that each block does not have duplicate elements
    for ( auto &block : blocks ) {
        for ( int d = 0; d < static_cast<int>( GeomDim ); d++ ) {
            if ( isPeriodic[d] ) {
                auto &first = block.first;
                auto &last  = block.second;
                if ( first.index( d ) + d_globalSize[d] <= last.index( d ) ) {
                    first.index( d ) = 0;
                    last.index( d )  = d_globalSize[d] - 1;
                }
            }
        }
    }
    return blocks;
}
BoxMesh::ElementBlocks BoxMesh::intersect( const ElementBlocks &set1,
                                           const ElementBlocks &set2 ) const
{
    ElementBlocks set;
    for ( auto v1 : set1 ) {
        for ( auto v2 : set2 ) {
            if ( v1.first.type() != v2.first.type() || v1.first.side() != v2.first.side() )
                continue;
            // Perform the intersection
            auto v              = v1;
            v.first.index( 0 )  = std::max( v1.first.index( 0 ), v2.first.index( 0 ) );
            v.first.index( 1 )  = std::max( v1.first.index( 1 ), v2.first.index( 1 ) );
            v.first.index( 2 )  = std::max( v1.first.index( 2 ), v2.first.index( 2 ) );
            v.second.index( 0 ) = std::min( v1.second.index( 0 ), v2.second.index( 0 ) );
            v.second.index( 1 ) = std::min( v1.second.index( 1 ), v2.second.index( 1 ) );
            v.second.index( 2 ) = std::min( v1.second.index( 2 ), v2.second.index( 2 ) );
            if ( MeshElementIndex::numElements( v.first, v.second ) > 0 )
                set.push_back( v );
        }
    }
    return set;
}
MeshIterator BoxMesh::createIterator( const ElementBlocks &list ) const
{
    if ( list.empty() ) {
        return {};
    } else if ( list.size() == 1 ) {
        return structuredMeshIterator( list[0].first, list[0].second, this, 0 );
    } else {
        std::vector<MeshIterator> iterator_list;
        iterator_list.reserve( list.size() );
        for ( const auto &item : list ) {
            if ( MeshElementIndex::numElements( item.first, item.second ) ) {
                structuredMeshIterator it( item.first, item.second, this, 0 );
                iterator_list.push_back( it );
            }
        }
        return MultiIterator( iterator_list, 0 );
    }
}
MeshIterator BoxMesh::getIterator( const GeomType type, const int gcw ) const
{
    if ( type > GeomDim )
        return {};
    auto box   = getLocalBlock( d_rank );
    auto range = getIteratorRange( box, type, gcw );
    return createIterator( range );
}


/****************************************************************
 * Function to get an iterator over the surface                  *
 ****************************************************************/
MeshIterator BoxMesh::getSurfaceIterator( const GeomType type, const int gcw ) const
{
    if ( type > GeomDim || gcw < 0 )
        return {};
    int type2 = static_cast<int>( type );
    // Build and cache the surface data
    if ( (int) d_surface[type2].size() < gcw + 1 )
        d_surface[type2].resize( gcw + 1 );
    if ( !d_surface[type2][gcw] ) {
        d_surface[type2][gcw] = std::make_shared<std::vector<MeshElementIndex>>();
        for ( auto &elem : getIterator( type, gcw ) ) {
            auto elem2 = dynamic_cast<structuredMeshElement *>( elem.getRawElement() );
            if ( elem2->isOnSurface() )
                d_surface[type2][gcw]->push_back( elem2->getIndex() );
        }
    }
    // Create the iterator
    return structuredMeshIterator( d_surface[type2][gcw], this, 0 );
}


/****************************************************************
 * Functions to get the boundaries                               *
 ****************************************************************/
std::vector<int> BoxMesh::getBoundaryIDs() const
{
    std::set<int> ids;
    for ( int i = 0; i < 2 * static_cast<int>( GeomDim ); i++ ) {
        if ( d_surfaceId[i] >= 0 )
            ids.insert( d_surfaceId[i] );
    }
    return std::vector<int>( ids.begin(), ids.end() );
}
MeshIterator
BoxMesh::getBoundaryIDIterator( const GeomType type, const int id, const int gcw ) const
{
    if ( type > GeomDim || id < 0 || gcw < 0 )
        return {};
    int type2 = static_cast<int>( type );
    // Get the boundary id index
    auto ids  = getBoundaryIDs();
    int index = -1;
    for ( size_t i = 0; i < ids.size(); i++ ) {
        if ( id == ids[i] )
            index = i;
    }
    if ( index < 0 )
        return {};
    AMP_ASSERT( index < 6 );
    // Build and cache the surface data
    if ( (int) d_boundary[type2][index].size() < gcw + 1 )
        d_boundary[type2][index].resize( gcw + 1 );
    if ( !d_boundary[type2][index][gcw] ) {
        d_boundary[type2][index][gcw] = std::make_shared<std::vector<MeshElementIndex>>();
        auto &boundary                = *d_boundary[type2][index][gcw];
        for ( auto &elem : getIterator( type, gcw ) ) {
            auto elem2 = dynamic_cast<structuredMeshElement *>( elem.getRawElement() );
            if ( elem2->isOnBoundary( id ) )
                boundary.push_back( elem2->getIndex() );
        }
    }
    // Create the iterator
    if ( !d_boundary[type2][index][gcw] )
        return {};
    if ( d_boundary[type2][index][gcw]->empty() )
        return {};
    if ( type == GeomType::Edge )
        structuredMeshIterator( d_boundary[type2][index][gcw], this, 0 );
    return structuredMeshIterator( d_boundary[type2][index][gcw], this, 0 );
}
std::vector<int> BoxMesh::getBlockIDs() const { return { d_blockID }; }
MeshIterator BoxMesh::getBlockIDIterator( const GeomType type, const int id, const int gcw ) const
{
    if ( id == d_blockID )
        return getIterator( type, gcw );
    return {};
}
bool BoxMesh::onSide( const MeshElementIndex &index, int d, int s ) const
{
    bool test = false;
    if ( index.type() == GeomDim ) {
        // Volume type
        test = ( s == 0 && index.index( d ) == 0 ) ||
               ( s == 1 && index.index( d ) == d_globalSize[d] - 1 );
    } else if ( index.type() == GeomType::Vertex ) {
        // Vertex
        test = ( s == 0 && index.index( d ) == 0 ) ||
               ( s == 1 && index.index( d ) == d_globalSize[d] );
    } else if ( index.type() == GeomType::Edge ) {
        if ( PhysicalDim == 2 ) {
            // Edge in 2D space
            if ( d == 0 ) {
                test |= ( index.side() == d && s == 0 && index.index( 1 ) == 0 );
                test |= ( index.side() == d && s == 1 && index.index( 1 ) == d_globalSize[d] );
            } else {
                test |= ( index.side() == d && s == 0 && index.index( 0 ) == 0 );
                test |= ( index.side() == d && s == 1 && index.index( 0 ) == d_globalSize[d] );
            }
        } else {
            // Edge in 3D space
            test |= ( index.side() != d && s == 0 && index.index( d ) == 0 );
            test |= ( index.side() != d && s == 1 && index.index( d ) == d_globalSize[d] );
        }
    } else if ( index.type() == GeomType::Face ) {
        // Face (assumed 3D space)
        test |= ( index.side() == d && s == 0 && index.side() == d && index.index( d ) == 0 );
        test |= ( index.side() == d && s == 1 && index.side() == d &&
                  index.index( d ) == d_globalSize[d] );
    }
    return test;
}
bool BoxMesh::isOnSurface( const MeshElementIndex &index ) const
{
    bool test = false;
    for ( int i = 0; i < 6; i++ ) {
        if ( d_surfaceId[i] >= 0 ) {
            int d = i / 2;
            int s = i % 2;
            test  = test | onSide( index, d, s );
        }
    }
    return test;
}
bool BoxMesh::isOnBoundary( const MeshElementIndex &index, int id ) const
{
    bool test = false;
    for ( int i = 0; i < 6; i++ ) {
        if ( d_surfaceId[i] == id ) {
            int d = i / 2;
            int s = i % 2;
            test  = test | onSide( index, d, s );
        }
    }
    return test;
}


/****************************************************************
 * Create an ArrayVector over the mesh                           *
 ****************************************************************/
std::shared_ptr<AMP::LinearAlgebra::Vector> BoxMesh::createVector( const std::string &name,
                                                                   int gcw )
{
    AMP_ASSERT( getComm().getSize() == 1 );
    auto mesh = shared_from_this();
    auto size = getLocalBox().size();
    auto type = GeomDim;
    auto var  = std::make_shared<AMP::LinearAlgebra::Variable>( name );
    auto DOFs = AMP::Discretization::simpleDOFManager::create( mesh, type, gcw, 1, true );
    auto ops  = std::make_shared<AMP::LinearAlgebra::VectorOperationsDefault<double>>();
    auto data = AMP::LinearAlgebra::ArrayVectorData<double>::create( size );
    auto vec  = std::make_shared<AMP::LinearAlgebra::Vector>( data, ops, var, DOFs );
    AMP_ASSERT( vec->getLocalSize() == getIterator( type ).size() );
    return vec;
}


/****************************************************************
 * Check if two meshes are equal                                 *
 ****************************************************************/
bool BoxMesh::operator==( const Mesh &rhs ) const
{
    // Check if we are comparing to *this
    if ( &rhs == this )
        return true;
    // Check if we can cast to a BoxMesh
    auto mesh = dynamic_cast<const BoxMesh *>( &rhs );
    if ( !mesh )
        return false;
    // Perform basic comparison
    if ( d_numBlocks != mesh->d_numBlocks || d_indexSize != mesh->d_indexSize ||
         d_localIndex != mesh->d_localIndex )
        return false;
    if ( d_surfaceId != mesh->d_surfaceId )
        return false;
    if ( d_geometry != mesh->d_geometry ) {
        if ( !d_geometry || !mesh->d_geometry )
            return false;
        if ( *d_geometry != *mesh->d_geometry )
            return false;
    }
    for ( int d = 0; d < 3; d++ ) {
        if ( d_startIndex[d] != mesh->d_startIndex[d] || d_endIndex[d] != mesh->d_endIndex[d] )
            return false;
    }
    return true;
}


/****************************************************************
 * Print the index                                               *
 ****************************************************************/
std::ostream &operator<<( std::ostream &out, const BoxMesh::MeshElementIndex &x )
{
    out << x.print();
    return out;
}
std::string BoxMesh::MeshElementIndex::print() const
{
    const char *type[] = { "Vertex", "Edge", "Face", "Volume" };
    char tmp[128];
    snprintf(
        tmp, 128, "(%i,%i,%i,%s,%u)", d_index[0], d_index[1], d_index[2], type[d_type], d_side );
    return tmp;
}
std::string BoxMesh::Box::print() const
{
    char tmp[128];
    snprintf( tmp,
              128,
              "(%i,%i,%i)-(%i,%i,%i)",
              first[0],
              first[1],
              first[2],
              last[0],
              last[1],
              last[2] );
    return tmp;
}


} // namespace AMP::Mesh


/********************************************************
 * Explicit instantiations                               *
 ********************************************************/
#include "AMP/utils/Utilities.hpp"
AMP_INSTANTIATE_SORT( AMP::Mesh::BoxMesh::MeshElementIndex );
template void
AMP::Utilities::quicksort<int, AMP::Mesh::MeshElementID>( std::vector<int> &,
                                                          std::vector<AMP::Mesh::MeshElementID> & );
template void AMP::Utilities::quicksort<AMP::Mesh::MeshElementID, unsigned long>(
    std::vector<AMP::Mesh::MeshElementID> &, std::vector<unsigned long> & );
