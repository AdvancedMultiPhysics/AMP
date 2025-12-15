#ifndef included_AMP_BoxMeshInline
#define included_AMP_BoxMeshInline

#include "AMP/mesh/structured/BoxMesh.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>


namespace AMP::Mesh {


/****************************************************************
 * Box                                                           *
 ****************************************************************/
constexpr BoxMesh::Box::Box() : first{ 0, 0, 0 }, last{ 0, 0, 0 } {}
constexpr BoxMesh::Box::Box( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast )
    : first{ ifirst, jfirst, kfirst }, last{ ilast, jlast, klast }
{
}
constexpr ArraySize BoxMesh::Box::size() const
{
    return { (size_t) last[0] - first[0] + 1,
             (size_t) last[1] - first[1] + 1,
             (size_t) last[2] - first[2] + 1 };
}


/****************************************************************
 * MeshElementIndex                                              *
 ****************************************************************/
constexpr BoxMesh::MeshElementIndex::MeshElementIndex(
    GeomType type_in, uint8_t side_in, int x, int y, int z )
    : d_type( static_cast<uint8_t>( type_in ) ), d_side( side_in ), d_index{ x, y, z }
{
}
constexpr void BoxMesh::MeshElementIndex::reset()
{
    d_type     = 0;
    d_side     = 255;
    d_index[0] = 0;
    d_index[1] = 0;
    d_index[2] = 0;
}
constexpr void
BoxMesh::MeshElementIndex::reset( GeomType type_in, uint8_t side_in, int x, int y, int z )
{
    d_type     = static_cast<uint8_t>( type_in );
    d_side     = side_in;
    d_index[0] = x;
    d_index[1] = y;
    d_index[2] = z;
}
constexpr bool BoxMesh::MeshElementIndex::operator==( const MeshElementIndex &rhs ) const
{
    return d_type == rhs.d_type && d_side == rhs.d_side && d_index[0] == rhs.d_index[0] &&
           d_index[1] == rhs.d_index[1] && d_index[2] == rhs.d_index[2];
}
constexpr bool BoxMesh::MeshElementIndex::operator!=( const MeshElementIndex &rhs ) const
{
    return d_type != rhs.d_type || d_side != rhs.d_side || d_index[0] != rhs.d_index[0] ||
           d_index[1] != rhs.d_index[1] || d_index[2] != rhs.d_index[2];
}
constexpr bool BoxMesh::MeshElementIndex::operator>( const MeshElementIndex &rhs ) const
{
    if ( d_type < rhs.d_type ) {
        return false;
    } else if ( d_type > rhs.d_type ) {
        return true;
    }
    if ( d_side < rhs.d_side ) {
        return false;
    } else if ( d_side > rhs.d_side ) {
        return true;
    }
    for ( int i = 2; i >= 0; i-- ) {
        if ( d_index[i] < rhs.d_index[i] ) {
            return false;
        } else if ( d_index[i] > rhs.d_index[i] ) {
            return true;
        }
    }
    return false;
}
constexpr bool BoxMesh::MeshElementIndex::operator>=( const MeshElementIndex &rhs ) const
{
    return this->operator>( rhs ) || this->operator==( rhs );
}
constexpr bool BoxMesh::MeshElementIndex::operator<( const MeshElementIndex &rhs ) const
{
    return !this->operator>( rhs ) && !this->operator==( rhs );
}
constexpr bool BoxMesh::MeshElementIndex::operator<=( const MeshElementIndex &rhs ) const
{
    return !this->operator>( rhs );
}
constexpr size_t BoxMesh::MeshElementIndex::numElements( const MeshElementIndex &first,
                                                         const MeshElementIndex &last )
{
    if ( last.d_index[0] < first.d_index[0] || last.d_index[1] < first.d_index[1] ||
         last.d_index[2] < first.d_index[2] )
        return 0;
    return ( last.d_index[0] - first.d_index[0] + 1 ) * ( last.d_index[1] - first.d_index[1] + 1 ) *
           ( last.d_index[2] - first.d_index[2] + 1 );
}


/****************************************************************
 * getGlobalBox, getLocalBox                                     *
 ****************************************************************/
inline std::vector<bool> BoxMesh::periodic() const
{
    std::vector<bool> periodic = { d_surfaceId[1] == -1,
                                   d_surfaceId[3] == -1,
                                   d_surfaceId[5] == -1 };
    periodic.resize( static_cast<int>( GeomDim ) );
    return periodic;
}
inline ArraySize BoxMesh::size() const
{
    return ArraySize( { d_globalSize[0], d_globalSize[1], d_globalSize[2] },
                      static_cast<int>( GeomDim ) );
}
inline ArraySize BoxMesh::localSize() const
{
    auto local  = getLocalBlock( d_comm.getRank() );
    size_t N[3] = { (size_t) local[1] - local[0] + 1,
                    (size_t) local[3] - local[2] + 1,
                    (size_t) local[5] - local[4] + 1 };
    return ArraySize( static_cast<int>( GeomDim ), N );
}
inline std::vector<size_t> BoxMesh::numBlocks() const
{
    std::vector<size_t> size( static_cast<int>( GeomDim ) );
    for ( int d = 0; d < static_cast<int>( GeomDim ); d++ )
        size[d] = d_numBlocks[d];
    return size;
}
inline BoxMesh::Box BoxMesh::getGlobalBox( int gcw ) const
{
    Box box;
    bool isPeriodic[3] = { d_surfaceId[1] == -1, d_surfaceId[3] == -1, d_surfaceId[5] == -1 };
    for ( int d = 0; d < static_cast<int>( GeomDim ); d++ ) {
        box.first[d] = -gcw;
        box.last[d]  = d_globalSize[d] + gcw - 1;
        if ( !isPeriodic[d] ) {
            box.first[d] = std::max( box.first[d], 0 );
            box.last[d]  = std::min( box.last[d], d_globalSize[d] - 1 );
        }
    }
    return box;
}

inline BoxMesh::Box BoxMesh::getLocalBox( int gcw ) const
{
    auto range = getLocalBlock( d_rank );
    Box box;
    bool isPeriodic[3] = { d_surfaceId[1] == -1, d_surfaceId[3] == -1, d_surfaceId[5] == -1 };
    for ( int d = 0; d < static_cast<int>( GeomDim ); d++ ) {
        box.first[d] = range[2 * d + 0] - gcw;
        box.last[d]  = range[2 * d + 1] + gcw;
        if ( !isPeriodic[d] ) {
            box.first[d] = std::max( box.first[d], 0 );
            box.last[d]  = std::min( box.last[d], d_globalSize[d] - 1 );
        }
    }
    return box;
}


/****************************************************************
 * Helper function to return the indices of the local block      *
 * owned by the given processor                                  *
 ****************************************************************/
inline std::array<int, 6> BoxMesh::getLocalBlock( int rank ) const
{
    const int p[3] = { rank % d_numBlocks[0],
                       rank / d_numBlocks[0] % d_numBlocks[1],
                       rank / ( d_numBlocks[0] * d_numBlocks[1] ) };
    AMP_ASSERT( p[2] < d_numBlocks[2] );
    std::array<int, 6> range;
    range.fill( 0 );
    for ( int d = 0; d < static_cast<int>( GeomDim ); d++ ) {
        range[2 * d + 0] = d_startIndex[d][p[d]];
        range[2 * d + 1] = std::min( d_endIndex[d][p[d]], d_globalSize[d] ) - 1;
    }
    return range;
}


/****************************************************************
 * Convert between the different id types                        *
 ****************************************************************/
inline int BoxMesh::getRank( const MeshElementIndex &index ) const
{
    int i        = index.index( 0 );
    int j        = index.index( 1 );
    int k        = index.index( 2 );
    bool isLocal = i >= d_localIndex[0] && i < d_localIndex[1] && j >= d_localIndex[2] &&
                   j < d_localIndex[3] && k >= d_localIndex[4] && k < d_localIndex[5];
    if ( isLocal ) {
        return d_rank;
    } else {
        int px = std::min<int>( Utilities::findfirst( d_endIndex[0], i + 1 ), d_numBlocks[0] - 1 );
        int py = std::min<int>( Utilities::findfirst( d_endIndex[1], j + 1 ), d_numBlocks[1] - 1 );
        int pz = std::min<int>( Utilities::findfirst( d_endIndex[2], k + 1 ), d_numBlocks[2] - 1 );
        return px + d_numBlocks[0] * ( py + d_numBlocks[1] * pz );
    }
}
inline MeshElementID BoxMesh::convert( const BoxMesh::MeshElementIndex &index ) const
{
    int i        = index.index( 0 );
    int j        = index.index( 1 );
    int k        = index.index( 2 );
    bool isLocal = i >= d_localIndex[0] && i < d_localIndex[1] && j >= d_localIndex[2] &&
                   j < d_localIndex[3] && k >= d_localIndex[4] && k < d_localIndex[5];
    int rank = -1, i0, j0, k0;
    std::array<int, 3> size;
    if ( isLocal ) {
        rank = d_rank;
        i0   = d_localIndex[0];
        j0   = d_localIndex[2];
        k0   = d_localIndex[4];
        size = d_indexSize;
    } else {
        int px  = std::min<int>( Utilities::findfirst( d_endIndex[0], i + 1 ), d_numBlocks[0] - 1 );
        int py  = std::min<int>( Utilities::findfirst( d_endIndex[1], j + 1 ), d_numBlocks[1] - 1 );
        int pz  = std::min<int>( Utilities::findfirst( d_endIndex[2], k + 1 ), d_numBlocks[2] - 1 );
        rank    = px + d_numBlocks[0] * ( py + d_numBlocks[1] * pz );
        i0      = d_startIndex[0][px];
        j0      = d_startIndex[1][py];
        k0      = d_startIndex[2][pz];
        size[0] = std::min( d_endIndex[0][px], d_globalSize[0] ) - i0 + 2;
        size[1] = std::min( d_endIndex[1][py], d_globalSize[1] ) - j0 + 2;
        size[2] = std::min( d_endIndex[2][pz], d_globalSize[2] ) - k0 + 2;
    }
    i -= i0;
    j -= j0;
    k -= k0;
    unsigned int local_id = i + size[0] * ( j + size[1] * ( k + size[2] * index.side() ) );
    return MeshElementID( isLocal, (GeomType) index.type(), local_id, rank, d_meshID );
}
inline BoxMesh::MeshElementIndex BoxMesh::convert( const MeshElementID &id ) const
{
    int rank = id.owner_rank();
    int size[3], index0[3];
    if ( rank == d_rank ) {
        size[0]   = d_indexSize[0];
        size[1]   = d_indexSize[1];
        size[2]   = d_indexSize[2];
        index0[0] = d_localIndex[0];
        index0[1] = d_localIndex[2];
        index0[2] = d_localIndex[4];
    } else {
        auto block = getLocalBlock( rank );
        size[0]    = block[1] - block[0] + 3;
        size[1]    = block[3] - block[2] + 3;
        size[2]    = block[5] - block[4] + 3;
        index0[0]  = block[0];
        index0[1]  = block[2];
        index0[2]  = block[4];
    }
    uint32_t ijk = id.local_id();
    int i        = ijk % size[0];
    ijk /= size[0];
    int j = ijk % size[1];
    ijk /= size[1];
    int k = ijk % size[2];
    ijk /= size[2];
    int side = ijk;
    i += index0[0];
    j += index0[1];
    k += index0[2];
    return MeshElementIndex( id.type(), side, i, j, k );
}


/****************************************************************
 * Fix the element id for any periodic boundaries                *
 ****************************************************************/
inline void BoxMesh::fixPeriodic( MeshElementIndex &index ) const
{
    for ( int d = 0; d < PhysicalDim; d++ ) {
        if ( d_surfaceId[2 * d] == -1 ) {
            // Periodic boundary
            if ( index[d] < 0 )
                index[d] += d_globalSize[d];
            else if ( index[d] >= d_globalSize[d] )
                index[d] -= d_globalSize[d];
        } else if ( d_surfaceId[2 * d] == -2 || d_surfaceId[2 * d + 1] == -2 ) {
            // Mapped boundary
            if ( index[d] >= 0 || index[d] < d_globalSize[d] )
                continue;
            auto &[x, y] = d_surfaceMaps[d];
            auto i       = std::min( AMP::Utilities::findfirst( x, index ), x.size() - 1 );
            AMP_DEBUG_ASSERT( x[i] == index );
            index = y[i];
        }
    }
}


} // namespace AMP::Mesh

#endif
