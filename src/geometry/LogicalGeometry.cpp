#include "AMP/geometry/LogicalGeometry.h"
#include "AMP/IO/HDF.h"
#include "AMP/utils/UtilityMacros.h"

#include <vector>


namespace AMP::Geometry {


/********************************************************
 * Default constructor                                   *
 ********************************************************/
static inline std::array<int, 6> checkIDs( int logical, std::array<int, 6> ids )
{
    AMP_ASSERT( logical <= 3 );
    for ( int d = logical; d < 3; d++ ) {
        ids[2 * d + 0] = -3;
        ids[2 * d + 1] = -3;
    }
    for ( int d = 0; d < logical; d++ ) {
        if ( ids[2 * d + 0] == -1 || ids[2 * d + 1] == -1 )
            AMP_ASSERT( ids[2 * d + 0] == -1 && ids[2 * d + 1] == -1 );
        AMP_ASSERT( ids[2 * d + 0] > -3 );
        AMP_ASSERT( ids[2 * d + 1] > -3 );
    }
    return ids;
}
LogicalGeometry::LogicalGeometry( int physical, int logical, std::array<int, 6> ids )
    : Geometry( physical ), d_logicalDim( logical ), d_ids( checkIDs( logical, ids ) )
{
}


/********************************************************
 * Get geometric type                                    *
 ********************************************************/
AMP::Mesh::GeomType LogicalGeometry::getGeomType() const
{
    return static_cast<AMP::Mesh::GeomType>( d_logicalDim );
}


/********************************************************
 * Get periodic boundaries                               *
 ********************************************************/
std::array<bool, 3> LogicalGeometry::getPeriodicDim() const
{
    std::array<bool, 3> periodic = { false, false, false };
    for ( int d = 0; d < d_logicalDim; d++ ) {
        if ( d_ids[2 * d + 0] == -1 && d_ids[2 * d + 1] == -1 )
            periodic[d] = true;
    }
    return periodic;
}


/********************************************************
 * Write/Read restart data                               *
 ********************************************************/
void LogicalGeometry::writeRestart( int64_t fid ) const
{
    Geometry::writeRestart( fid );
    IO::writeHDF5( fid, "logicalDim", d_logicalDim );
    IO::writeHDF5( fid, "ids", d_ids );
}
template<class TYPE>
static inline TYPE read( int64_t fid, const std::string &name )
{
    TYPE x;
    IO::readHDF5( fid, name, x );
    return x;
}
LogicalGeometry::LogicalGeometry( int64_t fid )
    : Geometry( fid ),
      d_logicalDim( read<int>( fid, "logicalDim" ) ),
      d_ids( read<std::array<int, 6>>( fid, "ids" ) )
{
}


} // namespace AMP::Geometry
