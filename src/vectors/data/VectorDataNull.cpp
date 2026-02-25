#include "AMP/vectors/data/VectorDataNull.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/typeid.h"


namespace AMP::LinearAlgebra {


std::shared_ptr<VectorData> VectorDataNull::cloneData( const std::string & ) const
{
    return std::make_shared<VectorDataNull>( d_type );
}
const AMP_MPI &VectorDataNull::getComm() const
{
    static AMP_MPI comm( AMP_COMM_SELF );
    return comm;
}
void VectorDataNull::getValuesByLocalID( size_t N, const size_t *, void *, const typeID & ) const
{
    AMP_INSIST( N == 0, "Cannot get values in NullVectorData" );
}
void VectorDataNull::setValuesByLocalID( size_t N, const size_t *, const void *, const typeID & )
{
    AMP_INSIST( N == 0, "Cannot set values in NullVectorData" );
}
void VectorDataNull::addValuesByLocalID( size_t N, const size_t *, const void *, const typeID & )
{
    AMP_INSIST( N == 0, "Cannot add values in NullVectorData" );
}
void VectorDataNull::setGhostValuesByGlobalID( size_t N,
                                               const size_t *,
                                               const void *,
                                               const typeID & )
{
    AMP_INSIST( N == 0, "Cannot set values in NullVectorData" );
}
void VectorDataNull::addGhostValuesByGlobalID( size_t N,
                                               const size_t *,
                                               const void *,
                                               const typeID & )
{
    AMP_INSIST( N == 0, "Cannot add values in NullVectorData" );
}
void VectorDataNull::getGhostValuesByGlobalID( size_t N,
                                               const size_t *,
                                               void *,
                                               const typeID & ) const
{
    AMP_INSIST( N == 0, "Cannot get values in NullVectorData" );
}
void VectorDataNull::getGhostAddValuesByGlobalID( size_t N,
                                                  const size_t *,
                                                  void *,
                                                  const typeID & ) const
{
    AMP_INSIST( N == 0, "Cannot get values in NullVectorData" );
}
size_t VectorDataNull::getAllGhostValues( void *, const typeID & ) const { return 0; }


} // namespace AMP::LinearAlgebra
