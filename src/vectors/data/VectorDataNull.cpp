#include "AMP/vectors/data/VectorDataNull.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/typeid.h"


namespace AMP::LinearAlgebra {


template<typename TYPE>
const AMP_MPI &VectorDataNull<TYPE>::getComm() const
{
    static AMP_MPI comm( AMP_COMM_SELF );
    return comm;
}
template<typename TYPE>
void VectorDataNull<TYPE>::getValuesByLocalID( size_t N,
                                               const size_t *,
                                               void *,
                                               const typeID & ) const
{
    AMP_INSIST( N == 0, "Cannot get values in NullVectorData" );
}
template<typename TYPE>
void VectorDataNull<TYPE>::setValuesByLocalID( size_t N,
                                               const size_t *,
                                               const void *,
                                               const typeID & )
{
    AMP_INSIST( N == 0, "Cannot set values in NullVectorData" );
}
template<typename TYPE>
void VectorDataNull<TYPE>::addValuesByLocalID( size_t N,
                                               const size_t *,
                                               const void *,
                                               const typeID & )
{
    AMP_INSIST( N == 0, "Cannot add values in NullVectorData" );
}
template<typename TYPE>
void VectorDataNull<TYPE>::setGhostValuesByGlobalID( size_t N,
                                                     const size_t *,
                                                     const void *,
                                                     const typeID & )
{
    AMP_INSIST( N == 0, "Cannot set values in NullVectorData" );
}
template<typename TYPE>
void VectorDataNull<TYPE>::addGhostValuesByGlobalID( size_t N,
                                                     const size_t *,
                                                     const void *,
                                                     const typeID & )
{
    AMP_INSIST( N == 0, "Cannot add values in NullVectorData" );
}
template<typename TYPE>
void VectorDataNull<TYPE>::getGhostValuesByGlobalID( size_t N,
                                                     const size_t *,
                                                     void *,
                                                     const typeID & ) const
{
    AMP_INSIST( N == 0, "Cannot get values in NullVectorData" );
}
template<typename TYPE>
void VectorDataNull<TYPE>::getGhostAddValuesByGlobalID( size_t N,
                                                        const size_t *,
                                                        void *,
                                                        const typeID & ) const
{
    AMP_INSIST( N == 0, "Cannot get values in NullVectorData" );
}
template<typename TYPE>
size_t VectorDataNull<TYPE>::getAllGhostValues( void *, const typeID & ) const
{
    return 0;
}


/****************************************************************
 * Explicit instantiations                                       *
 ****************************************************************/
template class VectorDataNull<int>;
template class VectorDataNull<float>;
template class VectorDataNull<double>;


} // namespace AMP::LinearAlgebra
