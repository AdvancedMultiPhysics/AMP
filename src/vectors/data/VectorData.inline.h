#ifndef included_AMP_VectorData_inline
#define included_AMP_VectorData_inline

#include "AMP/utils/typeid.h"
#include "AMP/vectors/data/VectorDataIterator.h"

#include <algorithm>


namespace AMP::LinearAlgebra {


/****************************************************************
 * Get the size of the vector                                    *
 ****************************************************************/
inline std::shared_ptr<CommunicationList> VectorData::getCommunicationList() const
{
    return d_CommList;
}
inline void VectorData::aliasGhostBuffer( std::shared_ptr<VectorData> in )
{
    d_Ghosts = in->d_Ghosts;
}
inline bool VectorData::containsGlobalElement( size_t i )
{
    if ( ( i >= d_CommList->getStartGID() ) &&
         ( i < d_CommList->getStartGID() + d_CommList->numLocalRows() ) )
        return true;
    return std::find( d_CommList->getGhostIDList().begin(),
                      d_CommList->getGhostIDList().end(),
                      i ) != d_CommList->getGhostIDList().end();
}


/****************************************************************
 * Get the type of data                                          *
 ****************************************************************/
template<typename TYPE>
bool VectorData::isType() const
{
    bool test           = true;
    constexpr auto type = getTypeID<TYPE>();
    for ( size_t i = 0; i < numberOfDataBlocks(); i++ )
        test = test && isType( type, i );
    return test;
}
template<typename TYPE>
bool VectorData::isBlockType( size_t i ) const
{
    constexpr auto type = getTypeID<TYPE>();
    return isType( type, i );
}


/****************************************************************
 * Create vector iterators                                       *
 ****************************************************************/
template<class TYPE>
inline VectorDataIterator<TYPE> VectorData::begin()
{
    dataChanged();
    return VectorDataIterator<TYPE>( this, 0 );
}
template<class TYPE>
inline VectorDataIterator<const TYPE> VectorData::begin() const
{
    return VectorDataIterator<const TYPE>( const_cast<VectorData *>( this ), 0 );
}
template<class TYPE>
inline VectorDataIterator<const TYPE> VectorData::constBegin() const
{
    return VectorDataIterator<const TYPE>( const_cast<VectorData *>( this ), 0 );
}
template<class TYPE>
inline VectorDataIterator<TYPE> VectorData::end()
{
    dataChanged();
    return VectorDataIterator<TYPE>( this, getLocalSize() );
}
template<class TYPE>
inline VectorDataIterator<const TYPE> VectorData::constEnd() const
{
    return VectorDataIterator<const TYPE>( const_cast<VectorData *>( this ), getLocalSize() );
}
template<class TYPE>
inline VectorDataIterator<const TYPE> VectorData::end() const
{
    return VectorDataIterator<const TYPE>( const_cast<VectorData *>( this ), getLocalSize() );
}
inline size_t VectorData::getGhostSize() const { return d_Ghosts->size(); }
inline size_t VectorData::getNumberOfComponents() const { return 1; }


/****************************************************************
 * Update status                                                 *
 ****************************************************************/
inline VectorData::UpdateState VectorData::getUpdateStatus() const { return *d_UpdateState; }
inline void VectorData::setUpdateStatus( UpdateState state ) { *d_UpdateState = state; }
inline void VectorData::setUpdateStatusPtr( std::shared_ptr<UpdateState> rhs )
{
    d_UpdateState = rhs;
}
inline std::shared_ptr<VectorData::UpdateState> VectorData::getUpdateStatusPtr() const
{
    return d_UpdateState;
}


/****************************************************************
 * Get/Set raw data                                              *
 ****************************************************************/
template<class TYPE>
void VectorData::putRawData( const TYPE *buf )
{
    constexpr auto type = getTypeID<TYPE>();
    putRawData( buf, type );
}
template<class TYPE>
void VectorData::getRawData( TYPE *buf ) const
{
    constexpr auto type = getTypeID<TYPE>();
    getRawData( buf, type );
}
template<typename TYPE>
TYPE *VectorData::getRawDataBlock( size_t i )
{
    return static_cast<TYPE *>( this->getRawDataBlockAsVoid( i ) );
}
template<typename TYPE>
const TYPE *VectorData::getRawDataBlock( size_t i ) const
{
    return static_cast<const TYPE *>( this->getRawDataBlockAsVoid( i ) );
}


/****************************************************************
 * Get/Set values by global id                                   *
 ****************************************************************/
template<typename TYPE>
void VectorData::getValuesByLocalID( size_t N, const size_t *ndx, TYPE *vals ) const
{
    constexpr auto type = getTypeID<TYPE>();
    getValuesByLocalID( N, ndx, vals, type );
}
template<typename TYPE>
void VectorData::setValuesByLocalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    constexpr auto type = getTypeID<TYPE>();
    setValuesByLocalID( N, ndx, vals, type );
}
template<typename TYPE>
void VectorData::addValuesByLocalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    constexpr auto type = getTypeID<TYPE>();
    addValuesByLocalID( N, ndx, vals, type );
}
template<typename TYPE>
void VectorData::getLocalValuesByGlobalID( size_t N, const size_t *ndx, TYPE *vals ) const
{
    constexpr size_t N_max = 128;
    while ( N > N_max ) {
        getLocalValuesByGlobalID( N_max, ndx, vals );
        N -= N_max;
        ndx  = &ndx[N_max];
        vals = &vals[N_max];
    }
    size_t index[N_max];
    for ( size_t i = 0; i < N; i++ ) {
        AMP_ASSERT( ndx[i] >= d_localStart && ndx[i] < ( d_localStart + d_localSize ) );
        index[i] = ndx[i] - d_localStart;
    }
    constexpr auto type = getTypeID<TYPE>();
    getValuesByLocalID( N, index, vals, type );
}
template<typename TYPE>
void VectorData::addLocalValuesByGlobalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    constexpr size_t N_max = 128;
    while ( N > N_max ) {
        addLocalValuesByGlobalID( N_max, ndx, vals );
        N -= N_max;
        ndx  = &ndx[N_max];
        vals = &vals[N_max];
    }
    size_t index[N_max];
    for ( size_t i = 0; i < N; i++ ) {
        AMP_ASSERT( ndx[i] >= d_localStart && ndx[i] < ( d_localStart + d_localSize ) );
        index[i] = ndx[i] - d_localStart;
    }
    constexpr auto type = getTypeID<TYPE>();
    addValuesByLocalID( N, index, vals, type );
}
template<typename TYPE>
void VectorData::setLocalValuesByGlobalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    constexpr size_t N_max = 128;
    while ( N > N_max ) {
        setLocalValuesByGlobalID( N_max, ndx, vals );
        N -= N_max;
        ndx  = &ndx[N_max];
        vals = &vals[N_max];
    }
    size_t index[N_max];
    for ( size_t i = 0; i < N; i++ ) {
        AMP_ASSERT( ndx[i] >= d_localStart && ndx[i] < ( d_localStart + d_localSize ) );
        index[i] = ndx[i] - d_localStart;
    }
    constexpr auto type = getTypeID<TYPE>();
    setValuesByLocalID( N, index, vals, type );
}
template<typename TYPE>
void VectorData::getValuesByGlobalID( size_t N, const size_t *ndx, TYPE *vals ) const
{
    constexpr size_t N_max = 128;
    while ( N > N_max ) {
        getValuesByGlobalID( N_max, ndx, vals );
        N -= N_max;
        ndx  = &ndx[N_max];
        vals = &vals[N_max];
    }
    size_t N_local = 0, N_ghost = 0;
    size_t local_index[N_max], ghost_index[N_max];
    TYPE local_vals[N_max], ghost_vals[N_max];
    for ( size_t i = 0; i < N; i++ ) {
        if ( ( ndx[i] >= d_localStart ) && ( ndx[i] < ( d_localStart + d_localSize ) ) ) {
            local_index[N_local] = ndx[i] - d_localStart;
            N_local++;
        } else {
            ghost_index[N_ghost] = ndx[i];
            N_ghost++;
        }
    }
    constexpr auto type = getTypeID<TYPE>();
    if ( N_local > 0 )
        getValuesByLocalID( N_local, local_index, local_vals, type );
    if ( N_ghost > 0 )
        getGhostValuesByGlobalID( N_ghost, ghost_index, ghost_vals, type );
    N_local = 0;
    N_ghost = 0;
    for ( size_t i = 0; i < N; i++ ) {
        if ( ( ndx[i] >= d_localStart ) && ( ndx[i] < ( d_localStart + d_localSize ) ) ) {
            vals[i] = local_vals[N_local];
            N_local++;
        } else {
            vals[i] = ghost_vals[N_ghost];
            N_ghost++;
        }
    }
}
template<typename TYPE>
void VectorData::setValuesByGlobalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    constexpr size_t N_max = 128;
    while ( N > N_max ) {
        setValuesByGlobalID( N_max, ndx, vals );
        N -= N_max;
        ndx  = &ndx[N_max];
        vals = &vals[N_max];
    }
    size_t N_local = 0, N_ghost = 0;
    size_t local_index[N_max], ghost_index[N_max];
    TYPE local_vals[N_max], ghost_vals[N_max];
    for ( size_t i = 0; i < N; i++ ) {
        if ( ( ndx[i] >= d_localStart ) && ( ndx[i] < ( d_localStart + d_localSize ) ) ) {
            local_index[N_local] = ndx[i] - d_localStart;
            local_vals[N_local]  = vals[i];
            N_local++;
        } else {
            ghost_index[N_ghost] = ndx[i];
            ghost_vals[N_ghost]  = vals[i];
            N_ghost++;
        }
    }
    constexpr auto type = getTypeID<TYPE>();
    if ( N_local > 0 )
        setValuesByLocalID( N_local, local_index, local_vals, type );
    if ( N_ghost > 0 )
        setGhostValuesByGlobalID( N_ghost, ghost_index, ghost_vals, type );
}
template<typename TYPE>
void VectorData::addValuesByGlobalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    constexpr size_t N_max = 128;
    while ( N > N_max ) {
        addValuesByGlobalID( N_max, ndx, vals );
        N -= N_max;
        ndx  = &ndx[N_max];
        vals = &vals[N_max];
    }
    size_t N_local = 0, N_ghost = 0;
    size_t local_index[N_max], ghost_index[N_max];
    TYPE local_vals[N_max], ghost_vals[N_max];
    for ( size_t i = 0; i < N; i++ ) {
        if ( ( ndx[i] >= d_localStart ) && ( ndx[i] < ( d_localStart + d_localSize ) ) ) {
            local_index[N_local] = ndx[i] - d_localStart;
            local_vals[N_local]  = vals[i];
            N_local++;
        } else {
            ghost_index[N_ghost] = ndx[i];
            ghost_vals[N_ghost]  = vals[i];
            N_ghost++;
        }
    }
    constexpr auto type = getTypeID<TYPE>();
    if ( N_local > 0 )
        addValuesByLocalID( N_local, local_index, local_vals, type );
    if ( N_ghost > 0 )
        addGhostValuesByGlobalID( N_ghost, ghost_index, ghost_vals, type );
}


/****************************************************************
 * Get/Set ghost values by global id                             *
 ****************************************************************/
template<typename TYPE>
void VectorData::setGhostValuesByGlobalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    constexpr auto type = getTypeID<TYPE>();
    setGhostValuesByGlobalID( N, ndx, vals, type );
}
template<typename TYPE>
void VectorData::addGhostValuesByGlobalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    constexpr auto type = getTypeID<TYPE>();
    addGhostValuesByGlobalID( N, ndx, vals, type );
}
template<typename TYPE>
void VectorData::getGhostValuesByGlobalID( size_t N, const size_t *ndx, TYPE *vals ) const
{
    constexpr auto type = getTypeID<TYPE>();
    getGhostValuesByGlobalID( N, ndx, vals, type );
}
template<typename TYPE>
void VectorData::getGhostAddValuesByGlobalID( size_t N, const size_t *ndx, TYPE *vals ) const
{
    constexpr auto type = getTypeID<TYPE>();
    getGhostAddValuesByGlobalID( N, ndx, vals, type );
}


} // namespace AMP::LinearAlgebra

#endif
