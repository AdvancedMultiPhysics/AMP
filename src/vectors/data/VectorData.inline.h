#ifndef included_AMP_VectorData_inline
#define included_AMP_VectorData_inline

#include "AMP/utils/Algorithms.h"
#include "AMP/utils/typeid.h"
#include "AMP/vectors/data/VectorDataIterator.h"

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {


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
inline bool VectorData::isType( const typeID &id, size_t i ) const
{
    auto type = getType( i );
    return id == type;
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


/****************************************************************
 * Get/Set raw data                                              *
 ****************************************************************/
template<class TYPE>
void VectorData::putRawData( const TYPE *buf, AMP::Utilities::MemoryType buf_loc )
{
    constexpr auto type = getTypeID<TYPE>();
    putRawData( buf, type, buf_loc );
}
template<class TYPE>
void VectorData::putRawData( const TYPE *buf )
{
    constexpr auto type = getTypeID<TYPE>();
    putRawData( buf, type, AMP::Utilities::getMemoryType( buf ) );
}
template<class TYPE>
void VectorData::getRawData( TYPE *buf, AMP::Utilities::MemoryType buf_loc ) const
{
    constexpr auto type = getTypeID<TYPE>();
    getRawData( buf, type, buf_loc );
}
template<class TYPE>
void VectorData::getRawData( TYPE *buf ) const
{
    constexpr auto type = getTypeID<TYPE>();
    getRawData( buf, type, AMP::Utilities::getMemoryType( buf ) );
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
void VectorData::getValuesByLocalID( size_t N,
                                     const size_t *ndx,
                                     TYPE *vals,
                                     AMP::Utilities::MemoryType buf_loc ) const
{
    PROFILE( "VectorData::getValuesByLocalID" );

    constexpr auto type = getTypeID<TYPE>();
    getValuesByLocalID( N, ndx, vals, type, buf_loc );
}

template<typename TYPE>
void VectorData::getValuesByLocalID( size_t N, const size_t *ndx, TYPE *vals ) const
{
    PROFILE( "VectorData::getValuesByLocalID" );

    constexpr auto type = getTypeID<TYPE>();
    getValuesByLocalID( N, ndx, vals, type, AMP::Utilities::getMemoryType( vals ) );
}

template<typename TYPE>
void VectorData::setValuesByLocalID( size_t N,
                                     const size_t *ndx,
                                     const TYPE *vals,
                                     AMP::Utilities::MemoryType buf_loc )
{
    PROFILE( "VectorData::setValuesByLocalID" );

    constexpr auto type = getTypeID<TYPE>();
    setValuesByLocalID( N, ndx, vals, type, buf_loc );
}

template<typename TYPE>
void VectorData::setValuesByLocalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    PROFILE( "VectorData::setValuesByLocalID" );

    constexpr auto type = getTypeID<TYPE>();
    setValuesByLocalID( N, ndx, vals, type, AMP::Utilities::getMemoryType( vals ) );
}

template<typename TYPE>
void VectorData::addValuesByLocalID( size_t N,
                                     const size_t *ndx,
                                     const TYPE *vals,
                                     AMP::Utilities::MemoryType buf_loc )
{
    PROFILE( "VectorData::addValuesByLocalID" );

    constexpr auto type = getTypeID<TYPE>();
    addValuesByLocalID( N, ndx, vals, type, buf_loc );
}

template<typename TYPE>
void VectorData::addValuesByLocalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    PROFILE( "VectorData::addValuesByLocalID" );

    constexpr auto type = getTypeID<TYPE>();
    addValuesByLocalID( N, ndx, vals, type, AMP::Utilities::getMemoryType( vals ) );
}

template<typename TYPE>
void VectorData::getValuesByGlobalID( size_t N,
                                      const size_t *ndx_,
                                      TYPE *vals_,
                                      AMP::Utilities::MemoryType buf_loc ) const
{
    PROFILE( "VectorData::getValuesByGlobalID" );
    auto ndx        = ndx_;
    auto vals       = vals_;
    size_t *ndx_mem = nullptr;
    TYPE *vals_mem  = nullptr;
    if ( buf_loc >= AMP::Utilities::MemoryType::managed ) {
        ndx_mem = new size_t[N];
        AMP::Utilities::Algorithms::copy_n(
            ndx_mem, AMP::Utilities::MemoryType::host, ndx_, buf_loc, N );
        ndx      = ndx_mem;
        vals_mem = new TYPE[N];
        AMP::Utilities::Algorithms::copy_n(
            vals_mem, AMP::Utilities::MemoryType::host, vals_, buf_loc, N );
        vals = vals_mem;
    }
    constexpr size_t N_max = 128;
    while ( N != 0 ) {
        size_t N2      = std::min( N, N_max );
        size_t N_local = 0, N_ghost = 0;
        size_t local_index[N_max], ghost_index[N_max];
        TYPE local_vals[N_max], ghost_vals[N_max];
        for ( size_t i = 0; i < N2; i++ ) {
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
            getValuesByLocalID(
                N_local, local_index, local_vals, type, AMP::Utilities::MemoryType::host );
        if ( N_ghost > 0 )
            getGhostValuesByGlobalID(
                N_ghost, ghost_index, ghost_vals, type, AMP::Utilities::MemoryType::host );
        N_local = 0;
        N_ghost = 0;
        for ( size_t i = 0; i < N2; i++ ) {
            if ( ( ndx[i] >= d_localStart ) && ( ndx[i] < ( d_localStart + d_localSize ) ) ) {
                vals[i] = local_vals[N_local];
                N_local++;
            } else {
                vals[i] = ghost_vals[N_ghost];
                N_ghost++;
            }
        }
        N -= N2;
        ndx  = &ndx[N2];
        vals = &vals[N2];
    }
    delete[] ndx_mem;
    delete[] vals_mem;
}

template<typename TYPE>
void VectorData::getValuesByGlobalID( size_t N, const size_t *ndx_, TYPE *vals_ ) const
{
    this->getValuesByGlobalID( N, ndx_, vals_, AMP::Utilities::getMemoryType( vals_ ) );
}

template<typename TYPE>
void VectorData::setValuesByGlobalID( size_t N,
                                      const size_t *ndx_,
                                      const TYPE *vals_,
                                      AMP::Utilities::MemoryType buf_loc )
{
    PROFILE( "VectorData::setValuesByGlobalID" );
    auto ndx        = ndx_;
    auto vals       = vals_;
    size_t *ndx_mem = nullptr;
    TYPE *vals_mem  = nullptr;
    if ( buf_loc >= AMP::Utilities::MemoryType::managed ) {
        ndx_mem = new size_t[N];
        AMP::Utilities::Algorithms::copy_n(
            ndx_mem, AMP::Utilities::MemoryType::host, ndx_, buf_loc, N );
        ndx      = ndx_mem;
        vals_mem = new TYPE[N];
        AMP::Utilities::Algorithms::copy_n(
            vals_mem, AMP::Utilities::MemoryType::host, vals_, buf_loc, N );
        vals = vals_mem;
    }
    constexpr size_t N_max = 128;
    while ( N != 0 ) {
        size_t N2      = std::min( N, N_max );
        size_t N_local = 0, N_ghost = 0;
        size_t local_index[N_max], ghost_index[N_max];
        TYPE local_vals[N_max], ghost_vals[N_max];
        for ( size_t i = 0; i < N2; i++ ) {
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
            setValuesByLocalID(
                N_local, local_index, local_vals, type, AMP::Utilities::MemoryType::host );
        if ( N_ghost > 0 )
            setGhostValuesByGlobalID(
                N_ghost, ghost_index, ghost_vals, type, AMP::Utilities::MemoryType::host );
        N -= N2;
        ndx  = &ndx[N2];
        vals = &vals[N2];
    }
    delete[] ndx_mem;
    delete[] vals_mem;
}

template<typename TYPE>
void VectorData::setValuesByGlobalID( size_t N, const size_t *ndx_, const TYPE *vals_ )
{
    this->setValuesByGlobalID( N, ndx_, vals_, AMP::Utilities::getMemoryType( vals_ ) );
}

template<typename TYPE>
void VectorData::addValuesByGlobalID( size_t N,
                                      const size_t *ndx_,
                                      const TYPE *vals_,
                                      AMP::Utilities::MemoryType buf_loc )
{
    PROFILE( "VectorData::addValuesByGlobalID" );

    auto ndx        = ndx_;
    auto vals       = vals_;
    size_t *ndx_mem = nullptr;
    TYPE *vals_mem  = nullptr;
    if ( buf_loc >= AMP::Utilities::MemoryType::managed ) {
        ndx_mem = new size_t[N];
        AMP::Utilities::Algorithms::copy_n(
            ndx_mem, AMP::Utilities::MemoryType::host, ndx_, buf_loc, N );
        ndx      = ndx_mem;
        vals_mem = new TYPE[N];
        AMP::Utilities::Algorithms::copy_n(
            vals_mem, AMP::Utilities::MemoryType::host, vals_, buf_loc, N );
        vals = vals_mem;
    }
    constexpr size_t N_max = 128;
    while ( N != 0 ) {
        size_t N2      = std::min( N, N_max );
        size_t N_local = 0, N_ghost = 0;
        size_t local_index[N_max], ghost_index[N_max];
        TYPE local_vals[N_max], ghost_vals[N_max];
        for ( size_t i = 0; i < N2; i++ ) {
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
            addValuesByLocalID(
                N_local, local_index, local_vals, type, AMP::Utilities::MemoryType::host );
        if ( N_ghost > 0 )
            addGhostValuesByGlobalID(
                N_ghost, ghost_index, ghost_vals, type, AMP::Utilities::MemoryType::host );
        N -= N2;
        ndx  = &ndx[N2];
        vals = &vals[N2];
    }
    delete[] ndx_mem;
    delete[] vals_mem;
}

template<typename TYPE>
void VectorData::addValuesByGlobalID( size_t N, const size_t *ndx_, const TYPE *vals_ )
{
    this->addValuesByGlobalID( N, ndx_, vals_, AMP::Utilities::getMemoryType( vals_ ) );
}

/****************************************************************
 * Get/Set ghost values by global id                             *
 ****************************************************************/
template<typename TYPE>
void VectorData::setGhostValuesByGlobalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    PROFILE( "VectorData::setGhostValuesByGlobalID" );

    constexpr auto type = getTypeID<TYPE>();
    setGhostValuesByGlobalID( N, ndx, vals, type, AMP::Utilities::getMemoryType( vals ) );
}

template<typename TYPE>
void VectorData::setGhostValuesByGlobalID( size_t N,
                                           const size_t *ndx,
                                           const TYPE *vals,
                                           AMP::Utilities::MemoryType buf_loc )
{
    PROFILE( "VectorData::setGhostValuesByGlobalID" );

    constexpr auto type = getTypeID<TYPE>();
    setGhostValuesByGlobalID( N, ndx, vals, type, buf_loc );
}

template<typename TYPE>
void VectorData::addGhostValuesByGlobalID( size_t N, const size_t *ndx, const TYPE *vals )
{
    PROFILE( "VectorData::addGhostValuesByGlobalID" );

    constexpr auto type = getTypeID<TYPE>();
    addGhostValuesByGlobalID( N, ndx, vals, type, AMP::Utilities::getMemoryType( vals ) );
}

template<typename TYPE>
void VectorData::addGhostValuesByGlobalID( size_t N,
                                           const size_t *ndx,
                                           const TYPE *vals,
                                           AMP::Utilities::MemoryType buf_loc )
{
    PROFILE( "VectorData::addGhostValuesByGlobalID" );

    constexpr auto type = getTypeID<TYPE>();
    addGhostValuesByGlobalID( N, ndx, vals, type, buf_loc );
}

template<typename TYPE>
void VectorData::getGhostValuesByGlobalID( size_t N, const size_t *ndx, TYPE *vals ) const
{
    PROFILE( "VectorData::getGhostValuesByGlobalID" );

    constexpr auto type = getTypeID<TYPE>();
    getGhostValuesByGlobalID( N, ndx, vals, type, AMP::Utilities::getMemoryType( vals ) );
}

template<typename TYPE>
void VectorData::getGhostValuesByGlobalID( size_t N,
                                           const size_t *ndx,
                                           TYPE *vals,
                                           AMP::Utilities::MemoryType buf_loc ) const
{
    PROFILE( "VectorData::getGhostValuesByGlobalID" );

    constexpr auto type = getTypeID<TYPE>();
    getGhostValuesByGlobalID( N, ndx, vals, type, buf_loc );
}

template<typename TYPE>
void VectorData::getGhostAddValuesByGlobalID( size_t N, const size_t *ndx, TYPE *vals ) const
{
    PROFILE( "VectorData::getGhostAddValuesByGlobalID" );

    constexpr auto type = getTypeID<TYPE>();
    getGhostAddValuesByGlobalID( N, ndx, vals, type, AMP::Utilities::getMemoryType( vals ) );
}

template<typename TYPE>
void VectorData::getGhostAddValuesByGlobalID( size_t N,
                                              const size_t *ndx,
                                              TYPE *vals,
                                              AMP::Utilities::MemoryType buf_loc ) const
{
    PROFILE( "VectorData::getGhostAddValuesByGlobalID" );

    constexpr auto type = getTypeID<TYPE>();
    getGhostAddValuesByGlobalID( N, ndx, vals, type, buf_loc );
}

template<class TYPE>
size_t VectorData::getAllGhostValues( TYPE *vals ) const
{
    PROFILE( "VectorData::getAllGhostValues" );

    constexpr auto type = getTypeID<TYPE>();
    return getAllGhostValues( vals, type, AMP::Utilities::getMemoryType( vals ) );
}

template<class TYPE>
size_t VectorData::getAllGhostValues( TYPE *vals, AMP::Utilities::MemoryType buf_loc ) const
{
    PROFILE( "VectorData::getAllGhostValues" );

    constexpr auto type = getTypeID<TYPE>();
    return getAllGhostValues( vals, type, buf_loc );
}

} // namespace AMP::LinearAlgebra

#endif
