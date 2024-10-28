// This file contains the default instantiations for templated operations
// Note: Intel compilers need definitions before all default instantions to compile correctly
#ifndef included_AMP_MPI_I
#define included_AMP_MPI_I

#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/AMP_MPI_pack.hpp"
#include "AMP/utils/Array.h"
#include "AMP/utils/TypeTraits.h"
#include "AMP/utils/UtilityMacros.h"

#include "ProfilerApp.h"

#include <array>
#include <complex>
#include <cstddef>
#include <typeinfo>
#include <vector>


#define MPI_CLASS AMP_MPI
#define MPI_CLASS2 AMP::AMP_MPI
#define MPI_CLASS_ERROR AMP_ERROR
#define MPI_CLASS_ASSERT AMP_ASSERT
#define MPI_CLASS_INSIST AMP_INSIST
#define MPI_CLASS_WARNING AMP_WARNING


// Include mpi.h (if building with MPI)
#ifdef AMP_USE_MPI
    #include "mpi.h"
#endif


#define MPI_CLASS_ERROR_TYPE( MSG ) \
    MPI_CLASS_ERROR( std::string( MSG ) + " (" + typeid( TYPE ).name() + "(" );


/************************************************************************
 *  Instantiate common MPI functions for the given type                  *
 ************************************************************************/
#define INSTANTIATE_MPI_REDUCE( TYPE )                                                            \
    template TYPE MPI_CLASS2::sumReduce<TYPE>( const TYPE & ) const;                              \
    template TYPE MPI_CLASS2::minReduce<TYPE>( const TYPE & ) const;                              \
    template TYPE MPI_CLASS2::maxReduce<TYPE>( const TYPE & ) const;                              \
    template void MPI_CLASS2::sumReduce<TYPE>( TYPE *, int ) const;                               \
    template void MPI_CLASS2::minReduce<TYPE>( TYPE *, int ) const;                               \
    template void MPI_CLASS2::maxReduce<TYPE>( TYPE *, int ) const;                               \
    template void MPI_CLASS2::minReduce<TYPE>( TYPE *, int, int * ) const;                        \
    template void MPI_CLASS2::maxReduce<TYPE>( TYPE *, int, int * ) const;                        \
    template void MPI_CLASS2::sumReduce<TYPE>( const TYPE *, TYPE *, int ) const;                 \
    template void MPI_CLASS2::minReduce<TYPE>( const TYPE *, TYPE *, int, int * ) const;          \
    template void MPI_CLASS2::maxReduce<TYPE>( const TYPE *, TYPE *, int, int * ) const;          \
    template AMP::Array<TYPE> MPI_CLASS2::sumReduce<AMP::Array<TYPE>>( const AMP::Array<TYPE> & ) \
        const;                                                                                    \
    template AMP::Array<TYPE> MPI_CLASS2::minReduce<AMP::Array<TYPE>>( const AMP::Array<TYPE> & ) \
        const;                                                                                    \
    template AMP::Array<TYPE> MPI_CLASS2::maxReduce<AMP::Array<TYPE>>( const AMP::Array<TYPE> & ) \
        const;                                                                                    \
    template TYPE MPI_CLASS2::sumScan<TYPE>( const TYPE & ) const;                                \
    template TYPE MPI_CLASS2::minScan<TYPE>( const TYPE & ) const;                                \
    template TYPE MPI_CLASS2::maxScan<TYPE>( const TYPE & ) const;                                \
    template void MPI_CLASS2::sumScan<TYPE>( const TYPE *, TYPE *, int ) const;                   \
    template void MPI_CLASS2::minScan<TYPE>( const TYPE *, TYPE *, int ) const;                   \
    template void MPI_CLASS2::maxScan<TYPE>( const TYPE *, TYPE *, int ) const
#define INSTANTIATE_MPI_BCAST( TYPE )                                 \
    template TYPE MPI_CLASS2::bcast<TYPE>( const TYPE &, int ) const; \
    template void MPI_CLASS2::bcast<TYPE>( TYPE *, int, int ) const
#define INSTANTIATE_MPI_SENDRECV( TYPE )                                                       \
    template void MPI_CLASS2::send<TYPE>( const TYPE &, int, int ) const;                      \
    template void MPI_CLASS2::send<TYPE>( const TYPE *, int, int, int ) const;                 \
    template MPI_CLASS2::Request MPI_CLASS2::Isend<TYPE>( const TYPE &, int, int ) const;      \
    template MPI_CLASS2::Request MPI_CLASS2::Isend<TYPE>( const TYPE *, int, int, int ) const; \
    template TYPE MPI_CLASS2::recv<TYPE>( int, int ) const;                                    \
    template void MPI_CLASS2::recv<TYPE>( TYPE *, int, int, int ) const;                       \
    template void MPI_CLASS2::recv<TYPE>( TYPE *, int &, int, const bool, int ) const;         \
    template MPI_CLASS2::Request MPI_CLASS2::Irecv<TYPE>( TYPE & buf, int, int ) const;        \
    template MPI_CLASS2::Request MPI_CLASS2::Irecv<TYPE>( TYPE * buf, int, int, int ) const
#define INSTANTIATE_MPI_GATHER( TYPE )                                                         \
    template std::vector<TYPE> MPI_CLASS2::allGather<TYPE>( const TYPE & ) const;              \
    template std::vector<TYPE> MPI_CLASS2::allGather<TYPE>( const std::vector<TYPE> & ) const; \
    template void MPI_CLASS2::allGather<TYPE>( const TYPE &, TYPE * ) const;                   \
    template int MPI_CLASS2::allGather<TYPE>( const TYPE *, int, TYPE *, int *, int *, bool )  \
        const;                                                                                 \
    template void MPI_CLASS2::setGather<TYPE>( std::set<TYPE> & ) const;                       \
    template void MPI_CLASS2::allToAll<TYPE>( int, const TYPE *, TYPE * ) const;               \
    template int MPI_CLASS2::allToAll<TYPE>(                                                   \
        const TYPE *, const int[], const int[], TYPE *, int *, int *, bool ) const
#define INSTANTIATE_MPI_TYPE( TYPE )  \
    INSTANTIATE_MPI_REDUCE( TYPE );   \
    INSTANTIATE_MPI_BCAST( TYPE );    \
    INSTANTIATE_MPI_SENDRECV( TYPE ); \
    INSTANTIATE_MPI_GATHER( TYPE )


// Use AMP namespace
namespace AMP {


// Function to test if a type can be passed by MPI
template<typename TYPE>
struct is_mpi_copyable {
    template<typename T>
    static constexpr bool test()
    {
        if constexpr ( std::is_trivially_copyable_v<T> )
            return true;
        if constexpr ( is_pair_v<T> )
            return test<typename T::first_type>() && test<typename T::second_type>();
        return false;
    }
    static constexpr bool value = test<TYPE>();
};
template<class T>
inline constexpr bool is_mpi_copyable_v = is_mpi_copyable<T>::value;


/************************************************************************
 *  getType                                                              *
 ************************************************************************/
template<class TYPE>
struct TypeIntStruct {
    TYPE x;
    int i;
};
// clang-format off
template<typename T> struct TypeIntSelector;
template<> struct TypeIntSelector<char>     { using type = TypeIntStruct<short>; };
template<> struct TypeIntSelector<uint8_t>  { using type = TypeIntStruct<short>; };
template<> struct TypeIntSelector<int8_t>   { using type = TypeIntStruct<short>; };
template<> struct TypeIntSelector<uint16_t> { using type = TypeIntStruct<int>;   };
template<> struct TypeIntSelector<int16_t>  { using type = TypeIntStruct<short>; };
template<> struct TypeIntSelector<uint32_t> { using type = TypeIntStruct<long>;  };
template<> struct TypeIntSelector<int32_t>  { using type = TypeIntStruct<int>;   };
template<> struct TypeIntSelector<uint64_t> { using type = TypeIntStruct<long>;  };
template<> struct TypeIntSelector<int64_t>  { using type = TypeIntStruct<long>;  };
template<> struct TypeIntSelector<float>    { using type = TypeIntStruct<float>; };
template<> struct TypeIntSelector<double>   { using type = TypeIntStruct<double>; };
template<typename T> struct TypeIntSelector { using type = TypeIntStruct<T>;     };
// clang-format on
template<class TYPE>
constexpr bool isMPItype()
{
    return std::is_same_v<TYPE, char> || std::is_same_v<TYPE, uint8_t> ||
           std::is_same_v<TYPE, int8_t> || std::is_same_v<TYPE, uint16_t> ||
           std::is_same_v<TYPE, int16_t> || std::is_same_v<TYPE, uint32_t> ||
           std::is_same_v<TYPE, int32_t> || std::is_same_v<TYPE, uint64_t> ||
           std::is_same_v<TYPE, int64_t> || std::is_same_v<TYPE, float> ||
           std::is_same_v<TYPE, double> || std::is_same_v<TYPE, std::complex<float>> ||
           std::is_same_v<TYPE, std::complex<double>> ||
           std::is_same_v<TYPE, TypeIntStruct<short>> || std::is_same_v<TYPE, TypeIntStruct<int>> ||
           std::is_same_v<TYPE, TypeIntStruct<long>> ||
           std::is_same_v<TYPE, TypeIntStruct<float>> ||
           std::is_same_v<TYPE, TypeIntStruct<double>>;
}
template<class TYPE>
MPI_CLASS::Datatype MPI_getType()
{
#ifdef AMP_USE_MPI
    if constexpr ( std::is_same_v<TYPE, std::byte> )
        return MPI_BYTE;
    if constexpr ( std::is_same_v<TYPE, char> )
        return MPI_CHAR;
    if constexpr ( std::is_same_v<TYPE, uint8_t> )
        return MPI_UINT8_T;
    if constexpr ( std::is_same_v<TYPE, int8_t> )
        return MPI_INT8_T;
    if constexpr ( std::is_same_v<TYPE, uint16_t> )
        return MPI_UINT16_T;
    if constexpr ( std::is_same_v<TYPE, int16_t> )
        return MPI_INT16_T;
    if constexpr ( std::is_same_v<TYPE, uint32_t> )
        return MPI_UINT32_T;
    if constexpr ( std::is_same_v<TYPE, int32_t> )
        return MPI_INT32_T;
    if constexpr ( std::is_same_v<TYPE, uint64_t> )
        return MPI_UINT64_T;
    if constexpr ( std::is_same_v<TYPE, int64_t> )
        return MPI_INT64_T;
    if constexpr ( std::is_same_v<TYPE, float> )
        return MPI_FLOAT;
    if constexpr ( std::is_same_v<TYPE, double> )
        return MPI_DOUBLE;
    if constexpr ( std::is_same_v<TYPE, std::complex<float>> )
        return MPI_C_FLOAT_COMPLEX;
    if constexpr ( std::is_same_v<TYPE, std::complex<double>> )
        return MPI_C_DOUBLE_COMPLEX;
    if constexpr ( std::is_same_v<TYPE, TypeIntStruct<short>> )
        return MPI_SHORT_INT;
    if constexpr ( std::is_same_v<TYPE, TypeIntStruct<int>> )
        return MPI_2INT;
    if constexpr ( std::is_same_v<TYPE, TypeIntStruct<long>> )
        return MPI_LONG_INT;
    if constexpr ( std::is_same_v<TYPE, TypeIntStruct<float>> )
        return MPI_FLOAT_INT;
    if constexpr ( std::is_same_v<TYPE, TypeIntStruct<double>> )
        return MPI_DOUBLE_INT;
    return MPI_DATATYPE_NULL;
#else
    if constexpr ( isMPItype<TYPE>() )
        return typeid( TYPE ).hash_code();
    return 0;
#endif
}
inline size_t MPI_getTypeSize( MPI_CLASS::Datatype type )
{
    if ( type == MPI_getType<std::byte>() )
        return sizeof( std::byte );
    if ( type == MPI_getType<char>() )
        return sizeof( char );
    if ( type == MPI_getType<uint8_t>() )
        return sizeof( uint8_t );
    if ( type == MPI_getType<int8_t>() )
        return sizeof( int8_t );
    if ( type == MPI_getType<uint16_t>() )
        return sizeof( uint16_t );
    if ( type == MPI_getType<int16_t>() )
        return sizeof( int16_t );
    if ( type == MPI_getType<uint32_t>() )
        return sizeof( uint32_t );
    if ( type == MPI_getType<int32_t>() )
        return sizeof( int32_t );
    if ( type == MPI_getType<uint64_t>() )
        return sizeof( uint64_t );
    if ( type == MPI_getType<int64_t>() )
        return sizeof( int64_t );
    if ( type == MPI_getType<float>() )
        return sizeof( float );
    if ( type == MPI_getType<double>() )
        return sizeof( double );
    if ( type == MPI_getType<std::complex<float>>() )
        return sizeof( std::complex<float> );
    if ( type == MPI_getType<std::complex<double>>() )
        return sizeof( std::complex<double> );
    if ( type == MPI_getType<TypeIntStruct<short>>() )
        return sizeof( TypeIntStruct<short> );
    if ( type == MPI_getType<TypeIntStruct<int>>() )
        return sizeof( TypeIntStruct<int> );
    if ( type == MPI_getType<TypeIntStruct<long>>() )
        return sizeof( TypeIntStruct<long> );
    if ( type == MPI_getType<TypeIntStruct<float>>() )
        return sizeof( TypeIntStruct<float> );
    if ( type == MPI_getType<TypeIntStruct<double>>() )
        return sizeof( TypeIntStruct<double> );
    if ( type == MPI_getType<TypeIntStruct<int>>() )
        return sizeof( TypeIntStruct<int> );
    if ( type == MPI_getType<TypeIntStruct<int>>() )
        return sizeof( TypeIntStruct<int> );
    MPI_CLASS_ERROR( "Unknown MPI type" );
    return 0;
}
#define MPI_CLASS_BYTE MPI_getType<std::byte>()


/************************************************************************
 *  sumReduce                                                            *
 ************************************************************************/
template<class TYPE>
void MPI_CLASS::sumReduce( const TYPE *send, TYPE *recv, int n ) const
{
    if ( d_size == 1 ) {
        for ( int i = 0; i < n; i++ )
            recv[i] = send[i];
        return;
    }
#ifdef AMP_USE_MPI
    PROFILE( "sumReduce", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_CLASS_ASSERT( d_comm != MPI_COMM_NULL );
        MPI_Allreduce( send, recv, n, type, MPI_SUM, d_comm );
    } else {
        throw std::logic_error( "sumReduce is not valid for arbitrary types" );
    }
#endif
}
template<class TYPE>
TYPE MPI_CLASS::sumReduce( const TYPE &value ) const
{
    if constexpr ( AMP::is_vector_v<TYPE> || std::is_array_v<TYPE> || AMP::is_array_v<TYPE> ) {
        auto recv = value;
        sumReduce( value.data(), recv.data(), value.size() );
        return recv;
    } else if constexpr ( AMP::is_Array_v<TYPE> ) {
        auto recv = value;
        sumReduce( value.data(), recv.data(), value.length() );
        return recv;
    } else {
        if ( d_size > 1 ) {
            TYPE tmp = value;
            sumReduce( &value, &tmp, 1 );
            return tmp;
        } else {
            return value;
        }
    }
}
template<class TYPE>
void MPI_CLASS::sumReduce( TYPE *x, int n ) const
{
    if ( d_size > 1 ) {
        auto tmp = new TYPE[n];
        for ( int i = 0; i < n; i++ )
            tmp[i] = x[i];
        sumReduce( tmp, x, n );
        delete[] tmp;
    }
}


/************************************************************************
 *  minReduce                                                            *
 ************************************************************************/
template<class TYPE>
void MPI_CLASS::minReduce( const TYPE *send, TYPE *recv, int n ) const
{
    if ( d_size == 1 ) {
        for ( int i = 0; i < n; i++ )
            recv[i] = send[i];
        return;
    }
#ifdef AMP_USE_MPI
    PROFILE( "minReduce", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Allreduce( send, recv, n, type, MPI_MIN, d_comm );
    } else {
        throw std::logic_error( "minReduce is not valid for arbitrary types" );
    }
#endif
}
template<class TYPE>
void MPI_CLASS::minReduce( const TYPE *send, TYPE *recv, int n, int *rank ) const
{
    if ( rank == nullptr )
        return minReduce( send, recv, n );
    if ( d_size == 1 ) {
        for ( int i = 0; i < n; i++ ) {
            recv[i] = send[i];
            rank[i] = 0;
        }
        return;
    }
#ifdef AMP_USE_MPI
    using TYPE2 = typename TypeIntSelector<TYPE>::type;
    if constexpr ( !isMPItype<TYPE>() )
        throw std::logic_error( "minReduce with rank is not supported for arbitrary types" );
    PROFILE( "minReduce2", profile_level );
    auto send2 = new TYPE2[n];
    auto recv2 = new TYPE2[n];
    for ( int i = 0; i < n; i++ ) {
        send2[i].x = send[i];
        send2[i].i = d_rank;
    }
    auto type2 = MPI_getType<TYPE2>();
    MPI_Allreduce( send2, recv2, n, type2, MPI_MINLOC, d_comm );
    for ( int i = 0; i < n; i++ ) {
        recv[i] = recv2[i].x;
        rank[i] = recv2[i].i;
    }
    delete[] send2;
    delete[] recv2;
#endif
}
template<class TYPE>
TYPE MPI_CLASS::minReduce( const TYPE &value ) const
{
    if constexpr ( AMP::is_vector_v<TYPE> || std::is_array_v<TYPE> || AMP::is_array_v<TYPE> ) {
        auto recv = value;
        minReduce( value.data(), recv.data(), value.size() );
        return recv;
    } else if constexpr ( AMP::is_Array_v<TYPE> ) {
        auto recv = value;
        minReduce( value.data(), recv.data(), value.length() );
        return recv;
    } else {
        if ( d_size > 1 ) {
            TYPE tmp = value;
            minReduce( &value, &tmp, 1 );
            return tmp;
        } else {
            return value;
        }
    }
}
template<class TYPE>
void MPI_CLASS::minReduce( TYPE *x, int n ) const
{
    if ( d_size > 1 ) {
        auto tmp = new TYPE[n];
        for ( int i = 0; i < n; i++ )
            tmp[i] = x[i];
        minReduce( tmp, x, n );
        delete[] tmp;
    }
}
template<class TYPE>
void MPI_CLASS::minReduce( TYPE *x, int n, int *rank ) const
{
    if ( rank == nullptr )
        return minReduce( x, n );
    if ( d_size > 1 ) {
        auto tmp = new TYPE[n];
        for ( int i = 0; i < n; i++ )
            tmp[i] = x[i];
        minReduce( tmp, x, n, rank );
        delete[] tmp;
    } else {
        rank[0] = 0;
    }
}


/************************************************************************
 *  maxReduce                                                            *
 ************************************************************************/
template<class TYPE>
void MPI_CLASS::maxReduce( const TYPE *send, TYPE *recv, int n ) const
{
    if ( d_size == 1 ) {
        for ( int i = 0; i < n; i++ )
            recv[i] = send[i];
        return;
    }
#ifdef AMP_USE_MPI
    PROFILE( "maxReduce", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Allreduce( send, recv, n, type, MPI_MAX, d_comm );
    } else {
        throw std::logic_error( "maxReduce is not valid for arbitrary types" );
    }
#endif
}
template<class TYPE>
void MPI_CLASS::maxReduce( const TYPE *send, TYPE *recv, int n, int *rank ) const
{
    if ( rank == nullptr )
        return maxReduce( send, recv, n );
    if ( d_size == 1 ) {
        for ( int i = 0; i < n; i++ ) {
            recv[i] = send[i];
            rank[i] = 0;
        }
        return;
    }
#ifdef AMP_USE_MPI
    using TYPE2 = typename TypeIntSelector<TYPE>::type;
    if constexpr ( !isMPItype<TYPE>() )
        throw std::logic_error( "maxReduce with rank is not supported for arbitrary types" );
    PROFILE( "maxReduce2", profile_level );
    auto send2 = new TYPE2[n];
    auto recv2 = new TYPE2[n];
    for ( int i = 0; i < n; i++ ) {
        send2[i].x = send[i];
        send2[i].i = d_rank;
    }
    auto type2 = MPI_getType<TYPE2>();
    MPI_Allreduce( send2, recv2, n, type2, MPI_MAXLOC, d_comm );
    for ( int i = 0; i < n; i++ ) {
        recv[i] = recv2[i].x;
        rank[i] = recv2[i].i;
    }
    delete[] send2;
    delete[] recv2;
#endif
}
template<class TYPE>
TYPE MPI_CLASS::maxReduce( const TYPE &value ) const
{
    if constexpr ( AMP::is_vector_v<TYPE> || std::is_array_v<TYPE> || AMP::is_array_v<TYPE> ) {
        auto recv = value;
        maxReduce( value.data(), recv.data(), value.size() );
        return recv;
    } else if constexpr ( AMP::is_Array_v<TYPE> ) {
        auto recv = value;
        maxReduce( value.data(), recv.data(), value.length() );
        return recv;
    } else {
        if ( d_size > 1 ) {
            TYPE tmp = value;
            maxReduce( &value, &tmp, 1 );
            return tmp;
        } else {
            return value;
        }
    }
}
template<class TYPE>
void MPI_CLASS::maxReduce( TYPE *x, int n ) const
{
    if ( d_size > 1 ) {
        auto tmp = new TYPE[n];
        for ( int i = 0; i < n; i++ )
            tmp[i] = x[i];
        maxReduce( tmp, x, n );
        delete[] tmp;
    }
}
template<class TYPE>
void MPI_CLASS::maxReduce( TYPE *x, int n, int *rank ) const
{
    if ( rank == nullptr )
        return minReduce( x, n );
    if ( d_size > 1 ) {
        auto tmp = new TYPE[n];
        for ( int i = 0; i < n; i++ )
            tmp[i] = x[i];
        maxReduce( tmp, x, n, rank );
        delete[] tmp;
    } else {
        rank[0] = 0;
    }
}


/************************************************************************
 *  bcast                                                                *
 ************************************************************************/
#ifdef AMP_USE_MPI
template<class TYPE>
void MPI_CLASS::bcast( TYPE *x, int n, int root ) const
{
    if ( d_size == 1 || n == 0 )
        return;
    if ( root >= d_size )
        MPI_CLASS_ERROR( "root cannot be >= size in bcast" );
    PROFILE( "bcast", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Bcast( x, n, type, root, d_comm );
    } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
        MPI_Bcast( x, n * sizeof( TYPE ), MPI_CLASS_BYTE, root, d_comm );
    } else if constexpr ( is_vector_v<TYPE> ) {
        for ( int i = 0; i < n; i++ ) {
            size_t N = bcast( x[i].size(), root );
            x[i].resize( N );
            bcast( x[i].data(), N, root );
        }
    } else if constexpr ( std::is_pointer_v<TYPE> || is_shared_ptr_v<TYPE> ) {
        MPI_CLASS_INSIST( n == 1, "Broadcasting arrays of pointers of objects is not supported" );
        bcast( x->get(), 1, root );
    } else {
        size_t bytes   = 0;
        std::byte *buf = nullptr;
        if ( d_rank == root )
            std::tie( bytes, buf ) = packArray( x, n );
        MPI_Bcast( &bytes, 1, MPI_getType<size_t>(), root, d_comm );
        if ( buf == nullptr )
            buf = new std::byte[bytes];
        MPI_Bcast( buf, bytes, MPI_CLASS_BYTE, root, d_comm );
        if ( d_rank != root )
            unpackArray( x, n, buf );
        delete[] buf;
    }
}
#else
template<class TYPE>
void MPI_CLASS::bcast( TYPE *, int, int ) const
{
}
#endif
template<class TYPE>
TYPE MPI_CLASS::bcast( const TYPE &x, int root ) const
{
    if constexpr ( std::is_pointer_v<TYPE> || is_shared_ptr_v<TYPE> ) {
        using TYPE2 = typename AMP::remove_cvref_t<decltype( *( x.get() ) )>;
        auto y      = std::make_shared<TYPE2>( *x );
        if ( d_size > 1 )
            bcast( y.get(), 1, root );
        return y;
    } else {
        auto y = x;
        if ( d_size > 1 )
            bcast( &y, 1, root );
        return y;
    }
}


/************************************************************************
 *  send                                                                 *
 ************************************************************************/
template<class TYPE>
std::vector<TYPE *> getPointers( const std::shared_ptr<TYPE> *x, size_t N )
{
    std::vector<TYPE *> y( N, nullptr );
    for ( size_t i = 0; i < y.size(); i++ )
        y[i] = x[i].get();
    return y;
}
template<class TYPE>
void MPI_CLASS::send( const TYPE &buf, int recv_proc, int tag ) const
{
    send( &buf, 1, recv_proc, tag );
}
inline void MPI_Send2(
    const void *buf, int length, MPI_CLASS::Datatype type, int proc, int tag, MPI_CLASS::Comm comm )
{
#ifdef AMP_USE_MPI
    MPI_Send( buf, length, type, proc, tag, comm );
#else
    size_t bytes = length * MPI_getTypeSize( type );
    MPI_CLASS( comm ).sendBytes( buf, bytes, proc, tag );
#endif
}
template<class TYPE>
void MPI_CLASS::send( const TYPE *buf, int length, int recv_proc, int tag ) const
{
    // Set the tag to 0 if it is < 0
    tag = ( tag >= 0 ) ? tag : 0;
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    // Send the data
    PROFILE( "send", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Send2( buf, length, type, recv_proc, tag, d_comm );
    } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
        size_t bytes = length * sizeof( TYPE );
        MPI_Send2( buf, bytes, MPI_CLASS_BYTE, recv_proc, tag, d_comm );
    } else if constexpr ( std::is_pointer_v<TYPE> || is_shared_ptr_v<TYPE> ) {
        send( getPointers( buf, length ).data(), length, recv_proc, tag );
    } else {
        auto [bytes, buf2] = packArray( buf, length );
        MPI_Send2( buf2, bytes, MPI_CLASS_BYTE, recv_proc, tag, d_comm );
        delete[] buf2;
    }
}


/************************************************************************
 *  Isend                                                                *
 ************************************************************************/
template<class TYPE>
MPI_CLASS::Request MPI_CLASS::Isend( const TYPE &buf, int recv_proc, int tag ) const
{
    return Isend( &buf, 1, recv_proc, tag );
}
inline void MPI_Isend2( const void *buf,
                        int length,
                        MPI_CLASS::Datatype type,
                        int proc,
                        int tag,
                        MPI_CLASS::Comm comm,
                        MPI_CLASS::Request2 *request )
{
#ifdef AMP_USE_MPI
    MPI_Isend( buf, length, type, proc, tag, comm, request );
#else
    size_t bytes = length * MPI_getTypeSize( type );
    *request = MPI_CLASS( comm ).IsendBytes( buf, bytes, proc, tag );
#endif
}
template<class TYPE>
MPI_CLASS::Request MPI_CLASS::Isend( const TYPE *buf, int length, int recv_proc, int tag ) const
{
    // Set the tag to 0 if it is < 0
    tag = ( tag >= 0 ) ? tag : 0;
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    // Send the data
    PROFILE( "isend", profile_level );
    MPI_CLASS::Request request;
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Isend2( buf, length, type, recv_proc, tag, d_comm, request.get() );
    } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
        size_t bytes = length * sizeof( TYPE );
        MPI_Isend2( buf, bytes, MPI_CLASS_BYTE, recv_proc, tag, d_comm, request.get() );
    } else if constexpr ( std::is_pointer_v<TYPE> || is_shared_ptr_v<TYPE> ) {
        request = Isend( buf->get(), 1, recv_proc, tag );
    } else {
        MPI_CLASS::Request2 request2;
        auto [bytes, buf2] = packArray( buf, length );
        MPI_Isend2( buf2, bytes, MPI_CLASS_BYTE, recv_proc, tag, d_comm, &request2 );
        request = Request( request2, std::shared_ptr<std::byte>( buf2 ) );
    }
    return request;
}


/************************************************************************
 *  recv                                                                 *
 ************************************************************************/
template<class TYPE>
TYPE MPI_CLASS::recv( int send_proc, int tag ) const
{
    TYPE data;
    int length = 1;
    recv<TYPE>( &data, length, send_proc, false, tag );
    return data;
}
inline void MPI_Recv2(
    void *buf, int length, MPI_CLASS::Datatype type, int proc, int tag, MPI_CLASS::Comm comm )
{
#ifdef AMP_USE_MPI
    MPI_Recv( buf, length, type, proc, tag, comm, MPI_STATUS_IGNORE );
#else
    size_t bytes = length * MPI_getTypeSize( type );
    MPI_CLASS( comm ).recvBytes( buf, bytes, proc, tag );
#endif
}
template<class TYPE>
void MPI_CLASS::recv( TYPE *buf, int &length, int send_proc, bool get_length, int tag ) const
{
    // Set the tag to 0 if it is < 0
    tag = ( tag >= 0 ) ? tag : 0;
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    PROFILE( "recv", profile_level );
    // Get the receive length if necessary
    if ( get_length ) {
        int bytes       = probe( send_proc, tag );
        int recv_length = bytes / sizeof( TYPE );
        MPI_CLASS_INSIST( length >= recv_length, "Recived length is larger than allocated array" );
        length = recv_length;
    }
    // Receive the data
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Recv2( buf, length, type, send_proc, tag, d_comm );
    } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
        size_t bytes = length * sizeof( TYPE );
        MPI_Recv2( buf, bytes, MPI_CLASS_BYTE, send_proc, tag, d_comm );
    } else if constexpr ( std::is_pointer_v<TYPE> || is_shared_ptr_v<TYPE> ) {
        MPI_CLASS_INSIST( length == 1 && get_length == false,
                          "Receiving arrays of pointers of objects is not supported" );
        using TYPE2 = typename AMP::remove_cvref_t<decltype( *( buf->get() ) )>;
        buf->reset( new TYPE2() );
        recv( buf->get(), length, send_proc, false, tag );
    } else {
        size_t bytes = probe( send_proc, tag );
        auto buf2    = new std::byte[bytes];
        MPI_Recv2( buf2, bytes, MPI_CLASS_BYTE, send_proc, tag, d_comm );
        size_t N = unpackArray( buf, length, buf2 );
        delete[] buf2;
        MPI_CLASS_ASSERT( N == bytes );
    }
}
template<class TYPE>
void MPI_CLASS::recv( TYPE *buf, int length, int send_proc, int tag ) const
{
    recv<TYPE>( buf, length, send_proc, false, tag );
}


/************************************************************************
 *  Irecv                                                                *
 ************************************************************************/
template<class TYPE>
MPI_CLASS::Request MPI_CLASS::Irecv( TYPE &data, int send_proc, int tag ) const
{
    return Irecv<TYPE>( &data, 1, send_proc, tag );
}
inline void MPI_Irecv2( void *buf,
                        int length,
                        MPI_CLASS::Datatype type,
                        int proc,
                        int tag,
                        MPI_CLASS::Comm comm,
                        MPI_CLASS::Request2 *request )
{
#ifdef AMP_USE_MPI
    MPI_Irecv( buf, length, type, proc, tag, comm, request );
#else
    size_t bytes = length * MPI_getTypeSize( type );
    *request = MPI_CLASS( comm ).IrecvBytes( buf, bytes, proc, tag );
#endif
}
template<class TYPE>
MPI_CLASS::Request MPI_CLASS::Irecv( TYPE *buf, int length, int send_proc, int tag ) const
{
    MPI_CLASS_INSIST( tag <= d_maxTag, "Maximum tag value exceeded" );
    MPI_CLASS_INSIST( tag >= 0, "tag must be >= 0" );
    MPI_CLASS::Request request;
    PROFILE( "Irecv", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Irecv2( buf, length, type, send_proc, tag, d_comm, request.get() );
    } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
        size_t bytes = length * sizeof( TYPE );
        MPI_Irecv2( buf, bytes, MPI_CLASS_BYTE, send_proc, tag, d_comm, request.get() );
    } else if constexpr ( std::is_pointer_v<TYPE> || is_shared_ptr_v<TYPE> ) {
        MPI_CLASS_INSIST( length == 1, "Receiving arrays of pointers of objects is not supported" );
        using TYPE2 = typename AMP::remove_cvref_t<decltype( *( buf->get() ) )>;
        buf->reset( new TYPE2() );
        request = Irecv( buf->get(), 1, send_proc, tag );
    } else {
        NULL_USE( buf );
        NULL_USE( length );
        NULL_USE( send_proc );
        NULL_USE( tag );
        MPI_CLASS_ERROR( "Arbitrary types are not supported yet" );
    }
    return request;
}


/************************************************************************
 *  sendrecv                                                             *
 ************************************************************************/
template<class TYPE>
void MPI_CLASS::sendrecv( const TYPE *sendbuf,
                          int sendcount,
                          int dest,
                          int sendtag,
                          TYPE *recvbuf,
                          int recvcount,
                          int source,
                          int recvtag ) const
{
    if ( getSize() <= 1 ) {
        MPI_CLASS_ASSERT( dest == 0 );
        MPI_CLASS_ASSERT( source == 0 );
        MPI_CLASS_ASSERT( sendcount <= recvcount );
        MPI_CLASS_ASSERT( sendtag == recvtag );
        for ( int i = 0; i < sendcount; i++ )
            recvbuf[i] = sendbuf[i];
        return;
    }
#ifdef AMP_USE_MPI
    PROFILE( "sendrecv", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Sendrecv( sendbuf,
                      sendcount,
                      type,
                      dest,
                      sendtag,
                      recvbuf,
                      recvcount,
                      type,
                      source,
                      recvtag,
                      d_comm,
                      MPI_STATUS_IGNORE );
    } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
        size_t sendbytes = sendcount * sizeof( TYPE );
        size_t recvbytes = recvcount * sizeof( TYPE );
        MPI_Sendrecv( sendbuf,
                      sendbytes,
                      MPI_CLASS_BYTE,
                      dest,
                      sendtag,
                      recvbuf,
                      recvbytes,
                      MPI_CLASS_BYTE,
                      source,
                      recvtag,
                      d_comm,
                      MPI_STATUS_IGNORE );
    } else {
        MPI_CLASS_ERROR_TYPE( "Arbitrary types are not supported" );
    }
#endif
}


/************************************************************************
 *  gather                                                            *
 ************************************************************************/
template<class TYPE>
std::vector<TYPE> MPI_CLASS::gather( const TYPE &x, int root ) const
{
    if ( d_size == 1 )
        return { x };
    std::vector<int> recv_cnt, recv_disp;
    if ( d_rank == root ) {
        recv_cnt.resize( d_size, 1 );
        recv_disp.resize( d_size, 0 );
        for ( int i = 0; i < d_size; i++ )
            recv_disp[i] = i;
    }
    std::vector<TYPE> y( d_size );
    gather( &x, 1, y.data(), recv_cnt.data(), recv_disp.data(), root );
    return y;
}
template<class TYPE>
std::vector<TYPE> MPI_CLASS::gather( const std::vector<TYPE> &x, int root ) const
{
    if ( d_size == 1 )
        return x;
    auto recv_cnt = gather<int>( x.size(), root );
    std::vector<int> recv_disp( recv_cnt.size(), 0 );
    for ( size_t i = 1; i < recv_cnt.size(); i++ )
        recv_disp[i] = recv_disp[i - 1] + recv_cnt[i - 1];
    std::vector<TYPE> y;
    if ( d_rank == root ) {
        size_t N = 0;
        for ( auto s : recv_cnt )
            N += s;
        y.resize( N );
    }
    gather<TYPE>( x.data(), x.size(), y.data(), recv_cnt.data(), recv_disp.data(), root );
    return y;
}
template<class TYPE>
void MPI_CLASS::gather( const TYPE *sendbuf,
                        int sendcount,
                        TYPE *recvbuf,
                        const int *recv_cnt,
                        const int *recv_disp,
                        int root ) const
{
    NULL_USE( root );
    if ( d_size == 1 ) {
        MPI_CLASS_ASSERT( sendcount == recv_cnt[0] );
        for ( int i = 0; i < sendcount; i++ )
            recvbuf[i + recv_disp[0]] = sendbuf[i];
        return;
    }
#ifdef AMP_USE_MPI
    PROFILE( "gather", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Gatherv( sendbuf, sendcount, type, recvbuf, recv_cnt, recv_disp, type, root, d_comm );
    } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
        size_t N = sizeof( TYPE );
        std::vector<int> recv_cnt2( d_size );
        std::vector<int> recv_disp2( d_size );
        for ( int i = 0; i < d_size; i++ ) {
            recv_cnt2[i]  = N * recv_cnt[i];
            recv_disp2[i] = N * recv_disp[i];
        }
        MPI_Gatherv( sendbuf,
                     N * sendcount,
                     MPI_CLASS_BYTE,
                     recvbuf,
                     recv_cnt2.data(),
                     recv_disp2.data(),
                     MPI_CLASS_BYTE,
                     root,
                     d_comm );
    } else {
        STATIC_ERROR( "Arbitrary types are not supported yet" );
    }
#endif
}


/************************************************************************
 *  allGather                                                            *
 ************************************************************************/
template<class TYPE>
std::vector<TYPE> MPI_CLASS::allGather( const TYPE &x ) const
{
    if ( getSize() <= 1 )
        return { x };
    std::vector<TYPE> data( getSize() );
    allGather( x, data.data() );
    return data;
}
template<class TYPE>
std::vector<TYPE> MPI_CLASS::allGather( const std::vector<TYPE> &x ) const
{
    if ( getSize() <= 1 )
        return x;
    std::vector<int> count = allGather<int>( x.size() );
    std::vector<int> disp( getSize(), 0 );
    size_t N = count[0];
    for ( size_t i = 1; i < count.size(); i++ ) {
        disp[i] = disp[i - 1] + count[i - 1];
        N += count[i];
    }
    if ( N == 0 )
        return {};
    std::vector<TYPE> data( N );
    allGather<TYPE>( x.data(), x.size(), data.data(), count.data(), disp.data(), true );
    return data;
}
// Default instantiation of MPI_CLASS::allGather
template<class TYPE>
void MPI_CLASS::allGather( const TYPE &x, TYPE *y ) const
{
    NULL_USE( x );
    NULL_USE( y );
    if constexpr ( !std::is_copy_assignable_v<TYPE> ) {
        STATIC_ERROR( "allGather is not valid for non-assignable objects" );
    } else {
        MPI_CLASS_ASSERT( y );
        if ( d_size == 1 ) {
            y[0] = x;
            return;
        }
#ifdef AMP_USE_MPI
        PROFILE( "allGather", profile_level );
        if constexpr ( isMPItype<TYPE>() ) {
            auto type = MPI_getType<TYPE>();
            MPI_Allgather( &x, 1, type, y, 1, type, d_comm );
        } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
            size_t N = sizeof( TYPE );
            MPI_Allgather( &x, N, MPI_CLASS_BYTE, y, N, MPI_CLASS_BYTE, d_comm );
        } else if constexpr ( std::is_pointer_v<TYPE> || is_shared_ptr_v<TYPE> ) {
            STATIC_ERROR( "gathering array of pointers is not yet supported" );
        } else {
            auto [send_cnt, send_data] = packArray( &x, 1 );
            auto recv_cnt              = allGather<int>( send_cnt );
            std::vector<int> recv_disp( recv_cnt.size(), 0 );
            size_t bytes = recv_cnt[0];
            for ( size_t i = 1; i < recv_cnt.size(); i++ ) {
                bytes += recv_cnt[i];
                recv_disp[i] = recv_disp[i - 1] + recv_cnt[i - 1];
            }
            auto recv_data = new std::byte[bytes];
            allGather( send_data, send_cnt, recv_data, recv_cnt.data(), recv_disp.data(), true );
            size_t N = unpackArray( y, recv_cnt.size(), recv_data );
            MPI_CLASS_ASSERT( N == bytes );
        }
#endif
    }
}
template<class TYPE>
int MPI_CLASS::allGather( const TYPE *send_data,
                          int send_cnt,
                          TYPE *recv_data,
                          int *recv_cnt0,
                          int *recv_disp0,
                          bool known_recv ) const
{
    NULL_USE( send_data );
    NULL_USE( send_cnt );
    NULL_USE( recv_data );
    NULL_USE( recv_cnt0 );
    NULL_USE( recv_disp0 );
    NULL_USE( known_recv );
    if constexpr ( !std::is_copy_assignable_v<TYPE> ) {
        STATIC_ERROR( "allGather is not valid for non-assignable objects" );
        return 0;
    } else if constexpr ( std::is_pointer_v<TYPE> || is_shared_ptr_v<TYPE> ) {
        STATIC_ERROR( "allGather is not valid for pointers yet" );
    } else {
        if ( d_size == 1 ) {
            if ( !known_recv ) {
                for ( int i = 0; i < send_cnt; i++ )
                    recv_data[i] = send_data[i];
                if ( recv_cnt0 )
                    *recv_cnt0 = send_cnt;
                if ( recv_disp0 )
                    *recv_disp0 = 0;
            } else {
                for ( int i = 0; i < send_cnt; i++ )
                    recv_data[i + recv_disp0[0]] = send_data[i];
            }
            return send_cnt;
        }
#ifdef AMP_USE_MPI
        PROFILE( "allGather2", profile_level );
        // Get the sizes of the received data (if necessary)
        int *recv_cnt  = recv_cnt0;
        int *recv_disp = recv_disp0;
        if ( !known_recv ) {
            if ( !recv_cnt0 )
                recv_cnt = new int[d_size];
            if ( !recv_disp0 )
                recv_disp = new int[d_size];
            allGather( send_cnt, recv_cnt );
            recv_disp[0] = 0;
            for ( int i = 1; i < d_size; i++ )
                recv_disp[i] = recv_disp[i - 1] + recv_cnt[i - 1];
        }
        int N_recv = 0;
        for ( int i = 0; i < d_size; i++ )
            N_recv += recv_cnt[i];
        // Send/recv the data
        if constexpr ( isMPItype<TYPE>() ) {
            auto type = MPI_getType<TYPE>();
            MPI_Allgatherv(
                send_data, send_cnt, type, recv_data, recv_cnt, recv_disp, type, d_comm );
        } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
            auto cnt  = new int[d_size];
            auto disp = new int[d_size];
            for ( int i = 0; i < d_size; i++ ) {
                cnt[i]  = sizeof( TYPE ) * recv_cnt[i];
                disp[i] = sizeof( TYPE ) * recv_disp[i];
            }
            int N = send_cnt * sizeof( TYPE );
            MPI_Allgatherv(
                send_data, N, MPI_CLASS_BYTE, recv_data, cnt, disp, MPI_CLASS_BYTE, d_comm );
            delete[] cnt;
            delete[] disp;
        } else if constexpr ( std::is_pointer_v<TYPE> || is_shared_ptr_v<TYPE> ) {
            STATIC_ERROR( "Pointers are not supported yet" );
        } else {
            auto [send_cnt2, send_data2] = packArray( send_data, send_cnt );
            auto recv_cnt2               = allGather<int>( send_cnt2 );
            std::vector<int> recv_disp2( recv_cnt2.size(), 0 );
            size_t bytes = recv_cnt2[0];
            for ( size_t i = 1; i < recv_cnt2.size(); i++ ) {
                bytes += recv_cnt2[i];
                recv_disp2[i] = recv_disp2[i - 1] + recv_cnt2[i - 1];
            }
            auto recv_data2 = new std::byte[bytes];
            allGather(
                send_data2, send_cnt2, recv_data2, recv_cnt2.data(), recv_disp2.data(), true );
            size_t N = unpackArray( recv_data, N_recv, recv_data2 );
            MPI_CLASS_ASSERT( N == bytes );
            delete[] recv_data2;
        }
        // Delete any temporary memory
        if ( !recv_cnt0 )
            delete[] recv_cnt;
        if ( !recv_disp0 )
            delete[] recv_disp;
        return N_recv;
#endif
    }
    return 0;
}


/************************************************************************
 *  setGather                                                            *
 ************************************************************************/
template<class TYPE>
void MPI_CLASS::setGather( std::set<TYPE> &set ) const
{
    std::vector<TYPE> send_buf( set.begin(), set.end() );
    std::vector<int> recv_cnt( d_size, 0 );
    allGather<int>( (int) send_buf.size(), &recv_cnt[0] );
    std::vector<int> recv_disp( d_size, 0 );
    for ( int i = 1; i < d_size; i++ )
        recv_disp[i] = recv_disp[i - 1] + recv_cnt[i - 1];
    size_t N_recv_tot = 0;
    for ( int i = 0; i < d_size; i++ )
        N_recv_tot += recv_cnt[i];
    if ( N_recv_tot == 0 )
        return;
    std::vector<TYPE> recv_buf( N_recv_tot );
    TYPE *send_data = nullptr;
    if ( send_buf.size() > 0 ) {
        send_data = &send_buf[0];
    }
    TYPE *recv_data = &recv_buf[0];
    allGather<TYPE>(
        send_data, (int) send_buf.size(), recv_data, &recv_cnt[0], &recv_disp[0], true );
    for ( size_t i = 0; i < recv_buf.size(); i++ )
        set.insert( recv_buf[i] );
}


/************************************************************************
 *  mapGather                                                            *
 ************************************************************************/
template<class KEY, class DATA>
void MPI_CLASS::mapGather( std::map<KEY, DATA> &map ) const
{
    std::vector<KEY> send_id;
    std::vector<DATA> send_data;
    send_id.reserve( map.size() );
    send_data.reserve( map.size() );
    for ( auto it = map.begin(); it != map.end(); ++it ) {
        send_id.push_back( it->first );
        send_data.push_back( it->second );
    }
    int send_size = (int) send_id.size();
    std::vector<int> recv_cnt( d_size, 0 );
    allGather<int>( send_size, &recv_cnt[0] );
    std::vector<int> recv_disp( d_size, 0 );
    for ( int i = 1; i < d_size; i++ )
        recv_disp[i] = recv_disp[i - 1] + recv_cnt[i - 1];
    size_t N_recv_tot = 0;
    for ( int i = 0; i < d_size; i++ )
        N_recv_tot += recv_cnt[i];
    if ( N_recv_tot == 0 )
        return;
    std::vector<KEY> recv_id( N_recv_tot );
    std::vector<DATA> recv_data( N_recv_tot );
    KEY *send_data1  = nullptr;
    DATA *send_data2 = nullptr;
    if ( send_id.size() > 0 ) {
        send_data1 = &send_id[0];
        send_data2 = &send_data[0];
    }
    allGather<KEY>( send_data1, send_size, &recv_id[0], &recv_cnt[0], &recv_disp[0], true );
    allGather<DATA>( send_data2, send_size, &recv_data[0], &recv_cnt[0], &recv_disp[0], true );
    map = std::map<KEY, DATA>();
    for ( size_t i = 0; i < N_recv_tot; i++ )
        map.insert( std::pair<KEY, DATA>( recv_id[i], recv_data[i] ) );
}


/************************************************************************
 *  sumScan                                                              *
 ************************************************************************/
template<class TYPE>
TYPE MPI_CLASS::sumScan( const TYPE &x ) const
{
    TYPE y = x;
    sumScan<TYPE>( &x, &y, 1 );
    return y;
}
template<class TYPE>
void MPI_CLASS::sumScan( const TYPE *x, TYPE *y, int n ) const
{
    if ( d_size == 1 ) {
        for ( int i = 0; i < n; i++ )
            y[i] = x[i];
        return;
    }
#ifdef AMP_USE_MPI
    PROFILE( "sumScan", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_CLASS_ASSERT( d_comm != MPI_COMM_NULL );
        MPI_Scan( x, y, n, type, MPI_SUM, d_comm );
    } else {
        throw std::logic_error( "Default instantion of sumScan in parallel is not supported" );
    }
#endif
}


/************************************************************************
 *  minScan                                                              *
 ************************************************************************/
template<class TYPE>
TYPE MPI_CLASS::minScan( const TYPE &x ) const
{
    TYPE y = x;
    minScan<TYPE>( &x, &y, 1 );
    return y;
}
template<class TYPE>
void MPI_CLASS::minScan( const TYPE *x, TYPE *y, int n ) const
{
    if ( d_size == 1 ) {
        for ( int i = 0; i < n; i++ )
            y[i] = x[i];
        return;
    }
#ifdef AMP_USE_MPI
    PROFILE( "minScan", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_CLASS_ASSERT( d_comm != MPI_COMM_NULL );
        MPI_Scan( x, y, n, type, MPI_MIN, d_comm );
    } else {
        throw std::logic_error( "Default instantion of minScan in parallel is not supported" );
    }
#endif
}


/************************************************************************
 *  maxScan                                                              *
 ************************************************************************/
template<class TYPE>
TYPE MPI_CLASS::maxScan( const TYPE &x ) const
{
    TYPE y = x;
    maxScan<TYPE>( &x, &y, 1 );
    return y;
}
template<class TYPE>
void MPI_CLASS::maxScan( const TYPE *x, TYPE *y, int n ) const
{
    if ( d_size == 1 ) {
        for ( int i = 0; i < n; i++ )
            y[i] = x[i];
        return;
    }
#ifdef AMP_USE_MPI
    PROFILE( "maxScan", profile_level );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_CLASS_ASSERT( d_comm != MPI_COMM_NULL );
        MPI_Scan( x, y, n, type, MPI_MAX, d_comm );
    } else {
        throw std::logic_error( "Default instantion of maxScan in parallel is not supported" );
    }
#endif
}


/************************************************************************
 *  allToAll                                                             *
 ************************************************************************/
template<class TYPE>
void MPI_CLASS::allToAll( int n, const TYPE *send, TYPE *recv ) const
{
    if ( d_size == 1 ) {
        for ( int i = 0; i < n; i++ )
            recv[i] = send[i];
        return;
    }
#ifdef AMP_USE_MPI
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Alltoall( send, n, type, recv, n, type, d_comm );
    } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
        int N = n * sizeof( TYPE );
        MPI_Alltoall( send, N, MPI_CLASS_BYTE, recv, N, MPI_CLASS_BYTE, d_comm );
    } else {
        std::vector<int> count( d_size, n );
        std::vector<int> disp( d_size, 0 );
        for ( int i = 0; i < d_size; i++ )
            disp[i] = i * n;
        allToAll( send, count.data(), disp.data(), recv, count.data(), disp.data(), true );
    }
#endif
}


/************************************************************************
 *  allToAll                                                             *
 ************************************************************************/
template<class TYPE>
int MPI_CLASS::allToAll( const TYPE *send_data,
                         const int send_cnt[],
                         const int send_disp[],
                         TYPE *recv_data,
                         int *recv_cnt,
                         int *recv_disp,
                         bool known_recv ) const
{
    if ( d_size == 1 ) {
        // Special case for single-processor communicators
        if ( known_recv ) {
            if ( recv_cnt[0] != send_cnt[0] && send_cnt[0] > 0 )
                MPI_CLASS_ERROR( "Single processor send/recv are different sizes" );
        } else {
            if ( recv_cnt != nullptr )
                recv_cnt[0] = send_cnt[0];
            if ( recv_disp != nullptr )
                recv_disp[0] = send_disp[0];
        }
        for ( int i = 0; i < send_cnt[0]; i++ )
            recv_data[i + recv_disp[0]] = send_data[i + send_disp[0]];
        return send_cnt[0];
    }
#ifdef AMP_USE_MPI
    PROFILE( "allToAll", profile_level );
    // Get the recv sizes if needed
    int *recv_cnt2  = recv_cnt;
    int *recv_disp2 = recv_disp;
    if ( !known_recv ) {
        // Communicate the size we will be recieving from each processor
        if ( !recv_cnt )
            recv_cnt2 = new int[d_size];
        if ( !recv_disp )
            recv_disp2 = new int[d_size];
        calcAllToAllDisp( send_cnt, nullptr, recv_cnt2, recv_disp2 );
    }
    // Perform the communication
    MPI_CLASS_ASSERT( recv_cnt2 && recv_disp2 );
    if constexpr ( isMPItype<TYPE>() ) {
        auto type = MPI_getType<TYPE>();
        MPI_Alltoallv(
            send_data, send_cnt, send_disp, type, recv_data, recv_cnt2, recv_disp2, type, d_comm );
    } else if constexpr ( is_mpi_copyable_v<TYPE> ) {
        auto send_cnt3  = new int[d_size];
        auto send_disp3 = new int[d_size];
        auto recv_cnt3  = new int[d_size];
        auto recv_disp3 = new int[d_size];
        for ( int i = 0; i < d_size; i++ ) {
            send_cnt3[i]  = sizeof( TYPE ) * send_cnt[i];
            send_disp3[i] = sizeof( TYPE ) * send_disp[i];
            recv_cnt3[i]  = sizeof( TYPE ) * recv_cnt2[i];
            recv_disp3[i] = sizeof( TYPE ) * recv_disp2[i];
        }
        MPI_Alltoallv( send_data,
                       send_cnt3,
                       send_disp3,
                       MPI_CLASS_BYTE,
                       recv_data,
                       recv_cnt3,
                       recv_disp3,
                       MPI_CLASS_BYTE,
                       d_comm );
        delete[] send_cnt3;
        delete[] send_disp3;
        delete[] recv_cnt3;
        delete[] recv_disp3;
    } else {
        MPI_CLASS_ERROR( "Not finished" );
    }
    int N_received = 0;
    for ( int i = 0; i < d_size; i++ )
        N_received += recv_cnt2[i];
    // Delete the temporary memory if needed
    if ( !recv_cnt )
        delete[] recv_cnt2;
    if ( !recv_disp )
        delete[] recv_disp2;
    return N_received;
#else
    return 0;
#endif
}


} // namespace AMP


#endif
