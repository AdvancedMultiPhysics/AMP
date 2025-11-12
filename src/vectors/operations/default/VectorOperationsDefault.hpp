#ifndef included_AMP_VectorOperationsDefault_hpp
#define included_AMP_VectorOperationsDefault_hpp

#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Backend.h"
#include "AMP/utils/Memory.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"

#include <random>
#include <string_view>

namespace AMP::LinearAlgebra {


extern template class VectorOperationsDefault<double>; // Suppresses implicit instantiation below --
extern template class VectorOperationsDefault<float>;  // Suppresses implicit instantiation below --

static bool allDefaultDataType( const VectorData &x )
{
    constexpr std::string_view type1( "VectorDataDefault", 17 );
    constexpr std::string_view type2( "ArrayVector", 11 );
    const auto xtype               = x.VectorDataName();
    const std::string_view xtype_s = xtype;
    return ( xtype_s.compare( 0, 17, type1 ) == 0 ) || ( xtype_s.compare( 0, 11, type2 ) == 0 );
}
static bool allDefaultDataType( const VectorData &x, const VectorData &y )
{
    constexpr std::string_view type1( "VectorDataDefault", 17 );
    constexpr std::string_view type2( "ArrayVector", 11 );
    const auto xtype               = x.VectorDataName();
    const auto ytype               = y.VectorDataName();
    const std::string_view xtype_s = xtype;
    const std::string_view ytype_s = ytype;
    return ( xtype_s.compare( ytype_s ) == 0 && (( xtype_s.compare( 0, 17, type1 ) == 0 )|| ( xtype_s.compare( 0, 11, type2 ) == 0 ));
}
static bool allDefaultDataType( const VectorData &x, const VectorData &y, VectorData &z )
{
    constexpr std::string_view type1( "VectorDataDefault", 17 );
    constexpr std::string_view type2( "ArrayVector", 11 );
    const auto xtype               = x.VectorDataName();
    const auto ytype               = y.VectorDataName();
    const auto ztype               = z.VectorDataName();
    const std::string_view xtype_s = xtype;
    const std::string_view ytype_s = ytype;
    const std::string_view ztype_s = ztype;

    return ( xtype_s.compare( ytype_s ) == 0 && ytype_s.compare( ztype_s ) == 0 &&
                 ( xtype_s.compare( 0, 17, type ) == 0 ) ||
             ( xtype_s.compare( 0, 11, type2 ) == 0 ) );
}

/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
template<typename TYPE>
std::shared_ptr<VectorOperations> VectorOperationsDefault<TYPE>::cloneOperations() const
{
    auto ptr = std::make_shared<VectorOperationsDefault<TYPE>>();
    return ptr;
}


/****************************************************************
 * Get the class type                                            *
 ****************************************************************/
template<typename TYPE>
std::string VectorOperationsDefault<TYPE>::VectorOpName() const
{
    constexpr typeID id = getTypeID<TYPE>();
    return "VectorOperationsDefault<" + std::string( id.name ) + ">";
}


//**********************************************************************
// Static functions that operate on VectorData objects

template<typename TYPE>
void VectorOperationsDefault<TYPE>::zero( VectorData &x )
{
    AMP_DEBUG_ASSERT( x.isType<TYPE>() );
    static_assert( std::is_trivially_copyable_v<TYPE> );
    size_t N_blocks = x.numberOfDataBlocks();
    for ( size_t i = 0; i < N_blocks; i++ ) {
        auto data = x.getRawDataBlock<TYPE>( i );
        auto N    = x.sizeOfDataBlock( i );
        AMP::Utilities::zero( data, N * sizeof( TYPE ) );
    }
    x.fillGhosts( 0 );
    // Override the status state since we set the ghost values
    x.setUpdateStatus( UpdateState::UNCHANGED );
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::setToScalar( const Scalar &alpha_in, VectorData &x )
{
    auto const alpha = alpha_in.get<TYPE>();

    if ( allDefaultDataType( x ) ) {
        auto xdata   = x.getRawDataBlock<TYPE>();
        const auto N = x.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            xdata[i] = alpha;
        }
    } else {
        auto curMe = x.begin<TYPE>();
        auto last  = x.end<TYPE>();
        while ( curMe != last ) {
            *curMe = alpha;
            ++curMe;
        }
    }
    x.fillGhosts( alpha_in );
    // Override the status state since we set the ghost values
    x.setUpdateStatus( UpdateState::UNCHANGED );
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::setRandomValues( VectorData &x )
{
    std::random_device rd;
    std::mt19937 gen( rd() );
    if constexpr ( std::is_floating_point_v<TYPE> ) {
        std::uniform_real_distribution<TYPE> dis( 0, 1 );
        auto y    = x.begin<TYPE>();
        auto last = x.end<TYPE>();
        while ( y != last ) {
            *y = dis( gen );
            ++y;
        }
    } else if constexpr ( std::is_integral_v<TYPE> ) {
        std::uniform_int_distribution<TYPE> dis;
        auto y    = x.begin<TYPE>();
        auto last = x.end<TYPE>();
        while ( y != last ) {
            *y = dis( gen );
            ++y;
        }
    } else {
        AMP_ERROR( "Not finished" );
    }
    // Call makeConsistent to leave the vector in a consistent state
    x.makeConsistent( ScatterType::CONSISTENT_SET );
}


template<typename TYPE>
void VectorOperationsDefault<TYPE>::copy( const VectorData &x, VectorData &y )
{
    AMP_DEBUG_ASSERT( y.getLocalSize() == x.getLocalSize() );
    AMP_DEBUG_ASSERT( y.isType<TYPE>() );
    size_t N_blocks   = y.numberOfDataBlocks();
    size_t N_blocks_x = x.numberOfDataBlocks();
    if ( N_blocks_x == N_blocks && x.isType<TYPE>() ) {
        static_assert( std::is_trivially_copyable_v<TYPE> );
        for ( size_t i = 0; i < N_blocks; i++ ) {
            size_t N   = y.sizeOfDataBlock( i );
            auto xdata = x.getRawDataBlock<TYPE>( i );
            auto ydata = y.getRawDataBlock<TYPE>( i );
            AMP_ASSERT( xdata && ydata && N == x.sizeOfDataBlock( i ) );
            AMP::Utilities::memcpy( ydata, xdata, N * sizeof( TYPE ) );
        }
        y.copyGhostValues( x );
    } else if ( N_blocks_x == N_blocks && !x.isType<TYPE>() ) {
        for ( size_t i = 0; i < N_blocks; i++ ) {
            size_t N   = y.sizeOfDataBlock( i );
            auto ydata = y.getRawDataBlock<TYPE>( i );
            auto type  = x.getType( i );
            AMP_ASSERT( N == x.sizeOfDataBlock( i ) );
            if ( type == getTypeID<float>() ) {
                auto xdata = x.getRawDataBlock<float>( i );
                AMP::Utilities::copy( N, xdata, ydata );
            } else if ( type == getTypeID<double>() ) {
                auto xdata = x.getRawDataBlock<double>( i );
                AMP::Utilities::copy( N, xdata, ydata );
            } else {
                AMP_ERROR( "copy only implemented for float or doubles" );
            }
        }
        y.makeConsistent( ScatterType::CONSISTENT_SET );
    } else {
        std::copy( x.begin<TYPE>(), x.end<TYPE>(), y.begin<TYPE>() );
        y.copyGhostValues( x );
    }
}
template<typename TYPE>
void VectorOperationsDefault<TYPE>::copyCast( const VectorData &x, VectorData &y )
{
    VectorOperationsDefault<TYPE>::copy( x, y );
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::scale( const Scalar &alpha_in, VectorData &x )
{
    auto const alpha = alpha_in.get<TYPE>();

    if ( allDefaultDataType( x ) ) {
        auto xdata   = x.getRawDataBlock<TYPE>();
        const auto N = x.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            xdata[i] *= alpha;
        }
        x.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe = x.begin<TYPE>();
        auto last  = x.end<TYPE>();
        while ( curMe != last ) {
            *curMe *= alpha;
            ++curMe;
        }
    }
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::scale( const Scalar &alpha_in,
                                           const VectorData &x,
                                           VectorData &y )
{
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );
    auto const alpha = alpha_in.get<TYPE>();

    if ( allDefaultDataType( x, y ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto ydata       = y.getRawDataBlock<TYPE>();
        const auto N     = y.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            ydata[i] = alpha * xdata[i];
        }
    } else {
        auto curMe  = y.begin<TYPE>();
        auto last   = y.end<TYPE>();
        auto curRhs = x.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = alpha * *curRhs;
            ++curRhs;
            ++curMe;
        }
    }
    y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );
    if ( allDefaultDataType( x, y, z ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto const ydata = y.getRawDataBlock<TYPE>();
        auto zdata       = z.getRawDataBlock<TYPE>();
        const auto N     = z.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            zdata[i] = xdata[i] + ydata[i];
        }
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe   = z.begin<TYPE>();
        auto last    = z.end<TYPE>();
        auto curXRhs = x.begin<TYPE>();
        auto curYRhs = y.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = *curXRhs + *curYRhs;
            ++curXRhs;
            ++curYRhs;
            ++curMe;
        }
    }
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::subtract( const VectorData &x,
                                              const VectorData &y,
                                              VectorData &z )
{
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );

    if ( allDefaultDataType( x, y, z ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto const ydata = y.getRawDataBlock<TYPE>();
        auto zdata       = z.getRawDataBlock<TYPE>();
        const auto N     = z.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            zdata[i] = xdata[i] - ydata[i];
        }
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe   = z.begin<TYPE>();
        auto last    = z.end<TYPE>();
        auto curXRhs = x.begin<TYPE>();
        auto curYRhs = y.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = *curXRhs - *curYRhs;
            ++curXRhs;
            ++curYRhs;
            ++curMe;
        }
    }
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::multiply( const VectorData &x,
                                              const VectorData &y,
                                              VectorData &z )
{
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );

    if ( allDefaultDataType( x, y, z ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto const ydata = y.getRawDataBlock<TYPE>();
        auto zdata       = z.getRawDataBlock<TYPE>();
        const auto N     = z.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            zdata[i] = xdata[i] * ydata[i];
        }
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe   = z.begin<TYPE>();
        auto last    = z.end<TYPE>();
        auto curXRhs = x.begin<TYPE>();
        auto curYRhs = y.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = *curXRhs * *curYRhs;
            ++curXRhs;
            ++curYRhs;
            ++curMe;
        }
    }
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::divide( const VectorData &x,
                                            const VectorData &y,
                                            VectorData &z )
{
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );

    if ( allDefaultDataType( x, y, z ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto const ydata = y.getRawDataBlock<TYPE>();
        auto zdata       = z.getRawDataBlock<TYPE>();
        const auto N     = z.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            zdata[i] = xdata[i] / ydata[i];
        }
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe   = z.begin<TYPE>();
        auto last    = z.end<TYPE>();
        auto curXRhs = x.begin<TYPE>();
        auto curYRhs = y.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = *curXRhs / *curYRhs;
            ++curXRhs;
            ++curYRhs;
            ++curMe;
        }
    }
}


template<typename TYPE>
void VectorOperationsDefault<TYPE>::reciprocal( const VectorData &x, VectorData &y )
{
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );

    if ( allDefaultDataType( x, y ) ) {
        auto const xdata   = x.getRawDataBlock<TYPE>();
        auto ydata         = y.getRawDataBlock<TYPE>();
        const auto N       = y.sizeOfDataBlock( 0 );
        constexpr auto one = static_cast<TYPE>( 1.0 );
        for ( size_t i = 0; i < N; ++i ) {
            ydata[i] = one / xdata[i];
        }
        y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe  = y.begin<TYPE>();
        auto last   = y.end<TYPE>();
        auto curRhs = x.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = static_cast<TYPE>( 1.0 ) / *curRhs;
            ++curRhs;
            ++curMe;
        }
    }
}


template<typename TYPE>
void VectorOperationsDefault<TYPE>::linearSum( const Scalar &alpha_in,
                                               const VectorData &x,
                                               const Scalar &beta_in,
                                               const VectorData &y,
                                               VectorData &z )
{
    const auto alpha = alpha_in.get<TYPE>();
    const auto beta  = beta_in.get<TYPE>();
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );

    if ( allDefaultDataType( x, y, z ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto const ydata = y.getRawDataBlock<TYPE>();
        auto zdata       = z.getRawDataBlock<TYPE>();
        const auto N     = z.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            zdata[i] = alpha * xdata[i] + beta * ydata[i];
        }
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe   = z.begin<TYPE>();
        auto last    = z.end<TYPE>();
        auto curXRhs = x.begin<TYPE>();
        auto curYRhs = y.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = alpha * *curXRhs + beta * *curYRhs;
            ++curXRhs;
            ++curYRhs;
            ++curMe;
        }
    }
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::axpy( const Scalar &alpha_in,
                                          const VectorData &x,
                                          const VectorData &y,
                                          VectorData &z )
{
    const auto alpha = alpha_in.get<TYPE>();
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );

    if ( allDefaultDataType( x, y, z ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto const ydata = y.getRawDataBlock<TYPE>();
        auto zdata       = z.getRawDataBlock<TYPE>();
        const auto N     = z.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            zdata[i] = alpha * xdata[i] + ydata[i];
        }
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe   = z.begin<TYPE>();
        auto last    = z.end<TYPE>();
        auto curXRhs = x.begin<TYPE>();
        auto curYRhs = y.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = alpha * *curXRhs + *curYRhs;
            ++curXRhs;
            ++curYRhs;
            ++curMe;
        }
    }
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::axpby( const Scalar &alpha_in,
                                           const Scalar &beta_in,
                                           const VectorData &x,
                                           VectorData &y )
{
    const auto alpha = alpha_in.get<TYPE>();
    const auto beta  = beta_in.get<TYPE>();
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );

    if ( allDefaultDataType( x, y ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto ydata       = y.getRawDataBlock<TYPE>();
        const auto N     = y.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            ydata[i] = alpha * xdata[i] + beta * ydata[i];
        }
        y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe   = y.begin<TYPE>();
        auto last    = y.end<TYPE>();
        auto curXRhs = x.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = alpha * *curXRhs + beta * *curMe;
            ++curXRhs;
            ++curMe;
        }
    }
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::abs( const VectorData &x, VectorData &y )
{
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );

    if ( allDefaultDataType( x, y ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto ydata       = y.getRawDataBlock<TYPE>();
        const auto N     = y.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            ydata[i] = std::abs( xdata[i] );
        }
        y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe  = y.begin<TYPE>();
        auto last   = y.end<TYPE>();
        auto curRhs = x.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = fabs( *curRhs );
            ++curRhs;
            ++curMe;
        }
    }
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::addScalar( const VectorData &x,
                                               const Scalar &alpha_in,
                                               VectorData &y )
{
    const auto alpha = alpha_in.get<TYPE>();
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );

    if ( allDefaultDataType( x, y ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto ydata       = y.getRawDataBlock<TYPE>();
        const auto N     = y.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            ydata[i] = alpha + xdata[i];
        }
        y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        auto curMe   = y.begin<TYPE>();
        auto last    = y.end<TYPE>();
        auto curXRhs = x.begin<TYPE>();
        while ( curMe != last ) {
            *curMe = *curXRhs + alpha;
            ++curXRhs;
            ++curMe;
        }
    }
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::setMax( const Scalar &val, VectorData &x )
{
    auto alpha = val.get<TYPE>();
    auto curMe = x.begin<TYPE>();
    auto last  = x.end<TYPE>();
    while ( curMe != last ) {
        *curMe = std::min( alpha, *curMe );
        ++curMe;
    }
}

template<typename TYPE>
void VectorOperationsDefault<TYPE>::setMin( const Scalar &val, VectorData &x )
{
    auto alpha = val.get<TYPE>();
    auto curMe = x.begin<TYPE>();
    auto last  = x.end<TYPE>();
    while ( curMe != last ) {
        *curMe = std::max( alpha, *curMe );
        ++curMe;
    }
}

template<typename TYPE>
Scalar VectorOperationsDefault<TYPE>::localMin( const VectorData &x ) const
{
    const size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans              = std::numeric_limits<TYPE>::max();
    for ( size_t i = 0; i < N_blocks; i++ ) {
        const size_t size = x.sizeOfDataBlock( i );
        const TYPE *data  = x.getRawDataBlock<TYPE>( i );
        for ( size_t j = 0; j < size; j++ )
            ans = std::min( data[j], ans );
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsDefault<TYPE>::localMax( const VectorData &x ) const
{
    const size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans              = std::numeric_limits<TYPE>::lowest();
    for ( size_t i = 0; i < N_blocks; i++ ) {
        const size_t size = x.sizeOfDataBlock( i );
        const TYPE *data  = x.getRawDataBlock<TYPE>( i );
        for ( size_t j = 0; j < size; j++ )
            ans = std::max( data[j], ans );
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsDefault<TYPE>::localSum( const VectorData &x ) const
{
    const size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans              = 0;
    for ( size_t i = 0; i < N_blocks; i++ ) {
        const size_t size = x.sizeOfDataBlock( i );
        const TYPE *data  = x.getRawDataBlock<TYPE>( i );
        for ( size_t j = 0; j < size; j++ )
            ans += data[j];
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsDefault<TYPE>::localL1Norm( const VectorData &x ) const
{
    size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans        = 0;
    for ( size_t i = 0; i < N_blocks; i++ ) {
        size_t size      = x.sizeOfDataBlock( i );
        const TYPE *data = x.getRawDataBlock<TYPE>( i );
        for ( size_t j = 0; j < size; j++ )
            ans += std::abs( data[j] );
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsDefault<TYPE>::localL2Norm( const VectorData &x ) const
{
    const size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans              = 0;
    for ( size_t i = 0; i < N_blocks; i++ ) {
        const size_t size = x.sizeOfDataBlock( i );
        const TYPE *data  = x.getRawDataBlock<TYPE>( i );
        for ( size_t j = 0; j < size; j++ )
            ans += data[j] * data[j];
    }
    return std::sqrt( ans );
}

template<typename TYPE>
Scalar VectorOperationsDefault<TYPE>::localMaxNorm( const VectorData &x ) const
{
    const size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans              = 0;
    for ( size_t i = 0; i < N_blocks; i++ ) {
        const size_t size = x.sizeOfDataBlock( i );
        const TYPE *data  = x.getRawDataBlock<TYPE>( i );
        for ( size_t j = 0; j < size; j++ )
            ans = std::max( std::abs( data[j] ), ans );
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsDefault<TYPE>::localDot( const VectorData &x, const VectorData &y ) const
{
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );

    TYPE ans = 0;

    if ( allDefaultDataType( x, y ) ) {
        auto const xdata = x.getRawDataBlock<TYPE>();
        auto const ydata = y.getRawDataBlock<TYPE>();
        const auto N     = x.sizeOfDataBlock( 0 );
        for ( size_t i = 0; i < N; ++i ) {
            ans += xdata[i] * ydata[i];
        }
    } else {
        auto curMe   = y.constBegin<TYPE>();
        auto last    = y.constEnd<TYPE>();
        auto curXRhs = x.constBegin<TYPE>();
        while ( curMe != last ) {
            TYPE v1 = *curMe;
            TYPE v2 = *curXRhs;
            ans += v1 * v2;
            ++curXRhs;
            ++curMe;
        }
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsDefault<TYPE>::localMinQuotient( const VectorData &x,
                                                        const VectorData &y ) const
{
    auto curx = x.constBegin<TYPE>();
    auto endx = x.constEnd<TYPE>();
    auto cury = y.constBegin<TYPE>();
    TYPE ans  = std::numeric_limits<TYPE>::max();
    while ( curx != endx ) {
        if ( *cury != 0 ) {
            TYPE v1 = *curx;
            TYPE v2 = *cury;
            ans     = std::min( ans, v1 / v2 );
        }
        ++curx;
        ++cury;
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsDefault<TYPE>::localWrmsNorm( const VectorData &x,
                                                     const VectorData &y ) const
{
    auto curx = x.constBegin<TYPE>();
    auto endx = x.constEnd<TYPE>();
    auto cury = y.constBegin<TYPE>();
    TYPE ans  = 0;
    size_t N  = 0;
    while ( curx != endx ) {
        TYPE v1 = *curx;
        TYPE v2 = *cury;
        ans += v1 * v1 * v2 * v2;
        ++curx;
        ++cury;
        ++N;
    }
    return std::sqrt( ans / N );
}

template<typename TYPE>
Scalar VectorOperationsDefault<TYPE>::localWrmsNormMask( const VectorData &x,
                                                         const VectorData &mask,
                                                         const VectorData &y ) const
{
    auto curx = x.constBegin<TYPE>();
    auto endx = x.constEnd<TYPE>();
    auto cury = y.constBegin<TYPE>();
    auto curm = mask.constBegin<TYPE>();
    TYPE ans  = 0;
    size_t N  = 0;
    while ( curx != endx ) {
        if ( *curm > 0 ) {
            TYPE v1 = *curx;
            TYPE v2 = *cury;
            ans += v1 * v1 * v2 * v2;
        }
        ++curx;
        ++cury;
        ++curm;
        ++N;
    }
    return std::sqrt( ans / N );
}

template<typename TYPE>
bool VectorOperationsDefault<TYPE>::localEquals( const VectorData &x,
                                                 const VectorData &y,
                                                 const Scalar &tol_in ) const
{
    if ( ( x.getGlobalSize() != y.getGlobalSize() ) || ( x.getLocalSize() != y.getLocalSize() ) )
        return false;
    bool equal = true;
    auto cur1  = x.constBegin<TYPE>();
    auto cur2  = y.constBegin<TYPE>();
    auto last  = x.constEnd<TYPE>();
    auto tol   = tol_in.get<double>();
    while ( cur1 != last ) {
        if ( std::abs( *cur1 - *cur2 ) > tol ) {
            equal = false;
            break;
        }
        ++cur1;
        ++cur2;
    }
    return equal;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
template<typename TYPE>
void VectorOperationsDefault<TYPE>::registerChildObjects( AMP::IO::RestartManager * ) const
{
}
template<typename TYPE>
void VectorOperationsDefault<TYPE>::writeRestart( int64_t ) const
{
}
template<typename TYPE>
VectorOperationsDefault<TYPE>::VectorOperationsDefault( int64_t, AMP::IO::RestartManager * )
{
}


} // namespace AMP::LinearAlgebra

#endif
