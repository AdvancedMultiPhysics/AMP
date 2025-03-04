#ifndef included_AMP_VectorOperationsOpenMP_hpp
#define included_AMP_VectorOperationsOpenMP_hpp

#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/operations/OpenMP/VectorOperationsOpenMP.h"

#include <random>


namespace AMP::LinearAlgebra {


extern template class VectorOperationsOpenMP<double>; // Suppresses implicit instantiation below --
extern template class VectorOperationsOpenMP<float>;  // Suppresses implicit instantiation below --


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
template<typename TYPE>
std::shared_ptr<VectorOperations> VectorOperationsOpenMP<TYPE>::cloneOperations() const
{
    auto ptr = std::make_shared<VectorOperationsOpenMP<TYPE>>();
    return ptr;
}

//**********************************************************************
// functions that operate on VectorData objects

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::zero( VectorData &x )
{
    size_t N_blocks = x.numberOfDataBlocks();
    for ( size_t i = 0; i < N_blocks; i++ ) {
        size_t size = x.sizeOfDataBlock( i );
        TYPE *data  = x.getRawDataBlock<TYPE>( i );
#pragma omp parallel for
        for ( size_t j = 0; j < size; j++ )
            data[j] = 0.0;
    }

    if ( x.hasGhosts() ) {
        auto &ghosts = x.getGhosts();
        for ( size_t i = 0; i != ghosts.size(); i++ )
            ghosts[i] = 0;
    }
    // Override the status state since we set the ghost values
    x.setUpdateStatus( UpdateState::UNCHANGED );
}

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::setToScalar( const Scalar &alpha_in, VectorData &x )
{
    size_t N_blocks = x.numberOfDataBlocks();
    TYPE alpha      = alpha_in.get<TYPE>();
    for ( size_t i = 0; i < N_blocks; i++ ) {
        size_t size = x.sizeOfDataBlock( i );
        TYPE *data  = x.getRawDataBlock<TYPE>( i );
#pragma omp parallel for
        for ( size_t j = 0; j < size; j++ )
            data[j] = alpha;
    }
    if ( x.hasGhosts() ) {
        auto &ghosts = x.getGhosts();
        for ( size_t i = 0; i != ghosts.size(); i++ )
            ghosts[i] = alpha;
    }
    // Override the status state since we set the ghost values
    x.setUpdateStatus( UpdateState::UNCHANGED );
}

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::setRandomValues( VectorData &x )
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
void VectorOperationsOpenMP<TYPE>::copy( const VectorData &x, VectorData &y )
{
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );
    std::copy( x.begin<TYPE>(), x.end<TYPE>(), y.begin<TYPE>() );
    y.copyGhostValues( x );
}

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::copyCast( const VectorData &x, VectorData &y )
{
    // VectorOperationsDefault::copyCast( x, y );
    if ( x.numberOfDataBlocks() == y.numberOfDataBlocks() ) {
        for ( size_t block_id = 0; block_id < y.numberOfDataBlocks(); block_id++ ) {
            auto ydata = y.getRawDataBlock<TYPE>( block_id );
            auto N     = y.sizeOfDataBlock( block_id );
            AMP_ASSERT( N == x.sizeOfDataBlock( block_id ) );
            if ( x.getType( 0 ) == getTypeID<float>() ) {
                auto xdata = x.getRawDataBlock<float>( block_id );
                AMP::Utilities::copyCast<float, TYPE>( N, xdata, ydata );
            } else if ( x.getType( 0 ) == getTypeID<double>() ) {
                auto xdata = x.getRawDataBlock<double>( block_id );
                AMP::Utilities::copyCast<double, TYPE>( N, xdata, ydata );
            } else {
                AMP_ERROR( "CopyCast only implemented for float or doubles." );
            }
        }
    } else {
        AMP_ERROR( "Different number of blocks; CopyCast not implemented for non-matching "
                   "multiblock data." );
    }
    y.copyGhostValues( x );
}

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::scale( const Scalar &alpha_in, VectorData &x )
{
    TYPE alpha      = alpha_in.get<TYPE>();
    size_t N_blocks = x.numberOfDataBlocks();
    for ( size_t i = 0; i < N_blocks; i++ ) {
        size_t size = x.sizeOfDataBlock( i );
        TYPE *data  = x.getRawDataBlock<TYPE>( i );
#pragma omp parallel for
        for ( size_t j = 0; j < size; j++ )
            data[j] *= alpha;
    }
}

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::scale( const Scalar &alpha_in,
                                          const VectorData &x,
                                          VectorData &y )
{
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );
    TYPE alpha  = alpha_in.get<TYPE>();
    auto curMe  = y.begin<TYPE>();
    auto last   = y.end<TYPE>();
    auto curRhs = x.begin<TYPE>();
    while ( curMe != last ) {
        *curMe = alpha * *curRhs;
        ++curRhs;
        ++curMe;
    }
}

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );
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

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::subtract( const VectorData &x,
                                             const VectorData &y,
                                             VectorData &z )
{
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );
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

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::multiply( const VectorData &x,
                                             const VectorData &y,
                                             VectorData &z )
{
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );
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

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::divide( const VectorData &x, const VectorData &y, VectorData &z )
{
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );
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


template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::reciprocal( const VectorData &x, VectorData &y )
{
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );
    auto curMe  = y.begin<TYPE>();
    auto last   = y.end<TYPE>();
    auto curRhs = x.begin<TYPE>();
    while ( curMe != last ) {
        *curMe = 1.0 / *curRhs;
        ++curRhs;
        ++curMe;
    }
}


template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::linearSum( const Scalar &alpha_in,
                                              const VectorData &x,
                                              const Scalar &beta_in,
                                              const VectorData &y,
                                              VectorData &z )
{
    TYPE alpha = alpha_in.get<TYPE>();
    TYPE beta  = beta_in.get<TYPE>();
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );
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

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::axpy( const Scalar &alpha_in,
                                         const VectorData &x,
                                         const VectorData &y,
                                         VectorData &z )
{
    TYPE alpha = alpha_in.get<TYPE>();
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    AMP_ASSERT( z.getLocalSize() == y.getLocalSize() );
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

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::axpby( const Scalar &alpha_in,
                                          const Scalar &beta_in,
                                          const VectorData &x,
                                          VectorData &z )
{
    TYPE alpha = alpha_in.get<TYPE>();
    TYPE beta  = beta_in.get<TYPE>();
    AMP_ASSERT( z.getLocalSize() == x.getLocalSize() );
    auto curMe   = z.begin<TYPE>();
    auto last    = z.end<TYPE>();
    auto curXRhs = x.begin<TYPE>();
    while ( curMe != last ) {
        *curMe = alpha * *curXRhs + beta * *curMe;
        ++curXRhs;
        ++curMe;
    }
}

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::abs( const VectorData &x, VectorData &y )
{
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );
    auto curMe  = y.begin<TYPE>();
    auto last   = y.end<TYPE>();
    auto curRhs = x.begin<TYPE>();
    while ( curMe != last ) {
        *curMe = fabs( *curRhs );
        ++curRhs;
        ++curMe;
    }
}

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::addScalar( const VectorData &x,
                                              const Scalar &alpha_in,
                                              VectorData &y )
{
    TYPE alpha = alpha_in.get<TYPE>();
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );
    auto curMe   = y.begin<TYPE>();
    auto last    = y.end<TYPE>();
    auto curXRhs = x.begin<TYPE>();
    while ( curMe != last ) {
        *curMe = *curXRhs + alpha;
        ++curXRhs;
        ++curMe;
    }
}

template<typename TYPE>
void VectorOperationsOpenMP<TYPE>::setMax( const Scalar &val, VectorData &x )
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
void VectorOperationsOpenMP<TYPE>::setMin( const Scalar &val, VectorData &x )
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
Scalar VectorOperationsOpenMP<TYPE>::localMin( const VectorData &x ) const
{
    size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans        = std::numeric_limits<TYPE>::max();
    for ( size_t i = 0; i < N_blocks; i++ ) {
        size_t size      = x.sizeOfDataBlock( i );
        const TYPE *data = x.getRawDataBlock<TYPE>( i );
#pragma omp parallel for reduction( min : ans )
        for ( size_t j = 0; j < size; j++ )
            ans = std::min( data[j], ans );
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsOpenMP<TYPE>::localMax( const VectorData &x ) const
{
    size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans        = std::numeric_limits<TYPE>::lowest();
    for ( size_t i = 0; i < N_blocks; i++ ) {
        size_t size      = x.sizeOfDataBlock( i );
        const TYPE *data = x.getRawDataBlock<TYPE>( i );
#pragma omp parallel for reduction( max : ans )
        for ( size_t j = 0; j < size; j++ )
            ans = std::max( data[j], ans );
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsOpenMP<TYPE>::localSum( const VectorData &x ) const
{
    size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans        = std::numeric_limits<TYPE>::lowest();
    for ( size_t i = 0; i < N_blocks; i++ ) {
        size_t size      = x.sizeOfDataBlock( i );
        const TYPE *data = x.getRawDataBlock<TYPE>( i );
#pragma omp parallel for reduction( + : ans )
        for ( size_t j = 0; j < size; j++ )
            ans += data[j];
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsOpenMP<TYPE>::localL1Norm( const VectorData &x ) const
{
    size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans        = 0;
    for ( size_t i = 0; i < N_blocks; i++ ) {
        size_t size      = x.sizeOfDataBlock( i );
        const TYPE *data = x.getRawDataBlock<TYPE>( i );
#pragma omp parallel for reduction( + : ans )
        for ( size_t j = 0; j < size; j++ )
            ans += std::abs( data[j] );
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsOpenMP<TYPE>::localL2Norm( const VectorData &x ) const
{
    size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans        = 0;
    for ( size_t i = 0; i < N_blocks; i++ ) {
        size_t size      = x.sizeOfDataBlock( i );
        const TYPE *data = x.getRawDataBlock<TYPE>( i );
#pragma omp parallel for reduction( + : ans )
        for ( size_t j = 0; j < size; j++ ) {
            auto tmp = data[j];
            ans += tmp * tmp;
        }
    }
    return std::sqrt( ans );
}

template<typename TYPE>
Scalar VectorOperationsOpenMP<TYPE>::localMaxNorm( const VectorData &x ) const
{
    size_t N_blocks = x.numberOfDataBlocks();
    TYPE ans        = 0;
    for ( size_t i = 0; i < N_blocks; i++ ) {
        size_t size      = x.sizeOfDataBlock( i );
        const TYPE *data = x.getRawDataBlock<TYPE>( i );
#pragma omp parallel for reduction( max : ans )
        for ( size_t j = 0; j < size; j++ )
            ans = std::max( std::abs( data[j] ), ans );
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsOpenMP<TYPE>::localDot( const VectorData &x, const VectorData &y ) const
{
    AMP_ASSERT( y.getLocalSize() == x.getLocalSize() );
    auto curMe   = y.constBegin<TYPE>();
    auto last    = y.constEnd<TYPE>();
    auto curXRhs = x.constBegin<TYPE>();
    TYPE ans     = 0;
    while ( curMe != last ) {
        auto v1 = *curMe;
        auto v2 = *curXRhs;
        ans += v1 * v2;
        ++curXRhs;
        ++curMe;
    }
    return ans;
}

template<typename TYPE>
Scalar VectorOperationsOpenMP<TYPE>::localMinQuotient( const VectorData &x,
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
Scalar VectorOperationsOpenMP<TYPE>::localWrmsNorm( const VectorData &x, const VectorData &y ) const
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
Scalar VectorOperationsOpenMP<TYPE>::localWrmsNormMask( const VectorData &x,
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
bool VectorOperationsOpenMP<TYPE>::localEquals( const VectorData &x,
                                                const VectorData &y,
                                                const Scalar &tol_in ) const
{
    if ( ( x.getGlobalSize() != y.getGlobalSize() ) || ( x.getLocalSize() != y.getLocalSize() ) )
        return false;
    bool equal = true;
    auto cur1  = x.constBegin<TYPE>();
    auto cur2  = y.constBegin<TYPE>();
    auto last  = x.constEnd<TYPE>();
    double tol = tol_in.get<double>();
    while ( cur1 != last ) {
        if ( fabs( *cur1 - *cur2 ) > tol ) {
            equal = false;
            break;
        }
        ++cur1;
        ++cur2;
    }
    return equal;
}

} // namespace AMP::LinearAlgebra

#endif
