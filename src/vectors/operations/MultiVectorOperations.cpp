#include "AMP/vectors/operations/MultiVectorOperations.h"
#include "AMP/vectors/Scalar.h"
#include "AMP/vectors/Scalar.hpp"
#include "AMP/vectors/data/MultiVectorData.h"


namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
std::shared_ptr<VectorOperations> MultiVectorOperations::cloneOperations() const
{
    auto ptr = std::make_shared<MultiVectorOperations>();
    return ptr;
}

//**********************************************************************
// Static functions that operate on VectorData objects

VectorData *MultiVectorOperations::getVectorDataComponent( VectorData &x, size_t i )
{
    auto x2 = dynamic_cast<MultiVectorData *>( &x );
    AMP_ASSERT( x2 && ( i < x2->numberOfComponents() ) );
    return x2->getVectorData( i );
}
const VectorData *MultiVectorOperations::getVectorDataComponent( const VectorData &x, size_t i )
{
    auto x2 = dynamic_cast<const MultiVectorData *>( &x );
    AMP_ASSERT( x2 && ( i < x2->numberOfComponents() ) );
    return x2->getVectorData( i );
}

const MultiVectorData *MultiVectorOperations::getMultiVectorData( const VectorData &x )
{
    return dynamic_cast<const MultiVectorData *>( &x );
}

MultiVectorData *MultiVectorOperations::getMultiVectorData( VectorData &x )
{
    return dynamic_cast<MultiVectorData *>( &x );
}

void MultiVectorOperations::zero( VectorData &x )
{
    auto mData = getMultiVectorData( x );

    for ( size_t i = 0; i != mData->numberOfComponents(); ++i ) {

        d_operations[i]->zero( *getVectorDataComponent( x, i ) );
    }
}

void MultiVectorOperations::setToScalar( const Scalar &alpha, VectorData &x )
{
    for ( size_t i = 0; i != d_operations.size(); i++ ) {
        d_operations[i]->setToScalar( alpha, *getVectorDataComponent( x, i ) );
    }
}

void MultiVectorOperations::setRandomValues( VectorData &x )
{
    for ( size_t i = 0; i != d_operations.size(); i++ ) {
        d_operations[i]->setRandomValues( *getVectorDataComponent( x, i ) );
    }
}

void MultiVectorOperations::setRandomValues( std::shared_ptr<RNG> rng, VectorData &x )
{
    for ( size_t i = 0; i != d_operations.size(); i++ ) {
        d_operations[i]->setRandomValues( rng, *getVectorDataComponent( x, i ) );
    }
}

void MultiVectorOperations::copy( const VectorData &x, VectorData &y )
{

    auto xc = getMultiVectorData( x );
    auto yc = getMultiVectorData( y );

    if ( xc && yc ) {
        // Both this and x are multivectors
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->copy( *getVectorDataComponent( x, i ),
                                   *getVectorDataComponent( y, i ) );
    } else {
        // x is not a multivector, try to call a default implementation
        AMP_ASSERT( x.getLocalSize() == y.getLocalSize() );
        if ( x.isType<double>() && y.isType<double>() ) {
            std::copy( x.begin<double>(), x.end<double>(), y.begin<double>() );
        } else if ( x.isType<float>() && y.isType<float>() ) {
            std::copy( x.begin<float>(), x.end<float>(), y.begin<float>() );
        } else {
            AMP_ERROR( "Unable to discern data types" );
        }
    }
}

void MultiVectorOperations::scale( const Scalar &alpha, VectorData &x )
{
    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() ) {
        return;
    }
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->scale( alpha, *getVectorDataComponent( x, i ) );
}

void MultiVectorOperations::scale( const Scalar &alpha, const VectorData &x, VectorData &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->scale(
                alpha, *getVectorDataComponent( x, i ), *getVectorDataComponent( y, i ) );

    } else {
        AMP_ERROR( "MultiVectorOperations::scale requires both x and y to be MultiVectorData" );
    }
}

void MultiVectorOperations::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        auto z2 = getMultiVectorData( y );
        AMP_ASSERT( z2 );
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->add( *getVectorDataComponent( x, i ),
                                  *getVectorDataComponent( y, i ),
                                  *getVectorDataComponent( z, i ) );
    } else {
        AMP_ERROR( "MultiVectorOperations::add requires x, y, z to be MultiVectorData" );
    }
}

void MultiVectorOperations::subtract( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        auto z2 = getMultiVectorData( y );
        AMP_ASSERT( z2 );
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->subtract( *getVectorDataComponent( x, i ),
                                       *getVectorDataComponent( y, i ),
                                       *getVectorDataComponent( z, i ) );
    } else {
        AMP_ERROR( "MultiVectorOperations::subtract requires x, y, z to be MultiVectorData" );
    }
}

void MultiVectorOperations::multiply( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        auto z2 = getMultiVectorData( y );
        AMP_ASSERT( z2 );
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->multiply( *getVectorDataComponent( x, i ),
                                       *getVectorDataComponent( y, i ),
                                       *getVectorDataComponent( z, i ) );
    } else {
        AMP_ERROR( "MultiVectorOperations::multiply requires x, y, z to be MultiVectorData" );
    }
}

void MultiVectorOperations::divide( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        auto z2 = getMultiVectorData( y );
        AMP_ASSERT( z2 );
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->divide( *getVectorDataComponent( x, i ),
                                     *getVectorDataComponent( y, i ),
                                     *getVectorDataComponent( z, i ) );
    } else {
        AMP_ERROR( "MultiVectorOperations::divide requires x, y, z to be MultiVectorData" );
    }
}

void MultiVectorOperations::reciprocal( const VectorData &x, VectorData &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        AMP_ASSERT( x2->numberOfComponents() == y2->numberOfComponents() );
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->reciprocal( *getVectorDataComponent( x, i ),
                                         *getVectorDataComponent( y, i ) );
    } else {
        AMP_ERROR(
            "MultiVectorOperations::reciprocal requires both x and y to be MultiVectorData" );
    }
}

void MultiVectorOperations::linearSum( const Scalar &alpha_in,
                                       const VectorData &x,
                                       const Scalar &beta_in,
                                       const VectorData &y,
                                       VectorData &z )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        auto z2 = getMultiVectorData( y );
        AMP_ASSERT( z2 );
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == z2->numberOfComponents() );
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->linearSum( alpha_in,
                                        *getVectorDataComponent( x, i ),
                                        beta_in,
                                        *getVectorDataComponent( y, i ),
                                        *getVectorDataComponent( z, i ) );

    } else {
        AMP_ASSERT( x.getLocalSize() == y.getLocalSize() );
        AMP_ASSERT( x.getLocalSize() == z.getLocalSize() );
        if ( x.isType<double>() && y.isType<double>() ) {
            auto xit     = x.begin<double>();
            auto yit     = y.begin<double>();
            auto zit     = z.begin<double>();
            auto xend    = x.end<double>();
            double alpha = alpha_in.get<double>();
            double beta  = beta_in.get<double>();
            while ( xit != xend ) {
                *zit = alpha * ( *xit ) + beta * ( *yit );
                ++xit;
                ++yit;
                ++zit;
            }
        } else if ( x.isType<float>() && y.isType<float>() ) {
            auto xit    = x.begin<float>();
            auto yit    = y.begin<float>();
            auto zit    = z.begin<float>();
            auto xend   = x.end<float>();
            float alpha = alpha_in.get<float>();
            float beta  = beta_in.get<float>();
            while ( xit != xend ) {
                *zit = alpha * ( *xit ) + beta * ( *yit );
                ++xit;
                ++yit;
                ++zit;
            }
        } else {
            AMP_ERROR( "Unable to discern data types" );
        }
    }
}

void MultiVectorOperations::axpy( const Scalar &alpha_in,
                                  const VectorData &x,
                                  const VectorData &y,
                                  VectorData &z )
{
    linearSum( alpha_in, x, 1.0, y, z );
}

void MultiVectorOperations::axpby( const Scalar &alpha_in,
                                   const Scalar &beta_in,
                                   const VectorData &x,
                                   VectorData &z )
{
    linearSum( alpha_in, x, beta_in, z, z );
}

void MultiVectorOperations::abs( const VectorData &x, VectorData &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        for ( size_t i = 0; i != d_operations.size(); i++ ) {
            d_operations[i]->abs( *getVectorDataComponent( x, i ),
                                  *getVectorDataComponent( y, i ) );
        }
    } else {
        AMP_ERROR( "MultiVectorOperations::abs requires x, y to be MultiVectorData" );
    }
}

void MultiVectorOperations::addScalar( const VectorData &x, const Scalar &alpha_in, VectorData &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->addScalar(
                *getVectorDataComponent( x, i ), alpha_in, *getVectorDataComponent( y, i ) );
    } else {
        AMP_ERROR( "MultiVectorOperations::addScalar requires x, y to be MultiVectorData" );
    }
}

Scalar MultiVectorOperations::localMin( const VectorData &x ) const
{
    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0.0;
    auto ans = d_operations[0]->localMin( *getVectorDataComponent( x, 0 ) );
    for ( size_t i = 1; i != d_operations.size(); i++ )
        ans = std::min( ans, d_operations[i]->localMin( *getVectorDataComponent( x, i ) ) );
    if ( !ans.has_value() )
        return 0.0;
    return ans;
}

Scalar MultiVectorOperations::localMax( const VectorData &x ) const
{
    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0.0;
    auto ans = d_operations[0]->localMax( *getVectorDataComponent( x, 0 ) );
    for ( size_t i = 1; i != d_operations.size(); i++ )
        ans = std::max( ans, d_operations[i]->localMax( *getVectorDataComponent( x, i ) ) );
    if ( !ans.has_value() )
        return 0.0;
    return ans;
}

Scalar MultiVectorOperations::localL1Norm( const VectorData &x ) const
{
    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0;
    Scalar ans;
    for ( size_t i = 0; i != d_operations.size(); i++ )
        ans = ans + d_operations[i]->localL1Norm( *getVectorDataComponent( x, i ) );
    if ( !ans.has_value() )
        return 0.0;
    return ans;
}

Scalar MultiVectorOperations::localL2Norm( const VectorData &x ) const
{
    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0;
    Scalar ans;
    for ( size_t i = 0; i != d_operations.size(); i++ ) {
        auto tmp = d_operations[i]->localL2Norm( *getVectorDataComponent( x, i ) );
        ans      = ans + tmp * tmp;
    }
    if ( !ans.has_value() )
        return 0.0;
    return ans.sqrt();
}

Scalar MultiVectorOperations::localMaxNorm( const VectorData &x ) const
{
    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0.0;
    auto ans = d_operations[0]->localMaxNorm( *getVectorDataComponent( x, 0 ) );
    for ( size_t i = 1; i != d_operations.size(); i++ )
        ans = std::max( ans, d_operations[i]->localMaxNorm( *getVectorDataComponent( x, i ) ) );
    if ( !ans.has_value() )
        return 0.0;
    return ans;
}

Scalar MultiVectorOperations::localDot( const VectorData &x, const VectorData &y ) const
{
    if ( d_operations.empty() )
        return 0;
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        Scalar ans;
        for ( size_t i = 0; i != d_operations.size(); i++ ) {
            auto xi = getVectorDataComponent( x, i );
            auto yi = getVectorDataComponent( y, i );
            ans     = ans + d_operations[i]->localDot( *xi, *yi );
        }
        if ( ans.has_value() )
            return ans;
    } else {
        AMP_ERROR( "MultiVectorOperations::localDot requires x, y to be MultiVectorData" );
    }
    return 0.0;
}

Scalar MultiVectorOperations::localMinQuotient( const VectorData &x, const VectorData &y ) const
{
    if ( d_operations.empty() )
        return std::numeric_limits<double>::max();
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        auto ans = d_operations[0]->localMinQuotient( *getVectorDataComponent( x, 0 ),
                                                      *getVectorDataComponent( y, 0 ) );
        for ( size_t i = 1; i != d_operations.size(); i++ )
            ans = std::min( ans,
                            d_operations[i]->localMinQuotient( *getVectorDataComponent( x, i ),
                                                               *getVectorDataComponent( y, i ) ) );
        if ( ans.has_value() )
            return ans;
    } else {
        AMP_ERROR( "MultiVectorOperations::localMinQuotient requires x, y to be MultiVectorData" );
    }
    return 0.0;
}

Scalar MultiVectorOperations::localWrmsNorm( const VectorData &x, const VectorData &y ) const
{
    if ( d_operations.empty() )
        return 0;
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        Scalar ans;
        for ( size_t i = 0; i < d_operations.size(); i++ ) {
            auto yi   = getVectorDataComponent( y, i );
            auto tmp  = d_operations[i]->localWrmsNorm( *getVectorDataComponent( x, i ), *yi );
            size_t N1 = yi->getLocalSize();
            ans       = ans + tmp * tmp * N1;
        }
        size_t N = y.getLocalSize();
        return ( ans * ( 1.0 / N ) ).sqrt();
    } else {
        AMP_ERROR( "MultiVectorOperations::localWrmsNorm requires x, y to be MultiVectorData" );
    }
    return 0.0;
}

Scalar MultiVectorOperations::localWrmsNormMask( const VectorData &x,
                                                 const VectorData &mask,
                                                 const VectorData &y ) const
{
    if ( d_operations.empty() )
        return 0;
    auto x2 = getMultiVectorData( x );
    auto m2 = getMultiVectorData( mask );
    auto y2 = getMultiVectorData( y );
    if ( x2 && m2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == m2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        Scalar ans;
        for ( size_t i = 0; i < d_operations.size(); i++ ) {
            auto yi  = getVectorDataComponent( y, i );
            auto tmp = d_operations[i]->localWrmsNormMask(
                *getVectorDataComponent( x, i ), *getVectorDataComponent( mask, i ), *yi );
            size_t N1 = yi->getLocalSize();
            ans       = ans + tmp * tmp * N1;
        }
        size_t N = y.getLocalSize();
        return ( ans * ( 1.0 / N ) ).sqrt();
    } else {
        AMP_ERROR(
            "MultiVectorOperations::localWrmsNormMask requires x, mask, y to be MultiVectorData" );
    }
    return 0.0;
}

bool MultiVectorOperations::localEquals( const VectorData &x,
                                         const VectorData &y,
                                         const Scalar &tol ) const
{
    if ( d_operations.empty() )
        return false;
    bool ans = true;
    auto x2  = getMultiVectorData( x );
    auto y2  = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
        AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
        for ( size_t i = 0; i < d_operations.size(); i++ ) {
            ans = ans && d_operations[i]->localEquals( *getVectorDataComponent( x, i ),
                                                       *getVectorDataComponent( y, i ),
                                                       tol );
        }
    } else {
        AMP_ERROR( "MultiVectorOperations::localEquals requires x, y to be MultiVectorData" );
    }
    return ans;
}

void MultiVectorOperations::resetVectorOperations(
    std::vector<std::shared_ptr<VectorOperations>> ops )
{
    d_operations = std::move( ops );
}


} // namespace AMP::LinearAlgebra
