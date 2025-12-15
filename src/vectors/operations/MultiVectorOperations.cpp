#include "AMP/vectors/operations/MultiVectorOperations.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/vectors/Scalar.h"
#include "AMP/vectors/data/MultiVectorData.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
std::shared_ptr<VectorOperations> MultiVectorOperations::cloneOperations() const
{
    auto ptr          = std::make_shared<MultiVectorOperations>();
    ptr->d_operations = d_operations;
    return ptr;
}
MultiVectorOperations::MultiVectorOperations( std::shared_ptr<VectorOperations> op )
    : VectorOperations(), d_operations( 1, op )
{
}


/****************************************************************
 * Static functions that operate on VectorData objects           *
 ****************************************************************/
VectorData *MultiVectorOperations::getVectorDataComponent( VectorData &x, size_t i )
{
    auto x2 = dynamic_cast<MultiVectorData *>( &x );
    if ( x2 ) {
        AMP_ASSERT( i < x2->getVectorDataSize() );
        return x2->getVectorData( i );
    } else {
        AMP_ASSERT( i == 0 );
        return &x;
    }
}
const VectorData *MultiVectorOperations::getVectorDataComponent( const VectorData &x, size_t i )
{
    auto x2 = dynamic_cast<const MultiVectorData *>( &x );
    if ( x2 ) {
        AMP_ASSERT( i < x2->getVectorDataSize() );
        return x2->getVectorData( i );
    } else {
        AMP_ASSERT( i == 0 );
        return &x;
    }
}
size_t MultiVectorOperations::getVectorDataSize( const VectorData &x )
{
    auto x2 = dynamic_cast<const MultiVectorData *>( &x );
    if ( x2 )
        return x2->getVectorDataSize();
    else
        return 1;
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
    PROFILE( "MultiVectorOperations::zero" );

    auto mData = getMultiVectorData( x );
    for ( size_t i = 0; i != mData->getVectorDataSize(); ++i )
        d_operations[i]->zero( *getVectorDataComponent( x, i ) );
    x.setUpdateStatus( UpdateState::UNCHANGED );
}

void MultiVectorOperations::setToScalar( const Scalar &alpha, VectorData &x )
{
    PROFILE( "MultiVectorOperations::setToScalar" );

    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->setToScalar( alpha, *getVectorDataComponent( x, i ) );
    x.setUpdateStatus( UpdateState::UNCHANGED );
}

void MultiVectorOperations::setRandomValues( VectorData &x )
{
    PROFILE( "MultiVectorOperations::setRandomValues" );

    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->setRandomValues( *getVectorDataComponent( x, i ) );
}

void MultiVectorOperations::copy( const VectorData &x, VectorData &y )
{
    PROFILE( "MultiVectorOperations::copy" );

    // Check if both x and y are MultVectorData objects (of the same size)
    auto xc = getMultiVectorData( x );
    auto yc = getMultiVectorData( y );
    if ( xc && yc ) {
        if ( d_operations.size() == xc->getVectorDataSize() &&
             d_operations.size() == yc->getVectorDataSize() ) {
            for ( size_t i = 0; i != d_operations.size(); i++ )
                d_operations[i]->copy( *getVectorDataComponent( x, i ),
                                       *getVectorDataComponent( y, i ) );
            return;
        }
    }
    // Check if we are dealing with vector data with multiple components
    size_t Nx = x.getNumberOfComponents();
    size_t Ny = y.getNumberOfComponents();
    if ( Nx == Ny ) {
        for ( size_t i = 0; i != Nx; i++ ) {
            std::shared_ptr<VectorOperations> op;
            auto x2 = x.getComponent( i );
            auto y2 = y.getComponent( i );
            if ( Nx == d_operations.size() ) {
                // Use the vector data operations on the multivector
                op = d_operations[i];
            } else if ( x2->isType<double>() && y2->isType<double>() ) {
                op = std::make_shared<VectorOperationsDefault<double>>();
            } else if ( x2->isType<float>() && y2->isType<float>() ) {
                op = std::make_shared<VectorOperationsDefault<float>>();
            } else {
                AMP_ERROR( "Unable to discern data types" );
            }
            op->copy( *x2, *y2 );
        }
        return;
    }
    // Try a default implementation using std::copy
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

void MultiVectorOperations::copyCast( const VectorData &x, VectorData &y )
{
    PROFILE( "MultiVectorOperations::copyCast" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( y ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->copyCast( *getVectorDataComponent( x, i ),
                                   *getVectorDataComponent( y, i ) );
}

void MultiVectorOperations::scale( const Scalar &alpha, VectorData &x )
{
    PROFILE( "MultiVectorOperations::scale" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->scale( alpha, *getVectorDataComponent( x, i ) );
}

void MultiVectorOperations::scale( const Scalar &alpha, const VectorData &x, VectorData &y )
{
    PROFILE( "MultiVectorOperations::scale" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( y ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->scale(
            alpha, *getVectorDataComponent( x, i ), *getVectorDataComponent( y, i ) );
}

void MultiVectorOperations::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    PROFILE( "MultiVectorOperations::add" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( y ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( z ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->add( *getVectorDataComponent( x, i ),
                              *getVectorDataComponent( y, i ),
                              *getVectorDataComponent( z, i ) );
}

void MultiVectorOperations::subtract( const VectorData &x, const VectorData &y, VectorData &z )
{
    PROFILE( "MultiVectorOperations::subtract" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( y ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( z ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->subtract( *getVectorDataComponent( x, i ),
                                   *getVectorDataComponent( y, i ),
                                   *getVectorDataComponent( z, i ) );
}

void MultiVectorOperations::multiply( const VectorData &x, const VectorData &y, VectorData &z )
{
    PROFILE( "MultiVectorOperations::multiply" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( y ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( z ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->multiply( *getVectorDataComponent( x, i ),
                                   *getVectorDataComponent( y, i ),
                                   *getVectorDataComponent( z, i ) );
}

void MultiVectorOperations::divide( const VectorData &x, const VectorData &y, VectorData &z )
{
    PROFILE( "MultiVectorOperations::divide" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( y ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( z ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->divide( *getVectorDataComponent( x, i ),
                                 *getVectorDataComponent( y, i ),
                                 *getVectorDataComponent( z, i ) );
}

void MultiVectorOperations::reciprocal( const VectorData &x, VectorData &y )
{
    PROFILE( "MultiVectorOperations::reciprocal" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( y ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->reciprocal( *getVectorDataComponent( x, i ),
                                     *getVectorDataComponent( y, i ) );
}

void MultiVectorOperations::linearSum( const Scalar &alpha_in,
                                       const VectorData &x,
                                       const Scalar &beta_in,
                                       const VectorData &y,
                                       VectorData &z )
{
    PROFILE( "MultiVectorOperations::linearSum" );

    if ( d_operations.empty() )
        return;
    bool match = d_operations.size() == getVectorDataSize( x ) &&
                 d_operations.size() == getVectorDataSize( y ) &&
                 d_operations.size() == getVectorDataSize( z );
    if ( match ) {
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
            auto xit   = x.begin<double>();
            auto yit   = y.begin<double>();
            auto zit   = z.begin<double>();
            auto xend  = x.end<double>();
            auto alpha = alpha_in.get<double>();
            auto beta  = beta_in.get<double>();
            while ( xit != xend ) {
                *zit = alpha * ( *xit ) + beta * ( *yit );
                ++xit;
                ++yit;
                ++zit;
            }
        } else if ( x.isType<float>() && y.isType<float>() ) {
            auto xit   = x.begin<float>();
            auto yit   = y.begin<float>();
            auto zit   = z.begin<float>();
            auto xend  = x.end<float>();
            auto alpha = alpha_in.get<float>();
            auto beta  = beta_in.get<float>();
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
    PROFILE( "MultiVectorOperations::axpy" );

    linearSum( alpha_in, x, 1.0, y, z );
}

void MultiVectorOperations::axpby( const Scalar &alpha_in,
                                   const Scalar &beta_in,
                                   const VectorData &x,
                                   VectorData &z )
{
    PROFILE( "MultiVectorOperations::axpby" );

    linearSum( alpha_in, x, beta_in, z, z );
}

void MultiVectorOperations::abs( const VectorData &x, VectorData &y )
{
    PROFILE( "MultiVectorOperations::abs" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( y ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->abs( *getVectorDataComponent( x, i ), *getVectorDataComponent( y, i ) );
}

void MultiVectorOperations::addScalar( const VectorData &x, const Scalar &alpha_in, VectorData &y )
{
    PROFILE( "MultiVectorOperations::addScalar" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    AMP_ASSERT( d_operations.size() == getVectorDataSize( y ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->addScalar(
            *getVectorDataComponent( x, i ), alpha_in, *getVectorDataComponent( y, i ) );
}

void MultiVectorOperations::setMax( const Scalar &alpha_in, VectorData &x )
{
    PROFILE( "MultiVectorOperations::setMax" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->setMax( alpha_in, *getVectorDataComponent( x, i ) );
}

void MultiVectorOperations::setMin( const Scalar &alpha_in, VectorData &x )
{
    PROFILE( "MultiVectorOperations::setMin" );

    if ( d_operations.empty() )
        return;
    AMP_ASSERT( d_operations.size() == getVectorDataSize( x ) );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->setMin( alpha_in, *getVectorDataComponent( x, i ) );
}

Scalar MultiVectorOperations::localMin( const VectorData &x ) const
{
    PROFILE( "MultiVectorOperations::localMin" );

    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0;
    auto ans = d_operations[0]->localMin( *getVectorDataComponent( x, 0 ) );
    for ( size_t i = 1; i != d_operations.size(); i++ )
        ans = std::min( ans, d_operations[i]->localMin( *getVectorDataComponent( x, i ) ) );
    if ( !ans.has_value() )
        return 0;
    return ans;
}

Scalar MultiVectorOperations::localMax( const VectorData &x ) const
{
    PROFILE( "MultiVectorOperations::localMax" );

    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0;
    auto ans = d_operations[0]->localMax( *getVectorDataComponent( x, 0 ) );
    for ( size_t i = 1; i != d_operations.size(); i++ )
        ans = std::max( ans, d_operations[i]->localMax( *getVectorDataComponent( x, i ) ) );
    if ( !ans.has_value() )
        return 0;
    return ans;
}

Scalar MultiVectorOperations::localSum( const VectorData &x ) const
{
    PROFILE( "MultiVectorOperations::localSum" );

    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0;
    auto ans = d_operations[0]->localSum( *getVectorDataComponent( x, 0 ) );
    for ( size_t i = 1; i != d_operations.size(); i++ )
        ans = ans + d_operations[i]->localSum( *getVectorDataComponent( x, i ) );
    if ( !ans.has_value() )
        return 0;
    return ans;
}

Scalar MultiVectorOperations::localL1Norm( const VectorData &x ) const
{
    PROFILE( "MultiVectorOperations::localL1Norm" );

    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0;
    Scalar ans;
    for ( size_t i = 0; i != d_operations.size(); i++ )
        ans = ans + d_operations[i]->localL1Norm( *getVectorDataComponent( x, i ) );
    if ( !ans.has_value() )
        return 0;
    return ans;
}

Scalar MultiVectorOperations::localL2Norm( const VectorData &x ) const
{
    PROFILE( "MultiVectorOperations::localL2Norm" );

    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0;
    Scalar ans;
    for ( size_t i = 0; i != d_operations.size(); i++ ) {
        auto tmp = d_operations[i]->localL2Norm( *getVectorDataComponent( x, i ) );
        ans      = ans + tmp * tmp;
    }
    if ( !ans.has_value() )
        return 0;
    return ans.sqrt();
}

Scalar MultiVectorOperations::localMaxNorm( const VectorData &x ) const
{
    PROFILE( "MultiVectorOperations::localMaxNorm" );

    AMP_ASSERT( getMultiVectorData( x ) );
    if ( d_operations.empty() )
        return 0;
    auto ans = d_operations[0]->localMaxNorm( *getVectorDataComponent( x, 0 ) );
    for ( size_t i = 1; i != d_operations.size(); i++ )
        ans = std::max( ans, d_operations[i]->localMaxNorm( *getVectorDataComponent( x, i ) ) );
    if ( !ans.has_value() )
        return 0;
    return ans;
}

Scalar MultiVectorOperations::localDot( const VectorData &x, const VectorData &y ) const
{
    PROFILE( "MultiVectorOperations::localDot" );

    if ( d_operations.empty() )
        return 0;
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->getVectorDataSize() );
        AMP_ASSERT( d_operations.size() == y2->getVectorDataSize() );
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
    return 0;
}

Scalar MultiVectorOperations::localMinQuotient( const VectorData &x, const VectorData &y ) const
{
    PROFILE( "MultiVectorOperations::localMinQuotient" );

    if ( d_operations.empty() )
        return std::numeric_limits<double>::max();
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->getVectorDataSize() );
        AMP_ASSERT( d_operations.size() == y2->getVectorDataSize() );
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
    return 0;
}

Scalar MultiVectorOperations::localWrmsNorm( const VectorData &x, const VectorData &y ) const
{
    PROFILE( "MultiVectorOperations::localWrmsNorm" );

    if ( d_operations.empty() )
        return 0;
    auto x2 = getMultiVectorData( x );
    auto y2 = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->getVectorDataSize() );
        AMP_ASSERT( d_operations.size() == y2->getVectorDataSize() );
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
    return 0;
}

Scalar MultiVectorOperations::localWrmsNormMask( const VectorData &x,
                                                 const VectorData &mask,
                                                 const VectorData &y ) const
{
    PROFILE( "MultiVectorOperations::localWrmsNormMask" );

    if ( d_operations.empty() )
        return 0;
    auto x2 = getMultiVectorData( x );
    auto m2 = getMultiVectorData( mask );
    auto y2 = getMultiVectorData( y );
    if ( x2 && m2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->getVectorDataSize() );
        AMP_ASSERT( d_operations.size() == m2->getVectorDataSize() );
        AMP_ASSERT( d_operations.size() == y2->getVectorDataSize() );
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
    return 0;
}

bool MultiVectorOperations::localEquals( const VectorData &x,
                                         const VectorData &y,
                                         const Scalar &tol ) const
{
    PROFILE( "MultiVectorOperations::localEquals" );

    if ( d_operations.empty() )
        return false;
    bool ans = true;
    auto x2  = getMultiVectorData( x );
    auto y2  = getMultiVectorData( y );
    if ( x2 && y2 ) {
        AMP_ASSERT( d_operations.size() == x2->getVectorDataSize() );
        AMP_ASSERT( d_operations.size() == y2->getVectorDataSize() );
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


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void MultiVectorOperations::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    for ( auto ops : d_operations )
        manager->registerObject( ops );
}
void MultiVectorOperations::writeRestart( int64_t fid ) const
{
    std::vector<uint64_t> opsHash( d_operations.size() );
    for ( size_t i = 0; i < d_operations.size(); i++ )
        opsHash[i] = d_operations[i]->getID();
    IO::writeHDF5( fid, "VectorOperationsHash", opsHash );
}
MultiVectorOperations::MultiVectorOperations( int64_t fid, AMP::IO::RestartManager *manager )
{
    std::vector<uint64_t> opsHash;
    IO::readHDF5( fid, "VectorOperationsHash", opsHash );
    d_operations.resize( opsHash.size() );
    for ( size_t i = 0; i < d_operations.size(); i++ )
        d_operations[i] = manager->getData<VectorOperations>( opsHash[i] );
}


} // namespace AMP::LinearAlgebra
