#include "vectors/operations/MultiVectorOperations.h"


namespace AMP {
namespace LinearAlgebra {


/****************************************************************
* Constructors                                                  *
****************************************************************/
AMP::shared_ptr<VectorOperations> MultiVectorOperations::cloneOperations() const
{
    auto ptr = AMP::make_shared<MultiVectorOperations>();
    return ptr;
}


/****************************************************************
* min, max, norms, etc.                                         *
****************************************************************/
bool MultiVectorOperations::localEquals( const VectorOperations &x, double tol ) const
{
    auto x2     = dynamic_cast<const MultiVectorOperations *>( &x );
    bool equals = true;
    if ( x2 ) {
        // Both this and x are multivectors
        AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
        for ( size_t i = 0; i < d_operations.size(); i++ )
            equals = equals && d_operations[i]->localEquals( *( x2->d_operations[i] ), tol );
    } else {
        // x is not a multivector, try to call the dot on x
        equals = x.localEquals( *this, tol );
    }
    return equals;
}
double MultiVectorOperations::localMin( void ) const
{
    double ans = 1e300;
    for ( auto &op : d_operations )
        ans = std::min( ans, op->localMin() );
    return ans;
}
double MultiVectorOperations::localMax( void ) const
{
    double ans = -1e300;
    for ( auto &op : d_operations )
        ans = std::max( ans, op->localMax() );
    return ans;
}
double MultiVectorOperations::localL1Norm() const
{
    double ans = 0.0;
    for ( auto &op : d_operations )
        ans += op->localL1Norm();
    return ans;
}
double MultiVectorOperations::localL2Norm() const
{
    double ans = 0.0;
    for ( auto &op : d_operations ) {
        double tmp = op->localL2Norm();
        ans += tmp * tmp;
    }
    return sqrt( ans );
}
double MultiVectorOperations::localMaxNorm() const
{
    double ans = 0.0;
    for ( auto &op : d_operations )
        ans = std::max( ans, op->localMaxNorm() );
    return ans;
}
double MultiVectorOperations::localDot( const VectorOperations &x ) const
{
    if ( d_operations.empty() ) {
        return 0;
    }
    auto x2    = dynamic_cast<const MultiVectorOperations *>( &x );
    double ans = 0.0;
    if ( x2 ) {
        // Both this and x are multivectors
        AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
        for ( size_t i = 0; i < d_operations.size(); i++ )
            ans += d_operations[i]->localDot( *( x2->d_operations[i] ) );
    } else {
        // x is not a multivector, try to call the dot on x
        ans = x.localDot( *this );
    }
    return ans;
}
double MultiVectorOperations::localMinQuotient( const VectorOperations &x ) const
{
    if ( d_operations.empty() ) {
        return std::numeric_limits<double>::max();
    }
    auto x2    = dynamic_cast<const MultiVectorOperations *>( &x );
    double ans = std::numeric_limits<double>::max();
    if ( x2 ) {
        // Both this and x are multivectors
        AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
        for ( size_t i = 0; i < d_operations.size(); i++ ) {
            double tmp = d_operations[i]->localMinQuotient( *( x2->d_operations[i] ) );
            ans        = std::min( ans, tmp );
        }
    } else {
        // x is not a multivector, try to call the localMinQuotient on x
        ans = x.localMinQuotient( *this );
    }
    return ans;
}
double MultiVectorOperations::localWrmsNorm( const VectorOperations &x ) const
{
    if ( d_operations.empty() ) {
        return 0;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    double ans = 0;
    for ( size_t i = 0; i < d_operations.size(); i++ ) {
        double tmp = d_operations[i]->localWrmsNorm( *( x2->d_operations[i] ) );
        size_t N1  = d_operations[i]->getVectorData()->getLocalSize();
        ans += tmp * tmp * N1;
    }
    size_t N = getVectorData()->getLocalSize();
    return sqrt( ans / N );
}
double MultiVectorOperations::localWrmsNormMask( const VectorOperations &x,
                                                 const VectorOperations &mask ) const
{
    if ( d_operations.empty() ) {
        return 0;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    auto m2 = dynamic_cast<const MultiVectorOperations *>( &mask );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_INSIST( m2 != nullptr, "mask is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    AMP_ASSERT( d_operations.size() == m2->d_operations.size() );
    double ans = 0;
    for ( size_t i = 0; i < d_operations.size(); i++ ) {
        double tmp = d_operations[i]->localWrmsNormMask( *( x2->d_operations[i] ),
                                                         *( m2->d_operations[i] ) );
        size_t N1 = d_operations[i]->getVectorData()->getLocalSize();
        ans += tmp * tmp * N1;
    }
    size_t N = getVectorData()->getLocalSize();
    return sqrt( ans / N );
}


/****************************************************************
* Functions to initalize the data                               *
****************************************************************/
void MultiVectorOperations::copy( const VectorOperations &x )
{
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    if ( x2 ) {
        // Both this and x are multivectors
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->copy( *( x2->d_operations[i] ) );
    } else {
        // x is not a multivector, try to call a default implimentation
        auto y2 = d_VectorData;
        auto x2 = x.getVectorData();
        AMP_ASSERT( x2->getLocalSize() == y2->getLocalSize() );
        if ( x2->isType<double>() && y2->isType<double>() ) {
            std::copy( x2->begin<double>(), x2->end<double>(), y2->begin<double>() );
        } else if ( x2->isType<float>() && y2->isType<float>() ) {
            std::copy( x2->begin<float>(), x2->end<float>(), y2->begin<float>() );
        } else {
            AMP_ERROR( "Unable to discern data types" );
        }
    }
}
void MultiVectorOperations::zero()
{
    for ( auto &op : d_operations )
        op->zero();
}
void MultiVectorOperations::setToScalar( double alpha )
{
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->setToScalar( alpha );
}
void MultiVectorOperations::setRandomValues()
{
    for ( auto &op : d_operations )
        op->setRandomValues();
}
void MultiVectorOperations::setRandomValues( RNG::shared_ptr rng )
{
    for ( auto &op : d_operations )
        op->setRandomValues( rng );
}
void MultiVectorOperations::addScalar( const VectorOperations &x, double alpha )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->addScalar( *( x2->d_operations[i] ), alpha );
}


/****************************************************************
* Basic linear algebra                                          *
****************************************************************/
void MultiVectorOperations::reciprocal( const VectorOperations &x )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->reciprocal( *( x2->d_operations[i] ) );
}
void MultiVectorOperations::add( const VectorOperations &x, const VectorOperations &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    auto y2 = dynamic_cast<const MultiVectorOperations *>( &y );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_INSIST( y2 != nullptr, "y is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    AMP_ASSERT( d_operations.size() == y2->d_operations.size() );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->add( *( x2->d_operations[i] ), *( y2->d_operations[i] ) );
}
void MultiVectorOperations::subtract( const VectorOperations &x, const VectorOperations &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    auto y2 = dynamic_cast<const MultiVectorOperations *>( &y );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_INSIST( y2 != nullptr, "y is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    AMP_ASSERT( d_operations.size() == y2->d_operations.size() );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->subtract( *( x2->d_operations[i] ), *( y2->d_operations[i] ) );
}
void MultiVectorOperations::multiply( const VectorOperations &x, const VectorOperations &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    auto y2 = dynamic_cast<const MultiVectorOperations *>( &y );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_INSIST( y2 != nullptr, "y is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    AMP_ASSERT( d_operations.size() == y2->d_operations.size() );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->multiply( *( x2->d_operations[i] ), *( y2->d_operations[i] ) );
}
void MultiVectorOperations::divide( const VectorOperations &x, const VectorOperations &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    auto y2 = dynamic_cast<const MultiVectorOperations *>( &y );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_INSIST( y2 != nullptr, "y is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    AMP_ASSERT( d_operations.size() == y2->d_operations.size() );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->divide( *( x2->d_operations[i] ), *( y2->d_operations[i] ) );
}
void MultiVectorOperations::linearSum( double alpha,
                                       const VectorOperations &x,
                                       double beta,
                                       const VectorOperations &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    auto y2 = dynamic_cast<const MultiVectorOperations *>( &y );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_INSIST( y2 != nullptr, "y is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    AMP_ASSERT( d_operations.size() == y2->d_operations.size() );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->linearSum(
            alpha, *( x2->d_operations[i] ), beta, *( y2->d_operations[i] ) );
}
void MultiVectorOperations::axpy( double alpha,
                                  const VectorOperations &x,
                                  const VectorOperations &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    auto y2 = dynamic_cast<const MultiVectorOperations *>( &y );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_INSIST( y2 != nullptr, "y is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    AMP_ASSERT( d_operations.size() == y2->d_operations.size() );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->axpy( alpha, *( x2->d_operations[i] ), *( y2->d_operations[i] ) );
}
void MultiVectorOperations::axpby( double alpha, double beta, const VectorOperations &x )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    if ( x2 != nullptr ) {
        // this and x are both multivectors
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->axpby( alpha, beta, *( x2->d_operations[i] ) );
    } else if ( d_operations.size() == 1 ) {
        // x is not a multivector, but this only contains one vector, try comparing
        d_operations[0]->axpby( alpha, beta, x );
    } else {
        AMP_ERROR( "x is not a multivector and this is" );
    }
}
void MultiVectorOperations::abs( const VectorOperations &x )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->abs( *( x2->d_operations[i] ) );
}
void MultiVectorOperations::scale( double alpha, const VectorOperations &x )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = dynamic_cast<const MultiVectorOperations *>( &x );
    AMP_INSIST( x2 != nullptr, "x is not a multivector and this is" );
    AMP_ASSERT( d_operations.size() == x2->d_operations.size() );
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->scale( alpha, *( x2->d_operations[i] ) );
}
void MultiVectorOperations::scale( double alpha )
{
    if ( d_operations.empty() ) {
        return;
    }
    for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->scale( alpha );
}


} // LinearAlgebra namespace
} // AMP namespace
