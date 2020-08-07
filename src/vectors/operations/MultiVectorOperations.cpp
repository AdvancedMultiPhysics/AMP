#include "AMP/vectors/operations/MultiVectorOperations.h"
#include "AMP/vectors/data/MultiVectorData.h"


namespace AMP {
namespace LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
std::shared_ptr<VectorOperations> MultiVectorOperations::cloneOperations() const
{
    auto ptr = std::make_shared<MultiVectorOperations>();
    return ptr;
}

#if 0
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
double MultiVectorOperations::localMin() const
{
    double ans = 1e300;
    for ( auto &op : d_operations )
        ans = std::min( ans, op->localMin() );
    return ans;
}
double MultiVectorOperations::localMax() const
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
        size_t N1  = d_operations[i]->getVectorData()->getLocalSize();
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
    auto xc = dynamic_cast<const MultiVectorOperations *>( &x );
    if ( xc ) {
        // Both this and x are multivectors
        for ( size_t i = 0; i != d_operations.size(); i++ )
            d_operations[i]->copy( *( xc->d_operations[i] ) );
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
#endif
//**********************************************************************
// Static functions that operate on VectorData objects

VectorData *MultiVectorOperations::getVectorDataComponent( VectorData &x, size_t i )
{
  auto x2 = dynamic_cast<MultiVectorData *>( &x );
  AMP_ASSERT(x2 && (i<x2->numberOfComponents()));
  return x2->getVectorData(i);
}
const VectorData *MultiVectorOperations::getVectorDataComponent( const VectorData &x, size_t i )
{
  auto x2 = dynamic_cast<const MultiVectorData *>( &x );
  AMP_ASSERT(x2 && (i<x2->numberOfComponents()));
  return x2->getVectorData(i);
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
  auto mData = getMultiVectorData(x);
  
  for ( size_t i = 0; i != mData->numberOfComponents(); ++i ) {
    
    d_operations[i]->zero( *getVectorDataComponent(x,i) );
  }
}

void MultiVectorOperations::setToScalar( double alpha, VectorData &x )
{
  for ( size_t i = 0; i != d_operations.size(); i++ ) {
    d_operations[i]->setToScalar( alpha,  *getVectorDataComponent(x,i) );
  }
}

void MultiVectorOperations::setRandomValues( VectorData &x )
{
  for ( size_t i = 0; i != d_operations.size(); i++ ) {
    d_operations[i]->setRandomValues( *getVectorDataComponent(x,i) );
  }
}

void MultiVectorOperations::setRandomValues( RNG::shared_ptr rng, VectorData &x )
{
  for ( size_t i = 0; i != d_operations.size(); i++ ) {
    d_operations[i]->setRandomValues( rng, *getVectorDataComponent(x,i) );
  }
}

void MultiVectorOperations::copy( const VectorData &x, VectorData &y )
{
  
  auto xc = getMultiVectorData(x);
  auto yc = getMultiVectorData(y);

  if ( xc && yc ) {
        // Both this and x are multivectors
        for ( size_t i = 0; i != d_operations.size(); i++ )
	  d_operations[i]->copy( *getVectorDataComponent(x,i), *getVectorDataComponent(y,i) );
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

void MultiVectorOperations::scale( double alpha, VectorData &x )
{
    AMP_ASSERT(getMultiVectorData(x));
    if ( d_operations.empty() ) {
        return;
    }
    for ( size_t i = 0; i != d_operations.size(); i++ )
      d_operations[i]->scale( alpha, *getVectorDataComponent(x,i));
}

void MultiVectorOperations::scale( double alpha, const VectorData &x, VectorData &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {
      AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
      for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->scale( alpha, *getVectorDataComponent(x,i), *getVectorDataComponent(y,i) );
      
    } else {
      AMP_ERROR("MultiVectorOperations::scale requires both x and y to be MultiVectorData");
    }
}

void MultiVectorOperations::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {
        auto z2 = getMultiVectorData(y);
	AMP_ASSERT( z2 );
	AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
	AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
	for ( size_t i = 0; i != d_operations.size(); i++ )
	  d_operations[i]->add( *getVectorDataComponent(x,i),
				*getVectorDataComponent(y,i),
				*getVectorDataComponent(z,i) ) ;
    } else {
       AMP_ERROR("MultiVectorOperations::add requires x, y, z to be MultiVectorData");
    }
}

void MultiVectorOperations::subtract( const VectorData &x, const VectorData &y, VectorData &z  )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {
        auto z2 = getMultiVectorData(y);
	AMP_ASSERT( z2 );
	AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
	AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
	for ( size_t i = 0; i != d_operations.size(); i++ )
	  d_operations[i]->subtract( *getVectorDataComponent(x,i),
				*getVectorDataComponent(y,i),
				*getVectorDataComponent(z,i) ) ;
    } else {
       AMP_ERROR("MultiVectorOperations::subtract requires x, y, z to be MultiVectorData");
    }
}

void MultiVectorOperations::multiply( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {
        auto z2 = getMultiVectorData(y);
	AMP_ASSERT( z2 );
	AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
	AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
	for ( size_t i = 0; i != d_operations.size(); i++ )
	  d_operations[i]->multiply( *getVectorDataComponent(x,i),
				     *getVectorDataComponent(y,i),
				     *getVectorDataComponent(z,i) ) ;
    } else {
       AMP_ERROR("MultiVectorOperations::multiply requires x, y, z to be MultiVectorData");
    }
}

void MultiVectorOperations::divide( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {
        auto z2 = getMultiVectorData(y);
	AMP_ASSERT( z2 );
	AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
	AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
	for ( size_t i = 0; i != d_operations.size(); i++ )
	  d_operations[i]->divide( *getVectorDataComponent(x,i),
				   *getVectorDataComponent(y,i),
				   *getVectorDataComponent(z,i) ) ;
    } else {
       AMP_ERROR("MultiVectorOperations::divide requires x, y, z to be MultiVectorData");
    }
}

void MultiVectorOperations::reciprocal( const VectorData &x, VectorData &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {
      AMP_ASSERT( d_operations.size()  == y2->numberOfComponents() );
      AMP_ASSERT( x2->numberOfComponents() == y2->numberOfComponents() );
      for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->reciprocal( *getVectorDataComponent(x,i),
				     *getVectorDataComponent(y,i) );
    } else {
       AMP_ERROR("MultiVectorOperations::reciprocal requires both x and y to be MultiVectorData");
    }
}

void MultiVectorOperations::linearSum( double alpha_in,
				       const VectorData &x,
				       double beta_in,
				       const VectorData &y,
				       VectorData &z)
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {
        auto z2 = getMultiVectorData(y);
	AMP_ASSERT( z2 );
	AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
	AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
	AMP_ASSERT( d_operations.size() == z2->numberOfComponents() );
	for ( size_t i = 0; i != d_operations.size(); i++ )
	  d_operations[i]->linearSum( alpha_in,
				      *getVectorDataComponent(x,i),
				      beta_in,
				      *getVectorDataComponent(y,i),
				      *getVectorDataComponent(z,i) );
	
    } else {
       AMP_ERROR("MultiVectorOperations::linearSum requires x, y, z to be MultiVectorData");
    }
}

void MultiVectorOperations::axpy( double alpha_in, const VectorData &x, const VectorData &y, VectorData &z )
{
  linearSum( alpha_in, x, 1.0, y, z);
}

void MultiVectorOperations::axpby( double alpha_in, double beta_in, const VectorData &x, VectorData &z )
{
  linearSum( alpha_in, x, beta_in, z, z);
}

void MultiVectorOperations::abs( const VectorData &x, VectorData &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {
      AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
      AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
      for ( size_t i = 0; i != d_operations.size(); i++ ){
	std::cout << "MultiVector component x " << i <<  std::endl;
	getVectorDataComponent(x,i)->dumpOwnedData(AMP::pout);
	std::cout << "MultiVector component y " << i <<  std::endl;
	getVectorDataComponent(y,i)->dumpOwnedData(AMP::pout);
        d_operations[i]->abs( *getVectorDataComponent(x,i),
			      *getVectorDataComponent(y,i) );
      }
    } else {
       AMP_ERROR("MultiVectorOperations::abs requires x, y to be MultiVectorData");
    }
}

void MultiVectorOperations::addScalar( const VectorData &x, double alpha_in, VectorData &y )
{
    if ( d_operations.empty() ) {
        return;
    }
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {      
      AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
      AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
      for ( size_t i = 0; i != d_operations.size(); i++ )
        d_operations[i]->addScalar( *getVectorDataComponent(x,i),
				    alpha_in,
				    *getVectorDataComponent(y,i) );
    } else {
       AMP_ERROR("MultiVectorOperations::linearSum requires x, y to be MultiVectorData");
    }
}

double MultiVectorOperations::localMin( const VectorData &x ) const
{
    double ans = 1e300;
    AMP_ASSERT(getMultiVectorData(x));
    if ( d_operations.empty() ) {
        return 0;
    }
    for ( size_t i = 0; i != d_operations.size(); i++ )
        ans = std::min( ans, d_operations[i]->localMin( *getVectorDataComponent(x,i)) );
    return ans;
}

double MultiVectorOperations::localMax( const VectorData &x )  const
{
    double ans = -1e300;
    AMP_ASSERT(getMultiVectorData(x));
    if ( d_operations.empty() ) {
        return 0;
    }
    for ( size_t i = 0; i != d_operations.size(); i++ )
        ans = std::max( ans, d_operations[i]->localMax( *getVectorDataComponent(x,i)) );
    return ans;
}

double MultiVectorOperations::localL1Norm( const VectorData &x ) const 
{
    double ans = 0.0;
    AMP_ASSERT(getMultiVectorData(x));
    if ( d_operations.empty() ) {
        return 0;
    }
    for ( size_t i = 0; i != d_operations.size(); i++ )
        ans += d_operations[i]->localL1Norm( *getVectorDataComponent(x,i));
    return ans;
}

double MultiVectorOperations::localL2Norm( const VectorData &x )  const
{
    double ans = 0.0;
    AMP_ASSERT(getMultiVectorData(x));
    if ( d_operations.empty() ) {
        return 0;
    }
    for ( size_t i = 0; i != d_operations.size(); i++ ) {
        const auto tmp = d_operations[i]->localL2Norm( *getVectorDataComponent(x,i)); 
        ans += tmp*tmp;
    }
	return sqrt(ans);
}

double MultiVectorOperations::localMaxNorm( const VectorData &x )  const
{
    double ans = 0.0;
    AMP_ASSERT(getMultiVectorData(x));
    if ( d_operations.empty() ) {
        return 0;
    }
    for ( size_t i = 0; i != d_operations.size(); i++ )
      ans = std::max(ans, d_operations[i]->localMaxNorm( *getVectorDataComponent(x,i)));
    return ans;
}

double MultiVectorOperations::localDot( const VectorData &x, const VectorData &y ) const
{
    if ( d_operations.empty() ) {
        return 0;
    }

    double ans = 0.0;
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {      
      AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
      AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
      for ( size_t i = 0; i != d_operations.size(); i++ )
        ans+= d_operations[i]->localDot( *getVectorDataComponent(x,i),
					 *getVectorDataComponent(y,i) );
    } else {
       AMP_ERROR("MultiVectorOperations::localMinQuotient requires x, y to be MultiVectorData");
    }
    return ans;
}

double MultiVectorOperations::localMinQuotient( const VectorData &x, const VectorData &y ) const
{
    if ( d_operations.empty() ) {
        return std::numeric_limits<double>::max();
    }

    double ans = std::numeric_limits<double>::max();
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {      
      AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
      AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
      for ( size_t i = 0; i != d_operations.size(); i++ )
        ans = std::min(ans, d_operations[i]->localMinQuotient( *getVectorDataComponent(x,i),
							       *getVectorDataComponent(y,i) ) );
    } else {
       AMP_ERROR("MultiVectorOperations::localMinQuotient requires x, y to be MultiVectorData");
    }
    return ans;
}

double MultiVectorOperations::localWrmsNorm( const VectorData &x, const VectorData &y ) const
{
    if ( d_operations.empty() ) {
        return 0;
    }
    double ans = 0.0;
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {      
      AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
      AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
      for ( size_t i = 0; i < d_operations.size(); i++ ) {
	auto yi = getVectorDataComponent(y,i);
        double tmp = d_operations[i]->localWrmsNorm( *getVectorDataComponent(x,i),
						     *yi );
        size_t N1  = yi->getLocalSize();
        ans += tmp * tmp * N1;
      }
      size_t N = y.getLocalSize();
      return sqrt( ans / N );
    } else {
       AMP_ERROR("MultiVectorOperations::localWrmsNorm requires x, y to be MultiVectorData");
    }
    return ans;
}

double MultiVectorOperations::localWrmsNormMask( const VectorData &x, const VectorData &mask, const VectorData &y ) const
{
    if ( d_operations.empty() ) {
        return 0;
    }
    double ans = 0.0;
    auto x2 = getMultiVectorData(x);
    auto m2 = getMultiVectorData(mask);
    auto y2 = getMultiVectorData(y);
    if ( x2 && m2 && y2 ) {      
      AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
      AMP_ASSERT( d_operations.size() == m2->numberOfComponents() );
      AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
      for ( size_t i = 0; i < d_operations.size(); i++ ) {
	auto yi = getVectorDataComponent(y,i);
        double tmp = d_operations[i]->localWrmsNormMask( *getVectorDataComponent(x,i),
							 *getVectorDataComponent(mask,i),
							 *yi );
        size_t N1  = yi->getLocalSize();
        ans += tmp * tmp * N1;
      }
      size_t N = y.getLocalSize();
      return sqrt( ans / N );
    } else {
       AMP_ERROR("MultiVectorOperations::localWrmsNormMask requires x, mask, y to be MultiVectorData");
    }
    return ans;
}

bool MultiVectorOperations::localEquals( const VectorData &x, const VectorData &y, double tol ) const
{
    if ( d_operations.empty() ) {
        return false;
    }
    bool ans = 0.0;
    auto x2 = getMultiVectorData(x);
    auto y2 = getMultiVectorData(y);
    if ( x2 && y2 ) {      
      AMP_ASSERT( d_operations.size() == x2->numberOfComponents() );
      AMP_ASSERT( d_operations.size() == y2->numberOfComponents() );
      for ( size_t i = 0; i < d_operations.size(); i++ ) {
        ans = ans && d_operations[i]->localEquals( *getVectorDataComponent(x,i),
						   *getVectorDataComponent(y,i),
						   tol);
      }
    } else {
       AMP_ERROR("MultiVectorOperations::localEquals requires x, y to be MultiVectorData");
    }
    return ans;
}

} // namespace LinearAlgebra
} // namespace AMP
