#include "AMP/vectors/trilinos/epetra/EpetraVectorOperations.h"
#include "AMP/vectors/trilinos/epetra/EpetraVectorEngine.h"

namespace AMP {
namespace LinearAlgebra {

static inline const Epetra_Vector &getEpetraVector( const VectorData &vec )
{
    auto epetraData = dynamic_cast<const EpetraVectorData *>( &vec );
    if ( epetraData )
        return epetraData->getEpetra_Vector();
    AMP_ERROR( "Not an EpetraVectorData" );
}
static inline Epetra_Vector &getEpetraVector( VectorData &vec )
{
    auto epetraData = dynamic_cast<EpetraVectorData *>( &vec );
    if ( epetraData )
        return epetraData->getEpetra_Vector();
    AMP_ERROR( "Not an EpetraVectorData" );
}
//**********************************************************************
// Functions that operate on VectorData objects

void EpetraVectorOperations::setToScalar( double alpha, VectorData &x )
{
    getEpetraVector( x ).PutScalar( alpha );
}

void EpetraVectorOperations::setRandomValues( VectorData &x )
{
    getEpetraVector( x ).Random();
    abs( x, x );
}

void EpetraVectorOperations::scale( double alpha, const VectorData &x, VectorData &y )
{
    getEpetraVector( y ).Scale( alpha, getEpetraVector( x ) );
}

void EpetraVectorOperations::scale( double alpha, VectorData &x )
{
    getEpetraVector( x ).Scale( alpha );
}

void EpetraVectorOperations::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    getEpetraVector( z ).Update( 1., getEpetraVector( x ), 1., getEpetraVector( y ), 0. );
}

void EpetraVectorOperations::subtract( const VectorData &x, const VectorData &y, VectorData &z )
{
    getEpetraVector( z ).Update( 1., getEpetraVector( x ), -1., getEpetraVector( y ), 0. );
}

void EpetraVectorOperations::multiply( const VectorData &x, const VectorData &y, VectorData &z )
{
    getEpetraVector( z ).Multiply( 1., getEpetraVector( x ), getEpetraVector( y ), 0. );
}

void EpetraVectorOperations::divide( const VectorData &x, const VectorData &y, VectorData &z )
{
    getEpetraVector( z ).ReciprocalMultiply( 1., getEpetraVector( y ), getEpetraVector( x ), 0. );
}

void EpetraVectorOperations::reciprocal( const VectorData &x, VectorData &y )
{
    getEpetraVector( y ).Reciprocal( getEpetraVector( x ) );
}

void EpetraVectorOperations::linearSum(
    double alpha, const VectorData &x, double beta, const VectorData &y, VectorData &z )
{
    getEpetraVector( z ).Update( alpha, getEpetraVector( x ), beta, getEpetraVector( y ), 0. );
}

void EpetraVectorOperations::axpy( double alpha,
                                   const VectorData &x,
                                   const VectorData &y,
                                   VectorData &z )
{
    linearSum( alpha, x, 1.0, y, z );
}

void EpetraVectorOperations::axpby( double alpha, double beta, const VectorData &x, VectorData &z )
{
    linearSum( alpha, x, beta, z, z );
}

void EpetraVectorOperations::abs( const VectorData &x, VectorData &y )
{
    getEpetraVector( y ).Abs( getEpetraVector( x ) );
}

double EpetraVectorOperations::min( const VectorData &x ) const
{
    double retVal;
    getEpetraVector( x ).MinValue( &retVal );
    return retVal;
}

double EpetraVectorOperations::max( const VectorData &x ) const
{
    double retVal;
    getEpetraVector( x ).MaxValue( &retVal );
    return retVal;
}

double EpetraVectorOperations::dot( const VectorData &x, const VectorData &y ) const
{
    double retVal;
    getEpetraVector( y ).Dot( getEpetraVector( x ), &retVal );
    return retVal;
}

double EpetraVectorOperations::L1Norm( const VectorData &x ) const
{
    double retVal;
    getEpetraVector( x ).Norm1( &retVal );
    return retVal;
}

double EpetraVectorOperations::L2Norm( const VectorData &x ) const
{
    double retVal;
    getEpetraVector( x ).Norm2( &retVal );
    return retVal;
}

double EpetraVectorOperations::maxNorm( const VectorData &x ) const
{
    double retVal;
    getEpetraVector( x ).NormInf( &retVal );
    return retVal;
}

} // namespace LinearAlgebra
} // namespace AMP
