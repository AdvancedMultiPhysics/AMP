#include "AMP/vectors/trilinos/epetra/EpetraVectorOperations.h"
#include "AMP/vectors/trilinos/epetra/EpetraVectorData.h"

namespace AMP::LinearAlgebra {

static inline const Epetra_Vector &getEpetraVector( const VectorData &vec )
{
    auto epetraData = dynamic_cast<const EpetraVectorData *>( &vec );
    AMP_INSIST( epetraData, "Not EpetraVectorData" );
    return epetraData->getEpetra_Vector();
}
static inline Epetra_Vector &getEpetraVector( VectorData &vec )
{
    auto epetraData = dynamic_cast<EpetraVectorData *>( &vec );
    AMP_INSIST( epetraData, "Not EpetraVectorData" );
    return epetraData->getEpetra_Vector();
}

static bool allEpetraDataType( const VectorData &x )
{
    constexpr std::string_view type( "EpetraVectorData", 17 );
    const auto xtype               = x.VectorDataName();
    const std::string_view xtype_s = xtype;
    return ( xtype_s.compare( 0, 17, type ) == 0 );
}
static bool allEpetraDataType( const VectorData &x, const VectorData &y )
{
    constexpr std::string_view type( "EpetraVectorData", 17 );
    const auto xtype               = x.VectorDataName();
    const auto ytype               = y.VectorDataName();
    const std::string_view xtype_s = xtype;
    const std::string_view ytype_s = ytype;
    return ( xtype_s.compare( ytype_s ) == 0 && xtype_s.compare( 0, 17, type ) == 0 );
}
static bool allEpetraDataType( const VectorData &x, const VectorData &y, VectorData &z )
{
    constexpr std::string_view type( "EpetraVectorData", 17 );
    const auto xtype               = x.VectorDataName();
    const auto ytype               = y.VectorDataName();
    const auto ztype               = z.VectorDataName();
    const std::string_view xtype_s = xtype;
    const std::string_view ytype_s = ytype;
    const std::string_view ztype_s = ztype;

    return ( xtype_s.compare( ytype_s ) == 0 && ytype_s.compare( ztype_s ) == 0 &&
             xtype_s.compare( 0, 17, type ) == 0 );
}

//**********************************************************************
// Functions that operate on VectorData objects

void EpetraVectorOperations::setToScalar( const Scalar &alpha, VectorData &x )
{
    if ( allEpetraDataType( x ) ) {
        getEpetraVector( x ).PutScalar( alpha.get<double>() );
    } else {
        VectorOperationsDefault<double>::setToScalar( alpha, x );
    }
}

void EpetraVectorOperations::setRandomValues( VectorData &x )
{
    getEpetraVector( x ).Random();
    abs( x, x );
}

void EpetraVectorOperations::scale( const Scalar &alpha, const VectorData &x, VectorData &y )
{
    if ( allEpetraDataType( x, y ) ) {
        getEpetraVector( y ).Scale( alpha.get<double>(), getEpetraVector( x ) );
    } else {
        VectorOperationsDefault<double>::scale( alpha, x, y );
    }
}

void EpetraVectorOperations::scale( const Scalar &alpha, VectorData &x )
{
    if ( allEpetraDataType( x ) ) {
        getEpetraVector( x ).Scale( alpha.get<double>() );
    } else {
        VectorOperationsDefault<double>::scale( alpha, x );
    }
}

void EpetraVectorOperations::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( allEpetraDataType( x, y, z ) ) {
        getEpetraVector( z ).Update( 1., getEpetraVector( x ), 1., getEpetraVector( y ), 0. );
    } else {
        VectorOperationsDefault<double>::add( x, y, z );
    }
}

void EpetraVectorOperations::subtract( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( allEpetraDataType( x, y, z ) ) {
        getEpetraVector( z ).Update( 1., getEpetraVector( x ), -1., getEpetraVector( y ), 0. );
    } else {
        VectorOperationsDefault<double>::subtract( x, y, z );
    }
}

void EpetraVectorOperations::multiply( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( allEpetraDataType( x, y, z ) ) {
        getEpetraVector( z ).Multiply( 1., getEpetraVector( x ), getEpetraVector( y ), 0. );
    } else {
        VectorOperationsDefault<double>::multiply( x, y, z );
    }
}

void EpetraVectorOperations::divide( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( allEpetraDataType( x, y, z ) ) {
        getEpetraVector( z ).ReciprocalMultiply(
            1., getEpetraVector( y ), getEpetraVector( x ), 0. );
    } else {
        VectorOperationsDefault<double>::divide( x, y, z );
    }
}

void EpetraVectorOperations::reciprocal( const VectorData &x, VectorData &y )
{
    if ( allEpetraDataType( x, y ) ) {
        getEpetraVector( y ).Reciprocal( getEpetraVector( x ) );
    } else {
        VectorOperationsDefault<double>::reciprocal( x, y );
    }
}

void EpetraVectorOperations::linearSum( const Scalar &alpha,
                                        const VectorData &x,
                                        const Scalar &beta,
                                        const VectorData &y,
                                        VectorData &z )
{
    if ( allEpetraDataType( x, y, z ) ) {
        getEpetraVector( z ).Update( alpha.get<double>(),
                                     getEpetraVector( x ),
                                     beta.get<double>(),
                                     getEpetraVector( y ),
                                     0. );
    } else {
        VectorOperationsDefault<double>::linearSum( alpha, x, beta, y, z );
    }
}

void EpetraVectorOperations::axpy( const Scalar &alpha,
                                   const VectorData &x,
                                   const VectorData &y,
                                   VectorData &z )
{
    if ( allEpetraDataType( x, y, z ) ) {
        linearSum( alpha.get<double>(), x, 1.0, y, z );
    } else {
        VectorOperationsDefault<double>::linearSum( alpha, x, 1.0, y, z );
    }
}

void EpetraVectorOperations::axpby( const Scalar &alpha,
                                    const Scalar &beta,
                                    const VectorData &x,
                                    VectorData &z )
{
    if ( allEpetraDataType( x, z ) ) {
        linearSum( alpha.get<double>(), x, beta.get<double>(), z, z );
    } else {
        VectorOperationsDefault<double>::linearSum( alpha, x, beta, z, z );
    }
}

void EpetraVectorOperations::abs( const VectorData &x, VectorData &y )
{
    if ( allEpetraDataType( x, y ) ) {
        getEpetraVector( y ).Abs( getEpetraVector( x ) );
    } else {
        VectorOperationsDefault<double>::abs( x, y );
    }
}

Scalar EpetraVectorOperations::min( const VectorData &x ) const
{
    double retVal;
    getEpetraVector( x ).MinValue( &retVal );
    return retVal;
}

Scalar EpetraVectorOperations::max( const VectorData &x ) const
{
    double retVal;
    getEpetraVector( x ).MaxValue( &retVal );
    return retVal;
}

Scalar EpetraVectorOperations::dot( const VectorData &x, const VectorData &y ) const
{
    if ( allEpetraDataType( x, y ) ) {
        double retVal;
        getEpetraVector( y ).Dot( getEpetraVector( x ), &retVal );
        return retVal;
    } else {
        return VectorOperationsDefault<double>::dot( x, y );
    }
}

Scalar EpetraVectorOperations::L1Norm( const VectorData &x ) const
{
    double retVal;
    getEpetraVector( x ).Norm1( &retVal );
    return retVal;
}

Scalar EpetraVectorOperations::L2Norm( const VectorData &x ) const
{
    double retVal;
    getEpetraVector( x ).Norm2( &retVal );
    return retVal;
}

Scalar EpetraVectorOperations::maxNorm( const VectorData &x ) const
{
    double retVal;
    getEpetraVector( x ).NormInf( &retVal );
    return retVal;
}

} // namespace AMP::LinearAlgebra
