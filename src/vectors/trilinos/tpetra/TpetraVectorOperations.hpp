#ifndef included_AMP_NativeTpetraVectorOperations_HPP_
#define included_AMP_NativeTpetraVectorOperations_HPP_

#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/trilinos/tpetra/TpetraVectorData.h"
#include "AMP/vectors/trilinos/tpetra/TpetraVectorOperations.h"

#include <Kokkos_Core.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Tpetra_Core.hpp>


namespace AMP::LinearAlgebra {


template<typename ST, typename LO, typename GO, typename NT>
static inline const Tpetra::Vector<ST, LO, GO, NT> &getTpetraVector( const VectorData &vec )
{
    auto tpetraData = dynamic_cast<const TpetraVectorData<ST, LO, GO, NT> *>( &vec );
    AMP_INSIST( tpetraData, "Not TpetraVectorData" );
    return *( tpetraData->getTpetraVector() );
}

template<typename ST, typename LO, typename GO, typename NT>
static inline Tpetra::Vector<ST, LO, GO, NT> &getTpetraVector( VectorData &vec )
{
    auto data = dynamic_cast<TpetraVectorData<ST, LO, GO, NT> *>( &vec );
    AMP_INSIST( data, "Not TpetraVectorData" );
    return *( data->getTpetraVector() );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::setToScalar( const AMP::Scalar &alpha, VectorData &x )
{
    getTpetraVector<ST, LO, GO, NT>( x ).putScalar( static_cast<ST>( alpha ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::addScalar( const VectorData &x,
                                                        const Scalar &val,
                                                        VectorData &y )
{
    const auto &xt = getTpetraVector<ST, LO, GO, NT>( x );
    auto &yt       = getTpetraVector<ST, LO, GO, NT>( y );
    AMP_ASSERT( xt.getNumVectors() == 1 && yt.getNumVectors() == 1 );
    auto xData = xt.getData( 0 );
    auto yData = yt.getDataNonConst( 0 );

    AMP_DEBUG_ASSERT( xData.size() == yData.size() );
    using size_type = typename decltype( xData )::size_type;
    size_type i     = 0;
    for ( ; i < xData.size(); ++i ) {
        yData[i] = xData[i] + static_cast<ST>( val );
    }
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::setMin( const AMP::Scalar &, VectorData & )
{
    AMP_ERROR( "TpetraVectorOperations::setMin not implemented" );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::setMax( const AMP::Scalar &, VectorData & )
{
    AMP_ERROR( "TpetraVectorOperations::setMax not implemented" );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::setRandomValues( VectorData &x )
{
    getTpetraVector<ST, LO, GO, NT>( x ).randomize( 0, 1 );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::copy( const VectorData &x, VectorData &z )
{
    if ( x.VectorDataName() == z.VectorDataName() ) {
        // assume both are TpetraVectorData
        deep_copy( getTpetraVector<ST, LO, GO, NT>( z ), getTpetraVector<ST, LO, GO, NT>( x ) );
    } else {
        if ( x.numberOfDataBlocks() == z.numberOfDataBlocks() && x.numberOfDataBlocks() == 1 ) {
            auto typeST = getTypeID<ST>();
            AMP_ASSERT( x.getType( 0 ) == z.getType( 0 ) && x.getType( 0 ) == typeST );
            AMP_ASSERT( x.sizeOfDataBlock() == z.sizeOfDataBlock() );
            const auto xvData = x.getRawDataBlockAsVoid( 0 );
            z.putRawData( xvData, typeST );
        } else {
            AMP_ERROR( "TpetraVectorOperations::copy for different VectorData only implemented for "
                       "one data block" );
        }
    }
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::scale( const Scalar &alpha,
                                                    const VectorData &x,
                                                    VectorData &y )
{
    getTpetraVector<ST, LO, GO, NT>( y ).scale( static_cast<ST>( alpha ),
                                                getTpetraVector<ST, LO, GO, NT>( x ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::scale( const Scalar &alpha, VectorData &x )
{
    getTpetraVector<ST, LO, GO, NT>( x ).scale( static_cast<ST>( alpha ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::add( const VectorData &x,
                                                  const VectorData &y,
                                                  VectorData &z )
{
    getTpetraVector<ST, LO, GO, NT>( z ).update( static_cast<ST>( 1.0 ),
                                                 getTpetraVector<ST, LO, GO, NT>( x ),
                                                 static_cast<ST>( 1.0 ),
                                                 getTpetraVector<ST, LO, GO, NT>( y ),
                                                 static_cast<ST>( 0.0 ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::subtract( const VectorData &x,
                                                       const VectorData &y,
                                                       VectorData &z )
{
    getTpetraVector<ST, LO, GO, NT>( z ).update( static_cast<ST>( 1.0 ),
                                                 getTpetraVector<ST, LO, GO, NT>( x ),
                                                 static_cast<ST>( -1.0 ),
                                                 getTpetraVector<ST, LO, GO, NT>( y ),
                                                 static_cast<ST>( 0.0 ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::multiply( const VectorData &x,
                                                       const VectorData &y,
                                                       VectorData &z )
{
    getTpetraVector<ST, LO, GO, NT>( z ).elementWiseMultiply( static_cast<ST>( 1.0 ),
                                                              getTpetraVector<ST, LO, GO, NT>( x ),
                                                              getTpetraVector<ST, LO, GO, NT>( y ),
                                                              static_cast<ST>( 0.0 ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::divide( const VectorData &x,
                                                     const VectorData &y,
                                                     VectorData &z )
{
    const auto &xt = getTpetraVector<ST, LO, GO, NT>( x );
    const auto &yt = getTpetraVector<ST, LO, GO, NT>( y );
    auto &zt       = getTpetraVector<ST, LO, GO, NT>( z );
    AMP_ASSERT( xt.getNumVectors() == 1 && yt.getNumVectors() == 1 && zt.getNumVectors() == 1 );
    const auto xData = xt.getData( 0 );
    const auto yData = yt.getData( 0 );
    auto zData       = zt.getDataNonConst( 0 );

    AMP_DEBUG_ASSERT( xData.size() == yData.size() && xData.size() == zData.size() );
    using size_type = typename decltype( xData )::size_type;
    size_type i     = 0;
    for ( ; i < xData.size(); ++i ) {
        zData[i] = xData[i] / yData[i];
    }
    // seems to be a bug in the elementWiseMultiply that zeroes out z
    // getTpetraVector<ST, LO, GO, NT>( z ).reciprocal( getTpetraVector<ST, LO, GO, NT>( y ) );
    // getTpetraVector<ST, LO, GO, NT>( z ).elementWiseMultiply( static_cast<ST>( 1.0 ),
    //                                                           getTpetraVector<ST, LO, GO, NT>( x
    //                                                           ), getTpetraVector<ST, LO, GO, NT>(
    //                                                           z ), static_cast<ST>( 0.0 ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::reciprocal( const VectorData &x, VectorData &y )
{
    getTpetraVector<ST, LO, GO, NT>( y ).reciprocal( getTpetraVector<ST, LO, GO, NT>( x ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::linearSum( const Scalar &alpha,
                                                        const VectorData &x,
                                                        const Scalar &beta,
                                                        const VectorData &y,
                                                        VectorData &z )
{
    getTpetraVector<ST, LO, GO, NT>( z ).update( static_cast<ST>( alpha ),
                                                 getTpetraVector<ST, LO, GO, NT>( x ),
                                                 static_cast<ST>( beta ),
                                                 getTpetraVector<ST, LO, GO, NT>( y ),
                                                 static_cast<ST>( 0.0 ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::axpy( const Scalar &alpha,
                                                   const VectorData &x,
                                                   const VectorData &y,
                                                   VectorData &z )
{
    getTpetraVector<ST, LO, GO, NT>( z ).update( static_cast<ST>( alpha ),
                                                 getTpetraVector<ST, LO, GO, NT>( x ),
                                                 static_cast<ST>( 1.0 ),
                                                 getTpetraVector<ST, LO, GO, NT>( y ),
                                                 static_cast<ST>( 0.0 ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::axpby( const Scalar &alpha,
                                                    const Scalar &beta,
                                                    const VectorData &x,
                                                    VectorData &y )
{
    getTpetraVector<ST, LO, GO, NT>( y ).update(
        static_cast<ST>( alpha ), getTpetraVector<ST, LO, GO, NT>( x ), static_cast<ST>( beta ) );
}

template<typename ST, typename LO, typename GO, typename NT>
void TpetraVectorOperations<ST, LO, GO, NT>::abs( const VectorData &x, VectorData &z )
{
    getTpetraVector<ST, LO, GO, NT>( z ).abs( getTpetraVector<ST, LO, GO, NT>( x ) );
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::min( const VectorData &x ) const
{
    auto lmin  = localMin( x );
    auto &comm = x.getComm();
    return comm.minReduce( lmin );
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::max( const VectorData &x ) const
{
    auto lmax  = localMax( x );
    auto &comm = x.getComm();
    return comm.maxReduce( lmax );
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::L1Norm( const VectorData &x ) const
{
    return getTpetraVector<ST, LO, GO, NT>( x ).norm1();
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::L2Norm( const VectorData &x ) const
{
    return getTpetraVector<ST, LO, GO, NT>( x ).norm2();
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::maxNorm( const VectorData &x ) const
{
    return getTpetraVector<ST, LO, GO, NT>( x ).normInf();
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::dot( const VectorData &x, const VectorData &y ) const
{
    return getTpetraVector<ST, LO, GO, NT>( x ).dot( getTpetraVector<ST, LO, GO, NT>( y ) );
}

template<typename ST, typename LO, typename GO, typename NT>
static Teuchos::RCP<Tpetra::Map<LO, GO, NT>>
getLocalMap( Teuchos::RCP<const Tpetra::Map<LO, GO, NT>> map )
{
    Teuchos::RCP<Tpetra::Map<LO, GO, NT>> lmap(
        new Tpetra::Map<LO, GO, NT>( map->getLocalNumElements(),
                                     map->getIndexBase(),
                                     map->getComm(),
                                     Tpetra::LocallyReplicated ) );
    return lmap;
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::localL1Norm( const VectorData &x ) const
{
    const auto &xt   = getTpetraVector<ST, LO, GO, NT>( x );
    const auto &lmap = getLocalMap<ST, LO, GO, NT>( xt.getMap() );
    const auto &xl   = xt.offsetView( lmap, 0 );
    return xl->norm1();
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::localL2Norm( const VectorData &x ) const
{
    const auto &xt   = getTpetraVector<ST, LO, GO, NT>( x );
    const auto &lmap = getLocalMap<ST, LO, GO, NT>( xt.getMap() );
    const auto &xl   = xt.offsetView( lmap, 0 );
    return xl->norm2();
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::localMaxNorm( const VectorData &x ) const
{
    const auto &xt   = getTpetraVector<ST, LO, GO, NT>( x );
    const auto &lmap = getLocalMap<ST, LO, GO, NT>( xt.getMap() );
    const auto &xl   = xt.offsetView( lmap, 0 );
    return xl->normInf();
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::localDot( const VectorData &x,
                                                         const VectorData &y ) const
{
    const auto &xt    = getTpetraVector<ST, LO, GO, NT>( x );
    const auto &xlmap = getLocalMap<ST, LO, GO, NT>( xt.getMap() );
    const auto &xl    = xt.offsetView( xlmap, 0 );

    const auto &yt    = getTpetraVector<ST, LO, GO, NT>( y );
    const auto &ylmap = getLocalMap<ST, LO, GO, NT>( yt.getMap() );
    const auto &yl    = yt.offsetView( ylmap, 0 );

    return xl->dot( *yl );
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar
TpetraVectorOperations<ST, LO, GO, NT>::localMin( const AMP::LinearAlgebra::VectorData &x ) const
{
    const auto &xt = getTpetraVector<ST, LO, GO, NT>( x );
    auto xData     = xt.getData( 0 );
    return AMP::Utilities::Algorithms<ST>::min_element( xData.get(), xData.size() );
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar
TpetraVectorOperations<ST, LO, GO, NT>::localMax( const AMP::LinearAlgebra::VectorData &x ) const
{
    const auto &xt = getTpetraVector<ST, LO, GO, NT>( x );
    auto xData     = xt.getData( 0 );
    return AMP::Utilities::Algorithms<ST>::max_element( xData.get(), xData.size() );
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar
TpetraVectorOperations<ST, LO, GO, NT>::localSum( const AMP::LinearAlgebra::VectorData &x ) const
{
    const auto &xt = getTpetraVector<ST, LO, GO, NT>( x );
    AMP_ASSERT( xt.getNumVectors() == 1 );
    auto xData = xt.getData( 0 );
    return AMP::Utilities::Algorithms<ST>::accumulate(
        xData.get(), xData.size(), static_cast<ST>( 0 ) );
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::localMinQuotient(
    const AMP::LinearAlgebra::VectorData &, const AMP::LinearAlgebra::VectorData & ) const
{
    AMP_ERROR( "TpetraVectorOperations::localMinQuotient not implemented" );
    return 0;
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::localWrmsNorm(
    const AMP::LinearAlgebra::VectorData &, const AMP::LinearAlgebra::VectorData & ) const
{
    AMP_ERROR( "TpetraVectorOperations::localWrmsNorm not implemented" );
}

template<typename ST, typename LO, typename GO, typename NT>
Scalar TpetraVectorOperations<ST, LO, GO, NT>::localWrmsNormMask(
    const AMP::LinearAlgebra::VectorData &,
    const AMP::LinearAlgebra::VectorData &,
    const AMP::LinearAlgebra::VectorData & ) const
{
    AMP_ERROR( "TpetraVectorOperations::localWrmsNormMask not implemented" );
    return 0;
}

template<typename ST, typename LO, typename GO, typename NT>
bool TpetraVectorOperations<ST, LO, GO, NT>::localEquals( const AMP::LinearAlgebra::VectorData &x,
                                                          const AMP::LinearAlgebra::VectorData &y,
                                                          const Scalar &tol ) const
{
    auto rval      = true;
    const auto &xt = getTpetraVector<ST, LO, GO, NT>( x );
    const auto &yt = getTpetraVector<ST, LO, GO, NT>( y );
    AMP_ASSERT( xt.getNumVectors() == 1 && yt.getNumVectors() == 1 );
    auto xData = xt.getData( 0 );
    auto yData = yt.getData( 0 );

    if ( xData.size() != yData.size() )
        return false;

    for ( int i = 0; i < xData.size(); ++i ) {
        if ( !AMP::Utilities::approx_equal( xData[i], yData[i], static_cast<ST>( tol ) ) ) {
            rval = false;
            break;
        }
    }

    return rval;
}

} // namespace AMP::LinearAlgebra

#endif
