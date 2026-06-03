#include "AMP/AMP_TPLs.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"
#include "AMP/vectors/operations/kokkos/VectorOperationsKokkos.h"

#include "ProfilerApp.h"

#ifdef AMP_USE_KOKKOS

    #include "Kokkos_Core.hpp"
    #include "Kokkos_Random.hpp"

namespace AMP::LinearAlgebra {

template<typename T>
auto wrapVecDataKokkos( const VectorData &x )
{
    AMP_ASSERT( x.numberOfDataBlocks() == 1 );
    const auto N = x.sizeOfDataBlock( 0 );
    return Kokkos::View<const T *, Kokkos::AnonymousSpace>( x.getRawDataBlock<T>( 0 ), N );
}

template<typename T>
auto wrapVecDataKokkos( VectorData &x )
{
    AMP_ASSERT( x.numberOfDataBlocks() == 1 );
    const auto N = x.sizeOfDataBlock( 0 );
    return Kokkos::View<T *, Kokkos::AnonymousSpace>( x.getRawDataBlock<T>( 0 ), N );
}

template<typename T>
std::shared_ptr<VectorOperations> VectorOperationsKokkos<T>::cloneOperations() const
{
    return std::make_shared<VectorOperationsKokkos<T>>();
}

template<typename T>
void VectorOperationsKokkos<T>::zero( VectorData &x )
{
    PROFILE( "VectorOperationsKokkos::zero" );

    setToScalar( T{ 0 }, x );
}

template<typename T, class ExecSpace, class ViewT>
void set_scalar_kernel( ExecSpace exec, const T alpha, ViewT xv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::set_scalar", pol, KOKKOS_LAMBDA( const int i ) {
            xv( i ) = alpha;
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::setToScalar( const Scalar &alpha_in, VectorData &x )
{
    PROFILE( "VectorOperationsKokkos::scale" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    const T alpha = alpha_in.get<T>();
    auto xv       = wrapVecDataKokkos<T>( x );

    if ( !device_exec ) {
        Kokkos::deep_copy( d_exec_host, xv, alpha );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        Kokkos::deep_copy( d_exec_device, xv, alpha );
        d_exec_device.fence();
    #endif
    }
    x.fillGhosts( alpha_in );
    x.setUpdateStatus( UpdateState::UNCHANGED );
}

template<class ExecSpace, class ViewT>
void random_kernel( ExecSpace exec, ViewT xv )
{
    using T = typename ViewT::non_const_value_type;
    // adapted from example in Kokkos docs
    std::random_device rd;
    uint64_t seed = rd();
    Kokkos::Random_XorShift64_Pool<ExecSpace> random_pool( seed );
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::random", pol, KOKKOS_LAMBDA( const int i ) {
            auto gen = random_pool.get_state();
            xv( i )  = static_cast<T>( gen.drand( 0.0, 1.0 ) );
            // do not forget to release the state of the engine
            random_pool.free_state( gen );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::setRandomValues( VectorData &x )
{
    PROFILE( "VectorOperationsKokkos::setRandomValues" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    auto xv = wrapVecDataKokkos<T>( x );
    if ( !device_exec ) {
        random_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        random_kernel( d_exec_device, xv );
    #endif
    }
    x.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<typename T>
void VectorOperationsKokkos<T>::copy( const VectorData &x, VectorData &y )
{
    d_default_ops->copy( x, y );
}


template<typename T>
void VectorOperationsKokkos<T>::copyCast( const VectorData &x, VectorData &y )
{
    d_default_ops->copyCast( x, y );
}

template<typename T, class ExecSpace, class ViewT>
void scale_kernel( ExecSpace exec, const T alpha, ViewT xv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::scale", pol, KOKKOS_LAMBDA( const int i ) { xv( i ) *= alpha; } );
}

template<typename T>
void VectorOperationsKokkos<T>::scale( const Scalar &alpha_in, VectorData &x )
{
    PROFILE( "VectorOperationsKokkos::scale" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    const T alpha = alpha_in.get<T>();

    auto xv = wrapVecDataKokkos<T>( x );
    if ( !device_exec ) {
        scale_kernel( d_exec_host, alpha, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        scale_kernel( d_exec_device, alpha, xv );
    #endif
    }
    x.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<typename T, class ExecSpace, class ViewCT, class ViewT>
void scale_kernel( ExecSpace exec, const T alpha, ViewCT xv, ViewT yv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::scale", pol, KOKKOS_LAMBDA( const int i ) {
            yv( i ) = alpha * xv( i );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::scale( const Scalar &alpha_in, const VectorData &x, VectorData &y )
{
    PROFILE( "VectorOperationsKokkos::scale" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation() );

    const T alpha = alpha_in.get<T>();
    auto xv       = wrapVecDataKokkos<T>( x );
    auto yv       = wrapVecDataKokkos<T>( y );

    if ( !device_exec ) {
        scale_kernel( d_exec_host, alpha, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        scale_kernel( d_exec_device, alpha, xv, yv );
    #endif
    }
    y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<typename T>
void VectorOperationsKokkos<T>::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    PROFILE( "VectorOperationsKokkos::add" );

    linearSum( T{ 1 }, x, T{ 1 }, y, z );
}

template<typename T>
void VectorOperationsKokkos<T>::subtract( const VectorData &x, const VectorData &y, VectorData &z )
{
    PROFILE( "VectorOperationsKokkos::subtract" );

    linearSum( T{ 1 }, x, T{ -1 }, y, z );
}

template<class ExecSpace, class ViewCT, class ViewT>
void multiply_kernel( ExecSpace exec, ViewCT xv, ViewCT yv, ViewT zv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::multiply", pol, KOKKOS_LAMBDA( const int i ) {
            zv( i ) = xv( i ) * yv( i );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::multiply( const VectorData &x, const VectorData &y, VectorData &z )
{
    PROFILE( "VectorOperationsKokkos::multiply" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation(), z.getMemoryLocation() );

    auto xv = wrapVecDataKokkos<T>( x );
    auto yv = wrapVecDataKokkos<T>( y );
    auto zv = wrapVecDataKokkos<T>( z );

    if ( !device_exec ) {
        multiply_kernel( d_exec_host, xv, yv, zv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        multiply_kernel( d_exec_device, xv, yv, zv );
    #endif
    }
    z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<class ExecSpace, class ViewCT, class ViewT>
void divide_kernel( ExecSpace exec, ViewCT xv, ViewCT yv, ViewT zv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::divide", pol, KOKKOS_LAMBDA( const int i ) {
            zv( i ) = xv( i ) / yv( i );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::divide( const VectorData &x, const VectorData &y, VectorData &z )
{
    PROFILE( "VectorOperationsKokkos::divide" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation(), z.getMemoryLocation() );

    auto xv = wrapVecDataKokkos<T>( x );
    auto yv = wrapVecDataKokkos<T>( y );
    auto zv = wrapVecDataKokkos<T>( z );

    if ( !device_exec ) {
        divide_kernel( d_exec_host, xv, yv, zv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        divide_kernel( d_exec_device, xv, yv, zv );
    #endif
    }
    z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<class ExecSpace, class ViewCT, class ViewT>
void reciprocal_kernel( ExecSpace exec, ViewCT xv, ViewT yv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::reciprocal", pol, KOKKOS_LAMBDA( const int i ) {
            yv( i ) = 1.0 / xv( i );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::reciprocal( const VectorData &x, VectorData &y )
{
    PROFILE( "VectorOperationsKokkos::reciprocal" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation() );

    auto xv = wrapVecDataKokkos<T>( x );
    auto yv = wrapVecDataKokkos<T>( y );

    if ( !device_exec ) {
        reciprocal_kernel( d_exec_host, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        reciprocal_kernel( d_exec_device, xv, yv );
    #endif
    }
    y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<typename T, class ExecSpace, class ViewCT, class ViewT>
void linsum_kernel( ExecSpace exec, const T alpha, ViewCT xv, const T beta, ViewCT yv, ViewT zv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::add", pol, KOKKOS_LAMBDA( const int i ) {
            zv( i ) = alpha * xv( i ) + beta * yv( i );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::linearSum( const Scalar &alpha_in,
                                           const VectorData &x,
                                           const Scalar &beta_in,
                                           const VectorData &y,
                                           VectorData &z )
{
    PROFILE( "VectorOperationsKokkos::linearSum" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation(), z.getMemoryLocation() );

    const T alpha = alpha_in.get<T>();
    const T beta  = beta_in.get<T>();
    auto xv       = wrapVecDataKokkos<T>( x );
    auto yv       = wrapVecDataKokkos<T>( y );
    auto zv       = wrapVecDataKokkos<T>( z );

    if ( !device_exec ) {
        linsum_kernel( d_exec_host, alpha, xv, beta, yv, zv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        linsum_kernel( d_exec_device, alpha, xv, beta, yv, zv );
    #endif
    }
    z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<typename T>
void VectorOperationsKokkos<T>::axpy( const Scalar &alpha_in,
                                      const VectorData &x,
                                      const VectorData &y,
                                      VectorData &z )
{
    PROFILE( "VectorOperationsKokkos::axpy" );

    VectorOperationsKokkos<T>::linearSum( alpha_in, x, 1.0, y, z );
}

template<typename T>
void VectorOperationsKokkos<T>::axpby( const Scalar &alpha_in,
                                       const Scalar &beta_in,
                                       const VectorData &x,
                                       VectorData &z )
{
    PROFILE( "VectorOperationsKokkos::axpby" );

    VectorOperationsKokkos<T>::linearSum( alpha_in, x, beta_in, z, z );
}

template<class ExecSpace, class ViewCT, class ViewT>
void abs_kernel( ExecSpace exec, ViewCT xv, ViewT yv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::abs", pol, KOKKOS_LAMBDA( const int i ) {
            yv( i ) = Kokkos::fabs( xv( i ) );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::abs( const VectorData &x, VectorData &y )
{
    PROFILE( "VectorOperationsKokkos::abs" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation() );

    auto xv = wrapVecDataKokkos<T>( x );
    auto yv = wrapVecDataKokkos<T>( y );

    if ( !device_exec ) {
        abs_kernel( d_exec_host, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        abs_kernel( d_exec_device, xv, yv );
    #endif
    }
    y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<typename T, class ExecSpace, class ViewCT, class ViewT>
void add_scalar_kernel( ExecSpace exec, const T alpha, ViewCT xv, ViewT yv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::add_scalar", pol, KOKKOS_LAMBDA( const int i ) {
            yv( i ) = alpha + xv( i );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::addScalar( const VectorData &x,
                                           const Scalar &alpha_in,
                                           VectorData &y )
{
    PROFILE( "VectorOperationsKokkos::addScalar" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation() );

    const T alpha = alpha_in.get<T>();
    auto xv       = wrapVecDataKokkos<T>( x );
    auto yv       = wrapVecDataKokkos<T>( y );

    if ( !device_exec ) {
        add_scalar_kernel( d_exec_host, alpha, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        add_scalar_kernel( d_exec_device, alpha, xv, yv );
    #endif
    }
    y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<typename T, class ExecSpace, class ViewT>
void set_min_kernel( ExecSpace exec, const T alpha, ViewT xv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::set_min", pol, KOKKOS_LAMBDA( const int i ) {
            xv( i ) = xv( i ) < alpha ? alpha : xv( i );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::setMin( const Scalar &alpha_in, VectorData &x )
{
    PROFILE( "VectorOperationsKokkos::setMin" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    const T alpha = alpha_in.get<T>();
    auto xv       = wrapVecDataKokkos<T>( x );

    if ( !device_exec ) {
        set_min_kernel( d_exec_host, alpha, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        set_min_kernel( d_exec_device, alpha, xv );
    #endif
    }
    x.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<typename T, class ExecSpace, class ViewT>
void set_max_kernel( ExecSpace exec, const T alpha, ViewT xv )
{
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::set_max", pol, KOKKOS_LAMBDA( const int i ) {
            xv( i ) = xv( i ) > alpha ? alpha : xv( i );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::setMax( const Scalar &alpha_in, VectorData &x )
{
    PROFILE( "VectorOperationsKokkos::setMax" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    const T alpha = alpha_in.get<T>();
    auto xv       = wrapVecDataKokkos<T>( x );

    if ( !device_exec ) {
        set_max_kernel( d_exec_host, alpha, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        set_max_kernel( d_exec_device, alpha, xv );
    #endif
    }
    x.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<class ExecSpace, class ViewCT>
typename ViewCT::non_const_value_type min_kernel( ExecSpace exec, ViewCT xv )
{
    using T   = typename ViewCT::non_const_value_type;
    T min_val = std::numeric_limits<T>::max();
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::min",
        pol,
        KOKKOS_LAMBDA( const int i, T &lmin ) { lmin = xv( i ) < lmin ? xv( i ) : lmin; },
        Kokkos::Min<T>( min_val ) );
    return min_val;
}

template<typename T>
Scalar VectorOperationsKokkos<T>::localMin( const VectorData &x ) const
{
    PROFILE( "VectorOperationsKokkos::localMin" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    T min_val;
    auto xv = wrapVecDataKokkos<T>( x );

    if ( !device_exec ) {
        min_val = min_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        min_val      = min_kernel( d_exec_device, xv );
    #endif
    }

    return min_val;
}

template<class ExecSpace, class ViewCT>
typename ViewCT::non_const_value_type max_kernel( ExecSpace exec, ViewCT xv )
{
    using T   = typename ViewCT::non_const_value_type;
    T max_val = std::numeric_limits<T>::min();
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::max",
        pol,
        KOKKOS_LAMBDA( const int i, T &lmax ) { lmax = xv( i ) > lmax ? xv( i ) : lmax; },
        Kokkos::Max<T>( max_val ) );
    return max_val;
}

template<typename T>
Scalar VectorOperationsKokkos<T>::localMax( const VectorData &x ) const
{
    PROFILE( "VectorOperationsKokkos::localMax" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    T max_val;
    auto xv = wrapVecDataKokkos<T>( x );

    if ( !device_exec ) {
        max_val = max_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        max_val      = max_kernel( d_exec_device, xv );
    #endif
    }

    return max_val;
}

template<class ExecSpace, class ViewCT>
typename ViewCT::non_const_value_type sum_kernel( ExecSpace exec, ViewCT xv )
{
    using T = typename ViewCT::non_const_value_type;
    T sum   = 0.0;
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::sum",
        pol,
        KOKKOS_LAMBDA( const int i, T &lsum ) { lsum += xv( i ); },
        sum );
    return sum;
}

template<typename T>
Scalar VectorOperationsKokkos<T>::localSum( const VectorData &x ) const
{
    PROFILE( "VectorOperationsKokkos::localSum" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    T sum;
    auto xv = wrapVecDataKokkos<T>( x );

    if ( !device_exec ) {
        sum = sum_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        sum          = sum_kernel( d_exec_device, xv );
    #endif
    }

    return sum;
}

template<class ExecSpace, class ViewCT>
typename ViewCT::non_const_value_type l1_norm_kernel( ExecSpace exec, ViewCT xv )
{
    using T = typename ViewCT::non_const_value_type;
    T sum   = 0.0;
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::l1_norm",
        pol,
        KOKKOS_LAMBDA( const int i, T &lsum ) { lsum += Kokkos::fabs( xv( i ) ); },
        sum );
    return sum;
}

template<typename T>
Scalar VectorOperationsKokkos<T>::localL1Norm( const VectorData &x ) const
{
    PROFILE( "VectorOperationsKokkos::localL1Norm" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    T norm;
    auto xv = wrapVecDataKokkos<T>( x );

    if ( !device_exec ) {
        norm = l1_norm_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        norm         = l1_norm_kernel( d_exec_device, xv );
    #endif
    }

    return norm;
}

template<class ExecSpace, class ViewCT>
typename ViewCT::non_const_value_type l2_norm_kernel( ExecSpace exec, ViewCT xv )
{
    using T = typename ViewCT::non_const_value_type;
    T sum   = 0.0;
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::l2_norm",
        pol,
        KOKKOS_LAMBDA( const int i, T &lsum ) { lsum += xv( i ) * xv( i ); },
        sum );
    return sum;
}

template<typename T>
Scalar VectorOperationsKokkos<T>::localL2Norm2( const VectorData &x ) const
{
    PROFILE( "VectorOperationsKokkos::localL2Norm2" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    T norm  = 0.0;
    auto xv = wrapVecDataKokkos<T>( x );

    if ( !device_exec ) {
        norm = l2_norm_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        norm         = l2_norm_kernel( d_exec_device, xv );
    #endif
    }

    return norm;
}

template<class ExecSpace, class ViewCT>
typename ViewCT::non_const_value_type max_norm_kernel( ExecSpace exec, ViewCT xv )
{
    using T   = typename ViewCT::non_const_value_type;
    T max_val = std::numeric_limits<T>::min();
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::max_norm",
        pol,
        KOKKOS_LAMBDA( const int i, T &lmax ) {
            const auto axv = Kokkos::fabs( xv( i ) );
            lmax           = axv > lmax ? axv : lmax;
        },
        Kokkos::Max<T>( max_val ) );
    return max_val;
}

template<typename T>
Scalar VectorOperationsKokkos<T>::localMaxNorm( const VectorData &x ) const
{
    PROFILE( "VectorOperationsKokkos::localMaxNorm" );

    const auto device_exec =
        AMP::Utilities::memoryLocationsDeviceAccessible( x.getMemoryLocation() );

    T norm;
    auto xv = wrapVecDataKokkos<T>( x );

    if ( !device_exec ) {
        norm = max_norm_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        norm         = max_norm_kernel( d_exec_device, xv );
    #endif
    }

    return norm;
}

template<class ExecSpace, class ViewCT>
typename ViewCT::non_const_value_type dot_kernel( ExecSpace exec, ViewCT xv, ViewCT yv )
{
    using T = typename ViewCT::non_const_value_type;
    T sum   = 0.0;
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::dot",
        pol,
        KOKKOS_LAMBDA( const int i, T &lsum ) { lsum += xv( i ) * yv( i ); },
        sum );
    return sum;
}

template<typename T>
Scalar VectorOperationsKokkos<T>::localDot( const VectorData &x, const VectorData &y ) const
{
    PROFILE( "VectorOperationsKokkos::localDot" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation() );

    T dot;
    auto xv = wrapVecDataKokkos<T>( x );
    auto yv = wrapVecDataKokkos<T>( y );

    if ( !device_exec ) {
        dot = dot_kernel( d_exec_host, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        dot          = dot_kernel( d_exec_device, xv, yv );
    #endif
    }

    return dot;
}

template<class ExecSpace, class ViewCT>
typename ViewCT::non_const_value_type min_quotient_kernel( ExecSpace exec, ViewCT xv, ViewCT yv )
{
    using T   = typename ViewCT::non_const_value_type;
    T min_val = std::numeric_limits<T>::max();
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::min",
        pol,
        KOKKOS_LAMBDA( const int i, T &lmin ) {
            const auto q = xv( i ) / yv( i );
            lmin         = q < lmin ? q : lmin;
        },
        Kokkos::Min<T>( min_val ) );
    return min_val;
}

template<typename T>
Scalar VectorOperationsKokkos<T>::localMinQuotient( const VectorData &x, const VectorData &y ) const
{
    PROFILE( "VectorOperationsKokkos::localMinQuotient" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation() );

    T min_quotient;
    auto xv = wrapVecDataKokkos<T>( x );
    auto yv = wrapVecDataKokkos<T>( y );

    if ( !device_exec ) {
        min_quotient = min_quotient_kernel( d_exec_host, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        min_quotient = min_quotient_kernel( d_exec_device, xv, yv );
    #endif
    }

    return min_quotient;
}

template<class ExecSpace, class ViewCT>
typename ViewCT::non_const_value_type wrms_kernel( ExecSpace exec, ViewCT xv, ViewCT yv )
{
    using T = typename ViewCT::non_const_value_type;
    T sum   = 0.0;
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::wrms",
        pol,
        KOKKOS_LAMBDA( const int i, T &lsum ) { lsum += xv( i ) * xv( i ) * yv( i ) * yv( i ); },
        sum );
    return sum;
}

template<typename T>
Scalar VectorOperationsKokkos<T>::localWrmsNorm( const VectorData &x, const VectorData &y ) const
{
    PROFILE( "VectorOperationsKokkos::localWrmsNorm" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation() );

    T norm;
    auto xv = wrapVecDataKokkos<T>( x );
    auto yv = wrapVecDataKokkos<T>( y );

    if ( !device_exec ) {
        norm = wrms_kernel( d_exec_host, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        norm         = wrms_kernel( d_exec_device, xv, yv );
    #endif
    }

    return norm;
}

template<class ExecSpace, class ViewCT>
typename ViewCT::non_const_value_type
wrms_mask_kernel( ExecSpace exec, ViewCT mv, ViewCT xv, ViewCT yv )
{
    using T = typename ViewCT::non_const_value_type;
    T sum   = 0.0;
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::wrms",
        pol,
        KOKKOS_LAMBDA( const int i, T &lsum ) {
            if ( mv( i ) > 0 ) {
                lsum += xv( i ) * xv( i ) * yv( i ) * yv( i );
            }
        },
        sum );
    return sum;
}

template<typename T>
Scalar VectorOperationsKokkos<T>::localWrmsNormMask( const VectorData &x,
                                                     const VectorData &mask,
                                                     const VectorData &y ) const
{
    PROFILE( "VectorOperationsKokkos::localWrmsNormMask" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), mask.getMemoryLocation(), y.getMemoryLocation() );

    T norm;
    auto xv = wrapVecDataKokkos<T>( x );
    auto yv = wrapVecDataKokkos<T>( y );
    auto mv = wrapVecDataKokkos<T>( mask );

    if ( !device_exec ) {
        norm = wrms_mask_kernel( d_exec_host, mv, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        norm         = wrms_mask_kernel( d_exec_device, mv, xv, yv );
    #endif
    }

    return norm;
}

template<typename T, class ExecSpace, class ViewCT>
bool equals_kernel( ExecSpace exec, const T tol, ViewCT xv, ViewCT yv )
{
    bool equal = true;
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_reduce(
        "VectorOperationsKokkos::equal",
        pol,
        KOKKOS_LAMBDA( const int i, bool &lequal ) {
            const auto diff = Kokkos::fabs( xv( i ) - yv( i ) );
            if ( diff > tol ) {
                lequal = false;
            }
        },
        Kokkos::LAnd<bool>( equal ) );
    return equal;
}

template<typename T>
bool VectorOperationsKokkos<T>::localEquals( const VectorData &x,
                                             const VectorData &y,
                                             const Scalar &tol_in ) const
{
    PROFILE( "VectorOperationsKokkos::localEquals" );

    const auto device_exec = AMP::Utilities::memoryLocationsDeviceAccessible(
        x.getMemoryLocation(), y.getMemoryLocation() );

    bool equals;
    const T tol = tol_in.get<T>();
    auto xv     = wrapVecDataKokkos<T>( x );
    auto yv     = wrapVecDataKokkos<T>( y );

    if ( !device_exec ) {
        equals = equals_kernel( d_exec_host, tol, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        equals       = equals_kernel( d_exec_device, tol, xv, yv );
    #endif
    }

    return equals;
}


} // namespace AMP::LinearAlgebra

#endif
