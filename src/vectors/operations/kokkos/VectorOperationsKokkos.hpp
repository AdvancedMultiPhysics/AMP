#include "AMP/AMP_TPLs.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"
#include "AMP/vectors/operations/kokkos/VectorOperationsKokkos.h"

#include "ProfilerApp.h"

#ifdef AMP_USE_KOKKOS

    #include "Kokkos_Core.hpp"
    #include "Kokkos_Random.hpp"

namespace AMP::LinearAlgebra {

template<typename T, class ViewSpace>
auto wrapVecDataKokkos( const VectorData &x )
{
    AMP_ASSERT( x.numberOfDataBlocks() == 1 );
    const auto N = x.sizeOfDataBlock( 0 );
    return Kokkos::View<const T *, Kokkos::LayoutRight, ViewSpace>( x.getRawDataBlock<T>( 0 ), N );
}

template<typename T, class ViewSpace>
auto wrapVecDataKokkos( VectorData &x )
{
    AMP_ASSERT( x.numberOfDataBlocks() == 1 );
    const auto N = x.sizeOfDataBlock( 0 );
    return Kokkos::View<T *, Kokkos::LayoutRight, ViewSpace>( x.getRawDataBlock<T>( 0 ), N );
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

template<typename T>
void VectorOperationsKokkos<T>::setToScalar( const Scalar &alpha_in, VectorData &x )
{
    PROFILE( "VectorOperationsKokkos::setToScalar" );

    const T alpha = alpha_in.get<T>();

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        Kokkos::deep_copy( xv, alpha );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            Kokkos::deep_copy( xv, alpha );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            Kokkos::deep_copy( xv, alpha );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    // seed is hardcoded
    Kokkos::Random_XorShift64_Pool<> random_pool( 12345 );
    Kokkos::RangePolicy<ExecSpace> pol( exec, 0, xv.extent( 0 ) );
    Kokkos::parallel_for(
        "VectorOperationsKokkos::random", pol, KOKKOS_LAMBDA( const int i ) {
            auto gen = random_pool.get_state();
            xv( i )  = static_cast<T>( gen.drand( 0.0, 1.0 ) );
        } );
}

template<typename T>
void VectorOperationsKokkos<T>::setRandomValues( VectorData &x )
{
    PROFILE( "VectorOperationsKokkos::setRandomValues" );

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        random_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            random_kernel( d_exec_device, xv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            random_kernel( d_exec_device, xv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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

    const T alpha = alpha_in.get<T>();

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        scale_kernel( d_exec_host, alpha, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            scale_kernel( d_exec_device, alpha, xv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            scale_kernel( d_exec_device, alpha, xv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::scale" );

    const T alpha = alpha_in.get<T>();

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        scale_kernel( d_exec_host, alpha, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            scale_kernel( d_exec_device, alpha, xv, yv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            scale_kernel( d_exec_device, alpha, xv, yv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
    #endif
    }
    y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
}

template<typename T>
void VectorOperationsKokkos<T>::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    AMP_ASSERT( x.getMemoryLocation() == z.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::add" );

    linearSum( T{ 1 }, x, T{ 1 }, y, z );
}

template<typename T>
void VectorOperationsKokkos<T>::subtract( const VectorData &x, const VectorData &y, VectorData &z )
{
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    AMP_ASSERT( x.getMemoryLocation() == z.getMemoryLocation() );
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    AMP_ASSERT( x.getMemoryLocation() == z.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::multiply" );

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        auto zv = wrapVecDataKokkos<T, ViewSpaceHost>( z );
        multiply_kernel( d_exec_host, xv, yv, zv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            auto zv = wrapVecDataKokkos<T, ViewSpaceManaged>( z );
            multiply_kernel( d_exec_device, xv, yv, zv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            auto zv = wrapVecDataKokkos<T, ViewSpaceDevice>( z );
            multiply_kernel( d_exec_device, xv, yv, zv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    AMP_ASSERT( x.getMemoryLocation() == z.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::divide" );

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        auto zv = wrapVecDataKokkos<T, ViewSpaceHost>( z );
        divide_kernel( d_exec_host, xv, yv, zv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            auto zv = wrapVecDataKokkos<T, ViewSpaceManaged>( z );
            divide_kernel( d_exec_device, xv, yv, zv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            auto zv = wrapVecDataKokkos<T, ViewSpaceDevice>( z );
            divide_kernel( d_exec_device, xv, yv, zv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::reciprocal" );

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        reciprocal_kernel( d_exec_host, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            reciprocal_kernel( d_exec_device, xv, yv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            reciprocal_kernel( d_exec_device, xv, yv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    AMP_ASSERT( x.getMemoryLocation() == z.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::linearSum" );

    const T alpha = alpha_in.get<T>();
    const T beta  = beta_in.get<T>();

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        auto zv = wrapVecDataKokkos<T, ViewSpaceHost>( z );
        linsum_kernel( d_exec_host, alpha, xv, beta, yv, zv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            auto zv = wrapVecDataKokkos<T, ViewSpaceManaged>( z );
            linsum_kernel( d_exec_device, alpha, xv, beta, yv, zv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            auto zv = wrapVecDataKokkos<T, ViewSpaceDevice>( z );
            linsum_kernel( d_exec_device, alpha, xv, beta, yv, zv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::abs" );

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        abs_kernel( d_exec_host, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            abs_kernel( d_exec_device, xv, yv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            abs_kernel( d_exec_device, xv, yv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::addScalar" );

    const T alpha = alpha_in.get<T>();

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        add_scalar_kernel( d_exec_host, alpha, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            add_scalar_kernel( d_exec_device, alpha, xv, yv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            add_scalar_kernel( d_exec_device, alpha, xv, yv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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

    const T alpha = alpha_in.get<T>();

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        set_min_kernel( d_exec_host, alpha, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            set_min_kernel( d_exec_device, alpha, xv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            set_min_kernel( d_exec_device, alpha, xv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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

    const T alpha = alpha_in.get<T>();

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        set_max_kernel( d_exec_host, alpha, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            set_max_kernel( d_exec_device, alpha, xv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            set_max_kernel( d_exec_device, alpha, xv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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

    T min_val;

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        min_val = min_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            min_val = min_kernel( d_exec_device, xv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            min_val = min_kernel( d_exec_device, xv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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

    T max_val;

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        max_val = max_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            max_val = max_kernel( d_exec_device, xv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            max_val = max_kernel( d_exec_device, xv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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

    T sum;

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        sum     = sum_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            sum     = sum_kernel( d_exec_device, xv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            sum     = sum_kernel( d_exec_device, xv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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

    T norm;

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        norm    = l1_norm_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            norm    = l1_norm_kernel( d_exec_device, xv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            norm    = l1_norm_kernel( d_exec_device, xv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
Scalar VectorOperationsKokkos<T>::localL2Norm( const VectorData &x ) const
{
    PROFILE( "VectorOperationsKokkos::localL2Norm" );

    T norm = 0.0;

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        norm    = l2_norm_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            norm    = l2_norm_kernel( d_exec_device, xv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            norm    = l2_norm_kernel( d_exec_device, xv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
    #endif
    }

    return std::sqrt( norm );
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

    T norm;

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        norm    = max_norm_kernel( d_exec_host, xv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            norm    = max_norm_kernel( d_exec_device, xv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            norm    = max_norm_kernel( d_exec_device, xv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::localDot" );

    T dot;

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        dot     = dot_kernel( d_exec_host, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            dot     = dot_kernel( d_exec_device, xv, yv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            dot     = dot_kernel( d_exec_device, xv, yv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::localMinQuotient" );

    T min_quotient;

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv      = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv      = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        min_quotient = min_quotient_kernel( d_exec_host, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv      = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv      = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            min_quotient = min_quotient_kernel( d_exec_device, xv, yv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv      = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv      = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            min_quotient = min_quotient_kernel( d_exec_device, xv, yv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::localWrmsNorm" );

    T norm;

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        norm    = wrms_kernel( d_exec_host, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            norm    = wrms_kernel( d_exec_device, xv, yv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            norm    = wrms_kernel( d_exec_device, xv, yv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    AMP_ASSERT( x.getMemoryLocation() == mask.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::localWrmsNormMask" );

    T norm;

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        auto mv = wrapVecDataKokkos<T, ViewSpaceHost>( mask );
        norm    = wrms_mask_kernel( d_exec_host, mv, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            auto mv = wrapVecDataKokkos<T, ViewSpaceManaged>( mask );
            norm    = wrms_mask_kernel( d_exec_device, mv, xv, yv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            auto mv = wrapVecDataKokkos<T, ViewSpaceDevice>( mask );
            norm    = wrms_mask_kernel( d_exec_device, mv, xv, yv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
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
    AMP_ASSERT( x.getMemoryLocation() == y.getMemoryLocation() );
    PROFILE( "VectorOperationsKokkos::localEquals" );

    bool equals;
    const T tol = tol_in.get<T>();

    if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::host ) {
        auto xv = wrapVecDataKokkos<T, ViewSpaceHost>( x );
        auto yv = wrapVecDataKokkos<T, ViewSpaceHost>( y );
        equals  = equals_kernel( d_exec_host, tol, xv, yv );
    } else {
    #ifndef AMP_USE_DEVICE
        AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
    #else
        if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::managed ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceManaged>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceManaged>( y );
            equals  = equals_kernel( d_exec_device, tol, xv, yv );
        } else if ( x.getMemoryLocation() == AMP::Utilities::MemoryType::device ) {
            auto xv = wrapVecDataKokkos<T, ViewSpaceDevice>( x );
            auto yv = wrapVecDataKokkos<T, ViewSpaceDevice>( y );
            equals  = equals_kernel( d_exec_device, tol, xv, yv );
        } else {
            AMP_ERROR( "VectorOperationsKokkos: Unrecognized memory space" );
        }
    #endif
    }

    return equals;
}


} // namespace AMP::LinearAlgebra

#endif
