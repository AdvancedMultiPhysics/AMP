#ifndef included_AMP_VectorOperationsDevice_hpp
#define included_AMP_VectorOperationsDevice_hpp

#include "AMP/utils/UtilityMacros.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.hpp"
#include "AMP/vectors/operations/device/DeviceOperationsHelpers.h"
#include "AMP/vectors/operations/device/VectorOperationsDevice.h"


namespace AMP {
namespace LinearAlgebra {

extern template class VectorOperationsDevice<double>; // Suppresses implicit instantiation below --
extern template class VectorOperationsDevice<float>;  // Suppresses implicit instantiation below --


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
template<typename TYPE>
std::shared_ptr<VectorOperations> VectorOperationsDevice<TYPE>::cloneOperations() const
{
    return std::make_shared<VectorOperationsDevice<TYPE>>();
}

template<typename TYPE>
VectorOperationsDevice<TYPE>::~VectorOperationsDevice<TYPE>()
{
    if ( d_default_ops )
        delete d_default_ops;
}


/****************************************************************
 * Check that all data can be passed to device                     *
 ****************************************************************/
inline bool checkData( const VectorData &x ) { return x.numberOfDataBlocks() == 1; }
template<class TYPE>
inline bool checkData( const VectorData &x, const VectorData &y )
{
    constexpr auto type = getTypeID<TYPE>();
    return x.numberOfDataBlocks() == 1 && y.numberOfDataBlocks() == 1 && y.getType( 0 ) == type;
}
template<class TYPE>
inline bool checkData( const VectorData &x, const VectorData &y, const VectorData &z )
{
    constexpr auto type = getTypeID<TYPE>();
    return x.numberOfDataBlocks() == 1 && y.numberOfDataBlocks() == 1 &&
           z.numberOfDataBlocks() == 1 && y.getType( 0 ) == type && z.getType( 0 ) == type;
}

template<typename TYPE>
inline VectorOperationsDefault<TYPE> *VectorOperationsDevice<TYPE>::getDefaultOps( void )
{
    if ( !d_default_ops )
        d_default_ops = new VectorOperationsDefault<TYPE>();
    return d_default_ops;
}

template<typename TYPE>
inline const VectorOperationsDefault<TYPE> *
VectorOperationsDevice<TYPE>::getDefaultOps( void ) const
{
    if ( !d_default_ops )
        d_default_ops = new VectorOperationsDefault<TYPE>();
    return d_default_ops;
}

//**********************************************************************
// Static functions that operate on VectorData objects

template<typename TYPE>
void VectorOperationsDevice<TYPE>::zero( VectorData &x )
{
    VectorOperationsDevice<TYPE>::setToScalar( 0.0, x );
    x.setUpdateStatus( UpdateState::UNCHANGED );
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::setToScalar( const Scalar &alpha_in, VectorData &x )
{
    bool useGPU = checkData( x );
    TYPE alpha  = alpha_in.get<TYPE>();
    if ( useGPU ) {
        TYPE *data = x.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        DeviceOperationsHelpers<TYPE>::setToScalar( alpha, N, data );
    } else {
        // Default to cpu version
        auto curMe = x.begin<TYPE>();
        auto last  = x.end<TYPE>();
        while ( curMe != last ) {
            *curMe = alpha;
            ++curMe;
        }
    }
    x.fillGhosts( alpha );
    // Override the status state since we set the ghost values
    x.setUpdateStatus( UpdateState::UNCHANGED );
    // Wait for cuda data to complete
    if ( useGPU )
        deviceSynchronize();
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::setRandomValues( VectorData &x )
{
    // Default to VectorOperationsDefault (on cpu)
    getDefaultOps()->setRandomValues( x );
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::copy( const VectorData &x, VectorData &y )
{
    if ( checkData<TYPE>( x, y ) ) {
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto N     = y.sizeOfDataBlock( 0 );
        DeviceOperationsHelpers<TYPE>::copy( N, xdata, ydata );
        deviceSynchronize();
        y.copyGhostValues( x );
        y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->copy( x, y );
    }
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::copyCast( const VectorData &x, VectorData &y )
{
    constexpr auto Hip_Cuda = AMP::Utilities::Backend::Hip_Cuda;
    if ( x.numberOfDataBlocks() == y.numberOfDataBlocks() ) {
        for ( size_t block_id = 0; block_id < y.numberOfDataBlocks(); block_id++ ) {
            auto ydata = y.getRawDataBlock<TYPE>( block_id );
            auto N     = y.sizeOfDataBlock( block_id );
            AMP_ASSERT( N == x.sizeOfDataBlock( block_id ) );
            if ( x.getType( 0 ) == getTypeID<float>() ) {
                auto xdata = x.getRawDataBlock<float>( block_id );
                AMP::Utilities::copyCast<float, TYPE, Hip_Cuda>( N, xdata, ydata );
            } else if ( x.getType( 0 ) == getTypeID<double>() ) {
                auto xdata = x.getRawDataBlock<double>( block_id );
                AMP::Utilities::copyCast<double, TYPE, Hip_Cuda>( N, xdata, ydata );
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
void VectorOperationsDevice<TYPE>::scale( const Scalar &alpha_in, VectorData &x )
{
    if ( checkData( x ) ) {
        TYPE *data = x.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        TYPE alpha = alpha_in.get<TYPE>();
        DeviceOperationsHelpers<TYPE>::scale( alpha, N, data );
        deviceSynchronize();
        x.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->scale( alpha_in, x );
    }
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::scale( const Scalar &alpha_in,
                                          const VectorData &x,
                                          VectorData &y )
{
    if ( checkData<TYPE>( x, y ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        auto N     = y.sizeOfDataBlock( 0 );
        auto alpha = alpha_in.get<TYPE>();
        DeviceOperationsHelpers<TYPE>::scale( alpha, N, xdata, ydata );
        deviceSynchronize();
        y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->scale( alpha_in, x, y );
    }
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( checkData<TYPE>( x, y ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        auto zdata = z.getRawDataBlock<TYPE>( 0 );
        auto N     = z.sizeOfDataBlock( 0 );
        DeviceOperationsHelpers<TYPE>::add( N, xdata, ydata, zdata );
        deviceSynchronize();
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->add( x, y, z );
    }
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::subtract( const VectorData &x,
                                             const VectorData &y,
                                             VectorData &z )
{
    if ( checkData<TYPE>( x, y, z ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        auto zdata = z.getRawDataBlock<TYPE>( 0 );
        size_t N   = z.sizeOfDataBlock( 0 );
        DeviceOperationsHelpers<TYPE>::subtract( N, xdata, ydata, zdata );
        deviceSynchronize();
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->subtract( x, y, z );
    }
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::multiply( const VectorData &x,
                                             const VectorData &y,
                                             VectorData &z )
{
    if ( checkData<TYPE>( x, y, z ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        auto zdata = z.getRawDataBlock<TYPE>( 0 );
        size_t N   = z.sizeOfDataBlock( 0 );
        DeviceOperationsHelpers<TYPE>::multiply( N, xdata, ydata, zdata );
        deviceSynchronize();
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->multiply( x, y, z );
    }
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::divide( const VectorData &x, const VectorData &y, VectorData &z )
{
    if ( checkData<TYPE>( x, y, z ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        auto zdata = z.getRawDataBlock<TYPE>( 0 );
        size_t N   = z.sizeOfDataBlock( 0 );
        DeviceOperationsHelpers<TYPE>::divide( N, xdata, ydata, zdata );
        deviceSynchronize();
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->divide( x, y, z );
    }
}


template<typename TYPE>
void VectorOperationsDevice<TYPE>::reciprocal( const VectorData &x, VectorData &y )
{
    if ( checkData<TYPE>( x, y ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        size_t N   = y.sizeOfDataBlock( 0 );
        DeviceOperationsHelpers<TYPE>::reciprocal( N, xdata, ydata );
        deviceSynchronize();
        y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->reciprocal( x, y );
    }
}


template<typename TYPE>
void VectorOperationsDevice<TYPE>::linearSum( const Scalar &alpha_in,
                                              const VectorData &x,
                                              const Scalar &beta_in,
                                              const VectorData &y,
                                              VectorData &z )
{
    if ( checkData<TYPE>( x, y, z ) ) {
        TYPE alpha = alpha_in.get<TYPE>();
        TYPE beta  = beta_in.get<TYPE>();
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        auto zdata = z.getRawDataBlock<TYPE>( 0 );
        size_t N   = z.sizeOfDataBlock( 0 );
        DeviceOperationsHelpers<TYPE>::linearSum( alpha, N, xdata, beta, ydata, zdata );
        deviceSynchronize();
        z.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->linearSum( alpha_in, x, beta_in, y, z );
    }
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::axpy( const Scalar &alpha_in,
                                         const VectorData &x,
                                         const VectorData &y,
                                         VectorData &z )
{
    VectorOperationsDevice<TYPE>::linearSum( alpha_in, x, 1.0, y, z );
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::axpby( const Scalar &alpha_in,
                                          const Scalar &beta_in,
                                          const VectorData &x,
                                          VectorData &z )
{
    VectorOperationsDevice<TYPE>::linearSum( alpha_in, x, beta_in, z, z );
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::abs( const VectorData &x, VectorData &y )
{
    if ( checkData<TYPE>( x, y ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        size_t N   = y.sizeOfDataBlock( 0 );
        DeviceOperationsHelpers<TYPE>::abs( N, xdata, ydata );
        deviceSynchronize();
        y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->abs( x, y );
    }
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::addScalar( const VectorData &x,
                                              const Scalar &alpha_in,
                                              VectorData &y )
{
    if ( checkData<TYPE>( x, y ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        size_t N   = y.sizeOfDataBlock( 0 );
        TYPE alpha = alpha_in.get<TYPE>();
        DeviceOperationsHelpers<TYPE>::addScalar( N, xdata, alpha, ydata );
        deviceSynchronize();
        y.setUpdateStatus( UpdateState::LOCAL_CHANGED );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->addScalar( x, alpha_in, y );
    }
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::setMin( const Scalar &alpha_in, VectorData &y )
{
    if ( checkData( y ) ) {
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        size_t N   = y.sizeOfDataBlock( 0 );
        TYPE alpha = alpha_in.get<TYPE>();
        DeviceOperationsHelpers<TYPE>::setMin( N, alpha, ydata );
        deviceSynchronize();
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->setMin( alpha_in, y );
    }
}

template<typename TYPE>
void VectorOperationsDevice<TYPE>::setMax( const Scalar &alpha_in, VectorData &y )
{
    if ( checkData( y ) ) {
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        size_t N   = y.sizeOfDataBlock( 0 );
        TYPE alpha = alpha_in.get<TYPE>();
        DeviceOperationsHelpers<TYPE>::setMax( N, alpha, ydata );
        deviceSynchronize();
    } else {
        // Default to VectorOperationsDefault (on cpu)
        getDefaultOps()->setMax( alpha_in, y );
    }
}

template<typename TYPE>
Scalar VectorOperationsDevice<TYPE>::localMin( const VectorData &x ) const
{
    if ( checkData( x ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        return DeviceOperationsHelpers<TYPE>::localMin( N, xdata );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        return getDefaultOps()->localMin( x );
    }
}

template<typename TYPE>
Scalar VectorOperationsDevice<TYPE>::localMax( const VectorData &x ) const
{
    if ( checkData( x ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        return DeviceOperationsHelpers<TYPE>::localMax( N, xdata );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        return getDefaultOps()->localMax( x );
    }
}

template<typename TYPE>
Scalar VectorOperationsDevice<TYPE>::localSum( const VectorData &x ) const
{
    if ( checkData( x ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        return DeviceOperationsHelpers<TYPE>::localSum( N, xdata );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        return getDefaultOps()->localSum( x );
    }
}

template<typename TYPE>
Scalar VectorOperationsDevice<TYPE>::localL1Norm( const VectorData &x ) const
{
    if ( checkData( x ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        return DeviceOperationsHelpers<TYPE>::localL1Norm( N, xdata );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        return getDefaultOps()->localL1Norm( x );
    }
}

template<typename TYPE>
Scalar VectorOperationsDevice<TYPE>::localL2Norm( const VectorData &x ) const
{
    if ( checkData( x ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        return DeviceOperationsHelpers<TYPE>::localL2Norm( N, xdata );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        return getDefaultOps()->localL2Norm( x );
    }
}

template<typename TYPE>
Scalar VectorOperationsDevice<TYPE>::localMaxNorm( const VectorData &x ) const
{
    if ( checkData( x ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        return DeviceOperationsHelpers<TYPE>::localMaxNorm( N, xdata );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        return getDefaultOps()->localMaxNorm( x );
    }
}

template<typename TYPE>
Scalar VectorOperationsDevice<TYPE>::localDot( const VectorData &x, const VectorData &y ) const
{
    if ( checkData<TYPE>( x, y ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        return DeviceOperationsHelpers<TYPE>::localDot( N, xdata, ydata );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        return getDefaultOps()->localDot( x, y );
    }
}

template<typename TYPE>
Scalar VectorOperationsDevice<TYPE>::localMinQuotient( const VectorData &x,
                                                       const VectorData &y ) const
{
    if ( checkData<TYPE>( x, y ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        return DeviceOperationsHelpers<TYPE>::localMinQuotient( N, xdata, ydata );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        return getDefaultOps()->localMinQuotient( x, y );
    }
}

template<typename TYPE>
Scalar VectorOperationsDevice<TYPE>::localWrmsNorm( const VectorData &x, const VectorData &y ) const
{
    if ( checkData<TYPE>( x, y ) ) {
        auto xdata = x.getRawDataBlock<TYPE>( 0 );
        auto ydata = y.getRawDataBlock<TYPE>( 0 );
        size_t N   = x.sizeOfDataBlock( 0 );
        return DeviceOperationsHelpers<TYPE>::localWrmsNorm( N, xdata, ydata );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        return getDefaultOps()->localWrmsNorm( x, y );
    }
}

template<typename TYPE>
Scalar VectorOperationsDevice<TYPE>::localWrmsNormMask( const VectorData &x,
                                                        const VectorData &mask,
                                                        const VectorData &y ) const
{
    // Default to VectorOperationsDefault (on cpu)
    return getDefaultOps()->localWrmsNormMask( x, mask, y );
}

template<typename TYPE>
bool VectorOperationsDevice<TYPE>::localEquals( const VectorData &x,
                                                const VectorData &y,
                                                const Scalar &tol_in ) const
{
    TYPE tol = tol_in.get<TYPE>();
    if ( checkData<TYPE>( x, y ) ) {
        // Call Cuda
        return getDefaultOps()->localEquals( x, y, tol );
    } else {
        // Default to VectorOperationsDefault (on cpu)
        return getDefaultOps()->localEquals( x, y, tol );
    }
}

} // namespace LinearAlgebra
} // namespace AMP

#endif
