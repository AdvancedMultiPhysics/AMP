#ifndef included_CSRMatrixOperationsDevice_HPP_
#define included_CSRMatrixOperationsDevice_HPP_

#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/device/CSRLocalMatrixOperationsDevice.h"
#include "AMP/matrices/operations/device/CSRMatrixOperationsDevice.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/typeid.h"

#include "AMP/utils/Memory.h"

#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/extrema.h"

#include <algorithm>

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

template<typename Config>
void CSRMatrixOperationsDevice<Config>::mult( std::shared_ptr<const Vector> in,
                                              MatrixData const &A,
                                              std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsDevice::mult" );
    AMP_DEBUG_ASSERT( in && out );
    AMP_DEBUG_ASSERT( in->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    out->zero();

    auto inData                 = in->getVectorData();
    const scalar_t *inDataBlock = inData->getRawDataBlock<scalar_t>( 0 );
    auto outData                = out->getVectorData();
    scalar_t *outDataBlock      = outData->getRawDataBlock<scalar_t>( 0 );

    AMP_DEBUG_INSIST( csrData->d_memory_location == AMP::Utilities::getMemoryType( inDataBlock ),
                      "Input vector from wrong memory space" );

    AMP_DEBUG_INSIST( csrData->d_memory_location == AMP::Utilities::getMemoryType( outDataBlock ),
                      "Output vector from wrong memory space" );

    AMP_DEBUG_INSIST(
        1 == inData->numberOfDataBlocks(),
        "CSRMatrixOperationsDevice::mult only implemented for vectors with one data block" );

    AMP_ASSERT( inDataBlock && outDataBlock );

    {
        PROFILE( "CSRMatrixOperationsDevice::mult(local)" );
        CSRLocalMatrixOperationsDevice<Config>::mult( inDataBlock, diagMatrix, outDataBlock );
    }

    if ( csrData->hasOffDiag() ) {
        PROFILE( "CSRMatrixOperationsDevice::mult(ghost)" );
        using scalarAllocator_t = typename std::allocator_traits<
            typename Config::allocator_type>::template rebind_alloc<scalar_t>;
        const auto nGhosts = offdMatrix->numUniqueColumns();
        scalarAllocator_t alloc;
        scalar_t *ghosts = alloc.allocate( nGhosts );
        if constexpr ( std::is_same_v<size_t, gidx_t> ) {
            // column map can be passed to get ghosts function directly
            auto *colMap = offdMatrix->getColumnMap();
            in->getGhostValuesByGlobalID( nGhosts, colMap, ghosts );
        } else if constexpr ( sizeof( size_t ) == sizeof( gidx_t ) ) {
            auto colMap = reinterpret_cast<size_t *>( offdMatrix->getColumnMap() );
            in->getGhostValuesByGlobalID( nGhosts, colMap, ghosts );
        } else {
            // this is inefficient and we should figure out a better approach
            AMP_WARN_ONCE(
                "CSRMatrixOperationsDevice::mult: Deep copy/cast of column map required" );
            using idxAllocator_t = typename std::allocator_traits<
                typename Config::allocator_type>::template rebind_alloc<size_t>;
            idxAllocator_t idx_alloc;
            size_t *idxMap = idx_alloc.allocate( nGhosts );
            auto *colMap   = offdMatrix->getColumnMap();
            AMP::Utilities::copy( nGhosts, colMap, idxMap );
            in->getGhostValuesByGlobalID( nGhosts, idxMap, ghosts );
            idx_alloc.deallocate( idxMap, nGhosts );
        }
        deviceSynchronize();
        CSRLocalMatrixOperationsDevice<Config>::mult( ghosts, offdMatrix, outDataBlock );
        alloc.deallocate( ghosts, nGhosts );
    }
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::multTranspose( std::shared_ptr<const Vector> in,
                                                       MatrixData const &A,
                                                       std::shared_ptr<Vector> out )
{
    AMP_WARNING( "multTranspose not enabled for device." );
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::scale( AMP::Scalar alpha_in, MatrixData &A )
{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    AMP_DEBUG_ASSERT( diagMatrix );

    auto alpha = static_cast<scalar_t>( alpha_in );
    CSRLocalMatrixOperationsDevice<Config>::scale( alpha, diagMatrix );

    if ( csrData->hasOffDiag() ) {
        auto offdMatrix = csrData->getOffdMatrix();
        AMP_DEBUG_ASSERT( offdMatrix );
        CSRLocalMatrixOperationsDevice<Config>::scale( alpha, offdMatrix );
    }
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::matMatMult( std::shared_ptr<MatrixData> A,
                                                    std::shared_ptr<MatrixData> B,
                                                    std::shared_ptr<MatrixData> C )
{
    auto csrDataA = std::dynamic_pointer_cast<CSRMatrixData<Config>>( A );
    auto csrDataB = std::dynamic_pointer_cast<CSRMatrixData<Config>>( B );
    auto csrDataC = std::dynamic_pointer_cast<CSRMatrixData<Config>>( C );

    AMP_DEBUG_ASSERT( csrDataA && csrDataB && csrDataC );

    // Verify that A and B have compatible dimensions
    const auto globalKa = csrDataA->numGlobalColumns();
    const auto globalKb = csrDataB->numGlobalRows();
    const auto localKa  = csrDataA->numLocalColumns();
    const auto localKb  = csrDataB->numLocalRows();
    AMP_INSIST( globalKa == globalKb,
                "CSRMatrixOperationsDefault::matMatMult got incompatible global dimensions" );
    AMP_INSIST( localKa == localKb,
                "CSRMatrixOperationsDefault::matMatMult got incompatible local dimensions" );

    // Verify that all matrices have the same memory space and that it isn't device
    const auto memLocA = csrDataA->getMemoryLocation();
    const auto memLocB = csrDataB->getMemoryLocation();
    const auto memLocC = csrDataC->getMemoryLocation();
    AMP_INSIST( memLocA == AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsDevice::matMatMult only implemented for device matrices" );
    AMP_INSIST( memLocA == memLocB,
                "CSRMatrixOperationsDevice::matMatMult A and B must have the same memory type" );
    AMP_INSIST( memLocA == memLocC,
                "CSRMatrixOperationsDevice::matMatMult A and C must have the same memory type" );

    // Check if an SpGEMM helper has already been constructed for this combination
    // of matrices. If not create it first and do symbolic phase, otherwise skip
    // ahead to numeric phase
    auto bcPair = std::make_pair( csrDataB, csrDataC );
    if ( d_SpGEMMHelpers.find( bcPair ) == d_SpGEMMHelpers.end() ) {
        AMP_INSIST( csrDataC->isEmpty(),
                    "CSRMatrixOperationsDevice::matMatMult A*B->C only applicable to non-empty C "
                    "if it came from same A and B input matrices originally" );
        d_SpGEMMHelpers[bcPair] = CSRMatrixSpGEMMDevice( csrDataA, csrDataB, csrDataC );
        d_SpGEMMHelpers[bcPair].multiply();
    } else {
        AMP_WARN_ONCE( "CSRMatrixOperationsDevice::matMatMult: Reuse of C not yet supported, "
                       "falling back to full calculation" );
        d_SpGEMMHelpers[bcPair].multiply();
    }
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::axpy( AMP::Scalar alpha_in,
                                              const MatrixData &X,
                                              MatrixData &Y )
{
    auto csrDataX = getCSRMatrixData<Config>( const_cast<MatrixData &>( X ) );
    auto csrDataY = getCSRMatrixData<Config>( const_cast<MatrixData &>( Y ) );

    AMP_DEBUG_ASSERT( csrDataX );
    AMP_DEBUG_ASSERT( csrDataY );

    AMP_DEBUG_INSIST( csrDataX->d_memory_location == csrDataY->d_memory_location,
                      "CSRMatrixOperationsDevice::axpy X and Y must be in same memory space" );

    auto diagMatrixX = csrDataX->getDiagMatrix();
    auto offdMatrixX = csrDataX->getOffdMatrix();

    auto diagMatrixY = csrDataY->getDiagMatrix();
    auto offdMatrixY = csrDataY->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrixX && offdMatrixX );
    AMP_DEBUG_ASSERT( diagMatrixY && offdMatrixY );

    auto alpha = static_cast<scalar_t>( alpha_in );
    CSRLocalMatrixOperationsDevice<Config>::axpy( alpha, diagMatrixX, diagMatrixY );
    if ( csrDataX->hasOffDiag() ) {
        CSRLocalMatrixOperationsDevice<Config>::axpy( alpha, offdMatrixX, offdMatrixY );
    }
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::setScalar( AMP::Scalar alpha_in, MatrixData &A )
{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    auto alpha = static_cast<scalar_t>( alpha_in );

    CSRLocalMatrixOperationsDevice<Config>::setScalar( alpha, diagMatrix );
    if ( csrData->hasOffDiag() ) {
        CSRLocalMatrixOperationsDevice<Config>::setScalar( alpha, offdMatrix );
    }
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::zero( MatrixData &A )
{
    setScalar( static_cast<scalar_t>( 0.0 ), A );
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::setDiagonal( std::shared_ptr<const Vector> in,
                                                     MatrixData &A )
{
    // constrain to one data block for now
    AMP_DEBUG_ASSERT( in && in->numberOfDataBlocks() == 1 && in->isType<scalar_t>( 0 ) );

    const scalar_t *vvals_p = in->getRawDataBlock<scalar_t>();

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    CSRLocalMatrixOperationsDevice<Config>::setDiagonal( vvals_p, diagMatrix );
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::extractDiagonal( MatrixData const &A,
                                                         std::shared_ptr<Vector> buf )

{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    scalar_t *buf_p = buf->getRawDataBlock<scalar_t>();
    CSRLocalMatrixOperationsDevice<Config>::extractDiagonal( diagMatrix, buf_p );
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::setIdentity( MatrixData &A )
{
    zero( A );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );
    CSRLocalMatrixOperationsDevice<Config>::setIdentity( diagMatrix );
}

template<typename Config>
AMP::Scalar CSRMatrixOperationsDevice<Config>::LinfNorm( MatrixData const &A ) const

{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    const auto nRows = csrData->numLocalRows();
    thrust::device_vector<scalar_t> rowSums( nRows, 0.0 );

    CSRLocalMatrixOperationsDevice<Config>::LinfNorm( diagMatrix, rowSums.data().get() );
    if ( csrData->hasOffDiag() ) {
        CSRLocalMatrixOperationsDevice<Config>::LinfNorm( offdMatrix, rowSums.data().get() );
    }

    // Reduce row sums to get global Linf norm
    auto max_norm = *thrust::max_element( thrust::device, rowSums.begin(), rowSums.end() );
    AMP_MPI comm  = csrData->getComm();
    return comm.maxReduce<scalar_t>( max_norm );
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::copy( const MatrixData &X, MatrixData &Y )
{
    auto csrDataX = getCSRMatrixData<Config>( const_cast<MatrixData &>( X ) );
    auto csrDataY = getCSRMatrixData<Config>( const_cast<MatrixData &>( Y ) );

    AMP_DEBUG_ASSERT( csrDataX );
    AMP_DEBUG_ASSERT( csrDataY );

    AMP_DEBUG_INSIST( csrDataX->d_memory_location == csrDataY->d_memory_location,
                      "CSRMatrixOperationsDevice::axpy X and Y must be in same memory space" );

    auto diagMatrixX = csrDataX->getDiagMatrix();
    auto offdMatrixX = csrDataX->getOffdMatrix();

    auto diagMatrixY = csrDataY->getDiagMatrix();
    auto offdMatrixY = csrDataY->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrixX && offdMatrixX );
    AMP_DEBUG_ASSERT( diagMatrixY && offdMatrixY );

    CSRLocalMatrixOperationsDevice<Config>::copy( diagMatrixX, diagMatrixY );
    if ( csrDataX->hasOffDiag() ) {
        CSRLocalMatrixOperationsDevice<Config>::copy( offdMatrixX, offdMatrixY );
    }
}

template<typename Config>
void CSRMatrixOperationsDevice<Config>::copyCast( const MatrixData &X, MatrixData &Y )
{
    auto csrDataY = getCSRMatrixData<Config>( Y );
    AMP_DEBUG_ASSERT( csrDataY );
    if ( X.getCoeffType() == getTypeID<double>() ) {
        using ConfigIn = typename Config::template set_scalar_t<scalar::f64>::template set_alloc_t<
            Config::allocator>;
        auto csrDataX = getCSRMatrixData<ConfigIn>( const_cast<MatrixData &>( X ) );
        AMP_DEBUG_ASSERT( csrDataX );

        copyCast<ConfigIn>( csrDataX, csrDataY );
    } else if ( X.getCoeffType() == getTypeID<float>() ) {
        using ConfigIn = typename Config::template set_scalar_t<scalar::f32>::template set_alloc_t<
            Config::allocator>;
        auto csrDataX = getCSRMatrixData<ConfigIn>( const_cast<MatrixData &>( X ) );
        AMP_DEBUG_ASSERT( csrDataX );

        copyCast<ConfigIn>( csrDataX, csrDataY );
    } else {
        AMP_ERROR( "Can't copyCast from the given matrix, policy not supported" );
    }
}

template<typename Config>
template<typename ConfigIn>
void CSRMatrixOperationsDevice<Config>::copyCast(
    CSRMatrixData<typename ConfigIn::template set_alloc_t<Config::allocator>> *X, matrixdata_t *Y )
{

    AMP_DEBUG_INSIST( X->d_memory_location == Y->d_memory_location,
                      "CSRMatrixOperationsDevice::copyCast X and Y must be in same memory space" );

    auto diagMatrixX = X->getDiagMatrix();
    auto offdMatrixX = X->getOffdMatrix();

    auto diagMatrixY = Y->getDiagMatrix();
    auto offdMatrixY = Y->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrixX && offdMatrixX );
    AMP_DEBUG_ASSERT( diagMatrixY && offdMatrixY );

    localops_t::template copyCast<ConfigIn>( diagMatrixX, diagMatrixY );
    if ( X->hasOffDiag() ) {
        localops_t::template copyCast<ConfigIn>( offdMatrixX, offdMatrixY );
    }
}

} // namespace AMP::LinearAlgebra

#endif
