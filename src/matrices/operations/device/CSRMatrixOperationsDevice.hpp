#ifndef included_CSRMatrixOperationsDevice_HPP_
#define included_CSRMatrixOperationsDevice_HPP_

#include "AMP/matrices/CSRPolicy.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/device/CSRLocalMatrixOperationsDevice.h"
#include "AMP/matrices/operations/device/CSRMatrixOperationsDevice.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/typeid.h"

#include "AMP/utils/memory.h"

#include "thrust/device_vector.h"
#include "thrust/execution_policy.h"
#include "thrust/extrema.h"

#include <algorithm>

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::mult(
    std::shared_ptr<const Vector> in, MatrixData const &A, std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsDevice::mult" );
    AMP_DEBUG_ASSERT( in && out );
    AMP_DEBUG_ASSERT( in->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    using scalar_t = typename Policy::scalar_t;
    using lidx_t   = typename Policy::lidx_t;

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    out->zero();

    auto inData                 = in->getVectorData();
    const scalar_t *inDataBlock = inData->getRawDataBlock<scalar_t>( 0 );
    auto outData                = out->getVectorData();
    scalar_t *outDataBlock      = outData->getRawDataBlock<scalar_t>( 0 );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDevice is not implemented for device memory" );

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
        CSRLocalMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::mult(
            inDataBlock, diagMatrix, outDataBlock );
    }

    if ( csrData->hasOffDiag() ) {
        PROFILE( "CSRMatrixOperationsDevice::mult(ghost)" );

        // Possible mismatch between Policy::gidx_t and size_t forces a deep copy
        // of the colMap from inside offdMatrix
        std::vector<size_t> colMap;
        offdMatrix->getColumnMap( colMap );
        std::vector<scalar_t> ghosts_h( colMap.size() );
        in->getGhostValuesByGlobalID( colMap.size(), colMap.data(), ghosts_h.data() );

        AMP_DEBUG_ASSERT( static_cast<typename Policy::lidx_t>( ghosts_h.size() ) ==
                          offdMatrix->numUniqueColumns() );

        CSRLocalMatrixOperationsDevice<Policy, Allocator>::mult(
            ghosts_h.data(), offdMatrix, outDataBlock );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::multTranspose(
    std::shared_ptr<const Vector> in, MatrixData const &A, std::shared_ptr<Vector> out )
{
    AMP_WARNING( "multTranspose not enabled for device." );
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::scale( AMP::Scalar alpha_in,
                                                                          MatrixData &A )
{
    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    AMP_DEBUG_ASSERT( diagMatrix );

    auto alpha = static_cast<scalar_t>( alpha_in );
    CSRLocalMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::scale( alpha, diagMatrix );

    if ( csrData->hasOffDiag() ) {
        auto offdMatrix = csrData->getOffdMatrix();
        AMP_DEBUG_ASSERT( offdMatrix );
        CSRLocalMatrixOperationsDevice<Policy, Allocator>::scale( alpha, offdMatrix );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::matMultiply( MatrixData const &,
                                                                                MatrixData const &,
                                                                                MatrixData & )
{
    AMP_WARNING( "SpGEMM for CSRMatrixOperationsDevice not implemented" );
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::axpy( AMP::Scalar alpha_in,
                                                                         const MatrixData &X,
                                                                         MatrixData &Y )
{
    using scalar_t = typename Policy::scalar_t;

    auto csrDataX =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( X ) );
    auto csrDataY =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( Y ) );

    AMP_DEBUG_ASSERT( csrDataX );
    AMP_DEBUG_ASSERT( csrDataY );

    AMP_DEBUG_INSIST( csrDataX->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDevice is not implemented for device memory" );
    AMP_DEBUG_INSIST( csrDataY->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDevice is not implemented for device memory" );
    AMP_DEBUG_INSIST( csrDataX->d_memory_location == csrDataY->d_memory_location,
                      "CSRMatrixOperationsDevice::axpy X and Y must be in same memory space" );

    auto diagMatrixX = csrDataX->getDiagMatrix();
    auto offdMatrixX = csrDataX->getOffdMatrix();

    auto diagMatrixY = csrDataY->getDiagMatrix();
    auto offdMatrixY = csrDataY->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrixX && offdMatrixX );
    AMP_DEBUG_ASSERT( diagMatrixY && offdMatrixY );

    auto alpha = static_cast<scalar_t>( alpha_in );
    CSRLocalMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::axpy(
        alpha, diagMatrixX, diagMatrixY );
    if ( csrDataX->hasOffDiag() ) {
        CSRLocalMatrixOperationsDevice<Policy, Allocator>::axpy( alpha, offdMatrixX, offdMatrixY );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::copyCast( const MatrixData &X,
                                                                             MatrixData &Y )
{
    auto csrDataY = getCSRMatrixData<Policy, Allocator, DiagMatrixData>( Y );
    AMP_DEBUG_ASSERT( csrDataY );
    if ( X.getCoeffType() == getTypeID<double>() ) {
        using PolicyIn =
            AMP::LinearAlgebra::CSRPolicy<typename Policy::gidx_t, typename Policy::lidx_t, double>;
        auto csrDataX =
            getCSRMatrixData<PolicyIn, Allocator, CSRLocalMatrixData<PolicyIn, Allocator>>(
                const_cast<MatrixData &>( X ) );
        AMP_DEBUG_ASSERT( csrDataX );

        copyCast<PolicyIn>( csrDataX, csrDataY );
    } else if ( X.getCoeffType() == getTypeID<float>() ) {
        using PolicyIn =
            AMP::LinearAlgebra::CSRPolicy<typename Policy::gidx_t, typename Policy::lidx_t, float>;
        auto csrDataX =
            getCSRMatrixData<PolicyIn, Allocator, CSRLocalMatrixData<PolicyIn, Allocator>>(
                const_cast<MatrixData &>( X ) );
        AMP_DEBUG_ASSERT( csrDataX );

        copyCast<PolicyIn>( csrDataX, csrDataY );
    } else {
        AMP_ERROR( "Can't copyCast from the given matrix, policy not supported" );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
template<typename PolicyIn>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::copyCast(
    CSRMatrixData<PolicyIn, Allocator, CSRLocalMatrixData<PolicyIn, Allocator>> *X,
    CSRMatrixData<Policy, Allocator, DiagMatrixData> *Y )
{

    AMP_DEBUG_INSIST( X->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsKokkos is not implemented for device memory" );
    AMP_DEBUG_INSIST( Y->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsKokkos is not implemented for device memory" );
    AMP_DEBUG_INSIST( X->d_memory_location == Y->d_memory_location,
                      "CSRMatrixOperationsKokkos::copyCast X and Y must be in same memory space" );

    auto diagMatrixX = X->getDiagMatrix();
    auto offdMatrixX = X->getOffdMatrix();

    auto diagMatrixY = Y->getDiagMatrix();
    auto offdMatrixY = Y->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrixX && offdMatrixX );
    AMP_DEBUG_ASSERT( diagMatrixY && offdMatrixY );


    CSRLocalMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::template copyCast<PolicyIn>(
        diagMatrixX, diagMatrixY );
    if ( X->hasOffDiag() ) {
        CSRLocalMatrixOperationsDevice<Policy, Allocator>::template copyCast<PolicyIn>(
            offdMatrixX, offdMatrixY );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::setScalar( AMP::Scalar alpha_in,
                                                                              MatrixData &A )
{
    using scalar_t = typename Policy::scalar_t;

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDevice is not implemented for device memory" );

    auto alpha = static_cast<scalar_t>( alpha_in );

    CSRLocalMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::setScalar( alpha,
                                                                                  diagMatrix );
    if ( csrData->hasOffDiag() ) {
        CSRLocalMatrixOperationsDevice<Policy, Allocator>::setScalar( alpha, offdMatrix );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::zero( MatrixData &A )
{
    using scalar_t = typename Policy::scalar_t;
    setScalar( static_cast<scalar_t>( 0.0 ), A );
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::setDiagonal(
    std::shared_ptr<const Vector> in, MatrixData &A )
{
    using scalar_t = typename Policy::scalar_t;

    // constrain to one data block for now
    AMP_DEBUG_ASSERT( in && in->numberOfDataBlocks() == 1 && in->isType<scalar_t>( 0 ) );

    const scalar_t *vvals_p = in->getRawDataBlock<scalar_t>();

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    CSRLocalMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::setDiagonal( vvals_p,
                                                                                    diagMatrix );
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::extractDiagonal(
    MatrixData const &A, std::shared_ptr<Vector> buf )

{
    using scalar_t = typename Policy::scalar_t;

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    scalar_t *buf_p = buf->getRawDataBlock<scalar_t>();
    CSRLocalMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::extractDiagonal( diagMatrix,
                                                                                        buf_p );
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::setIdentity( MatrixData &A )
{
    zero( A );

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    CSRLocalMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::setIdentity( diagMatrix );
}

template<typename Policy, class Allocator, class DiagMatrixData>
AMP::Scalar
CSRMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::LinfNorm( MatrixData const &A ) const

{
    using scalar_t = typename Policy::scalar_t;

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDevice is not implemented for device memory" );

    const auto nRows = csrData->numLocalRows();
    thrust::device_vector<scalar_t> rowSums( nRows, 0.0 );

    CSRLocalMatrixOperationsDevice<Policy, Allocator, DiagMatrixData>::LinfNorm(
        diagMatrix, rowSums.data().get() );
    if ( csrData->hasOffDiag() ) {
        CSRLocalMatrixOperationsDevice<Policy, Allocator>::LinfNorm( offdMatrix,
                                                                     rowSums.data().get() );
    }

    // Reduce row sums to get global Linf norm
    auto max_norm = *thrust::max_element( thrust::device, rowSums.begin(), rowSums.end() );
    AMP_MPI comm  = csrData->getComm();
    return comm.maxReduce<scalar_t>( max_norm );
}

} // namespace AMP::LinearAlgebra

#endif
