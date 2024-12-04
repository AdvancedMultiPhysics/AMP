#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/kokkos/CSRMatrixOperationsKokkos.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/memory.h"
#include "AMP/vectors/Vector.h"

#include <algorithm>

#include "ProfilerApp.h"

#if defined( AMP_USE_KOKKOS ) || defined( AMP_USE_TRILINOS_KOKKOS )

    #include "Kokkos_Core.hpp"

namespace AMP::LinearAlgebra {

template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
void CSRMatrixOperationsKokkos<Policy,
                               Allocator,
                               ExecSpace,
                               ViewSpace,
                               DiagMatrixData,
                               OffdMatrixData>::mult( std::shared_ptr<const Vector> in,
                                                      MatrixData const &A,
                                                      std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsKokkos::mult" );
    AMP_DEBUG_ASSERT( in && out );
    AMP_DEBUG_ASSERT( in->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    using scalar_t = typename Policy::scalar_t;
    using lidx_t   = typename Policy::lidx_t;

    auto csrData = getCSRMatrixData<Policy, Allocator, DiagMatrixData, OffdMatrixData>(
        const_cast<MatrixData &>( A ) );

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
                      "CSRMatrixOperationsKokkos is not implemented for device memory" );

    AMP_DEBUG_INSIST( csrData->d_memory_location == AMP::Utilities::getMemoryType( inDataBlock ),
                      "Input vector from wrong memory space" );

    AMP_DEBUG_INSIST( csrData->d_memory_location == AMP::Utilities::getMemoryType( outDataBlock ),
                      "Output vector from wrong memory space" );

    AMP_DEBUG_INSIST(
        1 == inData->numberOfDataBlocks(),
        "CSRMatrixOperationsKokkos::mult only implemented for vectors with one data block" );

    AMP_ASSERT( inDataBlock && outDataBlock );

    {
        PROFILE( "CSRMatrixOperationsKokkos::mult(local)" );
        d_localops_diag->mult( inDataBlock, diagMatrix, outDataBlock );
    }

    if ( csrData->hasOffDiag() ) {
        PROFILE( "CSRMatrixOperationsKokkos::mult(ghost)" );

        // Possible mismatch between Policy::gidx_t and size_t forces a deep copy
        // of the colMap from inside offdMatrix
        std::vector<size_t> colMap;
        offdMatrix->getColumnMap( colMap );
        std::vector<scalar_t> ghosts( colMap.size() );
        in->getGhostValuesByGlobalID( colMap.size(), colMap.data(), ghosts.data() );

        AMP_DEBUG_ASSERT( static_cast<typename Policy::lidx_t>( ghosts.size() ) ==
                          offdMatrix->numUniqueColumns() );

        Kokkos::View<scalar_t *, Kokkos::LayoutRight, Kokkos::HostSpace> ghostView_h(
            ghosts.data(), ghosts.size() );
        auto ghostView_d = Kokkos::create_mirror_view_and_copy( d_exec_space, ghostView_h );

        d_localops_offd->mult( ghostView_d.data(), offdMatrix, outDataBlock );
    }

    d_exec_space.fence(); // get rid of this eventually
}

template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
void CSRMatrixOperationsKokkos<Policy,
                               Allocator,
                               ExecSpace,
                               ViewSpace,
                               DiagMatrixData,
                               OffdMatrixData>::multTranspose( std::shared_ptr<const Vector> in,
                                                               MatrixData const &A,
                                                               std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsKokkos::multTranspose" );

    // this is not meant to be an optimized version. It is provided for completeness
    AMP_DEBUG_ASSERT( in && out );

    out->zero();

    using scalar_t = typename Policy::scalar_t;

    auto csrData = getCSRMatrixData<Policy, Allocator, DiagMatrixData, OffdMatrixData>(
        const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsKokkos is not implemented for device memory" );

    auto inData                 = in->getVectorData();
    const scalar_t *inDataBlock = inData->getRawDataBlock<scalar_t>( 0 );
    auto outData                = out->getVectorData();
    scalar_t *outDataBlock      = outData->getRawDataBlock<scalar_t>( 0 );

    {
        PROFILE( "CSRMatrixOperationsKokkos::multTranspose (local)" );

        d_localops_diag->multTranspose( inDataBlock, diagMatrix, outDataBlock );
    }

    if ( csrData->hasOffDiag() ) {
        PROFILE( "CSRMatrixOperationsKokkos::multTranspose (ghost)" );

        // Possible mismatch between Policy::gidx_t and size_t forces a deep copy
        // of the colMap from inside offdMatrix
        std::vector<size_t> rcols;
        offdMatrix->getColumnMap( rcols );

        Kokkos::View<scalar_t *, Kokkos::LayoutRight, ViewSpace> vvals_d( "multTrans vvals",
                                                                          rcols.size() );

        d_localops_offd->multTranspose( inDataBlock, offdMatrix, vvals_d.data() );
        // d_exec_space.fence();

        // now copy vvals_d back to host to write out
        auto vvals_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace(), vvals_d );
        d_exec_space.fence();

        // copy rcols and vvals into std::vectors and write out
        out->addValuesByGlobalID( rcols.size(), rcols.data(), vvals_h.data() );
    } else {
        d_exec_space.fence(); // still finish with a fence if no offd term present
    }
}

template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
void CSRMatrixOperationsKokkos<Policy,
                               Allocator,
                               ExecSpace,
                               ViewSpace,
                               DiagMatrixData,
                               OffdMatrixData>::scale( AMP::Scalar alpha_in, MatrixData &A )
{
    using scalar_t = typename Policy::scalar_t;

    auto csrData = getCSRMatrixData<Policy, Allocator, DiagMatrixData, OffdMatrixData>(
        const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsKokkos is not implemented for device memory" );

    auto alpha = static_cast<scalar_t>( alpha_in );

    d_localops_diag->scale( alpha, diagMatrix );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->scale( alpha, offdMatrix );
    }

    d_exec_space.fence();
}

template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
void CSRMatrixOperationsKokkos<Policy,
                               Allocator,
                               ExecSpace,
                               ViewSpace,
                               DiagMatrixData,
                               OffdMatrixData>::matMultiply( MatrixData const &,
                                                             MatrixData const &,
                                                             MatrixData & )
{
    AMP_WARNING( "SpGEMM for CSRMatrixOperationsKokkos not implemented" );
}

template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
void CSRMatrixOperationsKokkos<Policy,
                               Allocator,
                               ExecSpace,
                               ViewSpace,
                               DiagMatrixData,
                               OffdMatrixData>::axpy( AMP::Scalar alpha_in,
                                                      const MatrixData &X,
                                                      MatrixData &Y )
{
    using scalar_t = typename Policy::scalar_t;

    auto csrDataX = getCSRMatrixData<Policy, Allocator, DiagMatrixData, OffdMatrixData>(
        const_cast<MatrixData &>( X ) );
    auto csrDataY = getCSRMatrixData<Policy, Allocator, DiagMatrixData, OffdMatrixData>(
        const_cast<MatrixData &>( Y ) );

    AMP_DEBUG_ASSERT( csrDataX );
    AMP_DEBUG_ASSERT( csrDataY );

    AMP_DEBUG_INSIST( csrDataX->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsKokkos is not implemented for device memory" );
    AMP_DEBUG_INSIST( csrDataY->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsKokkos is not implemented for device memory" );
    AMP_DEBUG_INSIST( csrDataX->d_memory_location == csrDataY->d_memory_location,
                      "CSRMatrixOperationsKokkos::axpy X and Y must be in same memory space" );

    auto diagMatrixX = csrDataX->getDiagMatrix();
    auto offdMatrixX = csrDataX->getOffdMatrix();

    auto diagMatrixY = csrDataY->getDiagMatrix();
    auto offdMatrixY = csrDataY->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrixX && offdMatrixX );
    AMP_DEBUG_ASSERT( diagMatrixY && offdMatrixY );

    auto alpha = static_cast<scalar_t>( alpha_in );
    d_localops_diag->axpy( alpha, diagMatrixX, diagMatrixY );
    if ( csrDataX->hasOffDiag() ) {
        d_localops_offd->axpy( alpha, offdMatrixX, offdMatrixY );
    }

    d_exec_space.fence();
}

template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
void CSRMatrixOperationsKokkos<Policy,
                               Allocator,
                               ExecSpace,
                               ViewSpace,
                               DiagMatrixData,
                               OffdMatrixData>::setScalar( AMP::Scalar alpha_in, MatrixData &A )
{
    using scalar_t = typename Policy::scalar_t;

    auto csrData = getCSRMatrixData<Policy, Allocator, DiagMatrixData, OffdMatrixData>(
        const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsKokkos is not implemented for device memory" );

    auto alpha = static_cast<scalar_t>( alpha_in );

    d_localops_diag->setScalar( alpha, diagMatrix );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->setScalar( alpha, offdMatrix );
    }

    d_exec_space.fence();
}

template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
void CSRMatrixOperationsKokkos<Policy,
                               Allocator,
                               ExecSpace,
                               ViewSpace,
                               DiagMatrixData,
                               OffdMatrixData>::zero( MatrixData &A )
{
    setScalar( 0.0, A );
}

template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
void CSRMatrixOperationsKokkos<Policy,
                               Allocator,
                               ExecSpace,
                               ViewSpace,
                               DiagMatrixData,
                               OffdMatrixData>::setDiagonal( std::shared_ptr<const Vector> in,
                                                             MatrixData &A )
{
    using scalar_t = typename Policy::scalar_t;

    // constrain to one data block for now
    AMP_DEBUG_ASSERT( in && in->numberOfDataBlocks() == 1 && in->isType<scalar_t>( 0 ) );

    const scalar_t *vvals_p = in->getRawDataBlock<scalar_t>();

    auto csrData = getCSRMatrixData<Policy, Allocator, DiagMatrixData, OffdMatrixData>(
        const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    d_localops_diag->setDiagonal( vvals_p, diagMatrix );

    d_exec_space.fence();
}

template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
void CSRMatrixOperationsKokkos<Policy,
                               Allocator,
                               ExecSpace,
                               ViewSpace,
                               DiagMatrixData,
                               OffdMatrixData>::setIdentity( MatrixData &A )
{
    zero( A );

    auto csrData = getCSRMatrixData<Policy, Allocator, DiagMatrixData, OffdMatrixData>(
        const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    d_localops_diag->setIdentity( diagMatrix );

    d_exec_space.fence();
}


template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
void CSRMatrixOperationsKokkos<Policy,
                               Allocator,
                               ExecSpace,
                               ViewSpace,
                               DiagMatrixData,
                               OffdMatrixData>::extractDiagonal( MatrixData const &A,
                                                                 std::shared_ptr<Vector> buf )
{
    using scalar_t = typename Policy::scalar_t;

    auto csrData = getCSRMatrixData<Policy, Allocator, DiagMatrixData, OffdMatrixData>(
        const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    scalar_t *buf_p = buf->getRawDataBlock<scalar_t>();
    d_localops_diag->extractDiagonal( diagMatrix, buf_p );

    d_exec_space.fence();
}

template<typename Policy,
         typename Allocator,
         class ExecSpace,
         class ViewSpace,
         class DiagMatrixData,
         class OffdMatrixData>
AMP::Scalar
CSRMatrixOperationsKokkos<Policy, Allocator, ExecSpace, ViewSpace, DiagMatrixData, OffdMatrixData>::
    LinfNorm( MatrixData const &A ) const
{
    using scalar_t = typename Policy::scalar_t;

    auto csrData = getCSRMatrixData<Policy, Allocator, DiagMatrixData, OffdMatrixData>(
        const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsKokkos is not implemented for device memory" );

    const auto nRows = csrData->numLocalRows();
    std::vector<scalar_t> rowSums( nRows, 0.0 );

    d_localops_diag->LinfNorm( diagMatrix, rowSums.data() );
    d_exec_space.fence();
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->LinfNorm( offdMatrix, rowSums.data() );
        d_exec_space.fence();
    }

    // Reduce row sums to get global Linf norm
    auto max_norm = *std::max_element( rowSums.begin(), rowSums.end() );
    AMP_MPI comm  = csrData->getComm();
    return comm.maxReduce<scalar_t>( max_norm );
}

} // namespace AMP::LinearAlgebra

#endif // close check for Kokkos being defined
