#include "AMP/matrices/CSRPolicy.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/default/CSRMatrixOperationsDefault.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/typeid.h"
#include "AMP/vectors/Vector.h"

#include <algorithm>

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::mult(
    std::shared_ptr<const Vector> in, MatrixData const &A, std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsDefault::mult" );
    AMP_DEBUG_ASSERT( in && out );
    AMP_DEBUG_ASSERT( in->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    using gidx_t   = typename Policy::gidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    // this ensures that the separate on/off diag calls operate the same way
    // e.g. by accumulating into output data
    out->zero();

    // get all local in/out data buffers
    auto inData                 = in->getVectorData();
    const scalar_t *inDataBlock = inData->getRawDataBlock<scalar_t>( 0 );
    auto outData                = out->getVectorData();
    scalar_t *outDataBlock      = outData->getRawDataBlock<scalar_t>( 0 );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    AMP_DEBUG_INSIST(
        1 == inData->numberOfDataBlocks(),
        "CSRMatrixOperationsDefault::mult only implemented for vectors with one data block" );

    AMP_ASSERT( inDataBlock && outDataBlock );

    {
        PROFILE( "CSRMatrixOperationsDefault::mult (local)" );
        d_localops_diag->mult( inDataBlock, diagMatrix, outDataBlock );
    }

    if ( csrData->hasOffDiag() ) {
        PROFILE( "CSRMatrixOperationsDefault::mult (ghost)" );
        const auto nGhosts = offdMatrix->numUniqueColumns();
        std::vector<scalar_t> ghosts( nGhosts );
        if constexpr ( std::is_same_v<size_t, gidx_t> ) {
            // column map can be passed to get ghosts function directly
            size_t *colMap = offdMatrix->getColumnMap();
            in->getGhostValuesByGlobalID( nGhosts, colMap, ghosts.data() );
        } else {
            // type mismatch, need to copy/cast into temporary vector
            std::vector<size_t> colMap;
            offdMatrix->getColumnMap( colMap );
            in->getGhostValuesByGlobalID( nGhosts, colMap.data(), ghosts.data() );
        }
        d_localops_offd->mult( ghosts.data(), offdMatrix, outDataBlock );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::multTranspose(
    std::shared_ptr<const Vector> in, MatrixData const &A, std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsDefault::multTranspose" );

    // this is not meant to be an optimized version. It is provided for completeness
    AMP_DEBUG_ASSERT( in && out );

    out->zero();

    using scalar_t = typename Policy::scalar_t;

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    auto inData                 = in->getVectorData();
    const scalar_t *inDataBlock = inData->getRawDataBlock<scalar_t>( 0 );

    {
        PROFILE( "CSRMatrixOperationsDefault::multTranspose (d)" );

        std::vector<scalar_t> vvals;
        std::vector<size_t> rcols;
        d_localops_diag->multTranspose( inDataBlock, diagMatrix, vvals, rcols );
        out->addValuesByGlobalID( rcols.size(), rcols.data(), vvals.data() );
    }
    out->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    if ( csrData->hasOffDiag() ) {
        PROFILE( "CSRMatrixOperationsDefault::multTranspose (od)" );

        std::vector<scalar_t> vvals;
        std::vector<size_t> rcols;
        d_localops_offd->multTranspose( inDataBlock, offdMatrix, vvals, rcols );
        // Write out data, adding to any already present
        out->addValuesByGlobalID( rcols.size(), rcols.data(), vvals.data() );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::scale( AMP::Scalar alpha_in,
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
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    auto alpha = static_cast<scalar_t>( alpha_in );
    d_localops_diag->scale( alpha, diagMatrix );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->scale( alpha, offdMatrix );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::matMultiply(
    MatrixData const &A, MatrixData const &B, MatrixData &C )
{
    auto csrDataA =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );
    auto csrDataB =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( B ) );
    auto csrDataC =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( C ) );

    AMP_DEBUG_ASSERT( csrDataA );
    AMP_DEBUG_ASSERT( csrDataB );
    AMP_DEBUG_ASSERT( csrDataC );

    // Verify that A and B have compatible dimensions
    const auto globalKa = csrDataA->numGlobalColumns();
    const auto globalKb = csrDataB->numGlobalRows();
    const auto localKa  = csrDataA->numLocalColumns();
    const auto localKb  = csrDataB->numLocalRows();
    AMP_INSIST( globalKa == globalKb,
                "CSRMatrixOperationsDefault::matMultiply got incompatible global dimensions" );
    AMP_INSIST( localKa == localKb,
                "CSRMatrixOperationsDefault::matMultiply got incompatible local dimensions" );

    // Verify that all matrices have the same memory space and that it isn't device
    const auto memLocA = csrDataA->getMemoryLocation();
    const auto memLocB = csrDataB->getMemoryLocation();
    const auto memLocC = csrDataC->getMemoryLocation();
    AMP_INSIST( memLocA < AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsDefault::matMultiply not implemented for device matrices" );
    AMP_INSIST( memLocA == memLocB,
                "CSRMatrixOperationsDefault::matMultiply A and B must have the same memory type" );
    AMP_INSIST( memLocA == memLocC,
                "CSRMatrixOperationsDefault::matMultiply A and C must have the same memory type" );

    // Check if an SpGEMM helper has already been constructed for this combination
    // of matrices. If not create it first and do symbolic phase, otherwise skip
    // ahead to numeric phase
    auto bcPair = std::make_pair( csrDataB, csrDataC );
    if ( d_SpGEMMHelpers.find( bcPair ) == d_SpGEMMHelpers.end() ) {
        d_SpGEMMHelpers[bcPair] = CSRMatrixSpGEMMHelperDefault( csrDataA, csrDataB, csrDataC );
        d_SpGEMMHelpers[bcPair].symbolicMultiply();
    }

    d_SpGEMMHelpers[bcPair].numericMultiply();
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::axpy( AMP::Scalar alpha_in,
                                                                          const MatrixData &X,
                                                                          MatrixData &Y )
{
    using scalar_t = typename Policy::scalar_t;

    const auto csrDataX =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( X ) );
    auto csrDataY = getCSRMatrixData<Policy, Allocator, DiagMatrixData>( Y );

    AMP_DEBUG_INSIST( csrDataX->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    AMP_DEBUG_INSIST( csrDataY->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

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
}

template<typename Policy, typename Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::copyCast( const MatrixData &X,
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

template<typename Policy, typename Allocator, class DiagMatrixData>
template<typename PolicyIn>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::copyCast(
    CSRMatrixData<PolicyIn, Allocator, CSRLocalMatrixData<PolicyIn, Allocator>> *X,
    CSRMatrixData<Policy, Allocator, DiagMatrixData> *Y )
{
    AMP_DEBUG_INSIST( X->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );
    AMP_DEBUG_INSIST( Y->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );
    AMP_DEBUG_INSIST( X->d_memory_location == Y->d_memory_location,
                      "CSRMatrixOperationsDefault::copyCast X and Y must be in same memory space" );

    auto diagMatrixX = X->getDiagMatrix();
    auto offdMatrixX = X->getOffdMatrix();

    auto diagMatrixY = Y->getDiagMatrix();
    auto offdMatrixY = Y->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrixX && offdMatrixX );
    AMP_DEBUG_ASSERT( diagMatrixY && offdMatrixY );

    CSRLocalMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::template copyCast<PolicyIn>(
        diagMatrixX, diagMatrixY );
    if ( X->hasOffDiag() ) {
        CSRLocalMatrixOperationsDefault<Policy, Allocator>::template copyCast<PolicyIn>(
            offdMatrixX, offdMatrixY );
    }
}


template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::setScalar( AMP::Scalar alpha_in,
                                                                               MatrixData &A )
{
    using scalar_t = typename Policy::scalar_t;

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    auto alpha = static_cast<scalar_t>( alpha_in );
    d_localops_diag->setScalar( alpha, csrData->getDiagMatrix() );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->setScalar( alpha, csrData->getOffdMatrix() );
    }
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::zero( MatrixData &A )
{
    using scalar_t = typename Policy::scalar_t;
    setScalar( static_cast<scalar_t>( 0.0 ), A );
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::setDiagonal(
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

    d_localops_diag->setDiagonal( vvals_p, diagMatrix );
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::setIdentity( MatrixData &A )
{
    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    zero( A );

    d_localops_diag->setIdentity( csrData->getDiagMatrix() );
}

template<typename Policy, class Allocator, class DiagMatrixData>
void CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::extractDiagonal(
    MatrixData const &A, std::shared_ptr<Vector> buf )
{
    using scalar_t = typename Policy::scalar_t;

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_ASSERT( buf && buf->numberOfDataBlocks() == 1 );
    AMP_ASSERT( buf->isType<scalar_t>( 0 ) );

    auto *rawVecData = buf->getRawDataBlock<scalar_t>();
    auto memTypeV    = AMP::Utilities::getMemoryType( rawVecData );
    AMP_INSIST( memTypeV < AMP::Utilities::MemoryType::device &&
                    csrData->d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsDefault::extractDiagonal not implemented for device memory" );

    d_localops_diag->extractDiagonal( csrData->getDiagMatrix(), rawVecData );
}

template<typename Policy, class Allocator, class DiagMatrixData>
AMP::Scalar
CSRMatrixOperationsDefault<Policy, Allocator, DiagMatrixData>::LinfNorm( MatrixData const &A ) const
{
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;

    auto csrData =
        getCSRMatrixData<Policy, Allocator, DiagMatrixData>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    std::vector<scalar_t> rowSums( nRows, 0.0 );

    d_localops_diag->LinfNorm( csrData->getDiagMatrix(), rowSums.data() );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->LinfNorm( csrData->getOffdMatrix(), rowSums.data() );
    }

    // Reduce row sums to get global Linf norm
    auto max_norm = *std::max_element( rowSums.begin(), rowSums.end() );
    AMP_MPI comm  = csrData->getComm();
    return comm.maxReduce<scalar_t>( max_norm );
}

} // namespace AMP::LinearAlgebra
