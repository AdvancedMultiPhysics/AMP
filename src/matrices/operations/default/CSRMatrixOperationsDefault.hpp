#include "AMP/IO/HDF.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/default/CSRMatrixOperationsDefault.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/typeid.h"
#include "AMP/vectors/Vector.h"

#include <algorithm>

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {

template<typename Config>
void CSRMatrixOperationsDefault<Config>::mult( std::shared_ptr<const Vector> in,
                                               MatrixData const &A,
                                               std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsDefault::mult" );
    AMP_DEBUG_ASSERT( in && out );
    AMP_DEBUG_ASSERT( in->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

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
            auto colMap = offdMatrix->getColumnMap();
            in->getGhostValuesByGlobalID( nGhosts, colMap, ghosts.data() );
        } else if constexpr ( sizeof( size_t ) == sizeof( gidx_t ) ) {
            auto colMap = reinterpret_cast<size_t *>( offdMatrix->getColumnMap() );
            in->getGhostValuesByGlobalID( nGhosts, colMap, ghosts.data() );
        } else {
            // type mismatch, need to copy/cast into temporary vector
            AMP_WARN_ONCE(
                "CSRMatrixOperationsDefault::mult: Deep copy/cast of column map required" );
            std::vector<size_t> colMap;
            offdMatrix->getColumnMap( colMap );
            in->getGhostValuesByGlobalID( nGhosts, colMap.data(), ghosts.data() );
        }
        d_localops_offd->mult( ghosts.data(), offdMatrix, outDataBlock );
    }
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::multTranspose( std::shared_ptr<const Vector> in,
                                                        MatrixData const &A,
                                                        std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsDefault::multTranspose" );

    // this is not meant to be an optimized version. It is provided for completeness
    AMP_DEBUG_ASSERT( in && out );

    out->zero();

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

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

template<typename Config>
void CSRMatrixOperationsDefault<Config>::scale( AMP::Scalar alpha_in, MatrixData &A )
{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

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

template<typename Config>
void CSRMatrixOperationsDefault<Config>::scale( AMP::Scalar alpha_in,
                                                std::shared_ptr<const Vector> D,
                                                MatrixData &A )
{
    // constrain to one data block
    AMP_DEBUG_ASSERT( D && D->numberOfDataBlocks() == 1 && D->isType<scalar_t>( 0 ) );
    auto D_data                  = D->getVectorData();
    const scalar_t *D_data_block = D_data->getRawDataBlock<scalar_t>( 0 );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );
    AMP_DEBUG_ASSERT( csrData );
    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();
    AMP_DEBUG_ASSERT( diagMatrix );
    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    auto alpha = static_cast<scalar_t>( alpha_in );
    d_localops_diag->scale( alpha, D_data_block, diagMatrix );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->scale( alpha, D_data_block, offdMatrix );
    }
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::scaleInv( AMP::Scalar alpha_in,
                                                   std::shared_ptr<const Vector> D,
                                                   MatrixData &A )
{
    // constrain to one data block
    AMP_DEBUG_ASSERT( D && D->numberOfDataBlocks() == 1 && D->isType<scalar_t>( 0 ) );
    auto D_data                  = D->getVectorData();
    const scalar_t *D_data_block = D_data->getRawDataBlock<scalar_t>( 0 );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );
    AMP_DEBUG_ASSERT( csrData );
    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();
    AMP_DEBUG_ASSERT( diagMatrix );
    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    auto alpha = static_cast<scalar_t>( alpha_in );
    d_localops_diag->scaleInv( alpha, D_data_block, diagMatrix );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->scaleInv( alpha, D_data_block, offdMatrix );
    }
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::matMatMult( std::shared_ptr<MatrixData> A,
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
    AMP_INSIST( memLocA < AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsDefault::matMatMult not implemented for device matrices" );
    AMP_INSIST( memLocA == memLocB,
                "CSRMatrixOperationsDefault::matMatMult A and B must have the same memory type" );
    AMP_INSIST( memLocA == memLocC,
                "CSRMatrixOperationsDefault::matMatMult A and C must have the same memory type" );

    // Check if an SpGEMM helper has already been constructed for this combination
    // of matrices. If not create it first and do symbolic phase, otherwise skip
    // ahead to numeric phase
    auto bcPair = std::make_pair( csrDataB, csrDataC );
    if ( d_SpGEMMHelpers.find( bcPair ) == d_SpGEMMHelpers.end() ) {
        AMP_INSIST( csrDataC->isEmpty(),
                    "CSRMatrixOperationsDefault::matMatMult A*B->C only applicable to non-empty C "
                    "if it came from same A and B input matrices originally" );
        d_SpGEMMHelpers[bcPair] = CSRMatrixSpGEMMDefault( csrDataA, csrDataB, csrDataC, true );
        d_SpGEMMHelpers[bcPair].symbolicMultiply();
        d_SpGEMMHelpers[bcPair].numericMultiply();
    } else {
        d_SpGEMMHelpers[bcPair].numericMultiplyReuse();
    }
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::axpy( AMP::Scalar alpha_in,
                                               const MatrixData &X,
                                               MatrixData &Y )
{
    const auto csrDataX = getCSRMatrixData<Config>( const_cast<MatrixData &>( X ) );
    auto csrDataY       = getCSRMatrixData<Config>( Y );

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


template<typename Config>
void CSRMatrixOperationsDefault<Config>::setScalar( AMP::Scalar alpha_in, MatrixData &A )
{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    auto alpha = static_cast<scalar_t>( alpha_in );
    d_localops_diag->setScalar( alpha, csrData->getDiagMatrix() );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->setScalar( alpha, csrData->getOffdMatrix() );
    }
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::zero( MatrixData &A )
{
    setScalar( 0, A );
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::setDiagonal( std::shared_ptr<const Vector> in,
                                                      MatrixData &A )
{
    // constrain to one data block for now
    AMP_DEBUG_ASSERT( in && in->numberOfDataBlocks() == 1 && in->isType<scalar_t>( 0 ) );

    const scalar_t *vvals_p = in->getRawDataBlock<scalar_t>();

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    d_localops_diag->setDiagonal( vvals_p, diagMatrix );
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::setIdentity( MatrixData &A )
{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    zero( A );

    d_localops_diag->setIdentity( csrData->getDiagMatrix() );
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::extractDiagonal( MatrixData const &A,
                                                          std::shared_ptr<Vector> buf )
{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_ASSERT( buf && buf->numberOfDataBlocks() == 1 );
    AMP_ASSERT( buf->isType<scalar_t>( 0 ) );

    auto *rawVecData = buf->getRawDataBlock<scalar_t>();
    auto memTypeV    = AMP::Utilities::getMemoryType( rawVecData );
    AMP_INSIST( memTypeV < AMP::Utilities::MemoryType::device &&
                    csrData->d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsDefault::extractDiagonal not implemented for device memory" );

    d_localops_diag->extractDiagonal( csrData->getDiagMatrix(), rawVecData );
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::getRowSums( MatrixData const &A,
                                                     std::shared_ptr<Vector> buf )
{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_ASSERT( buf && buf->numberOfDataBlocks() == 1 );
    AMP_ASSERT( buf->isType<scalar_t>( 0 ) );

    auto *rawVecData = buf->getRawDataBlock<scalar_t>();
    AMP_ASSERT( rawVecData );
    auto memTypeV = AMP::Utilities::getMemoryType( rawVecData );
    AMP_INSIST( memTypeV < AMP::Utilities::MemoryType::device &&
                    csrData->d_memory_location < AMP::Utilities::MemoryType::device,
                "CSRMatrixOperationsDefault::getRowSums not implemented for device memory" );

    // zero out buffer so that the next two calls can accumulate into it
    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    AMP_ASSERT( buf->getLocalSize() == static_cast<size_t>( nRows ) );
    AMP::Utilities::Algorithms<scalar_t>::fill_n( rawVecData, nRows, 0 );

    d_localops_diag->getRowSums( csrData->getDiagMatrix(), rawVecData );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->getRowSums( csrData->getOffdMatrix(), rawVecData );
    }
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::getRowSumsAbsolute( MatrixData const &A,
                                                             std::shared_ptr<Vector> buf,
                                                             const bool remove_zeros )
{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_ASSERT( buf && buf->numberOfDataBlocks() == 1 );
    AMP_ASSERT( buf->isType<scalar_t>( 0 ) );

    auto *rawVecData = buf->getRawDataBlock<scalar_t>();
    AMP_ASSERT( rawVecData );
    auto memTypeV = AMP::Utilities::getMemoryType( rawVecData );
    AMP_INSIST(
        memTypeV < AMP::Utilities::MemoryType::device &&
            csrData->d_memory_location < AMP::Utilities::MemoryType::device,
        "CSRMatrixOperationsDefault::getRowSumsAbsolute not implemented for device memory" );

    // zero out buffer so that the next two calls can accumulate into it
    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    AMP_ASSERT( buf->getLocalSize() == static_cast<size_t>( nRows ) );
    AMP::Utilities::Algorithms<scalar_t>::fill_n( rawVecData, nRows, 0 );

    d_localops_diag->getRowSumsAbsolute( csrData->getDiagMatrix(), rawVecData );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->getRowSumsAbsolute( csrData->getOffdMatrix(), rawVecData );
    }

    if ( remove_zeros ) {
        for ( lidx_t row = 0; row < nRows; ++row ) {
            rawVecData[row] = rawVecData[row] != 0 ? rawVecData[row] : 1;
        }
    }
}

template<typename Config>
AMP::Scalar CSRMatrixOperationsDefault<Config>::LinfNorm( MatrixData const &A ) const
{
    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_INSIST( csrData->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    std::vector<scalar_t> rowSums( nRows, 0 );

    d_localops_diag->getRowSumsAbsolute( csrData->getDiagMatrix(), rowSums.data() );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->getRowSumsAbsolute( csrData->getOffdMatrix(), rowSums.data() );
    }

    // Reduce row sums to get global Linf norm
    auto max_norm = *std::max_element( rowSums.begin(), rowSums.end() );
    AMP_MPI comm  = csrData->getComm();
    return comm.maxReduce<scalar_t>( max_norm );
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::copy( const MatrixData &X, MatrixData &Y )
{
    const auto csrDataX = getCSRMatrixData<Config>( const_cast<MatrixData &>( X ) );
    auto csrDataY       = getCSRMatrixData<Config>( Y );

    AMP_DEBUG_INSIST( csrDataX->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    AMP_DEBUG_INSIST( csrDataY->d_memory_location != AMP::Utilities::MemoryType::device,
                      "CSRMatrixOperationsDefault is not implemented for device memory" );

    const auto diagMatrixX = csrDataX->getDiagMatrix();
    const auto offdMatrixX = csrDataX->getOffdMatrix();

    auto diagMatrixY = csrDataY->getDiagMatrix();
    auto offdMatrixY = csrDataY->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrixX && offdMatrixX );
    AMP_DEBUG_ASSERT( diagMatrixY && offdMatrixY );

    d_localops_diag->copy( diagMatrixX, diagMatrixY );
    if ( csrDataX->hasOffDiag() ) {
        d_localops_offd->copy( offdMatrixX, offdMatrixY );
    }
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::copyCast( const MatrixData &X, MatrixData &Y )
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
void CSRMatrixOperationsDefault<Config>::copyCast(
    CSRMatrixData<typename ConfigIn::template set_alloc_t<Config::allocator>> *X, matrixdata_t *Y )
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

    localops_t::template copyCast<ConfigIn>( diagMatrixX, diagMatrixY );
    if ( X->hasOffDiag() ) {
        localops_t::template copyCast<ConfigIn>( offdMatrixX, offdMatrixY );
    }
}

template<typename Config>
void CSRMatrixOperationsDefault<Config>::writeRestart( int64_t fid ) const
{
    MatrixOperations::writeRestart( fid );
    AMP::IO::writeHDF5( fid, "mode", static_cast<std::uint16_t>( Config::mode ) );
}

} // namespace AMP::LinearAlgebra
