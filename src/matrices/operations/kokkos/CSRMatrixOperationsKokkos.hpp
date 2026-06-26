#include "AMP/AMP_TPLs.h"
#include "AMP/IO/HDF.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/kokkos/CSRMatrixOperationsKokkos.h"
#include "AMP/utils/Algorithms.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/typeid.h"
#include "AMP/vectors/Vector.h"

#include <algorithm>

#include "ProfilerApp.h"

#ifdef AMP_USE_KOKKOS

    #include "Kokkos_Core.hpp"

    #ifdef AMP_USE_KOKKOSKERNELS
        #include "AMP/matrices/operations/kokkos/spgemm/CSRMatrixSpGEMMKokkos.hpp"
    #else
        #include "AMP/matrices/operations/default/spgemm/CSRMatrixSpGEMMDefault.h"
        #ifdef AMP_USE_DEVICE
            #include "AMP/matrices/operations/device/spgemm/CSRMatrixSpGEMMDevice.h"
        #endif
    #endif

namespace AMP::LinearAlgebra {

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::mult( std::shared_ptr<const Vector> in,
                                              MatrixData const &A,
                                              std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsKokkos::mult" );
    AMP_DEBUG_ASSERT( in && out );
    AMP_DEBUG_ASSERT( in->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );
    AMP_DEBUG_ASSERT( csrData );
    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    auto inData            = in->getVectorData();
    auto outData           = out->getVectorData();
    scalar_t *outDataBlock = outData->getRawDataBlock<scalar_t>( 0 );

    AMP_DEBUG_ASSERT( outDataBlock );

    if ( !diagMatrix->isEmpty() ) {
        PROFILE( "CSRMatrixOperationsKokkos::mult(local)" );
        AMP_DEBUG_INSIST(
            inData->numberOfDataBlocks() == 1,
            "CSRMatrixOperationsKokkos::mult only implemented for vectors with one data block" );
        const scalar_t *inDataBlock = inData->getRawDataBlock<scalar_t>( 0 );
        AMP_DEBUG_ASSERT( inDataBlock );

        d_localops_diag->mult( inDataBlock,
                               inData->getMemoryLocation(),
                               1.0,
                               diagMatrix,
                               0.0,
                               outDataBlock,
                               outData->getMemoryLocation() );
    }

    if ( csrData->hasOffDiag() ) {
        PROFILE( "CSRMatrixOperationsKokkos::mult(ghost)" );
        const auto nGhosts = offdMatrix->numUniqueColumns();
        auto ghosts        = offdMatrix->getGhostCache();
        if constexpr ( std::is_same_v<size_t, gidx_t> ) {
            // column map can be passed to get ghosts function directly
            auto colMap = offdMatrix->getColumnMap();
            inData->getGhostValuesByGlobalID( nGhosts, colMap, ghosts, Config::mem_loc );
        } else if constexpr ( sizeof( size_t ) == sizeof( gidx_t ) ) {
            auto colMap = reinterpret_cast<size_t *>( offdMatrix->getColumnMap() );
            inData->getGhostValuesByGlobalID( nGhosts, colMap, ghosts, Config::mem_loc );
        } else {
            // Fall back to forcing a copy-cast inside matrix data
            auto colMap = offdMatrix->getColumnMapSizeT();
            inData->getGhostValuesByGlobalID( nGhosts, colMap, ghosts, Config::mem_loc );
        }

        d_localops_offd->mult( ghosts,
                               offdMatrix->d_memory_location,
                               1.0,
                               offdMatrix,
                               1.0,
                               outDataBlock,
                               outData->getMemoryLocation() );
        fence();
    }
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::multTranspose( std::shared_ptr<const Vector> in,
                                                       MatrixData const &A,
                                                       std::shared_ptr<Vector> out )
{
    PROFILE( "CSRMatrixOperationsKokkos::multTranspose" );

    // this is not meant to be an optimized version. It is provided for completeness
    AMP_DEBUG_ASSERT( in && out );

    out->zero();

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    auto inData                 = in->getVectorData();
    const scalar_t *inDataBlock = inData->getRawDataBlock<scalar_t>( 0 );
    auto outData                = out->getVectorData();
    scalar_t *outDataBlock      = outData->getRawDataBlock<scalar_t>( 0 );

    {
        PROFILE( "CSRMatrixOperationsKokkos::multTranspose (local)" );

        d_localops_diag->multTranspose( inDataBlock,
                                        inData->getMemoryLocation(),
                                        diagMatrix,
                                        outDataBlock,
                                        outData->getMemoryLocation() );
    }

    if ( csrData->hasOffDiag() ) {
        PROFILE( "CSRMatrixOperationsKokkos::multTranspose (ghost)" );

        // Possible mismatch between Config::gidx_t and size_t forces a deep copy
        // of the colMap from inside offdMatrix
        std::vector<size_t> rcols;
        offdMatrix->getColumnMap( rcols );

        Kokkos::View<scalar_t *, Kokkos::LayoutRight, typename localops_t::csr_memspace_t> vvals_d(
            "multTrans vvals", rcols.size() );

        d_localops_offd->multTranspose( inDataBlock,
                                        inData->getMemoryLocation(),
                                        offdMatrix,
                                        vvals_d.data(),
                                        localmatrixdata_t::d_memory_location );

        // now copy vvals_d back to host to write out
        auto vvals_h = Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, vvals_d );
        fence();

        // copy rcols and vvals into std::vectors and write out
        outData->addValuesByGlobalID(
            rcols.size(), rcols.data(), vvals_h.data(), AMP::Utilities::MemoryType::host );
    } else {
        fence(); // still finish with a fence if no offd term present
    }
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::scale( AMP::Scalar alpha_in, MatrixData &A )
{
    PROFILE( "CSRMatrixOperationsKokkos::scale" );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    auto alpha = static_cast<scalar_t>( alpha_in );

    d_localops_diag->scale( alpha, diagMatrix );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->scale( alpha, offdMatrix );
    }

    fence();
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::scale( AMP::Scalar alpha_in,
                                               std::shared_ptr<const Vector> D,
                                               MatrixData &A )
{
    PROFILE( "CSRMatrixOperationsKokkos::scale" );

    // constrain to one data block
    AMP_DEBUG_ASSERT( D && D->numberOfDataBlocks() == 1 && D->isType<scalar_t>( 0 ) );
    auto D_data                  = D->getVectorData();
    const scalar_t *D_data_block = D_data->getRawDataBlock<scalar_t>( 0 );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );
    AMP_DEBUG_ASSERT( csrData );
    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();
    AMP_DEBUG_ASSERT( diagMatrix );

    auto alpha = static_cast<scalar_t>( alpha_in );
    d_localops_diag->scale( alpha, D_data_block, D_data->getMemoryLocation(), diagMatrix );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->scale( alpha, D_data_block, D_data->getMemoryLocation(), offdMatrix );
    }
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::scaleInv( AMP::Scalar alpha_in,
                                                  std::shared_ptr<const Vector> D,
                                                  MatrixData &A )
{
    PROFILE( "CSRMatrixOperationsKokkos::scaleInv" );

    // constrain to one data block
    AMP_DEBUG_ASSERT( D && D->numberOfDataBlocks() == 1 && D->isType<scalar_t>( 0 ) );
    auto D_data                  = D->getVectorData();
    const scalar_t *D_data_block = D_data->getRawDataBlock<scalar_t>( 0 );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );
    AMP_DEBUG_ASSERT( csrData );
    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();
    AMP_DEBUG_ASSERT( diagMatrix );

    auto alpha = static_cast<scalar_t>( alpha_in );
    d_localops_diag->scaleInv( alpha, D_data_block, D_data->getMemoryLocation(), diagMatrix );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->scaleInv( alpha, D_data_block, D_data->getMemoryLocation(), offdMatrix );
    }
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::matMatMult( std::shared_ptr<MatrixData> A,
                                                    std::shared_ptr<MatrixData> B,
                                                    std::shared_ptr<MatrixData> C )
{
    PROFILE( "CSRMatrixOperationsKokkos::matMatMult" );

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

    // construct SpGEMM helper and call multiply
    #ifdef AMP_USE_KOKKOSKERNELS
    if constexpr ( alloc_info<Config::allocator>::device_accessible ) {
        CSRMatrixSpGEMMKokkos<Config, Kokkos::DefaultExecutionSpace> spgemm(
            csrDataA, csrDataB, csrDataC );
        spgemm.multiply();
    } else {
        CSRMatrixSpGEMMKokkos<Config, Kokkos::DefaultHostExecutionSpace> spgemm(
            csrDataA, csrDataB, csrDataC );
        spgemm.multiply();
    }
    #else // don't have kokkos-kernels, forward to default or device ops as appropriate
    if ( !alloc_info<Config::allocator>::device_accessible ) {
        CSRMatrixSpGEMMDefault<Config> spgemm( csrDataA, csrDataB, csrDataC );
        spgemm.multiply();
    } else {
        #ifdef AMP_USE_DEVICE
        CSRMatrixSpGEMMDevice<Config> spgemm( csrDataA, csrDataB, csrDataC );
        spgemm.multiply();
        #else
        AMP_ERROR( "CSRMatrixOperationsKokkos::matMatMult Undefined memory location" );
        #endif
    }
    #endif
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::axpy( AMP::Scalar alpha_in,
                                              const MatrixData &X,
                                              MatrixData &Y )
{
    PROFILE( "CSRMatrixOperationsKokkos::axpy" );

    auto csrDataX = getCSRMatrixData<Config>( const_cast<MatrixData &>( X ) );
    auto csrDataY = getCSRMatrixData<Config>( const_cast<MatrixData &>( Y ) );

    AMP_DEBUG_ASSERT( csrDataX );
    AMP_DEBUG_ASSERT( csrDataY );

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

    fence();
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::setScalar( AMP::Scalar alpha_in, MatrixData &A )
{
    PROFILE( "CSRMatrixOperationsKokkos::setScalar" );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    auto alpha = static_cast<scalar_t>( alpha_in );

    d_localops_diag->setScalar( alpha, diagMatrix );
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->setScalar( alpha, offdMatrix );
    }

    fence();
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::zero( MatrixData &A )
{
    setScalar( 0.0, A );
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::setDiagonal( std::shared_ptr<const Vector> in,
                                                     MatrixData &A )
{
    PROFILE( "CSRMatrixOperationsKokkos::setDiagonal" );

    // constrain to one data block for now
    AMP_DEBUG_ASSERT( in && in->numberOfDataBlocks() == 1 && in->isType<scalar_t>( 0 ) );

    const scalar_t *vvals_p = in->getRawDataBlock<scalar_t>();

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    d_localops_diag->setDiagonal( vvals_p, in->getMemoryLocation(), diagMatrix );

    fence();
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::setIdentity( MatrixData &A )
{
    PROFILE( "CSRMatrixOperationsKokkos::setIdentity" );

    zero( A );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );
    d_localops_diag->setIdentity( diagMatrix );

    fence();
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::extractDiagonal( MatrixData const &A,
                                                         std::shared_ptr<Vector> buf )
{
    PROFILE( "CSRMatrixOperationsKokkos::extractDiagonal" );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();

    AMP_DEBUG_ASSERT( diagMatrix );

    scalar_t *buf_p = buf->getRawDataBlock<scalar_t>();
    d_localops_diag->extractDiagonal( diagMatrix, buf_p, buf->getMemoryLocation() );

    fence();
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::getRowSums( MatrixData const &A,
                                                    std::shared_ptr<Vector> buf )
{
    PROFILE( "CSRMatrixOperationsKokkos::getRowSums" );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_ASSERT( buf && buf->numberOfDataBlocks() == 1 );
    AMP_ASSERT( buf->isType<scalar_t>( 0 ) );

    auto *rawVecData = buf->getRawDataBlock<scalar_t>();
    AMP_ASSERT( rawVecData );

    // zero out buffer so that the next two calls can accumulate into it
    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    AMP_ASSERT( buf->getLocalSize() == static_cast<size_t>( nRows ) );

    d_localops_diag->getRowSums(
        csrData->getDiagMatrix(), rawVecData, buf->getMemoryLocation(), true );
    fence();
    if ( csrData->hasOffDiag() ) {
        d_localops_offd->getRowSums(
            csrData->getOffdMatrix(), rawVecData, buf->getMemoryLocation(), false );
        fence();
    }
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::getRowSumsAbsolute( MatrixData const &A,
                                                            std::shared_ptr<Vector> buf,
                                                            bool remove_zeros )
{
    PROFILE( "CSRMatrixOperationsKokkos::getRowSumsAbsolute" );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_ASSERT( buf && buf->numberOfDataBlocks() == 1 );
    AMP_ASSERT( buf->isType<scalar_t>( 0 ) );

    auto *rawVecData = buf->getRawDataBlock<scalar_t>();
    AMP_ASSERT( rawVecData );

    // zero out buffer so that the next two calls can accumulate into it
    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    AMP_ASSERT( buf->getLocalSize() == static_cast<size_t>( nRows ) );

    bool initialize_to_zero = true;
    d_localops_diag->getRowSumsAbsolute(
        csrData->getDiagMatrix(), rawVecData, buf->getMemoryLocation(), initialize_to_zero, false );
    fence();
    if ( csrData->hasOffDiag() ) {
        initialize_to_zero = false;
        d_localops_offd->getRowSumsAbsolute( csrData->getOffdMatrix(),
                                             rawVecData,
                                             buf->getMemoryLocation(),
                                             initialize_to_zero,
                                             remove_zeros );
        fence();
    }
}

template<typename Config>
AMP::Scalar CSRMatrixOperationsKokkos<Config>::LinfNorm( MatrixData const &A ) const
{
    PROFILE( "CSRMatrixOperationsKokkos::LinfNorm" );

    auto csrData = getCSRMatrixData<Config>( const_cast<MatrixData &>( A ) );

    AMP_DEBUG_ASSERT( csrData );

    auto diagMatrix = csrData->getDiagMatrix();
    auto offdMatrix = csrData->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrix && offdMatrix );

    const auto nRows = static_cast<lidx_t>( csrData->numLocalRows() );
    Kokkos::View<scalar_t *, Kokkos::LayoutRight, typename localops_t::csr_memspace_t> sums(
        "CSRMatrixOperationsKokkos::LinfNorm sum buffer", nRows );

    bool initialize_to_zero = true, remove_zeros = false;
    d_localops_diag->getRowSumsAbsolute( diagMatrix,
                                         sums.data(),
                                         localmatrixdata_t::d_memory_location,
                                         initialize_to_zero,
                                         remove_zeros );
    fence();
    if ( csrData->hasOffDiag() ) {
        initialize_to_zero = false;
        d_localops_offd->getRowSumsAbsolute( offdMatrix,
                                             sums.data(),
                                             localmatrixdata_t::d_memory_location,
                                             initialize_to_zero,
                                             remove_zeros );
        fence();
    }

    // Reduce row sums to get global Linf norm
    auto max_norm = AMP::Utilities::Algorithms::max_element( sums.data(), nRows, Config::mem_loc );
    AMP_MPI comm  = csrData->getComm();
    return comm.maxReduce<scalar_t>( max_norm );
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::copy( const MatrixData &X, MatrixData &Y )
{
    PROFILE( "CSRMatrixOperationsKokkos::copy" );

    auto csrDataX = getCSRMatrixData<Config>( const_cast<MatrixData &>( X ) );
    auto csrDataY = getCSRMatrixData<Config>( const_cast<MatrixData &>( Y ) );

    AMP_DEBUG_ASSERT( csrDataX );
    AMP_DEBUG_ASSERT( csrDataY );
    AMP_DEBUG_INSIST( csrDataX->d_memory_location == csrDataY->d_memory_location,
                      "CSRMatrixOperationsKokkos::axpy X and Y must be in same memory space" );

    auto diagMatrixX = csrDataX->getDiagMatrix();
    auto offdMatrixX = csrDataX->getOffdMatrix();

    auto diagMatrixY = csrDataY->getDiagMatrix();
    auto offdMatrixY = csrDataY->getOffdMatrix();

    AMP_DEBUG_ASSERT( diagMatrixX && offdMatrixX );
    AMP_DEBUG_ASSERT( diagMatrixY && offdMatrixY );

    d_localops_diag->copy( diagMatrixX, diagMatrixY );
    if ( csrDataX->hasOffDiag() ) {
        d_localops_offd->copy( offdMatrixX, offdMatrixY );
    }

    fence();
}

template<typename Config>
void CSRMatrixOperationsKokkos<Config>::copyCast( const MatrixData &X, MatrixData &Y )
{
    PROFILE( "CSRMatrixOperationsKokkos::copyCast" );

    // both X and Y must be CSRMatrixData's
    const auto mode_x = static_cast<csr_mode>( X.mode() ),
               mode_y = static_cast<csr_mode>( Y.mode() );
    AMP_ASSERT( mode_x != csr_mode::other && mode_y != csr_mode::other );

    // copyCast is only for handling the scalar values
    // memory location and index types need to match
    AMP_ASSERT( get_alloc( mode_x ) == get_alloc( mode_y ) );
    AMP_ASSERT( get_lidx( mode_x ) == get_lidx( mode_y ) );
    AMP_ASSERT( get_gidx( mode_x ) == get_gidx( mode_y ) );

    auto csrDataY = getCSRMatrixData<Config>( Y );
    AMP_DEBUG_ASSERT( csrDataY );
    if ( X.getCoeffType() == getTypeID<double>() ) {
        using ConfigIn = typename Config::template set_scalar_t<scalar::f64>;
        auto csrDataX  = getCSRMatrixData<ConfigIn>( const_cast<MatrixData &>( X ) );
        AMP_DEBUG_ASSERT( csrDataX );

        copyCast<ConfigIn>( csrDataX, csrDataY );
    } else if ( X.getCoeffType() == getTypeID<float>() ) {
        using ConfigIn = typename Config::template set_scalar_t<scalar::f32>;
        auto csrDataX  = getCSRMatrixData<ConfigIn>( const_cast<MatrixData &>( X ) );
        AMP_DEBUG_ASSERT( csrDataX );

        copyCast<ConfigIn>( csrDataX, csrDataY );
    } else {
        AMP_ERROR( "Can't copyCast from the given matrix, policy not supported" );
    }
}

template<typename Config>
template<typename ConfigIn>
void CSRMatrixOperationsKokkos<Config>::copyCast( CSRMatrixData<ConfigIn> *X, matrixdata_t *Y )
{
    PROFILE( "CSRMatrixOperationsKokkos::copyCast" );

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
void CSRMatrixOperationsKokkos<Config>::writeRestart( int64_t fid ) const
{
    MatrixOperations::writeRestart( fid );
    AMP::IO::writeHDF5( fid, "mode", static_cast<std::uint16_t>( Config::mode ) );
}

} // namespace AMP::LinearAlgebra

#endif // close check for Kokkos being defined
