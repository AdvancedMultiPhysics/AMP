#ifndef included_AMP_CSRMatrix_hpp
#define included_AMP_CSRMatrix_hpp

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/default/CSRMatrixOperationsDefault.h"
#include "AMP/utils/Memory.h"
#include "AMP/vectors/VectorBuilder.h"

#ifdef AMP_USE_DEVICE
    #include "AMP/matrices/operations/device/CSRMatrixOperationsDevice.h"
#endif

#ifdef AMP_USE_KOKKOS
    #include "AMP/matrices/operations/kokkos/CSRMatrixOperationsKokkos.h"
#endif

#include "ProfilerApp.h"

#include <cstdio>
#include <cstring>
#include <numeric>

namespace AMP::LinearAlgebra {

/********************************************************
 * Constructor/Destructor                                *
 ********************************************************/
template<typename Config>
CSRMatrix<Config>::CSRMatrix( std::shared_ptr<MatrixParametersBase> params ) : Matrix( params )
{
    PROFILE( "CSRMatrix::constructor" );

    bool set_ops = false;

    if ( params->d_backend == AMP::Utilities::Backend::Hip_Cuda ) {
#ifdef AMP_USE_DEVICE
        d_matrixOps = std::make_shared<CSRMatrixOperationsDevice<Config>>();
        set_ops     = true;
#else
        AMP_ERROR( "HIP or CUDA need to be loaded to be able to use hip_cuda backend." );
#endif
    }

    if ( params->d_backend == AMP::Utilities::Backend::Kokkos ) {
#ifdef AMP_USE_KOKKOS
        d_matrixOps = std::make_shared<CSRMatrixOperationsKokkos<Config>>();
        set_ops     = true;
#else
        AMP_ERROR( "KOKKOS need to be loaded to be able to use kokkos backend." );
#endif
    }

    // nothing above matched, fall back on default operations
    if ( !set_ops ) {
        d_matrixOps = std::make_shared<CSRMatrixOperationsDefault<Config>>();
    }

    d_matrixData = std::make_shared<matrixdata_t>( params );
}

template<typename Config>
CSRMatrix<Config>::CSRMatrix( std::shared_ptr<MatrixData> data ) : Matrix( data )
{
    PROFILE( "CSRMatrix::constructor" );

    auto backend = data->getBackend();
    bool set_ops = false;

    if ( backend == AMP::Utilities::Backend::Hip_Cuda ) {
#ifdef AMP_USE_DEVICE
        d_matrixOps = std::make_shared<CSRMatrixOperationsDevice<Config>>();
        set_ops     = true;
#else
        AMP_ERROR( "HIP or CUDA need to be loaded to be able to use hip_cuda backend." );
#endif
    }

    if ( backend == AMP::Utilities::Backend::Kokkos ) {
#ifdef AMP_USE_KOKKOS
        d_matrixOps = std::make_shared<CSRMatrixOperationsKokkos<Config>>();
        set_ops     = true;
#else
        AMP_ERROR( "KOKKOS need to be loaded to be able to use kokkos backend." );
#endif
    }

    // nothing above matched, fall back on default operations
    if ( !set_ops ) {
        d_matrixOps = std::make_shared<CSRMatrixOperationsDefault<Config>>();
    }
}

template<typename Config>
CSRMatrix<Config>::~CSRMatrix()
{
}

/********************************************************
 * Copy/transpose the matrix                             *
 ********************************************************/
template<typename Config>
std::shared_ptr<Matrix> CSRMatrix<Config>::clone() const
{
    PROFILE( "CSRMatrix::clone" );

    return std::make_shared<CSRMatrix<Config>>( d_matrixData->cloneMatrixData() );
}

template<typename Config>
std::shared_ptr<Matrix> CSRMatrix<Config>::migrate( AMP::Utilities::MemoryType memType ) const
{
    return migrate( memType, AMP::Utilities::getDefaultBackend( memType ) );
}

template<typename Config>
std::shared_ptr<Matrix> CSRMatrix<Config>::migrate( AMP::Utilities::MemoryType memType,
                                                    AMP::Utilities::Backend backend ) const
{
    PROFILE( "CSRMatrix::migrate" );

    using ConfigHost = typename Config::template set_alloc_t<alloc::host>;
#ifdef AMP_USE_DEVICE
    using ConfigManaged = typename Config::template set_alloc_t<alloc::managed>;
    using ConfigDevice  = typename Config::template set_alloc_t<alloc::device>;
#endif

    auto data = std::dynamic_pointer_cast<const CSRMatrixData<Config>>( getMatrixData() );

    if ( memType == AMP::Utilities::getAllocatorMemoryType<typename Config::allocator_type>() ) {
        return this->clone();
    } else if ( memType == AMP::Utilities::MemoryType::host ) {
        auto dataHost = data->template migrate<ConfigHost>( backend );
        return std::make_shared<CSRMatrix<ConfigHost>>( dataHost );
    }

#ifdef AMP_USE_DEVICE
    if ( memType == AMP::Utilities::MemoryType::managed ) {
        auto dataManaged = data->template migrate<ConfigManaged>( backend );
        return std::make_shared<CSRMatrix<ConfigManaged>>( dataManaged );
    } else if ( memType == AMP::Utilities::MemoryType::device ) {
        auto dataDevice = data->template migrate<ConfigDevice>( backend );
        return std::make_shared<CSRMatrix<ConfigDevice>>( dataDevice );
    }
#endif

    AMP_ERROR( "CSRMatrix::migrate: Invalid memory type" );
    return nullptr;
}

template<typename Config>
template<typename ConfigOut>
std::shared_ptr<Matrix> CSRMatrix<Config>::migrate() const
{
    return migrate<ConfigOut>( AMP::Utilities::getDefaultBackend(
        AMP::Utilities::getAllocatorMemoryType<typename ConfigOut::allocator_type>() ) );
}

template<typename Config>
template<typename ConfigOut>
std::shared_ptr<Matrix> CSRMatrix<Config>::migrate( AMP::Utilities::Backend backend ) const
{
    PROFILE( "CSRMatrix::migrate" );

    if constexpr ( std::is_same_v<Config, ConfigOut> )
        return this->clone();

    auto data    = std::dynamic_pointer_cast<const CSRMatrixData<Config>>( getMatrixData() );
    auto dataOut = data->template migrate<ConfigOut>( backend );
    return std::make_shared<CSRMatrix<ConfigOut>>( dataOut );
}

template<typename Config>
void CSRMatrix<Config>::setBackend( AMP::Utilities::Backend backend )
{
    PROFILE( "CSRMatrix::setBackend" );

    if ( backend == AMP::Utilities::Backend::Serial ||
         backend == AMP::Utilities::Backend::OpenMP ) {
        if ( std::dynamic_pointer_cast<CSRMatrixOperationsDefault<Config>>( d_matrixOps ) ) {
            return;
        }
        d_matrixOps = std::make_shared<CSRMatrixOperationsDefault<Config>>();
    } else if ( backend == AMP::Utilities::Backend::Hip_Cuda ) {
#ifdef AMP_USE_DEVICE
        if ( std::is_same_v<typename Config::allocator_type, AMP::HostAllocator<void>> ) {
            AMP_ERROR( "CSRMatrix::setBackend Can't set Hip_Cuda backend on host-stored matrix" );
        }
        if ( std::dynamic_pointer_cast<CSRMatrixOperationsDevice<Config>>( d_matrixOps ) ) {
            return;
        }
        d_matrixOps = std::make_shared<CSRMatrixOperationsDevice<Config>>();
        d_matrixData->setBackend( AMP::Utilities::Backend::Hip_Cuda );
#else
        AMP_ERROR( "CSRMatrix::setBackend Can't set Hip_Cuda backend in non-device build" );
#endif
    } else if ( backend == AMP::Utilities::Backend::Kokkos ) {
#ifdef AMP_USE_KOKKOS
        if ( std::dynamic_pointer_cast<CSRMatrixOperationsKokkos<Config>>( d_matrixOps ) ) {
            return;
        }
        d_matrixOps = std::make_shared<CSRMatrixOperationsKokkos<Config>>();
        d_matrixData->setBackend( AMP::Utilities::Backend::Kokkos );
#else
        AMP_ERROR( "CSRMatrix::setBackend Can't set Kokkos backend in non-Kokkos build" );
#endif
    } else {
        AMP_ERROR( "CSRMatrix::setBackend Unsupported backend selected" );
    }
}

template<typename Config>
std::shared_ptr<Matrix> CSRMatrix<Config>::transpose() const
{
    PROFILE( "CSRMatrix::transpose" );

    auto data = d_matrixData->transpose();
    AMP_ASSERT( data->getBackend() == d_matrixData->getBackend() );
    return std::make_shared<CSRMatrix<Config>>( data );
}

/********************************************************
 * Multiply two matrices                                *
 * result = this * other_op                             *
 * C(N,M) = A(N,K)*B(K,M)
 ********************************************************/
template<typename Config>
void CSRMatrix<Config>::multiply( std::shared_ptr<Matrix> other_op,
                                  std::shared_ptr<Matrix> &result )
{
    PROFILE( "CSRMatrix::multiply" );

    // pull out matrix data objects and ensure they are of correct type
    auto thisData  = std::dynamic_pointer_cast<matrixdata_t>( d_matrixData );
    auto otherData = std::dynamic_pointer_cast<matrixdata_t>( other_op->getMatrixData() );
    AMP_DEBUG_INSIST( thisData && otherData,
                      "CSRMatrix::multiply received invalid MatrixData types" );

    // if the result is empty then create it
    if ( result.get() == nullptr ) {
        // Build matrix parameters object for result from this op and the other op
        auto params = std::make_shared<AMP::LinearAlgebra::MatrixParameters>(
            getLeftDOFManager(),
            other_op->getRightDOFManager(),
            getComm(),
            thisData->getLeftVariable(),
            otherData->getRightVariable(),
            thisData->getBackend(),
            std::function<std::vector<size_t>( size_t )>() );

        // Create the matrix
        auto newData = std::make_shared<matrixdata_t>( params );
        std::shared_ptr<Matrix> newMatrix =
            std::make_shared<AMP::LinearAlgebra::CSRMatrix<Config>>( newData );
        AMP_ASSERT( newMatrix );
        result.swap( newMatrix );

        d_matrixOps->matMatMult( thisData, otherData, newData );
    } else {
        auto resultData = std::dynamic_pointer_cast<matrixdata_t>( result->getMatrixData() );
        d_matrixOps->matMatMult( thisData, otherData, resultData );
    }
}

/********************************************************
 * Get/Set the diagonal                                  *
 ********************************************************/
template<typename Config>
Vector::shared_ptr CSRMatrix<Config>::extractDiagonal( Vector::shared_ptr buf ) const
{
    PROFILE( "CSRMatrix::extractDiagonal" );

    Vector::shared_ptr out = buf;
    if ( !buf )
        out = this->createInputVector();

    d_matrixOps->extractDiagonal( *getMatrixData(), out );

    return out;
}

/********************************************************
 * Additional scaling operations                         *
 ********************************************************/
template<typename Config>
Vector::shared_ptr CSRMatrix<Config>::getRowSums( Vector::shared_ptr buf ) const
{
    PROFILE( "CSRMatrix::getRowSums" );

    Vector::shared_ptr out = buf;
    if ( !buf ) {
        out = this->createOutputVector();
    }
    out->setNoGhosts();

    d_matrixOps->getRowSums( *getMatrixData(), out );

    return out;
}

template<typename Config>
Vector::shared_ptr CSRMatrix<Config>::getRowSumsAbsolute( Vector::shared_ptr buf,
                                                          const bool remove_zeros ) const
{
    PROFILE( "CSRMatrix::getRowSumsAbsolute" );

    Vector::shared_ptr out = buf;
    if ( !buf ) {
        out = this->createOutputVector();
    }
    out->setNoGhosts();

    d_matrixOps->getRowSumsAbsolute( *getMatrixData(), out, remove_zeros );

    return out;
}

/********************************************************
 * Get the left/right vectors and DOFManagers            *
 ********************************************************/
template<typename Config>
Vector::shared_ptr CSRMatrix<Config>::createInputVector() const
{
    PROFILE( "CSRMatrix::createInputVector" );

    auto var          = std::dynamic_pointer_cast<matrixdata_t>( d_matrixData )->getRightVariable();
    auto backend      = d_matrixData->getBackend();
    const auto memloc = AMP::Utilities::getAllocatorMemoryType<allocator_type>();
    return createVector( getRightDOFManager(), var, true, memloc, backend );
}

template<typename Config>
Vector::shared_ptr CSRMatrix<Config>::createOutputVector() const
{
    PROFILE( "CSRMatrix::createOutputVector" );

    auto var          = std::dynamic_pointer_cast<matrixdata_t>( d_matrixData )->getLeftVariable();
    auto backend      = d_matrixData->getBackend();
    const auto memloc = AMP::Utilities::getAllocatorMemoryType<allocator_type>();
    return createVector( getLeftDOFManager(), var, true, memloc, backend );
}

template<typename Config>
CSRMatrix<Config>::CSRMatrix( int64_t fid, AMP::IO::RestartManager *manager )
    : Matrix( fid, manager )
{
}

} // namespace AMP::LinearAlgebra

#endif
