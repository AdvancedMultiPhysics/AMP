#ifndef included_AMP_CSRMatrix_hpp
#define included_AMP_CSRMatrix_hpp

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/matrices/operations/default/CSRMatrixOperationsDefault.h"

#ifdef USE_DEVICE
    #include "AMP/matrices/operations/device/CSRMatrixOperationsDevice.h"
#endif

#if ( defined( AMP_USE_KOKKOS ) || defined( AMP_USE_TRILINOS_KOKKOS ) )
    #include "AMP/matrices/operations/kokkos/CSRMatrixOperationsKokkos.h"
#endif

#include "AMP/utils/memory.h"
#include "AMP/vectors/VectorBuilder.h"

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
    bool set_ops = false;

    if ( params->d_backend == AMP::Utilities::Backend::Hip_Cuda ) {
#ifdef USE_DEVICE
        d_matrixOps = std::make_shared<CSRMatrixOperationsDevice<Config>>();
        set_ops     = true;
#else
        AMP_ERROR( "HIP or CUDA need to be loaded to be able to use hip_cuda backend." );
#endif
    }

    if ( params->d_backend == AMP::Utilities::Backend::Kokkos ) {
#if ( defined( AMP_USE_KOKKOS ) || defined( AMP_USE_TRILINOS_KOKKOS ) )
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
    auto backend = data->getBackend();
    bool set_ops = false;

    if ( backend == AMP::Utilities::Backend::Hip_Cuda ) {
#ifdef USE_DEVICE
        d_matrixOps = std::make_shared<CSRMatrixOperationsDevice<Config>>();
        set_ops     = true;
#else
        AMP_ERROR( "HIP or CUDA need to be loaded to be able to use hip_cuda backend." );
#endif
    }

    if ( backend == AMP::Utilities::Backend::Kokkos ) {
#if ( defined( AMP_USE_KOKKOS ) || defined( AMP_USE_TRILINOS_KOKKOS ) )
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
    return std::make_shared<CSRMatrix<Config>>( d_matrixData->cloneMatrixData() );
}

template<typename Config>
std::shared_ptr<Matrix> CSRMatrix<Config>::transpose() const
{
    PROFILE( "CSRMatrix<Config>::transpose" );

    auto data = d_matrixData->transpose();
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
    PROFILE( "CSRMatrix<Config>::multiply" );
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
    Vector::shared_ptr out = buf;
    if ( !buf )
        out = this->createInputVector();

    d_matrixOps->getRowSums( *getMatrixData(), out );

    return out;
}

template<typename Config>
Vector::shared_ptr CSRMatrix<Config>::getRowSumsAbsolute( Vector::shared_ptr buf ) const
{
    Vector::shared_ptr out = buf;
    if ( !buf )
        out = this->createInputVector();

    d_matrixOps->getRowSumsAbsolute( *getMatrixData(), out );

    return out;
}

/********************************************************
 * Get the left/right vectors and DOFManagers            *
 ********************************************************/
template<typename Config>
Vector::shared_ptr CSRMatrix<Config>::createInputVector() const
{
    auto var          = std::dynamic_pointer_cast<matrixdata_t>( d_matrixData )->getRightVariable();
    const auto memloc = AMP::Utilities::getAllocatorMemoryType<allocator_type>();
    return createVector( getRightDOFManager(), var, true, memloc );
}

template<typename Config>
Vector::shared_ptr CSRMatrix<Config>::createOutputVector() const
{
    auto var          = std::dynamic_pointer_cast<matrixdata_t>( d_matrixData )->getLeftVariable();
    const auto memloc = AMP::Utilities::getAllocatorMemoryType<allocator_type>();
    return createVector( getLeftDOFManager(), var, true, memloc );
}


} // namespace AMP::LinearAlgebra

#endif
