#ifndef included_AMP_CSRVisit
#define included_AMP_CSRVisit

#include "AMP/AMP_TPLs.h"
#include "AMP/matrices/AMPCSRMatrixParameters.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/RawCSRMatrixParameters.h"
#include "AMP/matrices/data/CSRMatrixData.h"

#include "AMP/matrices/operations/MatrixOperations.h"
#include "AMP/matrices/operations/default/CSRMatrixOperationsDefault.h"
#ifdef AMP_USE_DEVICE
    #include "AMP/matrices/operations/device/CSRMatrixOperationsDevice.h"
#endif
#ifdef AMP_USE_KOKKOS
    #include "AMP/matrices/operations/kokkos/CSRMatrixOperationsKokkos.h"
#endif

namespace AMP::LinearAlgebra {

template<typename T>
struct csr_construct {
    csr_mode mode;

    std::shared_ptr<T>
    operator()( int64_t fid, AMP::IO::RestartManager *manager, const std::string &class_type )
    {
        switch ( get_alloc( mode ) ) {
        case alloc::host:
            return check_lidx<alloc::host>( fid, manager, class_type );
        case alloc::device:
            return check_lidx<alloc::device>( fid, manager, class_type );
        case alloc::managed:
            return check_lidx<alloc::managed>( fid, manager, class_type );
        }
        AMP_ERROR( "csr_visitor: mode not found!" );
    }

private:
    template<alloc a, index l, index g, scalar s>
    std::shared_ptr<T>
    construct( int64_t fid, AMP::IO::RestartManager *manager, const std::string &class_type )
    {
        using config_t = CSRConfig<a, l, g, s>;
        if constexpr ( is_config_built<config_t> ) { // avoid linker errors for missing
                                                     // instantiations
            if constexpr ( std::is_same_v<Matrix, T> ) {
                return std::make_shared<CSRMatrix<config_t>>( fid, manager );
            } else if constexpr ( std::is_same_v<MatrixData, T> ) {
                return std::make_shared<CSRMatrixData<config_t>>( fid, manager );
            } else if constexpr ( std::is_same_v<MatrixOperations, T> ) {
                if ( class_type == "CSRMatrixOperationsDefault" ) {
                    return std::make_shared<CSRMatrixOperationsDefault<config_t>>( fid, manager );
                } else if ( class_type == "CSRMatrixOperationsDevice" ) {
#if defined( AMP_USE_DEVICE )
                    return std::make_shared<CSRMatrixOperationsDevice<config_t>>( fid, manager );
#else
                    AMP_ERROR( "AMP not configured for device" );
                    return nullptr;
#endif
                } else if ( class_type == "CSRMatrixOperationsKokkos" ) {
#if defined( AMP_USE_KOKKOS )
                    return std::make_shared<CSRMatrixOperationsKokkos<config_t>>( fid, manager );
#else
                    AMP_ERROR( "AMP not configured with Kokkos" );
                    return nullptr;
#endif
                } else {
                    AMP_ERROR( "Unknown MatrixOperations type" );
                    return nullptr;
                }
            } else if constexpr ( std::is_same_v<MatrixParametersBase, T> ) {
                // hack to differentiate the two types
                return std::make_shared<RawCSRMatrixParameters<config_t>>( fid, manager );
            } else if constexpr ( std::is_same_v<MatrixParameters, T> ) {
                return std::make_shared<AMPCSRMatrixParameters<config_t>>( fid, manager );
            } else {
                AMP_ERROR( "Can only construct CSRMatrix and CSRData at present" );
                return nullptr;
            }
        }
        AMP_ERROR( "csr_construct: mode not found!" );
    }
    template<alloc a, index l, index g>
    std::shared_ptr<T>
    check_scalar( int64_t fid, AMP::IO::RestartManager *manager, const std::string &class_type )
    {
        switch ( get_scalar( mode ) ) {
        case scalar::f32:
            return construct<a, l, g, scalar::f32>( fid, manager, class_type );
        case scalar::f64:
            return construct<a, l, g, scalar::f64>( fid, manager, class_type );
        case scalar::fld:
            return construct<a, l, g, scalar::fld>( fid, manager, class_type );
        }
        AMP_ERROR( "csr_visitor: mode not found!" );
    }
    template<alloc a, index l>
    std::shared_ptr<T>
    check_gidx( int64_t fid, AMP::IO::RestartManager *manager, const std::string &class_type )
    {
        switch ( get_gidx( mode ) ) {
        case index::i32:
            return check_scalar<a, l, index::i32>( fid, manager, class_type );
        case index::i64:
            return check_scalar<a, l, index::i64>( fid, manager, class_type );
        case index::ill:
            return check_scalar<a, l, index::ill>( fid, manager, class_type );
        }
        AMP_ERROR( "csr_visitor: mode not found!" );
    }
    template<alloc a>
    std::shared_ptr<T>
    check_lidx( int64_t fid, AMP::IO::RestartManager *manager, const std::string &class_type )
    {
        switch ( get_lidx( mode ) ) {
        case index::i32:
            return check_gidx<a, index::i32>( fid, manager, class_type );
        case index::i64:
            return check_gidx<a, index::i64>( fid, manager, class_type );
        case index::ill:
            return check_gidx<a, index::ill>( fid, manager, class_type );
        }
        AMP_ERROR( "csr_visitor: mode not found!" );
    }
};

template<typename T>
auto csrConstruct( csr_mode mode,
                   int64_t fid,
                   AMP::IO::RestartManager *manager,
                   const std::string &class_type = "" )
{
    csr_construct<T> constructCSR{ mode };
    return constructCSR( fid, manager, class_type );
}

} // namespace AMP::LinearAlgebra
#endif
