#ifndef included_AMP_CSRVisit
#define included_AMP_CSRVisit

#include "AMP/IO/RestartManager.h"
#include "AMP/matrices/CSRConfig.h"
#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/data/CSRMatrixData.h"

namespace AMP::LinearAlgebra {

namespace detail {
template<class C>
struct list_begin;
template<class Head, class... Rest>
struct list_begin<config_list<Head, Rest...>> {
    using type = Head;
};
} // namespace detail

template<class F,
         class MatrixType                                   = LinearAlgebra::Matrix,
         template<typename some_config> class CSRMatrixType = CSRMatrix>
struct csr_visitor {
    csr_mode mode;
    std::shared_ptr<MatrixType> mat;
    std::decay_t<F> f;

    // use first built config to determing return type
    using first_config = typename detail::list_begin<built_configs>::type;
    using ret_t =
        std::invoke_result_t<std::decay_t<F>, std::shared_ptr<CSRMatrixType<first_config>>>;

    ret_t operator()()
    {
        switch ( get_alloc( mode ) ) {
        case alloc::host:
            return check_lidx<alloc::host>();
        case alloc::device:
            return check_lidx<alloc::device>();
        case alloc::managed:
            return check_lidx<alloc::managed>();
        }
        AMP_ERROR( "csr_visitor: mode not found!" );
    }

private:
    template<alloc a, index l, index g, scalar s>
    ret_t visit()
    {
        using config_t = CSRConfig<a, l, g, s>;
        if constexpr ( is_config_built<config_t> ) { // avoid linker errors for missing
                                                     // instantiations
            auto ptr = std::dynamic_pointer_cast<CSRMatrixType<config_t>>( mat );
            AMP_DEBUG_ASSERT( ptr );
            return std::forward<F>( f )( ptr );
        }
        AMP_ERROR( "csr_visitor: mode not found!" );
    }
    template<alloc a, index l, index g>
    auto check_scalar()
    {
        switch ( get_scalar( mode ) ) {
        case scalar::f32:
            return visit<a, l, g, scalar::f32>();
        case scalar::f64:
            return visit<a, l, g, scalar::f64>();
        case scalar::fld:
            return visit<a, l, g, scalar::fld>();
        }
        AMP_ERROR( "csr_visitor: mode not found!" );
    }
    template<alloc a, index l>
    auto check_gidx()
    {
        switch ( get_gidx( mode ) ) {
        case index::i32:
            return check_scalar<a, l, index::i32>();
        case index::i64:
            return check_scalar<a, l, index::i64>();
        case index::ill:
            return check_scalar<a, l, index::ill>();
        }
        AMP_ERROR( "csr_visitor: mode not found!" );
    }
    template<alloc a>
    auto check_lidx()
    {
        switch ( get_lidx( mode ) ) {
        case index::i32:
            return check_gidx<a, index::i32>();
        case index::i64:
            return check_gidx<a, index::i64>();
        case index::ill:
            return check_gidx<a, index::ill>();
        }
        AMP_ERROR( "csr_visitor: mode not found!" );
    }
};

template<class F>
csr_visitor( csr_mode, std::shared_ptr<LinearAlgebra::Matrix>, F ) -> csr_visitor<F>;

template<class F>
csr_visitor( csr_mode, std::shared_ptr<LinearAlgebra::MatrixData>, F )
    -> csr_visitor<F, LinearAlgebra::MatrixData, LinearAlgebra::CSRMatrixData>;

/*!
  Helper to recover a CSRMatrix type from a type erased Matrix pointer
  @param[in] mat Generic Matrix pointer
  @param[in] f Callable that will be invoked with the CSRMatrix pointer
  @return Result of calling f with CSRMatrix pointer
 */
template<class F>
auto csrVisit( std::shared_ptr<Matrix> mat, F &&f )
{
    auto mode = static_cast<csr_mode>( mat->mode() );
    csr_visitor visit{ mode, mat, std::forward<F>( f ) };
    return visit();
}

/*!
  Helper to recover a CSRMatrixData type from a type erased MatrixData pointer
  @param[in] matData Generic MatrixData pointer
  @param[in] f Callable that will be invoked with the CSRMatrixData pointer
  @return Result of calling f with CSRMatrixData pointer
 */
template<class F>
auto csrVisit( std::shared_ptr<LinearAlgebra::MatrixData> matData, F &&f )
{
    auto mode = static_cast<csr_mode>( matData->mode() );
    csr_visitor visit{ mode, matData, std::forward<F>( f ) };
    return visit();
}

} // namespace AMP::LinearAlgebra
#endif
