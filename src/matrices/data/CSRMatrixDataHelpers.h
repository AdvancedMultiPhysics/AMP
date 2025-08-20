#ifndef included_AMP_CSRMatrixDataHelpers_h
#define included_AMP_CSRMatrixDataHelpers_h

namespace AMP {
namespace LinearAlgebra {

template<typename Config>
struct CSRMatrixDataHelpers {
    using gidx_t   = typename Config::gidx_t;
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    static void SortColumnsDiag(
        lidx_t *row_starts, gidx_t *cols, scalar_t *coeffs, lidx_t num_rows, gidx_t first_col );
    static void
    SortColumnsOffd( lidx_t *row_starts, gidx_t *cols, scalar_t *coeffs, lidx_t num_rows );
    static void GlobalToLocalDiag( gidx_t *cols, lidx_t nnz, gidx_t first_col, lidx_t *cols_loc );
    static void GlobalToLocalOffd(
        gidx_t *cols, lidx_t nnz, gidx_t *cols_unq, lidx_t ncols_unq, lidx_t *cols_loc );
};

} // namespace LinearAlgebra
} // namespace AMP

#endif
