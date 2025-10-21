#ifndef included_AMP_CSRMatrixDataHelpers_h
#define included_AMP_CSRMatrixDataHelpers_h

namespace AMP {
namespace LinearAlgebra {

template<typename Config>
struct CSRMatrixDataHelpers {
    using mask_t   = unsigned char;
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

    static void TransposeDiag( const lidx_t *in_row_starts,
                               const lidx_t *in_cols_loc,
                               const scalar_t *in_coeffs,
                               const lidx_t in_num_rows,
                               const lidx_t out_num_rows,
                               const gidx_t out_first_col,
                               lidx_t *out_row_starts,
                               gidx_t *out_cols,
                               scalar_t *out_coeffs );

    static void TransposeOffd( const lidx_t *in_row_starts,
                               const gidx_t *in_cols,
                               const scalar_t *in_coeffs,
                               const lidx_t in_num_rows,
                               const gidx_t in_first_col,
                               const lidx_t out_num_rows,
                               const gidx_t out_first_col,
                               lidx_t *out_row_starts,
                               gidx_t *out_cols,
                               scalar_t *out_coeffs );

#if 0
    static void TransposeDiagCountNNZ( const lidx_t *in_row_starts,
                                       const lidx_t *in_cols_loc,
                                       const lidx_t in_num_rows,
                                       const lidx_t out_num_rows,
                                       lidx_t *out_row_starts );


    static void TransposeDiagFill( const lidx_t *in_row_starts,
                                   const lidx_t *in_cols_loc,
                                   const scalar_t *in_coeffs,
                                   const lidx_t in_num_rows,
                                   const lidx_t out_num_rows,
                                   const gidx_t out_first_col,
                                   lidx_t *out_row_starts,
                                   gidx_t *out_cols,
                                   scalar_t *out_coeffs );

    static void TransposeOffdCountNNZ( const lidx_t *in_row_starts,
                                       const lidx_t *in_cols_loc,
                                       const gidx_t *in_cols_unq,
                                       const lidx_t in_num_rows,
                                       const gidx_t in_first_col,
                                       const lidx_t out_num_rows,
                                       lidx_t *out_row_starts );


    static void TransposeOffdFill( const lidx_t *in_row_starts,
                                   const lidx_t *in_cols_loc,
                                   const gidx_t *in_cols_unq,
                                   const scalar_t *in_coeffs,
                                   const lidx_t in_num_rows,
                                   const gidx_t in_first_col,
                                   const lidx_t out_num_rows,
                                   const gidx_t out_first_col,
                                   lidx_t *out_row_starts,
                                   gidx_t *out_cols,
                                   scalar_t *out_coeffs );
#endif

    static void RowSubsetCountNNZ( const gidx_t *rows,
                                   const lidx_t num_rows,
                                   const gidx_t first_row,
                                   const lidx_t *diag_row_starts,
                                   const lidx_t *offd_row_starts,
                                   lidx_t *counts );

    static void RowSubsetFill( const gidx_t *rows,
                               const lidx_t num_rows,
                               const gidx_t first_row,
                               const gidx_t first_col,
                               const lidx_t *diag_row_starts,
                               const lidx_t *offd_row_starts,
                               const lidx_t *diag_cols_loc,
                               const lidx_t *offd_cols_loc,
                               const scalar_t *diag_coeffs,
                               const scalar_t *offd_coeffs,
                               const gidx_t *offd_colmap,
                               const lidx_t *out_row_starts,
                               gidx_t *out_cols,
                               scalar_t *out_coeffs );

    static void ColSubsetCountNNZ( const gidx_t idx_lo,
                                   const gidx_t idx_up,
                                   const gidx_t first_col,
                                   const lidx_t *diag_row_starts,
                                   const lidx_t *diag_cols_loc,
                                   const lidx_t *offd_row_starts,
                                   const lidx_t *offd_cols_loc,
                                   const gidx_t *offd_cols_unq,
                                   const lidx_t num_rows,
                                   lidx_t *out_row_starts );

    static void ColSubsetFill( const gidx_t idx_lo,
                               const gidx_t idx_up,
                               const gidx_t first_col,
                               const lidx_t *diag_row_starts,
                               const lidx_t *diag_cols_loc,
                               const scalar_t *diag_coeffs,
                               const lidx_t *offd_row_starts,
                               const lidx_t *offd_cols_loc,
                               const gidx_t *offd_cols_unq,
                               const scalar_t *offd_coeffs,
                               const lidx_t num_rows,
                               lidx_t *out_row_starts,
                               gidx_t *out_cols,
                               scalar_t *out_coeffs );

    static void ConcatHorizontalCountNNZ( const lidx_t *in_row_starts,
                                          const lidx_t num_rows,
                                          lidx_t *out_row_starts );

    static void ConcatHorizontalFill( const lidx_t *in_row_starts,
                                      const gidx_t *in_cols,
                                      const scalar_t *in_coeffs,
                                      const lidx_t num_rows,
                                      const lidx_t *out_row_starts,
                                      lidx_t *row_nnz_ctrs,
                                      gidx_t *out_cols,
                                      scalar_t *out_coeffs );

    static void ConcatVerticalCountNNZ( const lidx_t *row_starts,
                                        const gidx_t *cols,
                                        const lidx_t num_rows,
                                        const gidx_t first_col,
                                        const gidx_t last_col,
                                        const bool keep_inside,
                                        lidx_t *counts );
    static void ConcatVerticalFill( const lidx_t *in_row_starts,
                                    const gidx_t *in_cols,
                                    const scalar_t *in_coeffs,
                                    const lidx_t num_rows,
                                    const gidx_t first_col,
                                    const gidx_t last_col,
                                    const bool keep_inside,
                                    const lidx_t row_offset,
                                    const lidx_t *out_row_starts,
                                    gidx_t *out_cols,
                                    scalar_t *out_coeffs );

    static void MaskCountNNZ( const lidx_t *in_row_starts,
                              const mask_t *mask,
                              const bool keep_first,
                              const lidx_t num_rows,
                              lidx_t *out_row_starts );

    static void MaskFillDiag( const lidx_t *in_row_starts,
                              const lidx_t *in_cols_loc,
                              const scalar_t *in_coeffs,
                              const mask_t *mask,
                              const bool keep_first,
                              const lidx_t num_rows,
                              const lidx_t *out_row_starts,
                              lidx_t *out_cols_loc,
                              scalar_t *out_coeffs );
};

} // namespace LinearAlgebra
} // namespace AMP

#endif
