#ifndef included_AMP_CSRLocalMatrixData_h
#define included_AMP_CSRLocalMatrixData_h

#include "AMP/matrices/MatrixParametersBase.h"
#include "AMP/matrices/data/MatrixData.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/memory.h"

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <tuple>

namespace AMP::Discretization {
class DOFManager;
}

namespace AMP::LinearAlgebra {

// Forward declare CSRMatrixData to make it a friend
template<typename P, class A, class DIAG>
class CSRMatrixData;

template<typename Policy, class Allocator>
class CSRLocalMatrixData :
    public AMP::enable_shared_from_this<CSRLocalMatrixData<Policy, Allocator>>
{
public:
    template<typename P, class A, class DIAG>
    friend class CSRMatrixData;

    using gidx_t   = typename Policy::gidx_t;
    using lidx_t   = typename Policy::lidx_t;
    using scalar_t = typename Policy::scalar_t;
    using gidxAllocator_t =
        typename std::allocator_traits<Allocator>::template rebind_alloc<gidx_t>;
    using lidxAllocator_t =
        typename std::allocator_traits<Allocator>::template rebind_alloc<lidx_t>;
    using scalarAllocator_t =
        typename std::allocator_traits<Allocator>::template rebind_alloc<scalar_t>;

    /** \brief Constructor
     * \param[in] outer Containing CSRMatrixData object
     * \param[in] params Description of the matrix
     * \param[in] is_diag True if this is the diag block, influences which dofs are used/ignored
     */
    explicit CSRLocalMatrixData( std::shared_ptr<MatrixParametersBase> params,
                                 AMP::Utilities::MemoryType memory_location,
                                 typename Policy::gidx_t first_row,
                                 typename Policy::gidx_t last_row,
                                 typename Policy::gidx_t first_col,
                                 typename Policy::gidx_t last_col,
                                 bool is_diag );

    //! Destructor
    virtual ~CSRLocalMatrixData();

    //! Get all data fields as tuple
    std::tuple<lidx_t *, gidx_t *, lidx_t *, scalar_t *> getDataFields()
    {
        return std::make_tuple(
            d_row_starts.get(), d_cols.get(), d_cols_loc.get(), d_coeffs.get() );
    }

    //! Get the memory space where data is stored
    auto getMemoryLocation() const { return d_memory_location; }

    //! Check if this is a diagonal block
    bool isDiag() const { return d_is_diag; }

    //! Get total number of nonzeros in block
    lidx_t numberOfNonZeros() const { return d_nnz; }

    //! Get number of local rows
    lidx_t numLocalRows() const { return d_last_row - d_first_row; }

    //! Get number of local columns in diagonal region
    lidx_t numLocalColumns() const { return d_last_col - d_first_col; }

    //! Get number of unique columns
    lidx_t numUniqueColumns() const { return d_ncols_unq; }

    //! Get global index of first row in block (inclusive)
    gidx_t beginRow() const { return d_first_row; }

    //! Get global index of last row in block (exclusive)
    gidx_t endRow() const { return d_last_row; }

    //! Get global index of first column in diagonal region (inclusive)
    gidx_t beginCol() const { return d_first_col; }

    //! Get global index of last column in diagonal region (exclusive)
    gidx_t endCol() const { return d_last_col; }

    //! Convert global column ids to local and free global columns
    void globalToLocalColumns();

    //! Get pointer to unique columns, only useful for off-diagonal block
    gidx_t *getColumnMap() const
    {
        if ( d_is_diag ) {
            return nullptr;
        }
        return d_cols_unq.get();
    }

    /** \brief  Copy unique columns into
     * \param[out] colMap  Vector to copy column indices into
     * \details  If this is a diagonal block the uniques are all columns
     * in diagonal region and is only supported for simplicity in calling
     * contexts. If this is an off-diagonal block this copies out the
     * the contents of d_cols_unq.
     */
    template<typename idx_t>
    void getColumnMap( std::vector<idx_t> &colMap ) const
    {
        // Don't do anything if offd and empty
        if ( !d_is_diag && d_is_empty ) {
            return;
        }

        AMP_INSIST( d_memory_location < AMP::Utilities::MemoryType::device,
                    "Copies from device to host memory not implemented yet" );

        colMap.resize( d_is_diag ? ( d_last_col - d_first_col ) : d_ncols_unq );

        if ( d_is_diag ) {
            std::iota( colMap.begin(), colMap.end(), d_first_col );
        } else {
            if constexpr ( std::is_same_v<idx_t, gidx_t> ) {
                std::copy( d_cols_unq.get(), d_cols_unq.get() + d_ncols_unq, colMap.begin() );
            } else {
                std::transform( d_cols_unq.get(),
                                d_cols_unq.get() + d_ncols_unq,
                                colMap.begin(),
                                []( gidx_t c ) -> idx_t { return c; } );
            }
        }
    }

    //! Set number of nonzeros in each row and allocate space accordingly
    void setNNZ( const std::vector<lidx_t> &nnz );

    //! Get pointers into d_cols at start of each row
    void getColPtrs( std::vector<gidx_t *> &col_ptrs );

    //! Print information about matrix block
    void printStats( bool show_zeros ) const
    {
        std::cout << ( d_is_diag ? "  diag block:" : "  offd block:" ) << std::endl;
        if ( d_is_empty ) {
            std::cout << "    EMPTY" << std::endl;
            return;
        }
        std::cout << "    first | last row: " << d_first_row << " | " << d_last_row << std::endl;
        std::cout << "    first | last col: " << d_first_col << " | " << d_last_col << std::endl;
        auto tot_nnz     = d_row_starts[d_num_rows];
        scalar_t avg_nnz = static_cast<scalar_t>( tot_nnz ) / static_cast<scalar_t>( d_num_rows );
        std::cout << "    avg nnz per row: " << avg_nnz << std::endl;
        std::cout << "    tot nnz: " << tot_nnz << std::endl;
        std::cout << "    row 0: ";
        for ( auto n = d_row_starts[0]; n < d_row_starts[1]; ++n ) {
            if ( d_coeffs[n] != 0 || show_zeros ) {
                std::cout << "(" << d_cols[n] << "," << d_coeffs[n] << "), ";
            }
        }
        std::cout << "    row last: ";
        for ( auto n = d_row_starts[d_num_rows - 1]; n < d_row_starts[d_num_rows]; ++n ) {
            if ( d_coeffs[n] != 0 || show_zeros ) {
                std::cout << "(" << d_cols[n] << "," << d_coeffs[n] << "), ";
            }
        }
        std::cout << std::endl << std::endl;
    }

protected:
    //! Make a clone of this matrix data
    std::shared_ptr<CSRLocalMatrixData> cloneMatrixData();

    /** \brief  Get columns and values from one row
     * \param[in]  local_row  Local index of desired row
     * \param[out] cols       Vector of global column ids to push onto
     * \param[out] values     Vector of values to push onto
     */
    void getRowByGlobalID( const size_t local_row,
                           std::vector<size_t> &cols,
                           std::vector<double> &values ) const;

    /** \brief  Get values from the matrix
     * \param[in]  local_row  Local row to query
     * \param[in]  num_cols   Number of cols to query
     * \param[in]  cols       The column indices to query
     * \param[out] values     Place to write retrieved values
     * \details  If the matrix has not allocated a particular col
     * specified those values will be set to zero.
     */
    void getValuesByGlobalID( const size_t local_row,
                              const size_t num_cols,
                              size_t *cols,
                              scalar_t *values ) const;

    /** \brief  Add to existing values at given column locations in a row
     * \param[in] num_cols   Number of columns/values passed in
     * \param[in] local_row  Local index row to alter
     * \param[in] cols       Global column indices where values are to be altered
     * \param[in] values     Values to add to existing ones
     * \details Entries in passed cols array that aren't the stored row are ignored
     */
    void addValuesByGlobalID( const size_t num_cols,
                              const size_t rows,
                              const size_t *cols,
                              const scalar_t *vals );

    /** \brief  Overwrite existing values at given column locations in a row
     * \param[in] num_cols   Number of columns/values passed in
     * \param[in] local_row  Local index row to alter
     * \param[in] cols       Global column indices where values are to be set
     * \param[in] values     Values to write
     * \details Entries in passed cols array that aren't the stored row are ignored
     */
    void setValuesByGlobalID( const size_t num_cols,
                              const size_t rows,
                              const size_t *cols,
                              const scalar_t *vals );

    /** \brief  Get columns and values from one row
     * \param[in] local_row  Local index of desired row
     * \param[out] cols      Vector of global column ids to push onto
     * \param[out] values    Vector of values to push onto
     */
    std::vector<size_t> getColumnIDs( const size_t local_row ) const;

    /** \brief  Sort the columns/values within each row
     * \details  This sorts within each row using the same ordering as
     * Hypre. Diagonal blocks will have the diagonal entry first, and
     * keep columns in ascending order after that. Off-diagonal blocks
     * have *local* columns in ascending order.
     */
    void sortColumns();

    // Data members passed from outer CSRMatrixData object
    //! Memory space where data lives, compatible with allocator template parameter
    const AMP::Utilities::MemoryType d_memory_location;
    //! Global index of first row of this block
    const gidx_t d_first_row;
    //! Global index of last row of this block
    const gidx_t d_last_row;
    //! Global index of first column of diagonal block
    const gidx_t d_first_col;
    //! Global index of last column of diagonal block
    const gidx_t d_last_col;

    //! Flag to indicate if this a diagonal block or not
    const bool d_is_diag;
    //! Flag to indicate if this is empty
    bool d_is_empty = true;

    //! Allocator for gidx_t matched to template parameter
    gidxAllocator_t d_gidxAllocator;
    //! Allocator for lidx_t matched to template parameter
    lidxAllocator_t d_lidxAllocator;
    //! Allocator for scalar_t matched to template parameter
    scalarAllocator_t d_scalarAllocator;

    //! Starting index of each row within other data arrays
    std::shared_ptr<lidx_t[]> d_row_starts;
    //! Global column indices for nonzeros in each row
    std::shared_ptr<gidx_t[]> d_cols;
    //! Unique set of indices from d_cols, unused for diagonal block
    std::shared_ptr<gidx_t[]> d_cols_unq;
    //! Local column indices for nonzeros in each row
    std::shared_ptr<lidx_t[]> d_cols_loc;
    //! Nonzero values in each row
    std::shared_ptr<scalar_t[]> d_coeffs;

    //! Number of locally stored rows, (d_local_row - d_first_row)
    const lidx_t d_num_rows;
    //! Total number of nonzeros
    lidx_t d_nnz = 0;
    //! Number of unique columns referenced by block
    lidx_t d_ncols_unq = 0;
};

} // namespace AMP::LinearAlgebra

#endif
