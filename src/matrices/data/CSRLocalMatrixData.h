#ifndef included_AMP_CSRLocalMatrixData_h
#define included_AMP_CSRLocalMatrixData_h

#include "AMP/matrices/MatrixParametersBase.h"
#include "AMP/matrices/data/MatrixData.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <tuple>
#include <unordered_map>

namespace AMP::Discretization {
class DOFManager;
}

namespace AMP::LinearAlgebra {

// Forward declare CSRMatrix{Data,Communicator} to make them friends
template<typename C>
class CSRMatrixData;
template<typename C>
class CSRMatrixCommunicator;
template<typename C>
class CSRMatrixSpGEMMDefault;
template<typename C>
class CSRMatrixSpGEMMDevice;

template<typename Config>
class CSRLocalMatrixData : public AMP::enable_shared_from_this<CSRLocalMatrixData<Config>>
{
public:
    template<typename C>
    friend class CSRMatrixData;
    template<typename C>
    friend class CSRLocalMatrixData;
    template<typename C>
    friend class CSRMatrixCommunicator;
    template<typename C>
    friend class CSRMatrixSpGEMMDefault;
    template<typename C>
    friend class CSRMatrixSpGEMMDevice;

    using mask_t         = unsigned char;
    using gidx_t         = typename Config::gidx_t;
    using lidx_t         = typename Config::lidx_t;
    using scalar_t       = typename Config::scalar_t;
    using allocator_type = typename Config::allocator_type;
    static_assert( std::is_same_v<typename allocator_type::value_type, void> );

    using gidxAllocator_t =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<gidx_t>;
    using lidxAllocator_t =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<lidx_t>;
    using scalarAllocator_t =
        typename std::allocator_traits<allocator_type>::template rebind_alloc<scalar_t>;

    /** \brief Constructor
     * \param[in] params           Description of the matrix
     * \param[in] memory_location  Memory space where data is located
     * \param[in] first_row        Global index of starting row (inclusive)
     * \param[in] last_row         Global index of final row (exclusive)
     * \param[in] first_col        Global index of starting column (inclusive)
     * \param[in] last_col         Global index of final column (exclusive)
     * \param[in] is_diag          True if this is the diag block, influences use of first/last col
     */
    explicit CSRLocalMatrixData( std::shared_ptr<MatrixParametersBase> params,
                                 AMP::Utilities::MemoryType memory_location,
                                 typename Config::gidx_t first_row,
                                 typename Config::gidx_t last_row,
                                 typename Config::gidx_t first_col,
                                 typename Config::gidx_t last_col,
                                 bool is_diag,
                                 bool is_symbolic = false );

    //! Destructor
    virtual ~CSRLocalMatrixData();

    //! Get all data fields as tuple
    std::tuple<lidx_t *, gidx_t *, lidx_t *, scalar_t *> getDataFields()
    {
        return std::make_tuple(
            d_row_starts.get(), d_cols.get(), d_cols_loc.get(), d_coeffs.get() );
    }

    std::tuple<const lidx_t *, const gidx_t *, const lidx_t *, const scalar_t *>
    getDataFields() const
    {
        return std::make_tuple(
            d_row_starts.get(), d_cols.get(), d_cols_loc.get(), d_coeffs.get() );
    }

    //! Swap data fields with another CSRLocalMatrix
    void swapDataFields( CSRLocalMatrixData<Config> &other );

    //! Get row pointers
    lidx_t *getRowStarts() { return d_row_starts.get(); }

    //! Get the memory space where data is stored
    auto getMemoryLocation() const { return d_memory_location; }

    //! Check if this is a diagonal block
    bool isDiag() const { return d_is_diag; }

    //! Check if empty
    bool isEmpty() const { return d_is_empty; }

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

        colMap.resize( d_is_diag ? ( d_last_col - d_first_col ) : d_ncols_unq );

        if ( d_is_diag ) {
            std::iota( colMap.begin(), colMap.end(), d_first_col );
        } else {
            AMP::Utilities::copy<gidx_t, idx_t>( d_ncols_unq, d_cols_unq.get(), colMap.data() );
        }
    }


    //! Set total number of nonzeros and allocate space accordingly
    void setNNZ( lidx_t tot_nnz );

    //! Set number of nonzeros in each row and allocate space accordingly
    void setNNZ( const std::vector<lidx_t> &nnz );

    //! setNNZ function that references d_row_starts and optionally does scan
    void setNNZ( bool do_accum );

    //! \brief Remove matrix entries from within given range
    void removeRange( const scalar_t bnd_lo, const scalar_t bnd_up );

    //! Get pointers into d_cols at start of each row
    void getColPtrs( std::vector<gidx_t *> &col_ptrs );

    //! Print information about matrix block
    void printStats( bool verbose, bool show_zeros ) const;

    //! Print all information in a matrix block
    void printAll( bool force = false ) const;

    //! Apply a mask to the entries of the matrix and return a new one
    std::shared_ptr<CSRLocalMatrixData> maskMatrixData( const mask_t *mask,
                                                        const bool is_symbolic ) const;

    static std::shared_ptr<CSRLocalMatrixData>
    ConcatHorizontal( std::shared_ptr<MatrixParametersBase> params,
                      std::map<int, std::shared_ptr<CSRLocalMatrixData>> blocks );

    static std::shared_ptr<CSRLocalMatrixData>
    ConcatVertical( std::shared_ptr<MatrixParametersBase> params,
                    std::map<int, std::shared_ptr<CSRLocalMatrixData>> blocks,
                    const gidx_t first_col,
                    const gidx_t last_col,
                    const bool is_diag );

protected:
    //! Helper function for getting a global col idx from local depending on diag/offd case
    gidx_t localToGlobal( const lidx_t loc_id ) const;

    /** \brief  Sort the columns/values within each row
     * \details  This sorts within each row using the same ordering as
     * Hypre. Diagonal blocks will have the diagonal entry first, and
     * keep columns in ascending order after that. Off-diagonal blocks
     * have *local* columns in ascending order.
     */
    void sortColumns();

    //! Make a clone of this matrix data
    std::shared_ptr<CSRLocalMatrixData> cloneMatrixData();

    //! Migrate data to new memory space
    template<typename ConfigOut>
    std::shared_ptr<CSRLocalMatrixData<ConfigOut>> migrate() const;

    //! Make matrix data for transpose
    std::shared_ptr<CSRLocalMatrixData>
    transpose( std::shared_ptr<MatrixParametersBase> params ) const;

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

    //! Flag to indicate if this is a diagonal block or not
    const bool d_is_diag;
    //! Flag to indicate if this is empty
    bool d_is_empty = true;
    //! Flag to indicate if this is a symbolic matrix
    const bool d_is_symbolic;

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
