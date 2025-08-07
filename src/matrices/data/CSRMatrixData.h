#ifndef included_AMP_CSRMatrixData_h
#define included_AMP_CSRMatrixData_h

#include "AMP/matrices/MatrixParametersBase.h"
#include "AMP/matrices/RawCSRMatrixParameters.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/data/MatrixData.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"

#include <functional>
#include <map>
#include <tuple>
#include <vector>

namespace AMP::Discretization {
class DOFManager;
}

namespace AMP::LinearAlgebra {

template<typename P>
class CSRMatrixSpGEMMDefault;

template<typename Config>
class CSRMatrixData : public MatrixData
{
public:
    template<typename P>
    friend class CSRMatrixSpGEMMDefault;

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
    using localmatrixdata_t = CSRLocalMatrixData<Config>;

    // Memory location, set by examining type of Allocator
    const AMP::Utilities::MemoryType d_memory_location;

    /** \brief  Constructor
     * \param[in] params  Description of the matrix
     */
    explicit CSRMatrixData( std::shared_ptr<MatrixParametersBase> params );

    //! Destructor
    virtual ~CSRMatrixData();

    //! Empty constructor
    CSRMatrixData();

    //! Copy constructor
    CSRMatrixData( const CSRMatrixData & ) = delete;

    //! Clone the data
    std::shared_ptr<MatrixData> cloneMatrixData() const override;

    //! Transpose
    std::shared_ptr<MatrixData> transpose() const override;

    //! Return the type of the matrix
    std::string type() const override { return "CSRMatrixData"; }

    /** \brief  Retrieve a row of the matrix in compressed format
     * \param[in]  row     Which row
     * \param[out] cols    The column ids of the returned values
     * \param[out] values  The values in the row
     */
    void getRowByGlobalID( size_t row,
                           std::vector<size_t> &cols,
                           std::vector<double> &values ) const override;

    /** \brief  Add values to those in the matrix
     * \param[in] num_rows The number of rows represented in values
     * \param[in] num_cols The number of cols represented in values
     * \param[in] rows     The row ids of values
     * \param[in] cols     The column ids of values
     * \param[in] values   The values to add to the matrix
     * \param[in] id       typeID of raw data
     * \details  If the matrix has not allocated a particular (row,col)
     * specified, those values will be ignored.
     */
    void addValuesByGlobalID( size_t num_rows,
                              size_t num_cols,
                              size_t *rows,
                              size_t *cols,
                              void *values,
                              const typeID &id ) override;

    /** \brief  Set values in the matrix
     * \param[in] num_rows The number of rows represented in values
     * \param[in] num_cols The number of cols represented in values
     * \param[in] rows     The row ids of values
     * \param[in] cols     The column ids of values
     * \param[in] values   The values to set to the matrix
     * \param[in] id       typeID of raw data
     * \details  If the matrix has not allocated a particular (row,col)
     * specified, those values will be ignored.
     */
    void setValuesByGlobalID( size_t num_rows,
                              size_t num_cols,
                              size_t *rows,
                              size_t *cols,
                              void *values,
                              const typeID &id ) override;

    /** \brief  Get values from the matrix
     * \param[in]  num_rows The number of rows represented in values
     * \param[in]  num_cols The number of cols represented in values
     * \param[in]  rows     The row ids of values
     * \param[in]  cols     The column ids of values
     * \param[out] values   Place to write retrieved values
     * \param[in]  id       typeID of raw data
     * \details  If the matrix has not allocated a particular (row,col)
     * specified those values will be set to zero.
     */
    void getValuesByGlobalID( size_t num_rows,
                              size_t num_cols,
                              size_t *rows,
                              size_t *cols,
                              void *values,
                              const typeID &id ) const override;

    //! Get the global indices of nonzeros in a given row
    std::vector<size_t> getColumnIDs( size_t row ) const override;

    //!  Perform communication to ensure values in the matrix are the same across cores
    void makeConsistent( AMP::LinearAlgebra::ScatterType t ) override;

    /** \brief Get the DOFManager associated with a left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$\mathbf{y}\f$ is
     * a left vector )
     * \return  The DOFManager associated with a left vector
     */
    std::shared_ptr<Discretization::DOFManager> getLeftDOFManager() const override;

    /** \brief Get the DOFManager associated with a right vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{x}\f$
     * is a right vector )
     * \return  The DOFManager associated with a right vector
     */
    std::shared_ptr<Discretization::DOFManager> getRightDOFManager() const override;

    /** \brief Get the CommList associated with a left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$\mathbf{y}\f$ is
     * a left vector )
     * \return  The CommList associated with a left vector
     */
    std::shared_ptr<CommunicationList> getLeftCommList() const;

    /** \brief Get the CommList associated with a right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$\mathbf{y}\f$ is
     * a right vector )
     * \return  The CommList associated with a right vector
     */
    std::shared_ptr<CommunicationList> getRightCommList() const;

    //!  Get the number of local rows in the matrix
    size_t numLocalRows() const override;

    //!  Get the number of global rows in the matrix
    size_t numGlobalRows() const override;

    //!  Get the number of local columns in the matrix
    size_t numLocalColumns() const override;

    //!  Get the number of global columns in the matrix
    size_t numGlobalColumns() const override;

    //!  Get the global id of the first stored row (inclusive)
    size_t beginRow() const override;

    //!  Get the global id of the last stored row (exclusive)
    size_t endRow() const override;

    //!  Get the global id of the first column in diagonal block (inclusive)
    size_t beginCol() const override;

    //!  Get the global id of the last column in diagonal block (exclusive)
    size_t endCol() const override;

    //! Return the typeid of the matrix coeffs
    typeID getCoeffType() const override
    {
        constexpr auto type = getTypeID<scalar_t>();
        return type;
    }

    //! Get pointer to diagonal block
    std::shared_ptr<localmatrixdata_t> getDiagMatrix() { return d_diag_matrix; }
    std::shared_ptr<const localmatrixdata_t> getDiagMatrix() const { return d_diag_matrix; }

    //! Get pointer to off-diagonal block
    std::shared_ptr<localmatrixdata_t> getOffdMatrix() { return d_offd_matrix; }
    std::shared_ptr<const localmatrixdata_t> getOffdMatrix() const { return d_offd_matrix; }

    //! Get row pointers from diagonal block
    lidx_t *getDiagRowStarts() { return d_diag_matrix->d_row_starts.get(); }

    //! Get row pointers from off-diagonal block
    lidx_t *getOffDiagRowStarts() { return d_offd_matrix->d_row_starts.get(); }

    //! Check if matrix is globally square
    bool isSquare() const noexcept { return d_is_square; }

    //! Check if matrix is globally square
    bool isEmpty() const noexcept { return d_diag_matrix->d_is_empty && d_offd_matrix->d_is_empty; }

    //! Get total number of nonzeros in both blocks
    auto numberOfNonZeros() const { return d_diag_matrix->d_nnz + d_offd_matrix->d_nnz; }

    //! Get total number of nonzeros in diagonal block
    auto numberOfNonZerosDiag() const { return d_diag_matrix->d_nnz; }

    //! Get total number of nonzeros in off-diagonal block
    auto numberOfNonZerosOffDiag() const { return d_offd_matrix->d_nnz; }

    //! Check if off-diagonal block is non-empty
    bool hasOffDiag() const { return !d_offd_matrix->d_is_empty; }

    //! Get the memory space where data is stored
    auto getMemoryLocation() const { return d_memory_location; }

    /** \brief  Set the number of nonzeros in each block and allocate space internally
     * \param[in] tot_nnz_diag   Number of nonzeros in whole diagonal block
     * \param[in] tot_nnz_offd   Number of nonzeros in whole off-diagonal block
     */
    void setNNZ( lidx_t tot_nnz_diag, lidx_t tot_nnz_offd );

    /** \brief  Set the number of nonzeros in each block and allocate space internally
     * \param[in] nnz_diag   Number of nonzeros in each row of diagonal block
     * \param[in] nnz_offd   Number of nonzeros in each row of off-diagonal block
     */
    void setNNZ( const std::vector<lidx_t> &nnz_diag, const std::vector<lidx_t> &nnz_offd );

    /** \brief  Set the number of nonzeros in each block and allocate space internally
     * \param[in] do_accum  Flag for whether entries in row pointers need to be accumulated
     * \details This version assumes that either the nnz per row have been written into the
     * row_pointer fields of the individual blocks, or the accumulated row pointers have been
     * written.
     */
    void setNNZ( bool do_accum );

    //! Convert global columns in blocks to local columns and free global columns
    void globalToLocalColumns();

    /** \brief  Replace left/right DOFManagers and CommunicationLists to match NNZ structure
     * \param[in] force_dm_reset Flag to force DOFManager/CommList resets
     * \details  This is necessary for matrices not created from pairs of vectors,
     * e.g. result matrices from SpGEMM and prolongators in AMG
     */
    void resetDOFManagers( bool force_dm_reset = false );

    /** \brief  Convenience function to triger internal setup
     * \param[in] force_dm_reset Flag to force DOFManager/CommList resets
     * \details  This just calls globalToLocalColumns and resetDOFManagers in succession.
     */
    void assemble( bool force_dm_reset = false );

    /** \brief  Remove matrix entries within given range
     * \param[in] bnd_lo Lower bound of range to discard
     * \param[in] bnd_up Upper bound of range to discard
     */
    void removeRange( AMP::Scalar bnd_lo, AMP::Scalar bnd_up ) override;

    //! Print information about matrix blocks
    void printStats( bool verbose, bool show_zeros ) const
    {
        std::cout << "CSRMatrixData stats:" << std::endl;
        std::cout << "  Global size: (" << numGlobalRows() << " x " << numGlobalColumns() << ")"
                  << std::endl;
        d_diag_matrix->printStats( verbose, show_zeros );
        d_offd_matrix->printStats( verbose, show_zeros );
    }

    /** \brief  Extract subset of locally owned rows into new local matrix
     * \param[in] rows  vector of global row indices to extract
     * \return  shared_ptr to CSRLocalMatrixData holding the extracted rows
     * \details  Returned matrix concatenates contributions for both diag and
     * offd components. Row extents are set to [0,rows.size) and column extents
     * are set to [0,numGlobalColumns).
     */
    std::shared_ptr<localmatrixdata_t> subsetRows( const std::vector<gidx_t> &rows ) const;

    /** \brief  Extract subset of each row containing global columns in some range
     * \param[in] idx_lo  Lower global column index (inclusive)
     * \param[in] idx_up  Upper global column index (exclusive)
     * \return  shared_ptr to CSRLocalMatrixData holding the extracted nonzeros
     * \details  Returned matrix concatenates contributions for both diag and
     * offd components. Row and column extents are inherited from this matrix,
     * but are neither sorted nor converted to local indices.
     */
    std::shared_ptr<localmatrixdata_t> subsetCols( const gidx_t idx_lo, const gidx_t idx_up ) const;

protected:
    bool d_is_square = true;
    //! Global index of first row of this block
    gidx_t d_first_row = 0;
    //! Global index of last row of this block
    gidx_t d_last_row = 0;
    //! Global index of first column of diagonal block
    gidx_t d_first_col = 0;
    //! Global index of last column of diagonal block
    gidx_t d_last_col = 0;

    //! Allocator for gidx_t matched to template parameter
    gidxAllocator_t d_gidxAllocator;
    //! Allocator for lidx_t matched to template parameter
    lidxAllocator_t d_lidxAllocator;
    //! Allocator for scalar_t matched to template parameter
    scalarAllocator_t d_scalarAllocator;

    //! Diagonal matrix block [d_first_row,d_last_row] x [d_first_col,d_last_col]
    std::shared_ptr<localmatrixdata_t> d_diag_matrix;
    //! Diagonal matrix block [d_first_row,d_last_row] x ]d_first_col,d_last_col[
    std::shared_ptr<localmatrixdata_t> d_offd_matrix;

    //! DOFManager for left vectors
    std::shared_ptr<Discretization::DOFManager> d_leftDOFManager;
    //! DOFManager for right vectors
    std::shared_ptr<Discretization::DOFManager> d_rightDOFManager;

    //! CommunicationList for left vectors
    std::shared_ptr<CommunicationList> d_leftCommList;
    //! CommunicationList for right vectors
    std::shared_ptr<CommunicationList> d_rightCommList;

    //!  \f$A_{i,j}\f$ storage of off core matrix data
    std::map<gidx_t, std::map<gidx_t, scalar_t>> d_other_data;

    //!  \f$A_{i,j}\f$ storage of off core matrix data to set
    std::map<gidx_t, std::map<gidx_t, scalar_t>> d_ghost_data;

    //!  Update matrix data off-core
    void setOtherData( std::map<gidx_t, std::map<gidx_t, scalar_t>> &,
                       AMP::LinearAlgebra::ScatterType );

    std::shared_ptr<localmatrixdata_t>
    transposeOffd( std::shared_ptr<MatrixParametersBase> params ) const;
};

template<typename Config>
static CSRMatrixData<Config> const *getCSRMatrixData( MatrixData const &A )
{
    auto ptr = dynamic_cast<CSRMatrixData<Config> const *>( &A );
    AMP_INSIST( ptr, "dynamic cast from const MatrixData to const CSRMatrixData failed" );
    return ptr;
}

template<typename Config>
static CSRMatrixData<Config> *getCSRMatrixData( MatrixData &A )
{
    auto ptr = dynamic_cast<CSRMatrixData<Config> *>( &A );
    AMP_INSIST( ptr, "dynamic cast from MatrixData to CSRMatrixData failed" );
    return ptr;
}

} // namespace AMP::LinearAlgebra

#endif
