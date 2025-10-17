#ifndef included_AMP_RawCSRMatrixParameters
#define included_AMP_RawCSRMatrixParameters

#include "AMP/matrices/MatrixParametersBase.h"

namespace AMP::LinearAlgebra {


/** \class RawCSRMatrixParameters
 * \brief  A class used to hold basic parameters for a matrix
 */
template<typename Config>
class RawCSRMatrixParameters : public MatrixParametersBase
{
public:
    using gidx_t   = typename Config::gidx_t;
    using lidx_t   = typename Config::lidx_t;
    using scalar_t = typename Config::scalar_t;

    // The diagonal and off-diagonal blocks need all the same parameters
    // Like in CSRMatrixData use a nested class to pack all this away
    struct RawCSRLocalMatrixParameters {
        lidx_t *d_row_starts = nullptr;
        gidx_t *d_cols       = nullptr;
        scalar_t *d_coeffs   = nullptr;
    };

    RawCSRMatrixParameters() = delete;

    /** \brief Constructor
     * \param[in] first_row     Index for first row
     * \param[in] last_row      Index for last row
     * \param[in] first_col     Index for first col
     * \param[in] last_col      Index for last col
     * \param[in] diag          Parameters for diag block
     * \param[in] off_diag      Parameters for offd block
     * \param[in] comm          Communicator for the matrix
     */
    explicit RawCSRMatrixParameters( gidx_t first_row,
                                     gidx_t last_row,
                                     gidx_t first_col,
                                     gidx_t last_col,
                                     const RawCSRLocalMatrixParameters &diag,
                                     const RawCSRLocalMatrixParameters &off_diag,
                                     const AMP_MPI &comm )
        : MatrixParametersBase( comm ),
          d_first_row( first_row ),
          d_last_row( last_row ),
          d_first_col( first_col ),
          d_last_col( last_col ),
          d_diag( diag ),
          d_off_diag( off_diag )
    {
    }

    /** \brief Constructor
     * \param[in] first_row     Index for first row
     * \param[in] last_row      Index for last row
     * \param[in] first_col     Index for first col
     * \param[in] last_col      Index for last col
     * \param[in] diag          Parameters for diag block
     * \param[in] off_diag      Parameters for offd block
     * \param[in] comm          Communicator for the matrix
     * \param[in] backend       Acceleration backend for matrix operations
     */
    explicit RawCSRMatrixParameters( gidx_t first_row,
                                     gidx_t last_row,
                                     gidx_t first_col,
                                     gidx_t last_col,
                                     const RawCSRLocalMatrixParameters &diag,
                                     const RawCSRLocalMatrixParameters &off_diag,
                                     const AMP_MPI &comm,
                                     AMP::Utilities::Backend backend )
        : MatrixParametersBase( comm, backend ),
          d_first_row( first_row ),
          d_last_row( last_row ),
          d_first_col( first_col ),
          d_last_col( last_col ),
          d_diag( diag ),
          d_off_diag( off_diag )
    {
    }

    /** \brief Constructor
     * \param[in] first_row     Index for first row
     * \param[in] last_row      Index for last row
     * \param[in] first_col     Index for first col
     * \param[in] last_col      Index for last col
     * \param[in] diag          Parameters for diag block
     * \param[in] off_diag      Parameters for offd block
     * \param[in] comm          Communicator for the matrix
     * \param[in] var_left      Variable for left vector
     * \param[in] var_right     Variable for right vector
     */
    explicit RawCSRMatrixParameters( gidx_t first_row,
                                     gidx_t last_row,
                                     gidx_t first_col,
                                     gidx_t last_col,
                                     const RawCSRLocalMatrixParameters &diag,
                                     const RawCSRLocalMatrixParameters &off_diag,
                                     const AMP_MPI &comm,
                                     std::shared_ptr<Variable> var_left,
                                     std::shared_ptr<Variable> var_right )
        : MatrixParametersBase( comm, var_left, var_right ),
          d_first_row( first_row ),
          d_last_row( last_row ),
          d_first_col( first_col ),
          d_last_col( last_col ),
          d_diag( diag ),
          d_off_diag( off_diag )
    {
    }

    /** \brief Constructor
     * \param[in] first_row     Index for first row
     * \param[in] last_row      Index for last row
     * \param[in] first_col     Index for first col
     * \param[in] last_col      Index for last col
     * \param[in] diag          Parameters for diag block
     * \param[in] off_diag      Parameters for offd block
     * \param[in] comm          Communicator for the matrix
     * \param[in] var_left      Variable for left vector
     * \param[in] var_right     Variable for right vector
     * \param[in] backend       Acceleration backend for matrix operations
     */
    explicit RawCSRMatrixParameters( gidx_t first_row,
                                     gidx_t last_row,
                                     gidx_t first_col,
                                     gidx_t last_col,
                                     const RawCSRLocalMatrixParameters &diag,
                                     const RawCSRLocalMatrixParameters &off_diag,
                                     const AMP_MPI &comm,
                                     std::shared_ptr<Variable> var_left,
                                     std::shared_ptr<Variable> var_right,
                                     AMP::Utilities::Backend backend )
        : MatrixParametersBase( comm, var_left, var_right, backend ),
          d_first_row( first_row ),
          d_last_row( last_row ),
          d_first_col( first_col ),
          d_last_col( last_col ),
          d_diag( diag ),
          d_off_diag( off_diag )
    {
    }

    //! Destructor
    virtual ~RawCSRMatrixParameters() = default;

    // Bulk information
    gidx_t d_first_row;
    gidx_t d_last_row;
    gidx_t d_first_col;
    gidx_t d_last_col;
    // Blockwise information
    RawCSRLocalMatrixParameters d_diag, d_off_diag;
};
} // namespace AMP::LinearAlgebra

#endif
