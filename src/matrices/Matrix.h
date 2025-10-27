#ifndef included_AMP_Matrix
#define included_AMP_Matrix

#include "AMP/matrices/MatrixParametersBase.h"
#include "AMP/matrices/data/MatrixData.h"
#include "AMP/matrices/operations/MatrixOperations.h"
#include "AMP/utils/ParameterBase.h"
#include "AMP/utils/enable_shared_from_this.h"
#include "AMP/utils/typeid.h"
#include "AMP/vectors/Vector.h"
#include <memory>


namespace AMP::LinearAlgebra {


/** \class Matrix
 * \brief  An abstract interface for using and manipulating matrices
 * \details  There are several different varieties of distributed
 * memory matrices.  While most operations between the varieties
 * can be abstracted away from the user, some cannot.  For this
 * reason, most of the time, this class will suffice as the
 * way to interact with a matrix.  Matrix creation may require
 * use of one of the derived classes.
 */

// commented out for now
// class Matrix : public AMP::enable_shared_from_this<Matrix>
class Matrix
{
public:
    //! Convenience typedef
    using shared_ptr       = std::shared_ptr<Matrix>;
    using const_shared_ptr = std::shared_ptr<const Matrix>;

    /** \brief Constructor
     * \param[in] params  Description of the matrix
     */
    explicit Matrix( std::shared_ptr<MatrixParametersBase> params );


    /** \brief Constructor
     * \param[in] data  MatrixData object associated with matrix
     */
    explicit Matrix( std::shared_ptr<MatrixData> data );

    //! Destructor
    virtual ~Matrix();

    //! Return the type of the matrix
    virtual std::string type() const = 0;

    //! Return CSR mode of the matrix.
    virtual std::uint16_t mode() const;

    //! Replace current backend with different one, no-op if same, no-op if not a CSRMatrix
    virtual void setBackend( AMP::Utilities::Backend );

    /** \brief  Matrix-vector multiplication
     * \param[in]  in  The vector to multiply
     * \param[out] out The resulting vectory
     * \details  Compute \f$\mathbf{Ain} = \mathbf{out}\f$.
     */
    void mult( std::shared_ptr<const Vector> in, std::shared_ptr<Vector> out );

    /** \brief  Matrix transpose-vector multiplication
     * \param[in]  in  The vector to multiply
     * \param[out] out The resulting vectory
     * \details  Compute \f$\mathbf{A}^T\mathbf{in} = \mathbf{out}\f$.
     */
    void multTranspose( std::shared_ptr<const Vector> in, std::shared_ptr<Vector> out );

    /** \brief  Scale the matrix by a scalar
     * \param[in] alpha  The value to scale by
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{A}\f$
     */
    void scale( AMP::Scalar alpha );

    /** \brief  Diagonally scale matrix
     * \param[in] alpha  Scaling applied to all rows
     * \param[in] D  Vector of scales, one for each row
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{D}\mathbf{A}\f$
     */
    void scale( AMP::Scalar alpha, Vector::const_shared_ptr D );

    /** \brief  Scale matrix by inverse of diagonal matrix
     * \param[in] alpha  Scaling applied to all rows
     * \param[in] D  Array of inverse scales, one for each row
     * \details  Compute \f$\mathbf{A} = \alpha\mathbf{D}^{-1}\mathbf{A}\f$
     */
    void scaleInv( AMP::Scalar alpha, Vector::const_shared_ptr D );

    /** \brief  Compute the linear combination of two matrices
     * \param[in] alpha  scalar
     * \param[in] X matrix
     * \details  Compute \f$\mathbf{THIS} = \alpha\mathbf{X} + \mathbf{THIS}\f$
     */
    void axpy( AMP::Scalar alpha, const Matrix &X );

    /** \brief  Set the non-zeros of the matrix to a scalar
     * \param[in]  alpha  The value to set the non-zeros to
     */
    void setScalar( AMP::Scalar alpha );

    /** \brief  Set the non-zeros of the matrix to zero
     * \details  May not deallocate space.
     */
    void zero();

    /** \brief  Set the diagonal to the values in a vector
     * \param[in] in The values to set the diagonal to
     */
    void setDiagonal( Vector::const_shared_ptr in );

    /** \brief  Set the matrix to the identity matrix
     */
    void setIdentity();

    /** \brief Compute the maximum row sum
     * \return  The L-infinity norm of the matrix
     */
    AMP::Scalar LinfNorm() const;

    /** \brief  Compute the product of two matrices
     * \param[in] A  Left multiplicand
     * \param[in] B  Right multiplicand
     * \return The product \f$\mathbf{AB}\f$.
     */
    static shared_ptr matMatMult( shared_ptr A, shared_ptr B );

    /** \brief  Compute the product of two matrices
     * \param[in] A     Left multiplicand
     * \param[in] B     Right multiplicand
     * \param[inout] C  Result matrix
     */
    static void matMatMult( shared_ptr A, shared_ptr B, shared_ptr C );

    /** \brief  Compute the linear combination of two matrices
     * \param[in] alpha  scalar
     * \param[in] X matrix
     * \details  Compute \f$\mathbf{THIS} = \alpha\mathbf{X} + \mathbf{THIS}\f$
     */
    void axpy( AMP::Scalar alpha, std::shared_ptr<const Matrix> X );

    /** \brief  Return a new matrix that is the transpose of this one
     * \return  A copy of this matrix transposed.
     */
    virtual std::shared_ptr<Matrix> transpose() const;

    /** \brief  Return a matrix with the same non-zero and distributed structure
     * \return  The new matrix
     */
    virtual shared_ptr clone() const = 0;

    /** \brief  Set <i>this</i> matrix with the same non-zero and distributed structure
     * as x and copy the coefficients
     */
    virtual void copy( std::shared_ptr<const Matrix> X );

    /** \brief  Set <i>this</i> matrix with the same non-zero and distributed structure
     * as x and copy the coefficients after up/down casting
     */
    void copyCast( std::shared_ptr<const Matrix> X );

    /** \brief  Extract the diagonal from a matrix
     * \param[in]  buf  An optional vector to use as a buffer
     * \return  A vector of the diagonal values
     */
    virtual Vector::shared_ptr
    extractDiagonal( Vector::shared_ptr buf = Vector::shared_ptr() ) const = 0;

    /** \brief  Get sum of each row in matrix
     * \param[in]  buf  An optional vector to use as a buffer
     * \return  A vector of the sums
     */
    virtual Vector::shared_ptr
    getRowSums( Vector::shared_ptr buf = Vector::shared_ptr() ) const = 0;

    /** \brief  Get absolute sum of each row in matrix
     * \param[in]  buf  An optional vector to use as a buffer
     * \param[in]  remove_zeros  Do we want to remove zeros
     * \return  A vector of the sums
     */
    virtual Vector::shared_ptr getRowSumsAbsolute( Vector::shared_ptr buf  = Vector::shared_ptr(),
                                                   const bool remove_zeros = false ) const = 0;

    /** \brief Get a right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{x}\f$ is a
     * right vector ) \return  A newly created right vector
     */
    virtual Vector::shared_ptr createInputVector() const = 0;

    /** \brief Get a left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{y}\f$ is a left
     * vector )
     * \return  A newly created left vector
     */
    virtual Vector::shared_ptr createOutputVector() const = 0;

    /**
     * Allow a matrix' structure and values to be modified
     */
    void enableModifications() { d_matrixData->enableModifications(); }

    /**
     * Disable modifying a matrix' structure and values
     */
    void disableModifications() { d_matrixData->disableModifications(); }

public:
    /** \brief  Add values to those in the matrix
     * \param[in] num_rows The number of rows represented in values
     * \param[in] num_cols The number of cols represented in values
     * \param[in] rows  The row ids of values
     * \param[in] cols  The column ids of values
     * \param[in] values  The values to add to the matrix (row-major ordering)
     * \details  This method may fail if the matrix has not
     * allocated a particular (row,col) specified, depending
     * on the actual subclass of matrix used.
     */
    template<typename T>
    void
    addValuesByGlobalID( size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, T *values )
    {
        d_matrixData->addValuesByGlobalID( num_rows, num_cols, rows, cols, values );
    }

    /** \brief  Set values in the matrix
     * \param[in] num_rows The number of rows represented in values
     * \param[in] num_cols The number of cols represented in values
     * \param[in] rows  The row ids of values
     * \param[in] cols  The column ids of values
     * \param[in] values  The values to set to the matrix (row-major ordering)
     * \details  This method may fail if the matrix has not
     * allocated a particular (row,col) specified, depending
     * on the actual subclass of matrix used.
     */
    template<typename T>
    void
    setValuesByGlobalID( size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, T *values )
    {
        d_matrixData->setValuesByGlobalID( num_rows, num_cols, rows, cols, values );
    }

    /** \brief  Get values in the matrix
     * \param[in] num_rows The number of rows represented in values
     * \param[in] num_cols The number of cols represented in values
     * \param[in] rows  The row ids of values
     * \param[in] cols  The column ids of values
     * \param[out] values  The values to get from the matrix (row-major ordering)
     * \details  This method will return zero for any entries that
     *   have not been allocated or are not ghosts on the current processor.
     */
    template<typename T>
    void getValuesByGlobalID(
        size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, T *values ) const
    {
        d_matrixData->getValuesByGlobalID( num_rows, num_cols, rows, cols, values );
    }

    /** \brief  Retrieve a row of the matrix in compressed format
     * \param[in]  row Which row
     * \param[out] cols  The column ids of the returned values
     * \param[out] values  The values in the row
     */
    template<typename T = double>
    void getRowByGlobalID( size_t row, std::vector<size_t> &cols, std::vector<T> &values ) const
    {
        d_matrixData->getRowByGlobalID( row, cols, values );
    }

    /** \brief  Given a row, retrieve the non-zero column indices of the matrix in compressed
     * format \param[in]  row Which row
     */
    std::vector<size_t> getColumnIDs( size_t row ) const
    {
        return d_matrixData->getColumnIDs( row );
    }

    /** \brief  Perform communication to ensure values in the
     * matrix are the same across cores.
     */
    void makeConsistent( AMP::LinearAlgebra::ScatterType t )
    {
        return d_matrixData->makeConsistent( t );
    }

    /** \brief  Get the number of local rows in the matrix
     * \return  The number of local rows
     */
    size_t numLocalRows() const { return d_matrixData->numLocalRows(); }

    /** \brief  Get the number of global rows in the matrix
     * \return  The number of global rows
     */
    size_t numGlobalRows() const { return d_matrixData->numGlobalRows(); }

    /** \brief  Get the number of local columns in the matrix
     * \return  The number of local columns
     */
    size_t numLocalColumns() const { return d_matrixData->numLocalColumns(); }

    /** \brief  Get the number of global columns in the matrix
     * \return  The number of global columns
     */
    size_t numGlobalColumns() const { return d_matrixData->numGlobalColumns(); }

    /** \brief  Get the global id of the beginning row
     * \return  beginning global row id
     */
    size_t beginRow() const { return d_matrixData->beginRow(); }

    /** \brief  Get the global id of the ending row
     * \return  ending global row id
     */
    size_t endRow() const { return d_matrixData->endRow(); }

    //! Get the comm
    AMP_MPI getComm() const { return d_matrixData->getComm(); }

    /** \brief Get the DOFManager associated with a right vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{x}\f$
     * is a right vector )
     * \return  The DOFManager associated with a right vector
     */
    virtual std::shared_ptr<Discretization::DOFManager> getRightDOFManager() const
    {
        return d_matrixData->getRightDOFManager();
    }

    /** \brief Get the DOFManager associated with a left vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{y}\f$ is a left vector ) \return  The
     * DOFManager associated with a left vector
     */
    virtual std::shared_ptr<Discretization::DOFManager> getLeftDOFManager() const
    {
        return d_matrixData->getLeftDOFManager();
    }

public:
    /** \brief  Add values to those in the matrix
     * \param[in] row  The row id of value
     * \param[in] col  The column id of value
     * \param[in] value  The value to add to the matrix
     * \details  This method may fail if the matrix has not
     * allocated a particular (row,col) specified, depending
     * on the actual subclass of matrix used.
     */
    template<typename T = double>
    void addValueByGlobalID( size_t row, size_t col, T value )
    {
        addValuesByGlobalID( 1, 1, &row, &col, &value );
    }

    /** \brief  Set values in the matrix
     * \param[in] row  The row id of value
     * \param[in] col  The column id of value
     * \param[in] value  The value to set to the matrix
     * \details  This method may fail if the matrix has not
     * allocated a particular (row,col) specified, depending
     * on the actual subclass of matrix used.
     */
    template<typename T = double>
    void setValueByGlobalID( size_t row, size_t col, T value )
    {
        setValuesByGlobalID( 1, 1, &row, &col, &value );
    }

    /** \brief  Get values in the matrix
     * \param[in] row  The row id of value
     * \param[in] col  The column id of value
     * \details  This method will return zero for any values that have not been allocated.
     */
    template<typename T = double>
    T getValueByGlobalID( size_t row, size_t col ) const
    {
        T ans;
        getValuesByGlobalID( 1, 1, &row, &col, &ans );
        return ans;
    }

    //! Return the pointer to the MatrixData
    std::shared_ptr<MatrixData> getMatrixData() { return d_matrixData; }

    //! Return the pointer to the MatrixData
    std::shared_ptr<const MatrixData> getMatrixData() const { return d_matrixData; }

    //! Get a unique id hash for the matrix
    uint64_t getID() const;

public: // Write/read restart data
    /**
     * \brief    Register any child objects
     * \details  This function will register child objects with the manager
     * \param manager   Restart manager
     */
    virtual void registerChildObjects( AMP::IO::RestartManager *manager ) const;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    virtual void writeRestart( int64_t fid ) const;

    /**
     * \brief    Read restart data to file
     * \details  This function will create a variable from the restart file
     * \param fid    File identifier to write
     * \param manager   Restart manager
     */
    Matrix( int64_t fid, AMP::IO::RestartManager *manager );

protected:
    //! Protected constructor
    Matrix();

    //! Protected copy constructor
    Matrix( const Matrix & );

    //! Protected assignment operator
    Matrix &operator=( const Matrix & ) = delete;


protected:
    /** \brief  Multiply two matrices and store in a third
     *    result = this * other_op
     * \param[in]  other_op  The other matrix to multiply
     * \param[out] result  The matrix to store the result
     */
    virtual void multiply( shared_ptr other_op, shared_ptr &result ) = 0;

    //! Pointer to data
    std::shared_ptr<MatrixData> d_matrixData;
    std::shared_ptr<MatrixOperations> d_matrixOps;
};

inline std::shared_ptr<Matrix> Matrix::transpose() const
{
    AMP_ERROR( "not implemented" );
    return std::shared_ptr<Matrix>();
}

std::ostream &operator<<( std::ostream &out, const Matrix &p );

inline std::ostream &operator<<( std::ostream &out, std::shared_ptr<const Matrix> p )
{
    return operator<<( out, *p );
}

} // namespace AMP::LinearAlgebra

#endif
