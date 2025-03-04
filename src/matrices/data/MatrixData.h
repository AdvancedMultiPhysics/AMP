#ifndef included_AMP_MatrixData_h
#define included_AMP_MatrixData_h

#include <memory>

#include "AMP/matrices/MatrixParametersBase.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/enable_shared_from_this.h"
#include "AMP/utils/typeid.h"
#include "AMP/vectors/CommunicationList.h"
#include "AMP/vectors/Variable.h"

namespace AMP::Discretization {
class DOFManager;
}

namespace AMP::LinearAlgebra {

class Vector;

class MatrixData : public AMP::enable_shared_from_this<MatrixData>
{
public:
    /** \brief Constructor
     * \param[in] params  Description of the matrix
     */
    explicit MatrixData( std::shared_ptr<MatrixParametersBase> params );

    //! Destructor
    virtual ~MatrixData();

    //! Empty constructor
    MatrixData();

    //! Copy constructor
    MatrixData( const MatrixData & ) = delete;

    //! Clone the data
    virtual std::shared_ptr<MatrixData> cloneMatrixData() const = 0;

    //! Transpose
    virtual std::shared_ptr<MatrixData> transpose() const = 0;

    //! Return the type of the matrix
    virtual std::string type() const = 0;

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
    template<class TYPE>
    void addValuesByGlobalID(
        size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, TYPE *values );

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
    template<class TYPE>
    void setValuesByGlobalID(
        size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, TYPE *values );

    /** \brief  Get values in the matrix
     * \param[in] num_rows The number of rows represented in values
     * \param[in] num_cols The number of cols represented in values
     * \param[in] rows  The row ids of values
     * \param[in] cols  The column ids of values
     * \param[out] values  The values to get from the matrix (row-major ordering)
     * \details  This method will return zero for any entries that
     *   have not been allocated or are not ghosts on the current processor.
     */
    template<class TYPE>
    void getValuesByGlobalID(
        size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, TYPE *values ) const;

    /** \brief  Retrieve a row of the matrix in compressed format
     * \param[in]  row Which row
     * \param[out] cols  The column ids of the returned values
     * \param[out] values  The values in the row
     */
    virtual void getRowByGlobalID( size_t row,
                                   std::vector<size_t> &cols,
                                   std::vector<double> &values ) const = 0;

    /** \brief  Add values to those in the matrix
     * \param[in] num_rows The number of rows represented in values
     * \param[in] num_cols The number of cols represented in values
     * \param[in] rows  The row ids of values
     * \param[in] cols  The column ids of values
     * \param[in] values  The values to add to the matrix (row-major ordering)
     * \param[in] id   typeID of raw data
     * \details  This method may fail if the matrix has not
     * allocated a particular (row,col) specified, depending
     * on the actual subclass of matrix used.
     */
    virtual void addValuesByGlobalID( size_t num_rows,
                                      size_t num_cols,
                                      size_t *rows,
                                      size_t *cols,
                                      void *values,
                                      const typeID &id ) = 0;

    /** \brief  Set values in the matrix
     * \param[in] num_rows The number of rows represented in values
     * \param[in] num_cols The number of cols represented in values
     * \param[in] rows  The row ids of values
     * \param[in] cols  The column ids of values
     * \param[in] values  The values to set to the matrix (row-major ordering)
     * \param[in] id   typeID of raw data
     * \details  This method may fail if the matrix has not
     * allocated a particular (row,col) specified, depending
     * on the actual subclass of matrix used.
     */
    virtual void setValuesByGlobalID( size_t num_rows,
                                      size_t num_cols,
                                      size_t *rows,
                                      size_t *cols,
                                      void *values,
                                      const typeID &id ) = 0;

    /** \brief  Get values in the matrix
     * \param[in] num_rows The number of rows represented in values
     * \param[in] num_cols The number of cols represented in values
     * \param[in] rows  The row ids of values
     * \param[in] cols  The column ids of values
     * \param[out] values  The values to get from the matrix (row-major ordering)
     * \param[in] id   typeID of raw data
     * \details  This method will return zero for any entries that
     *   have not been allocated or are not ghosts on the current processor.
     */
    virtual void getValuesByGlobalID( size_t num_rows,
                                      size_t num_cols,
                                      size_t *rows,
                                      size_t *cols,
                                      void *values,
                                      const typeID &id ) const = 0;

    /** \brief  Given a row, retrieve the non-zero column indices of the matrix in compressed format
     * \param[in]  row Which row
     */
    virtual std::vector<size_t> getColumnIDs( size_t row ) const = 0;

    /** \brief  Perform communication to ensure values in the
     * matrix are the same across cores.
     */
    virtual void makeConsistent( AMP::LinearAlgebra::ScatterType t ) = 0;

    /** \brief Get the DOFManager associated with a right vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{x}\f$
     * is a right vector )
     * \return  The DOFManager associated with a right vector
     */
    virtual std::shared_ptr<Discretization::DOFManager> getRightDOFManager() const = 0;

    /** \brief Get the DOFManager associated with a left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$\mathbf{y}\f$ is
     * a left vector )
     * \return  The DOFManager associated with a left vector
     */
    virtual std::shared_ptr<Discretization::DOFManager> getLeftDOFManager() const = 0;

    /** \brief Get the variable associated with a right vector ( For
     * \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$\mathbf{x}\f$
     * is a right vector )
     * \return  The variable associated with a right vector
     */
    virtual std::shared_ptr<Variable> getRightVariable() const
    {
        return d_pParameters->getRightVariable();
    }

    /** \brief Get the variable associated with a left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$,
     * \f$\mathbf{y}\f$ is
     * a left vector )
     * \return  The variable associated with a left vector
     */
    virtual std::shared_ptr<Variable> getLeftVariable() const
    {
        return d_pParameters->getLeftVariable();
    }

    /** \brief  Get the number of local rows in the matrix
     * \return  The number of local rows
     */
    virtual size_t numLocalRows() const;

    /** \brief  Get the number of global rows in the matrix
     * \return  The number of global rows
     */
    virtual size_t numGlobalRows() const;

    /** \brief  Get the number of local columns in the matrix
     * \return  The number of local columns
     */
    virtual size_t numLocalColumns() const;

    /** \brief  Get the number of global columns in the matrix
     * \return  The number of global columns
     */
    virtual size_t numGlobalColumns() const;

    /** \brief  Get the global id of the beginning row
     * \return  beginning global row id
     */
    virtual size_t beginRow() const;

    /** \brief  Get the global id of the ending row
     * \return  ending global row id
     */
    virtual size_t endRow() const;

    /** \brief  Get the global id of first column in diagonal block
     * \return  beginning global column id
     */
    virtual size_t beginCol() const;

    /** \brief  Get the global id of last column in diagonal block
     * \return  ending global column id
     */
    virtual size_t endCol() const;

    //! Get the comm
    inline virtual AMP_MPI getComm() const
    {
        AMP_ASSERT( d_pParameters );
        return d_pParameters->getComm();
    }

    //! Get the backend
    inline virtual AMP::Utilities::Backend getBackend() const
    {
        AMP_ASSERT( d_pParameters );
        return d_pParameters->d_backend;
    }

    /** \brief Return the typeid of the matrix coeffs
     */
    virtual typeID getCoeffType() const = 0;

protected:
    std::shared_ptr<MatrixParametersBase> d_pParameters;
};

} // namespace AMP::LinearAlgebra

#include "AMP/matrices/data/MatrixData.hpp"

#endif
