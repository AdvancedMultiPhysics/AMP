#ifndef included_AMP_ManagedEpetraMatrix
#define included_AMP_ManagedEpetraMatrix

#include <set>

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixParameters.h"


DISABLE_WARNINGS
#include "Epetra_CrsMatrix.h"
#include "Epetra_FECrsMatrix.h"
#include <EpetraExt_Transpose_RowMatrix.h>
ENABLE_WARNINGS

namespace AMP::LinearAlgebra {


/** \class ManagedEpetraMatrix
 * \brief  A class that wraps an Epetra_CrsMatrix
 * \details  This class stores an Epetra_FECrsMatrix and provides
 * the AMP interface to this matrix.
 */
class ManagedEpetraMatrix : public Matrix
{
protected:
    //!  Empty constructor
    ManagedEpetraMatrix() = delete;

    //!  Copy constructor
    ManagedEpetraMatrix( const ManagedEpetraMatrix &rhs );

    //!  Assignment operator
    ManagedEpetraMatrix &operator=( const ManagedEpetraMatrix &rhs ) = delete;

    void multiply( shared_ptr other_op, shared_ptr &result ) override;

    //! Return the type of the matrix
    std::string type() const override { return "ManagedEpetraMatrix"; }

public:
    /** \brief Constructor
     * \param[in] p  The description of the matrix
     */
    explicit ManagedEpetraMatrix( std::shared_ptr<MatrixParameters> p );

    ManagedEpetraMatrix( std::shared_ptr<MatrixData> data );

    /** \brief Constructor from Epetra_CrsMatrix
     * \param[in]  m  Matrix to wrap
     * \param[in]  dele  If true, this class deletes the matrix
     */
    explicit ManagedEpetraMatrix( Epetra_CrsMatrix *m, bool dele = false );

    //! Destructor
    virtual ~ManagedEpetraMatrix() {}

    Epetra_CrsMatrix &getEpetra_CrsMatrix();

    /** \brief  Return an Epetra_CrsMatrix
     * \return An Epetra_CrsMatrix view of this matrix
     */
    const Epetra_CrsMatrix &getEpetra_CrsMatrix() const;

    std::shared_ptr<Matrix> transpose() const override;

    Vector::shared_ptr
    extractDiagonal( Vector::shared_ptr buf = Vector::shared_ptr() ) const override;
    Vector::shared_ptr getRowSums( Vector::shared_ptr ) const override
    {
        AMP_ERROR( "Not implemented" );
    }
    Vector::shared_ptr getRowSumsAbsolute( Vector::shared_ptr, const bool ) const override
    {
        AMP_ERROR( "Not implemented" );
    }
    std::shared_ptr<Matrix> clone() const override;
    Vector::shared_ptr createInputVector() const override;
    Vector::shared_ptr createOutputVector() const override;
};


} // namespace AMP::LinearAlgebra

#endif
