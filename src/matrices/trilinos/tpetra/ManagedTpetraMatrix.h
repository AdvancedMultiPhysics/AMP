#ifndef included_AMP_ManagedTpetraMatrix
#define included_AMP_ManagedTpetraMatrix

#include <set>

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixParameters.h"


DISABLE_WARNINGS
#include "Tpetra_CrsMatrix_decl.hpp"
ENABLE_WARNINGS

namespace AMP::LinearAlgebra {


/** \class ManagedTpetraMatrix
 * \brief  A class that wraps an Tpetra_CrsMatrix
 * \details  This class stores an Tpetra_FECrsMatrix and provides
 * the AMP interface to this matrix.
 */
template<typename ST = double,
         typename LO = int32_t,
         typename GO = int64_t,
         typename NT = Tpetra::Vector<>::node_type>
class ManagedTpetraMatrix : public Matrix
{
protected:
    //!  Empty constructor
    ManagedTpetraMatrix() = delete;

    //!  Copy constructor
    ManagedTpetraMatrix( const ManagedTpetraMatrix<ST, LO, GO, NT> &rhs );

    //!  Assignment operator
    ManagedTpetraMatrix &operator=( const ManagedTpetraMatrix<ST, LO, GO, NT> &rhs ) = delete;

    void multiply( shared_ptr other_op, shared_ptr &result ) override;

    //! Return the type of the matrix
    std::string type() const override { return "ManagedTpetraMatrix"; }

public:
    /** \brief Constructor
     * \param[in] p  The description of the matrix
     */
    explicit ManagedTpetraMatrix( std::shared_ptr<MatrixParameters> p );

    ManagedTpetraMatrix( std::shared_ptr<MatrixData> data );

    /** \brief Constructor from Tpetra_CrsMatrix
     * \param[in]  m  Matrix to wrap
     * \param[in]  dele  If true, this class deletes the matrix
     */
    explicit ManagedTpetraMatrix( Teuchos::RCP<Tpetra::CrsMatrix<ST, LO, GO, NT>> m );

    //! Destructor
    virtual ~ManagedTpetraMatrix() {}

    Tpetra::CrsMatrix<ST, LO, GO, NT> &getTpetra_CrsMatrix();

    /** \brief  Return an Tpetra_CrsMatrix
     * \return An Tpetra_CrsMatrix view of this matrix
     */
    const Tpetra::CrsMatrix<ST, LO, GO, NT> &getTpetra_CrsMatrix() const;

    std::shared_ptr<Matrix> transpose() const override;

    Vector::shared_ptr
    extractDiagonal( Vector::shared_ptr buf = Vector::shared_ptr() ) const override;
    Vector::shared_ptr getRowSums( Vector::shared_ptr ) const override
    {
        AMP_ERROR( "Not implemented" );
    }
    Vector::shared_ptr getRowSumsAbsolute( Vector::shared_ptr ) const override
    {
        AMP_ERROR( "Not implemented" );
    }
    std::shared_ptr<Matrix> clone() const override;
    Vector::shared_ptr createInputVector() const override;
    Vector::shared_ptr createOutputVector() const override;
};


} // namespace AMP::LinearAlgebra

#endif
