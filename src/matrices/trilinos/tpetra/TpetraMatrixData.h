#ifndef included_AMP_TpetraMatrixData
#define included_AMP_TpetraMatrixData

#include "AMP/matrices/data/MatrixData.h"
#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.h"

#include <Tpetra_CrsMatrix_decl.hpp>

namespace AMP::LinearAlgebra {

class Vector;

/** \class TpetraMatrixData
  * \brief A Matrix with an Tpetra_CrsMatrix interface
  * \details  An TpetraMatrixData presents an Tpetra_Matrix class.
  * Given an AMP::LinearAlgebra::Matrix, this class can create an Tpetra view
  * without copying the data.  As such, this class serves three
  * purposes:
  *  -# Provides an Tpetra_CrsMatrix for derived classes to use, fill, manage, etc.
  *  -# Provides an interface for accessing this Tpetra_CrsMatrix independent of base or derived
  classes
  *  -# Provides a static method for creating an Tpetra_CrsMatrix view of an AMP matrix.
  */

template<typename ST = Tpetra_ST,
         typename LO = Tpetra_LO,
         typename GO = Tpetra_GO,
         typename NT = Tpetra::Vector<>::node_type>
class TpetraMatrixData : public MatrixData
{
private:
    TpetraMatrixData() = delete;

protected:
    /** \brief Bare pointer to an Tpetra_CrsMatrix
     */
    Teuchos::RCP<Tpetra::CrsMatrix<ST, LO, GO, NT>> d_tpetraMatrix;

    /** \brief Range map for the Tpetra_CrsMatrix
     */
    Teuchos::RCP<Tpetra::Map<LO, GO, NT>> d_RangeMap;

    /** \brief Domain map for the Tpetra_CrsMatrix
     */
    Teuchos::RCP<Tpetra::Map<LO, GO, NT>> d_DomainMap;

    //!  \f$A_{i,j}\f$ storage of off-core data
    std::map<int, std::map<GO, ST>> d_OtherData;

    //!  Update data off-core
    void setOtherData();


    /** \brief Ensure Tpetra methods return correctly
     * \param[in] err  The return value from the method
     * \param[in] func  The name of the Tpetra method called
     * \details  Throws an execption if err != 0
     */
    void VerifyTpetraReturn( int err, const char *func ) const;

public:
    explicit TpetraMatrixData( std::shared_ptr<MatrixParametersBase> params );

    TpetraMatrixData( const TpetraMatrixData<ST, LO, GO, NT> &rhs );

    /** \brief Constructor
     * \param[in] inMatrix  Matrix to wrap
     * \param[in] dele  If true, then this class will delete the Tpetra_CrsMatrix
     */
    explicit TpetraMatrixData( Teuchos::RCP<Tpetra::CrsMatrix<ST, LO, GO, NT>> inMatrix );

    std::shared_ptr<MatrixData> cloneMatrixData() const override;

    std::shared_ptr<MatrixData> transpose() const override;

    void removeRange( AMP::Scalar, AMP::Scalar ) override { AMP_ERROR( "Not implemented" ); }

    /** \brief Change the TpetraMaps for the matrix
     * \param[in] range  A vector that represents the range: y in y = A*x (row map)
     * \param[in] domain  A vector that represents the domain: x in y = A*x (column map)
     * \details  This does not change the matrix, just the maps stored above
     *
     */
    void setTpetraMaps( std::shared_ptr<Vector> range, std::shared_ptr<Vector> domain );

    TpetraMatrixData<ST, LO, GO, NT> &
    operator=( const TpetraMatrixData<ST, LO, GO, NT> & ) = delete;

    /** \brief Destructor
     */
    virtual ~TpetraMatrixData();

    //! Return the type of the matrix
    std::string type() const override { return "TpetraMatrixData"; }

    /** \brief  Return an Tpetra_CrsMatrix
     * \return An Tpetra_CrsMatrix view of this matrix
     */
    virtual Tpetra::CrsMatrix<ST, LO, GO, NT> &getTpetra_CrsMatrix();

    /** \brief  Return an Tpetra_CrsMatrix
     * \return An Tpetra_CrsMatrix view of this matrix
     */
    virtual const Tpetra::CrsMatrix<ST, LO, GO, NT> &getTpetra_CrsMatrix() const;

    /** \brief  Create an TpetraMatrixData view of an AMP::LinearAlgebra::Matrix
     * \param[in] p  The matrix to view
     * \return  An AMP:Matrix capable of casting to TpetraMatrixData
     */
    static std::shared_ptr<TpetraMatrixData<ST, LO, GO, NT>>
    createView( std::shared_ptr<MatrixData> p );

    /** \brief  A call-through to Tpetra_CrsMatrix fillComplete
     */
    void fillComplete();

    void createValuesByGlobalID( size_t, const std::vector<size_t> & );
    void addValuesByGlobalID( size_t num_rows,
                              size_t num_cols,
                              size_t *rows,
                              size_t *cols,
                              void *values,
                              const typeID &id ) override;
    void setValuesByGlobalID( size_t num_rows,
                              size_t num_cols,
                              size_t *rows,
                              size_t *cols,
                              void *values,
                              const typeID &id ) override;
    void getValuesByGlobalID( size_t num_rows,
                              size_t num_cols,
                              size_t *rows,
                              size_t *cols,
                              void *values,
                              const typeID &id ) const override;
    void getRowByGlobalID( size_t row,
                           std::vector<size_t> &cols,
                           std::vector<double> &values ) const override;
    /** \brief  Given a row, retrieve the non-zero column indices of the matrix in compressed format
     * \param[in]  row Which row
     */
    std::vector<size_t> getColumnIDs( size_t row ) const override;
    void makeConsistent( AMP::LinearAlgebra::ScatterType t ) override;
    size_t numLocalRows() const override;
    size_t numGlobalRows() const override;
    size_t numLocalColumns() const override;
    size_t numGlobalColumns() const override;
    AMP::AMP_MPI getComm() const override;
    std::shared_ptr<Discretization::DOFManager> getRightDOFManager() const override;
    std::shared_ptr<Discretization::DOFManager> getLeftDOFManager() const override;
    std::shared_ptr<Vector> createInputVector() const;
    std::shared_ptr<Vector> createOutputVector() const;

    /** \brief Return the typeid of the matrix coeffs
     */
    typeID getCoeffType() const override
    {
        // Tpetra matrix are double only for the moment
        constexpr auto type = getTypeID<ST>();
        return type;
    }
};


} // namespace AMP::LinearAlgebra


#endif
