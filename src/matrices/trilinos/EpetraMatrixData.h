#ifndef included_AMP_EpetraMatrixData
#define included_AMP_EpetraMatrixData

#include "AMP/matrices/data/MatrixData.h"


// Forward declare
class Epetra_Map;
class Epetra_CrsMatrix;

namespace AMP::LinearAlgebra {

class Vector;

/** \class EpetraMatrixData
  * \brief A Matrix with an Epetra_CrsMatrix interface
  * \details  An EpetraMatrixData presents an Epetra_Matrix class.
  * Given an AMP::LinearAlgebra::Matrix, this class can create an Epetra view
  * without copying the data.  As such, this class serves three
  * purposes:
  *  -# Provides an Epetra_CrsMatrix for derived classes to use, fill, manage, etc.
  *  -# Provides an interface for accessing this Epetra_CrsMatrix independent of base or derived
  classes
  *  -# Provides a static method for creating an Epetra_CrsMatrix view of an AMP matrix.
  */

class EpetraMatrixData : public MatrixData
{
private:
    EpetraMatrixData() = delete;

protected:
    /** \brief Bare pointer to an Epetra_CrsMatrix
     */
    Epetra_CrsMatrix *d_epetraMatrix;

    /** \brief Range map for the Epetra_CrsMatrix
     */
    std::shared_ptr<Epetra_Map> d_RangeMap;

    /** \brief Domain map for the Epetra_CrsMatrix
     */
    std::shared_ptr<Epetra_Map> d_DomainMap;

    /** \brief Indicates if the destructor calls delete
     */
    bool d_DeleteMatrix;

    //!  \f$A_{i,j}\f$ storage of off-core data
    std::map<int, std::map<int, double>> d_OtherData;

    //!  Update data off-core
    void setOtherData();


    /** \brief Ensure Epetra methods return correctly
     * \param[in] err  The return value from the method
     * \param[in] func  The name of the Epetra method called
     * \details  Throws an execption if err != 0
     */
    void VerifyEpetraReturn( int err, const char *func ) const;

public:
    explicit EpetraMatrixData( std::shared_ptr<MatrixParametersBase> params );

    EpetraMatrixData( const EpetraMatrixData &rhs );

    /** \brief Constructor
     * \param[in] inMatrix  Matrix to wrap
     * \param[in] dele  If true, then this class will delete the Epetra_CrsMatrix
     */
    explicit EpetraMatrixData( Epetra_CrsMatrix *inMatrix, bool dele = false );

    std::shared_ptr<MatrixData> cloneMatrixData() const override;

    std::shared_ptr<MatrixData> transpose() const override;

    /** \brief Change the EpetraMaps for the matrix
     * \param[in] range  A vector that represents the range: y in y = A*x (row map)
     * \param[in] domain  A vector that represents the domain: x in y = A*x (column map)
     * \details  This does not change the matrix, just the maps stored above
     *
     */
    void setEpetraMaps( std::shared_ptr<Vector> range, std::shared_ptr<Vector> domain );

    EpetraMatrixData &operator=( const EpetraMatrixData & ) = delete;

    /** \brief Destructor
     */
    virtual ~EpetraMatrixData();

    //! Return the type of the matrix
    std::string type() const override { return "EpetraMatrixData"; }

    /** \brief  Return an Epetra_CrsMatrix
     * \return An Epetra_CrsMatrix view of this matrix
     */
    virtual Epetra_CrsMatrix &getEpetra_CrsMatrix();

    /** \brief  Return an Epetra_CrsMatrix
     * \return An Epetra_CrsMatrix view of this matrix
     */
    virtual const Epetra_CrsMatrix &getEpetra_CrsMatrix() const;

    /** \brief  Create an EpetraMatrixData view of an AMP::LinearAlgebra::Matrix
     * \param[in] p  The matrix to view
     * \return  An AMP:Matrix capable of casting to EpetraMatrixData
     */
    static std::shared_ptr<EpetraMatrixData> createView( std::shared_ptr<MatrixData> p );

    /** \brief  A call-through to Epetra_CrsMatrix fillComplete
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
    std::shared_ptr<Vector> getRightVector() const;
    std::shared_ptr<Vector> getLeftVector() const;

    /** \brief Return the typeid of the matrix coeffs
     */
    typeID getCoeffType() const override
    {
        // Epetra matrix are double only for the moment
        constexpr auto type = getTypeID<double>();
        return type;
    }
};


} // namespace AMP::LinearAlgebra


#endif
