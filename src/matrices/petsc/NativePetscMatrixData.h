#ifndef included_AMP_Petsc_MatrixData
#define included_AMP_Petsc_MatrixData

// AMP files
#include "AMP/matrices/data/MatrixData.h"
#include "petscmat.h"

namespace AMP::LinearAlgebra {

/** \class NativePetscMatrixData
 * \brief  This is a thin wrapper around PETSc Mat
 * \details  As opposed to ManagedPetscMatrixData, this is a
 *    thin wrapper around a PETSc Mat.
 */
class NativePetscMatrixData : public MatrixData
{
protected:
    /** \brief Unused default constructor
     */
    NativePetscMatrixData() = delete;

public:
    /** \brief  Construct a matrix from a PETSc Mat.
     * \param[in] m  The Mat to wrap
     * \param[in] dele  Let this class deallocate the Mat
     */
    explicit NativePetscMatrixData( Mat m, bool dele = false );

    /** \brief Destructor
     */
    virtual ~NativePetscMatrixData();

    //! Return the type of the matrix
    virtual std::string type() const override { return "NativePetscMatrixData"; }

    /** \brief Create a NativePetscMatrixData with the same non-zero
     * structure
     * \param[in] m  The matrix to duplicate
     * \return A new matrix with the same non-zero structure
     */
    static std::shared_ptr<MatrixData> duplicateMat( Mat m );

    /** \brief Copy data from a PETSc Mat
     * \param[in] m  The matrix with the data
     */
    void copyFromMat( Mat m );

    std::shared_ptr<MatrixData> cloneMatrixData() const override;

    Vector::shared_ptr getRightVector() const;
    Vector::shared_ptr getLeftVector() const;
    Discretization::DOFManager::shared_ptr getRightDOFManager() const override;
    Discretization::DOFManager::shared_ptr getLeftDOFManager() const override;

    size_t numGlobalRows() const override;
    size_t numGlobalColumns() const override;

    void addValuesByGlobalID(
        size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, double *values ) override;
    void setValuesByGlobalID(
        size_t num_rows, size_t num_cols, size_t *rows, size_t *cols, double *values ) override;
    void getValuesByGlobalID( size_t num_rows,
                              size_t num_cols,
                              size_t *rows,
                              size_t *cols,
                              double *values ) const override;
    void getRowByGlobalID( size_t row,
                           std::vector<size_t> &cols,
                           std::vector<double> &values ) const override;

    std::vector<size_t> getColumnIDs( size_t row ) const override;

    void makeConsistent() override;

    Mat getMat() { return d_Mat; }

    AMP_MPI getComm() const override;

private:
    Mat d_Mat;
    bool d_MatCreatedInternally;
};


} // namespace AMP::LinearAlgebra


#endif
