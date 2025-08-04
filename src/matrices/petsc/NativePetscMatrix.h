
#ifndef included_AMP_Petsc_Matrix
#define included_AMP_Petsc_Matrix

// AMP files
#include "AMP/matrices/petsc/PetscMatrix.h"

namespace AMP::LinearAlgebra {

/** \class NativePetscMatrix
 * \brief  This is a thin wrapper around PETSc Mat
 * \details  As opposed to ManagedPetscMatrix, this is a
 *    thin wrapper around a PETSc Mat.
 */
class NativePetscMatrix : public Matrix
{
protected:
    void multiply( std::shared_ptr<Matrix> other_op, std::shared_ptr<Matrix> &result ) override;

public:
    NativePetscMatrix();
    NativePetscMatrix( std::shared_ptr<MatrixParametersBase> params );

    /** \brief  Construct a matrix from a PETSc Mat.
     * \param[in] m  The Mat to wrap
     * \param[in] dele  Let this class deallocate the Mat
     */
    explicit NativePetscMatrix( Mat m, bool dele = false );

    explicit NativePetscMatrix( std::shared_ptr<MatrixData> data );

    /** \brief Destructor
     */
    virtual ~NativePetscMatrix();

    //! Return the type of the matrix
    virtual std::string type() const override { return "NativePetscMatrix"; }

    /** \brief Create a NativePetscMatrix with the same non-zero
     * structure
     * \param[in] m  The matrix to duplicate
     * \return A new matrix with the same non-zero structure
     */
    static std::shared_ptr<Matrix> duplicateMat( Mat m );

    void copy( std::shared_ptr<const Matrix> X ) override;

    /** \brief Copy data from a PETSc Mat
     * \param[in] m  The matrix with the data
     */
    void copyFromMat( Mat m );

    std::shared_ptr<Matrix> clone() const override;

    Vector::shared_ptr getInputVector() const override;
    Vector::shared_ptr getOutputVector() const override;
    std::shared_ptr<Discretization::DOFManager> getRightDOFManager() const override;
    std::shared_ptr<Discretization::DOFManager> getLeftDOFManager() const override;

    Vector::shared_ptr
    extractDiagonal( Vector::shared_ptr p = Vector::shared_ptr() ) const override;
    Vector::shared_ptr getRowSums( Vector::shared_ptr ) const override
    {
        AMP_ERROR( "Not implemented" );
    }
    Vector::shared_ptr getRowSumsAbsolute( Vector::shared_ptr ) const override
    {
        AMP_ERROR( "Not implemented" );
    }

private:
};


} // namespace AMP::LinearAlgebra


#endif
