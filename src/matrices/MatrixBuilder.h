#ifndef included_AMP_MatrixBuider
#define included_AMP_MatrixBuider

#include "AMP/matrices/Matrix.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"

#include <functional>
#include <string>


extern "C" {
typedef struct _p_Mat *Mat;
}


namespace AMP::LinearAlgebra {


//! Check if we have a sparse matrix availible
bool haveSparseMatrix();


//! Return the default matrix type
std::string getDefaultMatrixType();


/**
 * \brief  This function will create a matrix from two vectors
 * \details  This function is responsible for creating matrices given a left and a right vector
 * \param right     Vector that will be used to create the matrix
 *                  The right is x in the expression y = A*x.
 * \param left      Vector that will be used to create the matrix.
 *                  The left is y in the expression y = A*x.
 * \param type      Type of matrix to build:
 *                      auto: Automatically determined based on build (default)
 *                      ManagedPetscMatrix
 *                      ManagedEpetraMatrix
 *                      DenseSerialMatrix
 * \param getColumnIDs Function to provide the column indices given the row index.
 *                      If not provided, with will default to calling the getRowDOFs function on the
 *                      DOFManager associated with the left vector.
 */
std::shared_ptr<Matrix>
createMatrix( AMP::LinearAlgebra::Vector::shared_ptr right,
              AMP::LinearAlgebra::Vector::shared_ptr left,
              std::string type = "auto",
              std::function<std::vector<size_t>( size_t row )> getColumnIDs =
                  std::function<std::vector<size_t>( size_t )>() );

/**
 * \brief  This function will create a matrix from two vectors
 * \details  This function is responsible for creating matrices given a left and a right vector
 * \param right     Vector that will be used to create the matrix
 *                  The right is x in the expression y = A*x.
 * \param left      Vector that will be used to create the matrix.
 *                  The left is y in the expression y = A*x.
 * \param accelerationBackend Backend used for matrix operations.
 * \param type      Type of matrix to build:
 *                      auto: Automatically determined based on build (default)
 *                      ManagedPetscMatrix
 *                      ManagedEpetraMatrix
 *                      DenseSerialMatrix
 * \param getColumnIDs Function to provide the column indices given the row index.
 *                      If not provided, with will default to calling the getRowDOFs function on the
 *                      DOFManager associated with the left vector.
 */
std::shared_ptr<Matrix>
createMatrix( AMP::LinearAlgebra::Vector::shared_ptr right,
              AMP::LinearAlgebra::Vector::shared_ptr left,
              AMP::Utilities::Backend accelerationBackend,
              std::string type = "auto",
              std::function<std::vector<size_t>( size_t row )> getColumnIDs =
                  std::function<std::vector<size_t>( size_t )>() );


#if defined( AMP_USE_PETSC )
/**
 * \brief  Create a matrix from an arbitrary PETSc Mat
 * \details  This function creates a matrix from an arbitrary PETSc Mat
 * \param[in] M             PETSc Mat
 * \param[in] deleteable    If true, ~Matrix() will call MatDestroy()
 */
std::shared_ptr<Matrix> createMatrix( Mat M, bool deleteable );
#endif


} // namespace AMP::LinearAlgebra

#endif
