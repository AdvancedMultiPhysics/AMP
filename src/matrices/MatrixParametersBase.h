#ifndef included_AMP_MatrixParametersBase
#define included_AMP_MatrixParametersBase

#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/Variable.h"

namespace AMP::LinearAlgebra {


/** \class MatrixParametersBase
 * \brief  A class used to hold basic parameters for a matrix
 */
class MatrixParametersBase
{
public:
    MatrixParametersBase() = delete;

    /** \brief Constructor, variable names set to default
     * \param[in] comm     Communicator for the matrix
     */
    explicit MatrixParametersBase( const AMP_MPI &comm )
        : d_comm( comm ),
          d_VariableLeft( std::make_shared<Variable>( "" ) ),
          d_VariableRight( std::make_shared<Variable>( "" ) )
    {
    }

    /** \brief Constructor, variable names provided
     * \param[in] comm      Communicator for the matrix
     * \param[in] varLeft   pointer to left variable
     * \param[in] varRight  pointer to right variable
     */
    explicit MatrixParametersBase( const AMP_MPI &comm,
                                   std::shared_ptr<Variable> varLeft,
                                   std::shared_ptr<Variable> varRight )
        : d_comm( comm ), d_VariableLeft( varLeft ), d_VariableRight( varRight )
    {
    }

    //! Deconstructor
    virtual ~MatrixParametersBase() = default;

    //!  Get the communicator for the matrix
    AMP::AMP_MPI &getComm() { return d_comm; }

    void setLeftVariable( std::shared_ptr<Variable> var ) { d_VariableLeft = var; }

    void setRightVariable( std::shared_ptr<Variable> var ) { d_VariableRight = var; }

    std::shared_ptr<Variable> getLeftVariable() const { return d_VariableLeft; }

    std::shared_ptr<Variable> getRightVariable() const { return d_VariableRight; }

protected:
    // The comm of the matrix
    AMP_MPI d_comm;

    //!  The variable for the left vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$y\f$ is a left
    //!  vector )
    std::shared_ptr<Variable> d_VariableLeft;

    //!  The variable for the right vector ( For \f$\mathbf{y}^T\mathbf{Ax}\f$, \f$x\f$ is a right
    //!  vector )
    std::shared_ptr<Variable> d_VariableRight;
};
} // namespace AMP::LinearAlgebra

#endif
