#ifndef included_AMP_MatrixParametersBase
#define included_AMP_MatrixParametersBase

#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"
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
          d_VariableRight( std::make_shared<Variable>( "" ) ),
          d_backend( AMP::Utilities::Backend::Serial )
    {
    }

    /** \brief Constructor, variable names set to default
     * \param[in] comm     Communicator for the matrix
     * \param[in] backend  Acceleration backend for matrix operations
     */
    explicit MatrixParametersBase( const AMP_MPI &comm, AMP::Utilities::Backend backend )
        : d_comm( comm ),
          d_VariableLeft( std::make_shared<Variable>( "" ) ),
          d_VariableRight( std::make_shared<Variable>( "" ) ),
          d_backend( backend )
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
        : d_comm( comm ),
          d_VariableLeft( varLeft ),
          d_VariableRight( varRight ),
          d_backend( AMP::Utilities::Backend::Serial )
    {
    }

    /** \brief Constructor, variable names provided
     * \param[in] comm      Communicator for the matrix
     * \param[in] varLeft   pointer to left variable
     * \param[in] varRight  pointer to right variable
     * \param[in] backend   Acceleration backend for matrix operations
     */
    explicit MatrixParametersBase( const AMP_MPI &comm,
                                   std::shared_ptr<Variable> varLeft,
                                   std::shared_ptr<Variable> varRight,
                                   AMP::Utilities::Backend backend )
        : d_comm( comm ),
          d_VariableLeft( varLeft ),
          d_VariableRight( varRight ),
          d_backend( backend )
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

public:
    // The backend used for cpus and/or gpu acceleration
    AMP::Utilities::Backend d_backend;
};
} // namespace AMP::LinearAlgebra

#endif
