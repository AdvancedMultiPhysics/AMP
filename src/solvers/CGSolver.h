#ifndef included_AMP_CGSolver
#define included_AMP_CGSolver

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/utils/AMP_MPI.h"

#include <string>

namespace AMP::Solver {

/**
 * The CGSolver class implements the Conjugate Gradient method namely the 2-term recurrence variant.
 * M.R. Hestenes, E. Stiefel. "Methods of conjugate gradients for solving linear systems"
 * J. Res. Natl. Bur. Stand., 49 (1952), pp. 409-436
 *
 * In addition it implements the IPCG variant  developed in
 * Golub, Gene H.; Ye, Qiang (1999). "Inexact Preconditioned Conjugate Gradient Method with
 * Inner-Outer Iteration". SIAM Journal on Scientific Computing 21 (4): 1305.
 * doi:10.1137/S1064827597323415 (http://dx.doi.org/10.1137%2FS1064827597323415) .
 *
 * and in addition the FCG method developed in
 * Axelsson, O.; Vassilevski, P.S. "Variable‐step multilevel preconditioning methods, I:
 * Self‐adjoint and positive definite elliptic problems." Numer. Linear Algebra Appl. 1994, 1,
 * 75–101, https://doi.org/10.1002/nla.1680010108.
 *
 * Note that the FCG implementation orthogonalizes against the last d_max_dimension search direction
 * vectors as described in
 * Notay, Y. "Flexible Conjugate Gradients", SIAM J. Sc. Comput 22(4), 2000
 * https://doi.org/10.1137/S1064827599362314
 *
 * By specifying the input string variant = "pcg", "ipcg" or "fcg" a user can switch between
 * the methods
 */

template<typename T = double>
class CGSolver : public SolverStrategy
{
public:
    /**
     * default constructor
     */
    CGSolver() = default;

    /**
     * main constructor
     @param [in] params The parameters object
     contains a database objects containing the following fields:

     1. type: double, name : relative_tolerance, default value of $1.0e-9$, relative tolerance for
    CG solver
    acceptable values (non-negative real values)

     2. type: bool, name : uses_preconditioner, default value false
        acceptable values (false, true),
        side effect: if false sets string pc_type to "none"
     */
    explicit CGSolver( std::shared_ptr<SolverStrategyParameters> params );

    /**
     * static create routine that is used by SolverFactory
     @param [in] params The parameters object
     contains a database objects with the fields listed for the constructor above
     */
    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return std::make_unique<CGSolver<T>>( params );
    }

    /**
     * Default destructor
     */
    virtual ~CGSolver() = default;

    std::string type() const override { return "CGSolver"; }

    /**
     * Solve the system \f$Au = 0\f$.
     * @param [in] f : shared pointer to right hand side vector
     * @param [out] u : shared pointer to approximate computed solution
     */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

    /**
     * Initialize the CGSolver. Should not be necessary for the user to call in general.
     * @param params
     */
    void initialize( std::shared_ptr<const SolverStrategyParameters> params ) override;

    /**
     * Register the operator that the solver will use during solves
     * @param [in] op shared pointer to operator $A()$ for equation \f$A(u) = f\f$
     */
    void registerOperator( std::shared_ptr<AMP::Operator::Operator> op ) override;

protected:
    void getFromInput( std::shared_ptr<AMP::Database> db );
    void allocateScratchVectors( std::shared_ptr<const AMP::LinearAlgebra::Vector> u );

private:
    T d_dDivergenceTolerance = 1e3;

    std::vector<T> d_gamma;

    bool d_bUsesPreconditioner = false;

    //! use flexible CG if true
    bool d_bFlexibleCG = false;

    //! maximum dimension of the stored search space for FCG
    int d_max_dimension = 0;

    //! variant being used, can be one of "pcg", "ipcg", or "fcg"
    std::string d_sVariant = "pcg";

    //! scratch vectors required for PCG
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_r;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_p;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_w;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_z;

    //! stores the search directions for IPCG/FCG if needed
    //! we do not preallocate by default
    std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>> d_vDirs;
};
} // namespace AMP::Solver

#endif
