#ifndef included_AMP_GMRESRSolver
#define included_AMP_GMRESRSolver

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Array.h"

#include <string>

namespace AMP::Solver {

/**
 * Implements the the family of nested GMRES methods for non-symmetric linear systems introduced in
 * H.A. van der Vorst and C. Vuik,
 * GMRESR: a family of nested GMRES methods,
 * Report 91-80, Faculty of Technical Mathematics and Informatics, Delft University of Technology,
 * or Num. Lin. Alg. Appl., vol. 1(4), 369--386 (1994)
 *
 * It will in future incorporate improvements proposed in
 * Kees Vuik
 * New insights in GMRES-like methods with variable preconditioners
 * Report 93-10, Faculty of Technical Mathematics and Informatics, Delft University of Technology
 * The algorithm as implemented presently uses a truncation strategy for the outer iteration.
 * It also includes a variant for GCR since this is trivial to implement
 */

template<typename T = double>
class GMRESRSolver : public SolverStrategy
{
public:
    /**
     * default constructor
     */
    GMRESRSolver();

    /**
     * main constructor
     @param [in] params The parameters object
     contains a database objects containing the following fields:

     1. type: double, name : relative_tolerance, default value of $1.0e-9$, relative tolerance for
    GMRESR solver
    acceptable values (non-negative real values)
     2. type: int, name : max_dimension, maximum size of space before truncation
     */
    explicit GMRESRSolver( std::shared_ptr<SolverStrategyParameters> params );

    /**
     * static create routine that is used by SolverFactory
     @param [in] params The parameters object
     contains a database objects with the fields listed for the constructor above
     */
    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return std::make_unique<GMRESRSolver<T>>( params );
    }

    /**
     * Default destructor
     */
    virtual ~GMRESRSolver() = default;

    std::string type() const override { return "GMRESRSolver"; }

    /**
     * Solve the system Au=f.
     * @param [in] f : const shared pointer to right hand side vector
     * @param [out] u : shared pointer to approximate computed solution
     */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                AMP::LinearAlgebra::Vector::shared_ptr u ) override;

    /**
     * Initialize the GMRESRSolver. Should not be necessary for the user to call in general.
     * @param params
     */
    void initialize( std::shared_ptr<const SolverStrategyParameters> params ) override;

protected:
    void getFromInput( std::shared_ptr<AMP::Database> db );

private:
    bool d_bRestart = false; //! whether to restart

    int d_iMaxKrylovDimension = 50; //! maximum dimension of the Krylov subspace before a restart or
                                    //! termination happens

    int d_restarts; //! logs number of times the solver is restarted

    //! string to specify variant - "gcr", "gmresr"
    std::string d_variant;

    //! stores the orthonormal basis for the Krylov space
    //! we do not preallocate by default
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> d_c;

    //! stores the orthonormal basis for the Krylov space in case of FGMRESR
    //! we do not preallocate by default
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> d_u;
};
} // namespace AMP::Solver

#endif
