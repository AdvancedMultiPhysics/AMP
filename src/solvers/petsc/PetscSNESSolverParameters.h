#ifndef included_AMP_PetscSNESSolverParameters
#define included_AMP_PetscSNESSolverParameters

#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include <memory>


namespace AMP::Solver {

/**
 * Class PetscSNESSolverParameters provides a uniform mechanism to pass
 * initialization parameters to the PetscSNESSolver solver. It contains
 * shared pointers to a PertscKrylovSolver object and a vector
 * for initial guesses. All member variables are public.
 */
class PetscSNESSolverParameters : public SolverStrategyParameters
{
public:
    PetscSNESSolverParameters() {}
    explicit PetscSNESSolverParameters( std::shared_ptr<AMP::Database> db );
    virtual ~PetscSNESSolverParameters() {}

    AMP_MPI d_comm;

    std::shared_ptr<PetscKrylovSolver> d_pKrylovSolver;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pInitialGuess;

protected:
private:
};


} // namespace AMP::Solver

#endif
