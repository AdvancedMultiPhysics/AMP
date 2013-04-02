#ifndef included_AMP_TrilinosNOXSolverParameters
#define included_AMP_TrilinosNOXSolverParameters

#include "boost/shared_ptr.hpp"
#include "utils/Database.h"
#include "utils/AMP_MPI.h"
#include "solvers/SolverStrategyParameters.h"


namespace AMP {
namespace Solver {

/**
 * Class TrilinosNOXSolverParameters provides a uniform mechanism to pass
 * initialization parameters to the PetscSNESSolver solver. It contains
 * shared pointers to a PertscKrylovSolver object and a vector
 * for initial guesses. All member variables are public.
 */
class TrilinosNOXSolverParameters: public SolverStrategyParameters{
public:
    TrilinosNOXSolverParameters(){}
    TrilinosNOXSolverParameters(const boost::shared_ptr<AMP::Database> &db):SolverStrategyParameters(db) {}
    virtual ~TrilinosNOXSolverParameters(){}

    AMP_MPI d_comm;
    AMP::LinearAlgebra::Vector::shared_ptr  d_pInitialGuess;
    AMP::Operator::Operator::shared_ptr  d_pLinearOperator;

protected:
private:
    
};


}
}

#endif
