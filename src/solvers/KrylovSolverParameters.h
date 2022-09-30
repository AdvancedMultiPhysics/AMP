#ifndef included_AMP_KrylovSolverParameters
#define included_AMP_KrylovSolverParameters

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/utils/Database.h"
#include <memory>

namespace AMP::Solver {

/**
 * Class KrylovSolverParameters provides a uniform mechanism to pass
 * initialization parameters to Krylov solvers. It contains
 * shared pointers to a database object and a preconditioner (which could be NULL).
 * All member variables are public.
 */
class KrylovSolverParameters : public SolverStrategyParameters
{
public:
    KrylovSolverParameters() {}
    explicit KrylovSolverParameters( std::shared_ptr<AMP::Database> db );
    virtual ~KrylovSolverParameters() {}

protected:
private:
};
} // namespace AMP::Solver

#endif
