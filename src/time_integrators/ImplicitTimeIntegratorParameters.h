#ifndef included_AMP_ImplicitTimeIntegratorParameters
#define included_AMP_ImplicitTimeIntegratorParameters

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/utils/Database.h"
#include "TimeIntegratorParameters.h"
#include <memory>


namespace AMP {
namespace TimeIntegrator {

/*!
  @brief Parameter class for implicit time integrators

  Class ImplicitTimeIntegratorParameters contains the parameters to
  initialize an implicit time integrator class. It contains a Database
  object and a pointer to a SolverStrategy object.

  @param d_solver pointer to SolverStrategy

  @see SolverStrategy
*/
class ImplicitTimeIntegratorParameters : public TimeIntegratorParameters
{
public:
    explicit ImplicitTimeIntegratorParameters( std::shared_ptr<AMP::Database> db );

    virtual ~ImplicitTimeIntegratorParameters();

    /**
     * Pointers to implicit equation and solver strategy objects
     * The strategies provide nonlinear equation and solver
     * routines for treating the nonlinear problem on the hierarchy.
     */
    std::shared_ptr<AMP::Solver::SolverStrategy> d_solver;

protected:
private:
    // not implemented
    ImplicitTimeIntegratorParameters();
    explicit ImplicitTimeIntegratorParameters( const ImplicitTimeIntegratorParameters & );
    void operator=( const ImplicitTimeIntegratorParameters & );
};
} // namespace TimeIntegrator
} // namespace AMP

#endif
