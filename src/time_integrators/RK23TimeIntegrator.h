#ifndef included_AMP_RK23TimeIntegrator
#define included_AMP_RK23TimeIntegrator

#include <string>

#ifndef included_AMP_TimeIntegrator
#include "TimeIntegrator.h"
#endif

#ifndef included_AMP_Vector
#include "AMP/vectors/Vector.h"
#endif

namespace AMP {
namespace TimeIntegrator {

/** \class RK23TimeIntegrator
 *
 * Class RK23TimeIntegrator is a concrete time integrator
 * that implements the explicit Bogacki-Shampine adaptive Runge-Kutta (Matlab ode23) method.
 */
class RK23TimeIntegrator : public TimeIntegrator
{
public:
    /**
     * Constructor that accepts parameter list.
     */
    explicit RK23TimeIntegrator( std::shared_ptr<TimeIntegratorParameters> parameters );

    /**
     * Destructor.
     */
    virtual ~RK23TimeIntegrator();

    /**
     * Initialize from parameter list.
     */
    void initialize( std::shared_ptr<TimeIntegratorParameters> parameters ) override;

    /**
     * Resets the internal state of the time integrator as needed.
     * A parameter argument is passed to allow for general flexibility
     * in determining what needs to be reset Typically used after a regrid.
     */
    void reset( std::shared_ptr<const TimeIntegratorParameters> parameters ) override;

    /**
     * Specify next time step to use.
     */
    double getNextDt( const bool good_solution ) override;

    /**
     * Determine whether time advanced solution is satisfactory.
     */
    bool checkNewSolution( void ) override;

    /**
     * Update state of the solution.
     */
    void updateSolution( void ) override;

    int advanceSolution( const double dt,
                         const bool first_step,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> in,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> out ) override;

private:
    /**
     * Constructor.
     */
    RK23TimeIntegrator();

    /**
     * Read data from input database.
     */
    void getFromInput( std::shared_ptr<AMP::Database> input_db );

    /**
     * setup the vectors used by RK23
     */
    void setupVectors( void );

    double d_safety_factor;
    double d_atol;

    std::shared_ptr<AMP::LinearAlgebra::Vector> d_new_solution;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_k1_vec;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_k2_vec;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_k3_vec;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_k4_vec;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_z_vec;
};
} // namespace TimeIntegrator
} // namespace AMP

#endif
