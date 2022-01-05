#ifndef included_AMP_OxideTimeIntegrator
#define included_AMP_OxideTimeIntegrator


#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/oxide/OxideTimeIntegratorParameters.h"
#include "AMP/vectors/Vector.h"

#include <string>


namespace AMP::TimeIntegrator {

/*!
  @brief This class solves the time-dependent oxide growth on a surface.
 */
class OxideTimeIntegrator : public TimeIntegrator
{
public:
    /**
     * The constructor for TimeIntegrator initializes the
     * default state of the integrator. Data members are
     * initialized from the input and restart databases.
     *
     * Note that no vectors are created in the constructor.  Vectors are
     * created and initialized in the initialize()
     * member function.
     *
     */
    explicit OxideTimeIntegrator( std::shared_ptr<TimeIntegratorParameters> parameters );

    //! Empty destructor for TimeIntegrator
    virtual ~OxideTimeIntegrator();

    /**
     * Initialize state of time integrator.  This includes creating
     * solution vector and initializing solver components.
     */
    void initialize( std::shared_ptr<TimeIntegratorParameters> parameters ) override;

    /**
     * Resets the internal state of the time integrator as needed.
     * A parameter argument is passed to allow for general flexibility
     * in determining what needs to be reset.
     */
    void reset( std::shared_ptr<const TimeIntegratorParameters> parameters ) override;

    /*!
     * @brief Integrate through the specified time increment in days.
     *
     * Integrate through the specified time increment in days.
     *
     * The boolean first_step argument is true when this is the very
     * first call to the advance function.  Otherwise it is
     * false.  Note that, when the argument is true, the use of extrapolation
     * to construct the initial guess for the advanced solution may not be
     * possible.
     *
     *
     * @param dt Time step size (days)
     * @param first_step Whether this is the first step
     *
     * @return value is the return code generated by the particular solver
     * package in use
     */
    int advanceSolution( const double dt,
                         const bool first_step,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> in,
                         std::shared_ptr<AMP::LinearAlgebra::Vector> out ) override;

    /**
     * Check time advanced solution to determine whether it is acceptable.
     * Return true if the solution is acceptable; return false otherwise.
     * The meaning of this value must be intepreted
     * properly by the user-supplied solution checking routine.
     */
    bool checkNewSolution( void ) override;

    /**
     * Update solution (e.g., reset pointers for solution data, update
     * dependent variables, etc.) after time advance.  It is assumed that
     * when this routine is invoked, an acceptable new solution has been
     * computed.  The double return value is the simulation time corresponding
     * to the advanced solution.
     */
    void updateSolution( void ) override;

    /**
     * Return time increment for next solution advance.  Timestep selection
     * is generally based on whether the solution meets some user-defined
     * criteria.  This routine assumes that, before it is called, the
     * routine checkNewSolution() was called.  The boolean argument is the
     * return value from that call.
     */
    double getNextDt( const bool good_solution ) override;

private:
    // declare the default constructor to be private
    OxideTimeIntegrator() = delete;

    // The mesh over which we define the oxide
    AMP::Mesh::Mesh::shared_ptr d_mesh;

    // Some internal vectors
    AMP::LinearAlgebra::Vector::shared_ptr d_oxide; // Oxide depth of each point (m)
    AMP::LinearAlgebra::Vector::shared_ptr d_alpha; // Alpha depth of each point (m)
    AMP::LinearAlgebra::Vector::shared_ptr d_temp;  // Temperature of each point (K)

    // Internal data for calculating the oxide
    std::vector<int> N_layer;
    AMP::LinearAlgebra::Vector::shared_ptr depth; // Depth of each layer (cm)
    AMP::LinearAlgebra::Vector::shared_ptr conc;  // Oxygen concentration of each layer (g/cm^3)
};
} // namespace AMP::TimeIntegrator

#endif
