//
// $Id: RK2TimeIntegrator.h,v 1.2 2006/02/07 17:36:01 philipb Exp $
// $Revision: 1.2 $
// $Date: 2006/02/07 17:36:01 $
//
// File:  RK2TimeIntegrator.h
// Copyright:  (c) 2005 The Regents of the University of California
// Description:  Concrete time integrator using backward Euler method
//

#ifndef included_RK2TimeIntegrator
#define included_RK2TimeIntegrator

#include <string>

#include "AMP/time_integrators/TimeIntegrator.h"

namespace AMP::TimeIntegrator {

class TimeIntegratorParameters;

/** \class RK2TimeIntegrator
 *
 * Class RK2TimeIntegrator is a concrete time integrator
 * that implements the explicit Runge-Kutta second order (RK2) method
 * also known as Heun's method.
 */
class RK2TimeIntegrator : public AMP::TimeIntegrator::TimeIntegrator
{
public:
    /**
     * Constructor that accepts parameter list.
     */
    explicit RK2TimeIntegrator( std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> );

    /**
     * Destructor.
     */
    ~RK2TimeIntegrator();

    static std::unique_ptr<AMP::TimeIntegrator::TimeIntegrator> createTimeIntegrator(
        std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> parameters )
    {
        return std::unique_ptr<AMP::TimeIntegrator::TimeIntegrator>(
            new RK2TimeIntegrator( parameters ) );
    }

    /**
     * Initialize from parameter list.
     */
    void initialize(
        std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> parameters ) override;

    /**
     * Resets the internal state of the time integrator as needed.
     * A parameter argument is passed to allow for general flexibility
     * in determining what needs to be reset Typically used after a regrid.
     */
    void reset(
        std::shared_ptr<const AMP::TimeIntegrator::TimeIntegratorParameters> parameters ) override;

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

    std::string type() const override { return "RK2"; }

public: // Write/read restart data
    /**
     * \brief    Register any child objects
     * \details  This function will register child objects with the manager
     * \param manager   Restart manager
     */
    void registerChildObjects( AMP::IO::RestartManager *manager ) const override;

    /**
     * \brief    Write restart data to file
     * \details  This function will write the mesh to an HDF5 file
     * \param fid    File identifier to write
     */
    void writeRestart( int64_t fid ) const override;

    /**
     * \brief    Read restart data to file
     * \details  This function will create a variable from the restart file
     * \param fid    File identifier to write
     * \param manager   Restart manager
     */
    RK2TimeIntegrator( int64_t fid, AMP::IO::RestartManager *manager );

private:
    /**
     * Constructor.
     */
    RK2TimeIntegrator() = delete;

    /**
     * setup the vectors used by RK2
     */
    void setupVectors( void );

    std::shared_ptr<AMP::LinearAlgebra::Vector> d_new_solution;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_k1_vec;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_k2_vec;
};
} // namespace AMP::TimeIntegrator

#endif
