#ifndef included_BackwardEulerTimeIntegrator
#define included_BackwardEulerTimeIntegrator

#include <string>

#ifndef included_ImplicitTimeIntegrator
#include "ImplicitTimeIntegrator.h"
#endif

namespace AMP{
namespace TimeIntegrator{

/** \class BackwardEulerTimeIntegrator
 * 
 * Class BackwardEulerTimeIntegrator is a concrete time integrator
 * that implements the backward Euler method.
 */
class BackwardEulerTimeIntegrator : public ImplicitTimeIntegrator
{
public:
   /**
    * Constructor that accepts parameter list.
    */
   BackwardEulerTimeIntegrator( AMP::shared_ptr< TimeIntegratorParameters > parameters );

   /**
    * Destructor.
    */
   virtual ~BackwardEulerTimeIntegrator();

   /**
    * Initialize from parameter list.
    */
   void initialize( AMP::shared_ptr< TimeIntegratorParameters > parameters );

   /**
   * Resets the internal state of the time integrator as needed.
   * A parameter argument is passed to allow for general flexibility
   * in determining what needs to be reset Typically used after a regrid.
   */
   void reset( AMP::shared_ptr< TimeIntegratorParameters > parameters);

   /**
    * Specify initial time step.
    */
   double getInitialDt();

   /**
    * Specify next time step to use.
    */
   double getNextDt( const bool good_solution );

   /**
   * Set an initial guess for the time advanced solution.
   */
   void setInitialGuess( const bool first_step, const double current_time, const double current_dt, const double old_dt ); 
      
   /**
   * Update state of the solution. 
   */
   void updateSolution( void );

   /**
    * Check time advanced solution to determine whether it is acceptable.
    * Return true if the solution is acceptable; return false otherwise.
    * The integer argument is the return code generated by the call to the
    * nonlinear solver "solve" routine.   The meaning of this value depends
    * on the particular nonlinear solver in use and must be intepreted 
    * properly by the user-supplied solution checking routine.
    */
   bool checkNewSolution(void) const;

 protected:
   
   void initializeTimeOperator(AMP::shared_ptr< TimeIntegratorParameters > parameters);

 private:
   /**
    * Constructor.
    */
   BackwardEulerTimeIntegrator();

   /**
    * Read data from input database.
    */
   void getFromInput( AMP::shared_ptr<AMP::Database> input_db );

   /**
   * setup the vectors used by BE
   */
   void setupVectors(void);

   int d_number_regrid_states;

   double d_initial_dt;

};

}
}

#endif
