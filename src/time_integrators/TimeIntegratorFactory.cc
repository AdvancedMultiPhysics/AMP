#include "utils/Utilities.h"

/*Design-By-Contract Macros*/
#include "utils/Utilities.h"

#ifndef included_TimeIntegratorFactory
#include "TimeIntegratorFactory.h"
#endif

#ifndef included_ExplicitEuler
#include "ExplicitEuler.h"
#endif

#ifndef included_RK2TimeIntegrator
#include "RK2TimeIntegrator.h"
#endif

#ifndef included_RK23TimeIntegrator
#include "RK23TimeIntegrator.h"
#endif

#ifndef included_RK4TimeIntegrator
#include "RK4TimeIntegrator.h"
#endif

#ifndef included_BackwardEulerTimeIntegrator
#include "BackwardEulerTimeIntegrator.h"
#endif


namespace AMP{
namespace TimeIntegrator{

TimeIntegratorFactory::TimeIntegratorFactory()
{
}

TimeIntegratorFactory::~TimeIntegratorFactory()
{
}

boost::shared_ptr<TimeIntegrator>
TimeIntegratorFactory::createTimeIntegrator( boost::shared_ptr<TimeIntegratorParameters > timeIntegratorParameters)
{
  AMP_ASSERT(timeIntegratorParameters.get()!=NULL);

  boost::shared_ptr<TimeIntegrator> timeIntegrator;
  
  std::string timeIntegratorName = "";
  
  boost::shared_ptr<AMP::Database> db( timeIntegratorParameters->d_db );

   if (db->keyExists("timeIntegrator_name")) 
   {
      timeIntegratorName = db->getString("timeIntegrator_name");
   } 
   else 
   {
      AMP_ERROR("TimeIntegratorFactory"
                 << " -- Required key `timeIntegrator_name'"
                 << " missing in input.");
   }

   if(timeIntegratorName=="ExplicitEuler")
   {
      timeIntegrator.reset(new ExplicitEuler(timeIntegratorParameters));
   }
   else if(timeIntegratorName=="RK2")
   {
      timeIntegrator.reset(new RK2TimeIntegrator(timeIntegratorParameters));
   }
   else if(timeIntegratorName=="RK4")
   {
      timeIntegrator.reset(new RK4TimeIntegrator(timeIntegratorParameters));
   }
   else if(timeIntegratorName=="RK23")
   {
      timeIntegrator.reset(new RK23TimeIntegrator(timeIntegratorParameters));
   }
   else if(timeIntegratorName=="BackwardEuler")
   {
      timeIntegrator.reset(new BackwardEulerTimeIntegrator(timeIntegratorParameters));
   }
   else
   {
      AMP::pout << "ERROR::" << "TimeIntegratorFactory" << " does not currently create timeIntegrator " << timeIntegratorName << std::endl;
      abort();
   }
   
   return timeIntegrator;
}

}
}

