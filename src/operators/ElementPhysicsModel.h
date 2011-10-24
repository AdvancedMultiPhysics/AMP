
#ifndef included_AMP_ElementPhysicsModel
#define included_AMP_ElementPhysicsModel

#include "ElementPhysicsModelParameters.h"
#include "boost/shared_ptr.hpp"

namespace AMP {
namespace Operator {

  /**
    An abstract base class for representing the physics (material) models in
    the finite element operators.
    */
  class ElementPhysicsModel{
    public :

      /**
        Constructor.
        */
      ElementPhysicsModel(const boost::shared_ptr<ElementPhysicsModelParameters>& params)  {
        d_iDebugPrintInfoLevel = (params->d_db)->getIntegerWithDefault("print_info_level", 0);
      }

      /**
        Destructor.
        */
      virtual ~ElementPhysicsModel() { }

      /**
       * Specify level of diagnostic information printed during iterations.
       * @param [in] print_level zero prints none or minimal information, higher numbers provide increasingly
       *        verbose debugging information.
       */
      virtual void setDebugPrintInfoLevel(int print_level) {
        d_iDebugPrintInfoLevel = print_level;
      }

    protected :

      int d_iDebugPrintInfoLevel; /**< Variable that controls the amount of 
                                    diagnostic information that gets 
                                    printed within this material model. */

  };

}
}

#endif


