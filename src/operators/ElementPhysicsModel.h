#ifndef included_AMP_ElementPhysicsModel
#define included_AMP_ElementPhysicsModel

#include "AMP/operators/ElementPhysicsModelParameters.h"
#include <memory>


namespace AMP::Operator {

/**
  An abstract base class for representing the physics (material) models in
  the finite element operators.
  */
class ElementPhysicsModel
{
public:
    /**
      Constructor.
      */
    explicit ElementPhysicsModel( std::shared_ptr<const ElementPhysicsModelParameters> params )
    {
        d_iDebugPrintInfoLevel = params->d_db->getWithDefault<int>( "print_info_level", 0 );
    }

    /**
      Destructor.
      */
    virtual ~ElementPhysicsModel() {}

    /**
     * Specify level of diagnostic information printed during iterations.
     * @param [in] print_level zero prints none or minimal information, higher numbers provide
     * increasingly
     *        verbose debugging information.
     */
    virtual void setDebugPrintInfoLevel( int print_level ) { d_iDebugPrintInfoLevel = print_level; }

protected:
    int d_iDebugPrintInfoLevel; /**< Variable that controls the amount of
                                  diagnostic information that gets
                                  printed within this material model. */
};
} // namespace AMP::Operator

#endif
