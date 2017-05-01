#ifndef included_AMP_SubchannelPhysicsModel
#define included_AMP_SubchannelPhysicsModel

#include "utils/shared_ptr.h"
#include <cstring>
#include <map>
#include <vector>

#include "materials/Material.h"
#include "operators/ElementPhysicsModel.h"
#include "operators/ElementPhysicsModelParameters.h"


namespace AMP {
namespace Operator {


/**
  This class acts a interface to the water material library for the subchannel
  flow operators.
*/
class SubchannelPhysicsModel : public ElementPhysicsModel
{
public:
    /**
      Constructor
      */
    explicit SubchannelPhysicsModel( const AMP::shared_ptr<ElementPhysicsModelParameters> &params );

    /**
      Destructor
      */
    virtual ~SubchannelPhysicsModel() {}

    /**
      Function to evaluate property functions
      @param [in]  property property identifier string
      @param [out] result   output vector
      @param [in]  args of input argument strings and input vectors
      */
    void getProperty( std::string property,
                      std::vector<double> &result,
                      std::map<std::string, AMP::shared_ptr<std::vector<double>>> &args );

    /**
      Function to return a pointer to the material
      */
    AMP::Materials::Material::shared_ptr getMaterial() { return d_material; }

protected:
    // pointer to the material
    AMP::Materials::Material::shared_ptr d_material;

    // map of property identifier strings and property pointers
    std::map<std::string, AMP::shared_ptr<AMP::Materials::Property<double>>> d_properties;
};
}
}

#endif
