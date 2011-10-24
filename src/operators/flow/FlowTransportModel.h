
#ifndef included_AMP_FlowTransportModel
#define included_AMP_FlowTransportModel

#include <cstring>

#include "ElementPhysicsModel.h"

#include "materials/Material.h"

#include "boost/shared_ptr.hpp"

namespace AMP {
namespace Operator {

  typedef ElementPhysicsModelParameters FlowTransportModelParameters;

  class FlowTransportModel : public ElementPhysicsModel
  {
    public :

      FlowTransportModel(const boost::shared_ptr<FlowTransportModelParameters>& params)
        : ElementPhysicsModel(params) { 
          d_useMaterialsLibrary = (params->d_db)->getBoolWithDefault("USE_MATERIALS_LIBRARY",false);

          if(d_useMaterialsLibrary == true) 
          {
            AMP_INSIST( (params->d_db->keyExists("Material")), "Key ''Material'' is missing!" );
            std::string matname = params->d_db->getString("Material");
            d_coolant = AMP::voodoo::Factory<AMP::Materials::Material>::instance().create(matname);
          }
          else{
            d_density = (params->d_db)->getDoubleWithDefault("DENSITY",1000);
            d_fmu     = (params->d_db)->getDoubleWithDefault("VISCOSITY",8.9e-7);
          }

        }


      /**
       * Destructor.
       */ 
      virtual ~FlowTransportModel() { }

      double getDensity(){
        return d_density;
      }

      double getViscosity(){
        return d_fmu;
      }

    protected :

      bool d_useMaterialsLibrary;

      double d_density;

      double d_fmu;

      AMP::Materials::Material::shared_ptr d_coolant; /**< Shared pointer to the materials object. */

    private :

  };

}
}

#endif

