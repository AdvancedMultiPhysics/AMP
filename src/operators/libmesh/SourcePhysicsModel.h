#ifndef included_AMP_SourcePhysicsModel
#define included_AMP_SourcePhysicsModel

#include <cstring>
#include <memory>

#include "AMP/materials/Material.h"
#include "AMP/operators/ElementPhysicsModel.h"
#include "AMP/operators/ElementPhysicsModelParameters.h"
#include "AMP/utils/UtilityMacros.h"


// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/point.h"
ENABLE_WARNINGS


namespace AMP::Operator {


typedef ElementPhysicsModelParameters SourcePhysicsModelParameters;


/**
  This class acts a interface to various ElementPhysicsModel and also
  for source fuction evaluation for the VolumeIntegral Operator. The
  Element Operation of this operator passes the input vectors and coordinates
  through the getConstitutiveProperty function for calculating the source terms.
*/
class SourcePhysicsModel : public ElementPhysicsModel
{
public:
    /**
     * This Constructor reads the value for the key USE_MATERIALS_LIBRARY (false by default)
     * and also USE_ELEMENT_PHYSICS from the database. These specify if material library or the
     * element physics model are used for calculating the source terms.
     */
    explicit SourcePhysicsModel( std::shared_ptr<const SourcePhysicsModelParameters> params );

    /**
     * Destructor.
     */
    virtual ~SourcePhysicsModel() {}

    void getConstitutiveProperty( std::vector<double> &result,
                                  const std::vector<std::vector<double>> &InputVec,
                                  const std::vector<std::vector<double>> &,
                                  const std::vector<libMesh::Point> &Coordinates );

protected:
    bool d_useMaterialsLibrary;

    std::string d_physicsName;
    double d_constantProperty; // Constant value of property if the material is not used.

    std::shared_ptr<AMP::Materials::Material> d_material;

    std::shared_ptr<AMP::Materials::Property> d_property;

private:
    double d_DefaultTemperature;
    double d_DefaultConcentration;
    double d_DefaultBurnup;

    std::vector<double> d_defaults;

    std::shared_ptr<ElementPhysicsModel> d_elementPhysicsModel;
    std::shared_ptr<ElementPhysicsModelParameters> d_elementPhysicsParams;

    // Cached variables that may or may not be used to improve perfomance
    std::map<std::string, std::shared_ptr<std::vector<double>>> d_inputMaterialParameters;
};
} // namespace AMP::Operator

#endif
