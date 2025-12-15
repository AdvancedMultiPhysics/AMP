#ifndef included_AMP_MassDensityModel
#define included_AMP_MassDensityModel

#include "AMP/materials/Material.h"
#include "AMP/operators/ElementPhysicsModel.h"
#include "AMP/utils/ManufacturedSolution.h"
#include <memory>

// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/point.h"
ENABLE_WARNINGS

#include <string>
#include <valarray>


namespace AMP::Operator {


typedef ElementPhysicsModelParameters MassDensityModelParameters;

class MassDensityModel : public ElementPhysicsModel
{
public:
    // This was moved from the AMP namespace due to a conflict with
    // MechanicsConstants.h.  --WKC
    enum class MassEquation { Mechanics, Thermal, Chemical, Manufactured, UnknownMassEquation };

    enum class ManufacturedEquation {
        ThermalSrc,
        FickSrc,
        SoretSrc,
        FickSoretSrc,
        UnknownManufacturedEquation
    };

    explicit MassDensityModel( std::shared_ptr<const MassDensityModelParameters> params );

    virtual ~MassDensityModel() {}

    virtual void getDensityMechanics( std::vector<double> &result,
                                      const std::vector<double> &T,
                                      const std::vector<double> &U,
                                      const std::vector<double> &B );

    virtual void getDensityThermal( std::vector<double> &result,
                                    const std::vector<double> &T,
                                    const std::vector<double> &U,
                                    const std::vector<double> &B );

    virtual void getDensityChemical( std::vector<double> &result,
                                     const std::vector<double> &T,
                                     const std::vector<double> &U,
                                     const std::vector<double> &B );

    virtual void getDensityManufactured( std::vector<double> &result,
                                         const std::vector<double> &T,
                                         const std::vector<double> &U,
                                         const std::vector<double> &B,
                                         const std::vector<libMesh::Point> &xyz );

    // For LinearOperator's Reset/Init

    virtual void preLinearAssembly() {}

    virtual void postLinearAssembly() {}

    virtual void preLinearElementOperation() {}

    virtual void postLinearElementOperation() {}

    virtual void preLinearGaussPointOperation() {}

    virtual void postLinearGaussPointOperation() {}

    std::shared_ptr<ManufacturedSolution> getManufacturedSolution()
    {
        return d_ManufacturedSolution;
    }

    MassEquation getEquation() { return d_equation; }

protected:
    std::shared_ptr<AMP::Materials::Material> d_material;

    /**
     * \brief Use a bilogarithmic scaling of material arguments
     *
     * See documentation for DiffusionTransportModel for background.
     * The Chemical mass matrix is 1 of not scaling.
     */
    bool d_UseBilogScaling;

    /**
     * \brief the material argument to which the bilogarithmic transformation applies
     * These must be one of the values returned by Material::get_arg_names().
     */
    std::string d_BilogVariable;

    MassEquation d_equation;

private:
    std::array<double, 2> d_BilogRange;

    size_t d_BilogIndex;

    std::shared_ptr<ManufacturedSolution> d_ManufacturedSolution;

    ManufacturedEquation d_ManufacturedEquation;

    std::string d_ManufacturedVariable;

    bool d_ManufacturedUseTemp;
    bool d_ManufacturedUseConc;

    std::string d_PropertyName;
    std::vector<double> d_Parameters;
};
} // namespace AMP::Operator

#endif
