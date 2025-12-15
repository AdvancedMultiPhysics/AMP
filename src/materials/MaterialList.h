// Define the materials in AMP
#ifndef included_AMP_MaterialList
#define included_AMP_MaterialList

#include "AMP/materials/Material.h"


namespace AMP::Materials {


class CylindricallySymmetric : public AMP::Materials::Material
{
public:
    CylindricallySymmetric();
    std::string materialName() const override { return "CylindricallySymmetric"; }
};

class Dr_nonlinear : public Material
{
public:
    Dr_nonlinear();
    std::string materialName() const override { return "Dr_nonlinear"; }
};

class FixedClad : public Material
{
public:
    FixedClad();
    std::string materialName() const override { return "FixedClad"; }
};

class FixedFuel : public Material
{
public:
    FixedFuel();
    std::string materialName() const override { return "FixedFuel"; }
};

class Independent : public Material
{
public:
    Independent();
    std::string materialName() const override { return "Independent"; }
};

class Ox_MSRZC_09 : public Material
{
public:
    Ox_MSRZC_09();
    std::string materialName() const override { return "Ox_MSRZC_09"; }
};

class Steel316_MSRZC_09 : public Material
{
public:
    Steel316_MSRZC_09();
    std::string materialName() const override { return "Steel316_MSRZC_09"; }
};

class UO2_MSRZC_09 : public Material
{
public:
    UO2_MSRZC_09();
    std::string materialName() const override { return "UO2_MSRZC_09"; }
};

class WaterLibrary : public Material
{
public:
    WaterLibrary();
    std::string materialName() const override { return "WaterLibrary"; }
};


} // namespace AMP::Materials

#endif
