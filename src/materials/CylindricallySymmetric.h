// Define the materials in AMP
#ifndef included_AMP_Materials_CylindricallySymmetric
#define included_AMP_Materials_CylindricallySymmetric

#include "AMP/materials/Property.h"


namespace AMP::Materials {


/** full cylindrically symmetric tensor diffusion coefficient
 *
 * The parameters are set as follows:
 * params[0] = number of parameters for radial
 * params[1]...params[ params[0] ] = parameters for radial
 * the rest are for the longitudinal
 * AuxiliaryInteger data "derivative" has values 0, 1, 2 for
 * zeroth, r- and z- derivatives, respectively.
 */
class CylindricallySymmetricTensor : public Property
{
public:
    CylindricallySymmetricTensor( const std::string &name,
                                  std::vector<double> params = { 1, 1, 1 } );

    void eval( AMP::Array<double> &result, const AMP::Array<double> &args ) const override;

private:
    std::vector<double> d_paramsRadial;
    std::vector<double> d_paramsLongitudinal;
};


} // namespace AMP::Materials

#endif
