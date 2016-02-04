#ifndef CylindricallySymmetric_H
#define CylindricallySymmetric_H

#include "materials/Material.h"


// Define the material
namespace AMP {
namespace Materials {
class CylindricallySymmetric : public AMP::Materials::Material
{
public:
    CylindricallySymmetric();
};
}
}


// Add static initialize to force symbols to be included
// It will register the material with the factory
static struct CylindricallySymmetric_INIT {
    CylindricallySymmetric_INIT()
    {
        static AMP::voodoo::Registration<AMP::Materials::Material,
                                         AMP::Materials::CylindricallySymmetric>
            reg( "CylindricallySymmetric" );
    }
} CylindricallySymmetric_init;


#endif
