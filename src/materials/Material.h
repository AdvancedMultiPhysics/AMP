#ifndef included_AMP_Material
#define included_AMP_Material

#include "AMP/materials/Property.h"
#include "AMP/utils/Factory.h"
#include "AMP/utils/Utilities.h"

#include <map>
#include <memory>
#include <string>
#include <vector>


// This macro is to be placed after each material class (UO2, Pu, etc.)
// It will register the material with the factory
#define REGISTER_MATERIAL( NAME )                                                          \
    static struct NAME##_INIT {                                                            \
        NAME##_INIT()                                                                      \
        {                                                                                  \
            static AMP::voodoo::Registration<AMP::Materials::Material, NAME> reg( #NAME ); \
        }                                                                                  \
    } NAME##_init


namespace AMP::Materials {


/**
 * Material base class.
 * Loose organizer to collect a group of properties.
 */
class Material
{
public:
    Material() {}

    virtual ~Material() {}

public:
    //! check if a property exists in the material
    bool hasProperty( const std::string &type ) const;

    //! Return the name of the material
    virtual std::string materialName() const = 0;

    //! get a pointer to a specific scalar property
    std::shared_ptr<Property> property( std::string type );

    //! return a list of all properties in this material
    std::vector<std::string> list() const;

protected:
    /// database of scalar properties
    std::map<std::string, std::shared_ptr<Property>> d_propertyMap;

protected:
    //! Add a constant-value fixed property
    template<class PROPERTY, class... Args>
    void addProperty( const std::string &name, Args &&...args )
    {
        auto name2          = materialName() + "::" + name;
        d_propertyMap[name] = std::make_shared<PROPERTY>( name2, args... );
    }

    //! Add a constant-value fixed property
    void addScalarProperty( const std::string &name,
                            double value,
                            const AMP::Units &unit = AMP::Units(),
                            std::string source     = "" );
};


//! Get a material
std::shared_ptr<Material> getMaterial( const std::string &name );

//! Get the list of materials available
std::vector<std::string> getMaterialList();


} // namespace AMP::Materials


#endif
