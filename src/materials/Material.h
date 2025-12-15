#ifndef included_AMP_Material
#define included_AMP_Material

#include "AMP/materials/Property.h"

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>


namespace AMP::Materials {


/**
 * Material base class.
 * Loose organizer to collect a group of properties.
 */
class Material
{
public:
    //! Default constructor
    Material() = default;

    //! Default destructor
    virtual ~Material() = default;

public:
    //! check if a property exists in the material
    bool hasProperty( const std::string &type ) const;

    //! Return the name of the material
    virtual std::string materialName() const = 0;

    /*!
     * \brief  Get the desired property
     * \details  Return a shared_ptr to the desired property if it exists.
     *    If it does not exist, return a nullptr.
     * \param type  Name of desired property
     */
    std::shared_ptr<Property> property( const std::string &type );

    /*!
     * \brief  Get the desired property
     * \details  Return a shared_ptr to the desired property if it exists.
     *    If it does not exist, return a nullptr.
     * \param type  Name of desired property
     */
    std::shared_ptr<const Property> property( const std::string &type ) const;

    //! return a list of all properties in this material
    std::vector<std::string> list() const;

public:
    //! Add a property
    template<class PROPERTY>
    void addProperty( std::string_view name, std::shared_ptr<PROPERTY> property )
    {
        d_propertyMap[std::string( name )] = property;
    }
    template<class PROPERTY, typename... Args>
    void addProperty( std::string_view name, Args &&...args )
    {
        auto name2 = materialName() + "::" + std::string( name );
        d_propertyMap[std::string( name )] =
            std::make_shared<PROPERTY>( name2, std::forward<Args>( args )... );
    }

    //! Add a constant-value fixed property
    void
    addStringProperty( std::string_view name, std::string value, std::string_view source = "" );

    //! Add a constant-value fixed property
    void addScalarProperty( std::string_view name,
                            double value,
                            const AMP::Units &unit  = AMP::Units(),
                            std::string_view source = "" );

    //! Add a constant-value fixed property
    void addScalarProperty( std::string_view name,
                            AMP::Array<double> value,
                            const AMP::Units &unit  = AMP::Units(),
                            std::string_view source = "" );

    //! Add a polynomial based property
    void addPolynomialProperty( std::string_view name,
                                const AMP::Units &unit                    = {},
                                std::vector<double> params                = {},
                                std::vector<std::string> args             = {},
                                std::vector<std::array<double, 2>> ranges = {},
                                std::vector<AMP::Units> argUnits          = {},
                                std::string_view source                   = "" );

    //! Add an equation based propoerty
    void addEquationProperty( std::string_view name,
                              const AMP::Units &unit,
                              std::string expression,
                              std::vector<std::string> args             = {},
                              std::vector<std::array<double, 2>> ranges = {},
                              std::vector<AMP::Units> argUnits          = {},
                              std::string_view source                   = "" );


protected:
    /// database of scalar properties
    std::map<std::string, std::shared_ptr<Property>> d_propertyMap;
};


//! Construct a material from a database
class DatabaseMaterial : public Material
{
public:
    //! Construct a material from the database
    DatabaseMaterial( std::string_view name, std::shared_ptr<Database> db );

    //! Return the name of the material
    std::string materialName() const override { return d_name; }

private:
    std::string d_name; //!< Name of material
};


//! Register a material with the factory
void registerMaterial( const std::string &name, std::function<std::unique_ptr<Material>()> fun );

//! Get a material
std::unique_ptr<Material> getMaterial( const std::string &name );

//! Get the list of materials available
std::vector<std::string> getMaterialList();

//! Check if the given material exists (is registered)
bool isMaterial( const std::string &name );


} // namespace AMP::Materials


#endif
