#include "AMP/materials/Material.h"
#include "AMP/materials/MaterialList.h"
#include "AMP/materials/ScalarProperty.h"
#include "AMP/utils/FactoryStrategy.hpp"


namespace AMP::Materials {


/********************************************************************
 * Get properties                                                    *
 ********************************************************************/
bool Material::hasProperty( const std::string &type ) const
{
    return d_propertyMap.find( type ) != d_propertyMap.end();
}
std::shared_ptr<Property> Material::property( const std::string &type )
{
    auto it = d_propertyMap.find( type );
    if ( it == d_propertyMap.end() )
        return nullptr;
    return it->second;
}
std::shared_ptr<const Property> Material::property( const std::string &type ) const
{
    auto it = d_propertyMap.find( type );
    if ( it == d_propertyMap.end() )
        return nullptr;
    return it->second;
}
std::vector<std::string> Material::list() const
{
    std::vector<std::string> result;
    for ( auto it : d_propertyMap )
        result.push_back( it.first );
    return result;
}


/********************************************************************
 * Add a property                                                    *
 ********************************************************************/
void Material::addStringProperty( std::string_view name,
                                  std::string value,
                                  std::string_view source )
{
    addProperty<StringProperty>( name, std::move( value ), source );
}
void Material::addScalarProperty( std::string_view name,
                                  double value,
                                  const AMP::Units &unit,
                                  std::string_view source )
{
    addProperty<ScalarProperty>( name, value, unit, source );
}
void Material::addScalarProperty( std::string_view name,
                                  AMP::Array<double> value,
                                  const AMP::Units &unit,
                                  std::string_view source )
{
    addProperty<ScalarProperty>( name, std::move( value ), unit, source );
}
void Material::addPolynomialProperty( std::string_view name,
                                      const AMP::Units &unit,
                                      std::vector<double> params,
                                      std::vector<std::string> args,
                                      std::vector<std::array<double, 2>> ranges,
                                      std::vector<AMP::Units> argUnits,
                                      std::string_view source )
{
    addProperty<PolynomialProperty>( name,
                                     source,
                                     unit,
                                     std::move( params ),
                                     std::move( args ),
                                     std::move( ranges ),
                                     std::move( argUnits ) );
}
void Material::addEquationProperty( std::string_view name,
                                    const AMP::Units &unit,
                                    std::string expression,
                                    std::vector<std::string> args,
                                    std::vector<std::array<double, 2>> ranges,
                                    std::vector<AMP::Units> argUnits,
                                    std::string_view source )
{
    addProperty<EquationProperty>( name,
                                   std::move( expression ),
                                   unit,
                                   std::move( args ),
                                   std::move( ranges ),
                                   std::move( argUnits ),
                                   source );
}

/********************************************************************
 * Construct a material from a database                              *
 ********************************************************************/
DatabaseMaterial::DatabaseMaterial( std::string_view name, std::shared_ptr<Database> db )
    : d_name( name )
{
    if ( !db )
        return;
    auto keys = db->getAllKeys();
    for ( auto &key : keys )
        d_propertyMap[key] = createProperty( key, *db );
}


/********************************************************************
 * Material factory functions                                        *
 ********************************************************************/
using MaterialFactory = AMP::FactoryStrategy<Material>;
std::vector<std::string> getMaterialList() { return MaterialFactory::getKeys(); }
std::unique_ptr<Material> getMaterial( const std::string &name )
{
    return MaterialFactory::create( name );
}
void registerMaterial( const std::string &name, std::function<std::unique_ptr<Material>()> fun )
{
    MaterialFactory::registerFactory( name, fun );
}
bool isMaterial( const std::string &name ) { return MaterialFactory::exists( name ); }


} // namespace AMP::Materials


/********************************************************************
 * Register default materials                                        *
 ********************************************************************/
#define REGISTER_MATERIAL( NAME ) d_factories[#NAME] = []() { return std::make_unique<NAME>(); };
template<>
void AMP::FactoryStrategy<AMP::Materials::Material>::registerDefault()
{
    using namespace AMP::Materials;
    REGISTER_MATERIAL( CylindricallySymmetric );
    REGISTER_MATERIAL( Dr_nonlinear );
    REGISTER_MATERIAL( FixedClad );
    REGISTER_MATERIAL( FixedFuel );
    REGISTER_MATERIAL( Independent );
    REGISTER_MATERIAL( WaterLibrary );
    REGISTER_MATERIAL( Steel316_MSRZC_09 );
    REGISTER_MATERIAL( Ox_MSRZC_09 );
    REGISTER_MATERIAL( UO2_MSRZC_09 );
}
