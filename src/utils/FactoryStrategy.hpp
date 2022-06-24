#ifndef AMP_FactoryStrategy_HPP_
#define AMP_FactoryStrategy_HPP_

#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>


namespace AMP {


// template for factories for operators and solvers
template<typename TYPE, class... Args>
class FactoryStrategy
{
public:
    using FunctionPtr = std::function<std::unique_ptr<TYPE>( Args... )>;


public:
    //! get a singleton instance of the factory
    static FactoryStrategy &getFactory()
    {
        static FactoryStrategy<TYPE, Args...> singletonInstance;
        return singletonInstance;
    }

    //! Factory method for generating objects
    static std::unique_ptr<TYPE> create( const std::string &name, Args... args )
    {
        auto &factory = getFactory();
        auto it       = factory.d_factories.find( name );
        if ( it == factory.d_factories.end() )
            AMP_ERROR( "Unable to create object " + name );
        return it->second( args... );
    }

    //! public interface for registering a function that creates an object
    static void registerFactory( const std::string &name, FactoryStrategy::FunctionPtr ptr )
    {
        getFactory().d_factories[name] = ptr;
    }

    //! Clear factory data and all registered functions
    static void clear() { getFactory().d_factories.clear(); }

    static std::vector<std::string> getKeys()
    {
        std::vector<std::string> keys;
        for ( const auto &tmp : getFactory().d_factories )
            keys.push_back( tmp.first );
        return keys;
    }


protected:
    //! Private constructor
    FactoryStrategy() = default;


protected:
    std::map<std::string, FunctionPtr> d_factories; //! maps from names to factories
};


} // namespace AMP
#endif
