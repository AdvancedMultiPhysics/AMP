#ifndef included_AMP_DOFManager_DOFManagerFactory
#define included_AMP_DOFManager_DOFManagerFactory

#include "AMP/utils/FactoryStrategy.hpp"


namespace AMP::IO {
class RestartManager;
}


namespace AMP::Discretization {

class DOFManager;

template<class DOFMANAGER>
std::unique_ptr<DOFMANAGER> createDOFManagerFromRestart( int64_t fid,
                                                         AMP::IO::RestartManager *manager )
{
    return std::make_unique<DOFMANAGER>( fid, manager );
}


//! DOFManager factory class
class DOFManagerFactory
{
public:
    //! get a singleton instance of the factory
    static DOFManagerFactory &getFactory()
    {
        static DOFManagerFactory singletonInstance;
        return singletonInstance;
    }

    //! Create the vector from the restart file
    static std::shared_ptr<DOFManager> create( int64_t fid, AMP::IO::RestartManager *manager );

    //! Register a vector with the factory
    template<class DOFMANAGER>
    static void registerDOFManager( const std::string &name )
    {
        FactoryStrategy<DOFManager, int64_t, AMP::IO::RestartManager *>::registerFactory(
            name, createDOFManagerFromRestart<DOFMANAGER> );
    }
};


} // namespace AMP::Discretization

#endif
