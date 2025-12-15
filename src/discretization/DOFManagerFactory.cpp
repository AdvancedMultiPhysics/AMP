#include "AMP/discretization/DOFManagerFactory.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/simpleDOF_Manager.h"


namespace AMP::Discretization {


std::shared_ptr<DOFManager> DOFManagerFactory::create( int64_t fid,
                                                       AMP::IO::RestartManager *manager )
{
    std::string type;
    IO::readHDF5( fid, "ClassType", type );
    std::shared_ptr<DOFManager> dofs;
    if ( type == "DOFManager" ) {
        dofs = std::make_shared<AMP::Discretization::DOFManager>( fid, manager );
    } else if ( type == "simpleDOFManager" ) {
        dofs = std::make_shared<AMP::Discretization::simpleDOFManager>( fid, manager );
    } else {
        dofs = FactoryStrategy<DOFManager, int64_t, AMP::IO::RestartManager *>::create(
            type, fid, manager );
    }
    return dofs;
}


} // namespace AMP::Discretization


template<>
void AMP::FactoryStrategy<AMP::Discretization::DOFManager, int64_t, AMP::IO::RestartManager *>::
    registerDefault()
{
}
