#include "AMP/vectors/operations/VectorOperationsFactory.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/vectors/operations/MultiVectorOperations.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"


namespace AMP::LinearAlgebra {


std::shared_ptr<VectorOperations>
VectorOperationsFactory::create( int64_t fid, AMP::IO::RestartManager *manager )
{
    std::string type;
    IO::readHDF5( fid, "ClassType", type );
    std::shared_ptr<VectorOperations> operations;
    if ( type == "MultiVectorOperations" ) {
        operations = std::make_shared<AMP::LinearAlgebra::MultiVectorOperations>( fid, manager );
    } else if ( type == "VectorOperationsDefault<double>" ) {
        operations =
            std::make_shared<AMP::LinearAlgebra::VectorOperationsDefault<double>>( fid, manager );
    } else if ( type == "VectorOperationsDefault<float>" ) {
        operations =
            std::make_shared<AMP::LinearAlgebra::VectorOperationsDefault<float>>( fid, manager );
    } else {
        operations = FactoryStrategy<VectorOperations, int64_t, AMP::IO::RestartManager *>::create(
            type, fid, manager );
    }
    return operations;
}


} // namespace AMP::LinearAlgebra


template<>
void AMP::FactoryStrategy<AMP::LinearAlgebra::VectorOperations,
                          int64_t,
                          AMP::IO::RestartManager *>::registerDefault()
{
}
