#include "AMP/vectors/data/VectorDataFactory.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/vectors/data/MultiVectorData.h"
#include "AMP/vectors/data/VectorDataDefault.h"

namespace AMP::LinearAlgebra {


std::shared_ptr<VectorData> VectorDataFactory::create( int64_t fid,
                                                       AMP::IO::RestartManager *manager )
{
    std::string type;
    AMP::IO::readHDF5( fid, "ClassType", type );
    std::shared_ptr<VectorData> data;
    if ( type == "MultiVectorData" ) {
        data = std::make_shared<AMP::LinearAlgebra::MultiVectorData>( fid, manager );
    } else if ( type == "VectorDataDefault<double>" ) {
        data = std::make_shared<AMP::LinearAlgebra::VectorDataDefault<double>>( fid, manager );
    } else if ( type == "VectorDataDefault<float>" ) {
        data = std::make_shared<AMP::LinearAlgebra::VectorDataDefault<float>>( fid, manager );
    } else {
        data = FactoryStrategy<VectorData, int64_t, AMP::IO::RestartManager *>::create(
            type, fid, manager );
    }
    return data;
}


} // namespace AMP::LinearAlgebra


template<>
void AMP::FactoryStrategy<AMP::LinearAlgebra::VectorData, int64_t, AMP::IO::RestartManager *>::
    registerDefault()
{
}
