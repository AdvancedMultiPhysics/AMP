#include "AMP/vectors/VectorFactory.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/vectors/MultiVector.h"


namespace AMP::LinearAlgebra {


std::shared_ptr<Vector> VectorFactory::create( int64_t fid, AMP::IO::RestartManager *manager )
{
    std::string type;
    IO::readHDF5( fid, "type", type );
    std::shared_ptr<Vector> vec;
    if ( type.substr( 0, 7 ) == "Vector<" ) {
        vec = std::make_shared<AMP::LinearAlgebra::Vector>( fid, manager );
    } else if ( type == "MultiVector" ) {
        vec = std::make_shared<AMP::LinearAlgebra::MultiVector>( fid, manager );
    } else {
        vec = FactoryStrategy<Vector, int64_t, AMP::IO::RestartManager *>::create(
            type, fid, manager );
    }
    return vec;
}


} // namespace AMP::LinearAlgebra


template<>
void AMP::FactoryStrategy<AMP::LinearAlgebra::Vector, int64_t, AMP::IO::RestartManager *>::
    registerDefault()
{
}
