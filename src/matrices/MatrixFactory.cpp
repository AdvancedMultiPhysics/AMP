#include "AMP/matrices/MatrixFactory.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/matrices/CSRConstruct.h"
#include "AMP/matrices/Matrix.h"


namespace AMP::LinearAlgebra {


std::shared_ptr<Matrix> MatrixFactory::create( int64_t fid, AMP::IO::RestartManager *manager )
{
    std::string type;
    std::shared_ptr<Matrix> matrix;
    IO::readHDF5( fid, "type", type );
    if ( type == "CSRMatrix" ) {
        std::uint16_t mode;
        IO::readHDF5( fid, "mode", mode );
        matrix = csrConstruct<Matrix>( static_cast<csr_mode>( mode ), fid, manager );
    } else {
        matrix = FactoryStrategy<Matrix, int64_t, AMP::IO::RestartManager *>::create(
            type, fid, manager );
    }
    return matrix;
}


} // namespace AMP::LinearAlgebra


template<>
void AMP::FactoryStrategy<AMP::LinearAlgebra::Matrix, int64_t, AMP::IO::RestartManager *>::
    registerDefault()
{
}
