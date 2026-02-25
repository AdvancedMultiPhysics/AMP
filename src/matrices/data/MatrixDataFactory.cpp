#include "AMP/matrices/data/MatrixDataFactory.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/matrices/CSRConstruct.h"
#include "AMP/matrices/data/MatrixData.h"

namespace AMP::LinearAlgebra {


std::shared_ptr<MatrixData> MatrixDataFactory::create( int64_t fid,
                                                       AMP::IO::RestartManager *manager )
{
    std::string type;
    AMP::IO::readHDF5( fid, "ClassType", type );
    std::shared_ptr<MatrixData> data;
    if ( type == "CSRMatrixData" ) {
        std::uint16_t mode;
        IO::readHDF5( fid, "mode", mode );
        data = csrConstruct<MatrixData>( static_cast<csr_mode>( mode ), fid, manager );
    } else {
        std::shared_ptr<MatrixData> data =
            FactoryStrategy<MatrixData, int64_t, AMP::IO::RestartManager *>::create(
                type, fid, manager );
    }
    return data;
}


} // namespace AMP::LinearAlgebra


template<>
void AMP::FactoryStrategy<AMP::LinearAlgebra::MatrixData, int64_t, AMP::IO::RestartManager *>::
    registerDefault()
{
}
