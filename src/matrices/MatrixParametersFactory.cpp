#include "AMP/matrices/MatrixParametersFactory.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/matrices/CSRConstruct.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/MatrixParameters.h"


namespace AMP::LinearAlgebra {


std::shared_ptr<MatrixParametersBase>
MatrixParametersFactory::create( int64_t fid, AMP::IO::RestartManager *manager )
{
    std::string type;
    std::shared_ptr<MatrixParametersBase> matrixParams;
    IO::readHDF5( fid, "type", type );
    if ( type == "MatrixParameters" )
        return std::make_shared<MatrixParameters>( fid, manager );
    else if ( type == "RawCSRMatrixParameters" || type == "AMPCSRMatrixParameters" ) {
        std::uint16_t mode;
        IO::readHDF5( fid, "mode", mode );
        if ( type == "RawCSRMatrixParameters" )
            matrixParams =
                csrConstruct<MatrixParametersBase>( static_cast<csr_mode>( mode ), fid, manager );
        else
            matrixParams =
                csrConstruct<MatrixParameters>( static_cast<csr_mode>( mode ), fid, manager );
    } else {
        matrixParams =
            FactoryStrategy<MatrixParametersBase, int64_t, AMP::IO::RestartManager *>::create(
                type, fid, manager );
    }
    return matrixParams;
}


} // namespace AMP::LinearAlgebra


template<>
void AMP::FactoryStrategy<AMP::LinearAlgebra::MatrixParametersBase,
                          int64_t,
                          AMP::IO::RestartManager *>::registerDefault()
{
}
