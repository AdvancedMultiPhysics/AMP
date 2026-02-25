#include "AMP/matrices/operations/MatrixOperationsFactory.h"
#include "AMP/IO/PIO.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/matrices/CSRConstruct.h"
#include "AMP/matrices/operations/MatrixOperations.h"


namespace AMP::LinearAlgebra {


std::shared_ptr<MatrixOperations>
MatrixOperationsFactory::create( int64_t fid, AMP::IO::RestartManager *manager )
{
    std::string type;
    IO::readHDF5( fid, "ClassType", type );
    std::shared_ptr<MatrixOperations> operations;
    if ( type == "CSRMatrixOperationsDefault" || type == "CSRMatrixOperationsDevice" ||
         type == "CSRMatrixOperationsKokkos" ) {
        std::uint16_t mode;
        IO::readHDF5( fid, "mode", mode );
        operations =
            csrConstruct<MatrixOperations>( static_cast<csr_mode>( mode ), fid, manager, type );
    } else {
        operations = FactoryStrategy<MatrixOperations, int64_t, AMP::IO::RestartManager *>::create(
            type, fid, manager );
    }
    return operations;
}


} // namespace AMP::LinearAlgebra


template<>
void AMP::FactoryStrategy<AMP::LinearAlgebra::MatrixOperations,
                          int64_t,
                          AMP::IO::RestartManager *>::registerDefault()
{
}
