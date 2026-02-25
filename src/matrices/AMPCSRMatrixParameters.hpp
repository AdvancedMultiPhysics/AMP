#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/AMPCSRMatrixParameters.h"
#include "AMP/matrices/MatrixParameters.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/Vector.h"

namespace AMP::LinearAlgebra {

template<typename Config>
AMPCSRMatrixParameters<Config>::AMPCSRMatrixParameters(
    std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
    std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
    const AMP_MPI &comm,
    std::shared_ptr<GetRowHelper> getRowHelper )
    : MatrixParameters( dofLeft, dofRight, comm ), d_getRowHelper( getRowHelper )
{
    AMP_ASSERT( d_getRowHelper.get() );
}

template<typename Config>
AMPCSRMatrixParameters<Config>::AMPCSRMatrixParameters(
    std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
    std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
    const AMP_MPI &comm,
    AMP::Utilities::Backend backend,
    std::shared_ptr<GetRowHelper> getRowHelper )
    : MatrixParameters( dofLeft, dofRight, comm, backend ), d_getRowHelper( getRowHelper )
{
    AMP_ASSERT( d_getRowHelper.get() );
}

template<typename Config>
AMPCSRMatrixParameters<Config>::AMPCSRMatrixParameters(
    std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
    std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
    const AMP_MPI &comm,
    std::shared_ptr<Variable> varLeft,
    std::shared_ptr<Variable> varRight,
    std::shared_ptr<GetRowHelper> getRowHelper )
    : MatrixParameters( dofLeft, dofRight, comm, varLeft, varRight ), d_getRowHelper( getRowHelper )
{
    AMP_ASSERT( d_getRowHelper.get() );
}

template<typename Config>
AMPCSRMatrixParameters<Config>::AMPCSRMatrixParameters(
    std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
    std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
    const AMP_MPI &comm,
    std::shared_ptr<Variable> varLeft,
    std::shared_ptr<Variable> varRight,
    AMP::Utilities::Backend backend,
    std::shared_ptr<GetRowHelper> getRowHelper )
    : MatrixParameters( dofLeft, dofRight, comm, varLeft, varRight, backend ),
      d_getRowHelper( getRowHelper )
{
    AMP_ASSERT( d_getRowHelper.get() );
}

template<typename Config>
AMPCSRMatrixParameters<Config>::AMPCSRMatrixParameters(
    std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
    std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
    const AMP_MPI &comm,
    std::shared_ptr<CommunicationList> commListLeft,
    std::shared_ptr<CommunicationList> commListRight,
    std::shared_ptr<GetRowHelper> getRowHelper )
    : MatrixParameters( dofLeft, dofRight, comm, commListLeft, commListRight ),
      d_getRowHelper( getRowHelper )
{
    AMP_ASSERT( d_getRowHelper.get() );
}

template<typename Config>
AMPCSRMatrixParameters<Config>::AMPCSRMatrixParameters(
    std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
    std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
    const AMP_MPI &comm,
    std::shared_ptr<CommunicationList> commListLeft,
    std::shared_ptr<CommunicationList> commListRight,
    AMP::Utilities::Backend backend,
    std::shared_ptr<GetRowHelper> getRowHelper )
    : MatrixParameters( dofLeft, dofRight, comm, commListLeft, commListRight, backend ),
      d_getRowHelper( getRowHelper )
{
    AMP_ASSERT( d_getRowHelper.get() );
}

template<typename Config>
AMPCSRMatrixParameters<Config>::AMPCSRMatrixParameters(
    std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
    std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
    const AMP_MPI &comm,
    std::shared_ptr<Variable> varLeft,
    std::shared_ptr<Variable> varRight,
    std::shared_ptr<CommunicationList> commListLeft,
    std::shared_ptr<CommunicationList> commListRight,
    std::shared_ptr<GetRowHelper> getRowHelper )
    : MatrixParameters( dofLeft, dofRight, comm, varLeft, varRight, commListLeft, commListRight ),
      d_getRowHelper( getRowHelper )
{
    AMP_ASSERT( d_getRowHelper.get() );
}

template<typename Config>
AMPCSRMatrixParameters<Config>::AMPCSRMatrixParameters(
    std::shared_ptr<AMP::Discretization::DOFManager> dofLeft,
    std::shared_ptr<AMP::Discretization::DOFManager> dofRight,
    const AMP_MPI &comm,
    std::shared_ptr<Variable> varLeft,
    std::shared_ptr<Variable> varRight,
    std::shared_ptr<CommunicationList> commListLeft,
    std::shared_ptr<CommunicationList> commListRight,
    AMP::Utilities::Backend backend,
    std::shared_ptr<GetRowHelper> getRowHelper )
    : MatrixParameters(
          dofLeft, dofRight, comm, varLeft, varRight, commListLeft, commListRight, backend ),
      d_getRowHelper( getRowHelper )
{
    AMP_ASSERT( d_getRowHelper.get() );
}

template<typename Config>
void AMPCSRMatrixParameters<Config>::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    MatrixParameters::registerChildObjects( manager );
}

template<typename Config>
void AMPCSRMatrixParameters<Config>::writeRestart( int64_t fid ) const
{
    MatrixParameters::writeRestart( fid );
}

template<typename Config>
AMPCSRMatrixParameters<Config>::AMPCSRMatrixParameters( int64_t fid,
                                                        AMP::IO::RestartManager *manager )
    : MatrixParameters( fid, manager )
{
}

} // namespace AMP::LinearAlgebra
