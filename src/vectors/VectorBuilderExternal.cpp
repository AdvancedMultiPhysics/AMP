#include "AMP/AMP_TPLs.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#ifdef AMP_USE_PETSC
    #include "AMP/vectors/petsc/NativePetscVectorData.h"
    #include "AMP/vectors/petsc/NativePetscVectorOperations.h"
    #include "AMP/vectors/petsc/PetscVector.h"
    #include "petscvec.h"
#endif
#ifdef AMP_USE_TRILINOS
    #include "AMP/vectors/trilinos/epetra/EpetraVector.h"
    #include "AMP/vectors/trilinos/epetra/EpetraVectorData.h"
    #include "AMP/vectors/trilinos/epetra/EpetraVectorOperations.h"
    #include "AMP/vectors/trilinos/thyra/NativeThyraVectorData.h"
    #include "AMP/vectors/trilinos/thyra/NativeThyraVectorOperations.h"
DISABLE_WARNINGS
    #include "Thyra_VectorDefaultBase_decl.hpp"
ENABLE_WARNINGS
    #ifdef AMP_USE_TRILINOS_TPETRA
        #include "AMP/vectors/trilinos/tpetra/TpetraVectorData.h"
        #include "AMP/vectors/trilinos/tpetra/TpetraVectorOperations.h"
    #endif
#else
namespace Teuchos {
template<class TYPE>
class RCP
{
};
} // namespace Teuchos
#endif


namespace AMP::LinearAlgebra {


/********************************************************
 * create vector from PETSc Vec                          *
 ********************************************************/
#if defined( AMP_USE_PETSC )
std::shared_ptr<Vector>
createVector( Vec v, bool deleteable, AMP_MPI comm, std::shared_ptr<Variable> var )
{
    if ( !var )
        var = std::make_shared<Variable>( "vec" );
    auto ops  = std::make_shared<NativePetscVectorOperations>();
    auto data = std::make_shared<NativePetscVectorData>( v, deleteable, comm );
    return std::make_shared<Vector>( data, ops, var, nullptr );
}
#else
std::shared_ptr<Vector> createVector( Vec, bool, AMP_MPI, std::shared_ptr<Variable> )
{
    AMP_ERROR( "PETSc support not enabled" );
    return nullptr;
}
#endif


/********************************************************
 * create vector from Trilinos Thyra vector              *
 ********************************************************/
#if defined( AMP_USE_TRILINOS ) && defined( AMP_USE_TRILINOS_THYRA )
std::shared_ptr<Vector> createVector( Teuchos::RCP<Thyra::VectorBase<double>> vec,
                                      size_t local,
                                      AMP_MPI comm,
                                      std::shared_ptr<Variable> var )
{
    if ( !var )
        var = std::make_shared<Variable>( "vec" );
    auto ops  = std::make_shared<NativeThyraVectorOperations>();
    auto data = std::make_shared<NativeThyraVectorData>( vec, local, comm );
    return std::make_shared<Vector>( data, ops, var, nullptr );
}
#else
std::shared_ptr<Vector>
createVector( Teuchos::RCP<Thyra::VectorBase<double>>, size_t, AMP_MPI, std::shared_ptr<Variable> )
{
    AMP_ERROR( "Thyra support not enabled" );
    return nullptr;
}
#endif


/********************************************************
 * create Trilinos Epetra vector                         *
 ********************************************************/
#if defined( AMP_USE_TRILINOS ) && defined( AMP_USE_TRILINOS_EPETRA )
std::shared_ptr<Vector> createEpetraVector( std::shared_ptr<CommunicationList> commList,
                                            std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
                                            std::shared_ptr<VectorData> buf )
{
    auto var    = std::make_shared<Variable>( "vec" );
    auto ops    = std::make_shared<EpetraVectorOperations>();
    auto params = std::make_shared<EpetraVectorEngineParameters>(
        DOFs->numLocalDOF(), DOFs->getComm(), commList );
    auto data = EpetraVectorData::create( params, buf );
    return std::make_shared<Vector>( data, ops, var, DOFs );
}
#else
std::shared_ptr<Vector> createEpetraVector( std::shared_ptr<CommunicationList>,
                                            std::shared_ptr<AMP::Discretization::DOFManager>,
                                            std::shared_ptr<VectorData> )
{
    AMP_ERROR( "Epetra support not enabled" );
    return nullptr;
}
#endif

/********************************************************
 * create Trilinos Tpetra vector                         *
 ********************************************************/
#if defined( AMP_USE_TRILINOS ) && defined( AMP_USE_TRILINOS_TPETRA )
std::shared_ptr<Vector> createTpetraVector( std::shared_ptr<CommunicationList> commList,
                                            std::shared_ptr<AMP::Discretization::DOFManager> DOFs )
{
    auto var  = std::make_shared<Variable>( "vec" );
    auto ops  = std::make_shared<TpetraVectorOperations<>>();
    auto data = std::make_shared<TpetraVectorData<>>( commList, DOFs );
    return std::make_shared<Vector>( data, ops, var, DOFs );
}
#else
std::shared_ptr<Vector> createTpetraVector( std::shared_ptr<CommunicationList>,
                                            std::shared_ptr<AMP::Discretization::DOFManager>,
                                            std::shared_ptr<VectorData> )
{
    AMP_ERROR( "Tpetra support not enabled" );
    return nullptr;
}
#endif

} // namespace AMP::LinearAlgebra
