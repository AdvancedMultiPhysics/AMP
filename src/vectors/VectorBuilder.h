#ifndef included_AMP_VectorBuider
#define included_AMP_VectorBuider

#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/Backend.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/VectorDataDefault.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"

#include <string>


// Forward declares
extern "C" {
typedef struct _p_Vec *Vec;
}
#if defined( AMP_USE_TRILINOS )
namespace Teuchos {
template<class TYPE>
class RCP;
}

    #if defined( AMP_USE_TRILINOS_THYRA )
namespace Thyra {
template<class TYPE>
class VectorBase;
}
    #endif

#endif

namespace AMP::LinearAlgebra {


// Forward declerations
class EpetraVectorEngineParameters;


/**
 * \brief  This function will create a vector from an arbitrary DOFManager
 * \details  This function is responsible for creating vectors from a DOFManager and variable.
 * \param[in] DOFs          DOFManager to use for constucting the vector
 * \param[in] variable      Variable for the vector
 * \param[in] split         If we are given a multiDOFManager, do we want to split the vector
 *                              based on the individual DOFManagers to create a MultiVector
 */
template<typename TYPE = double,
         typename OPS  = VectorOperationsDefault<TYPE>,
         typename DATA = VectorDataDefault<TYPE>>
AMP::LinearAlgebra::Vector::shared_ptr
createVector( std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
              std::shared_ptr<AMP::LinearAlgebra::Variable> variable,
              bool split = true );

/**
 * \brief  This function will create a vector from a DOFManager with a given ops backend
 */
template<typename TYPE = double, typename DATA = VectorDataDefault<TYPE>>
AMP::LinearAlgebra::Vector::shared_ptr
createVector( std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
              std::shared_ptr<AMP::LinearAlgebra::Variable> variable,
              bool split,
              AMP::Utilities::Backend backend );

/**
 * \brief  This function will create a vector from an arbitrary DOFManager
 * \details  This function is responsible for creating vectors from a DOFManager and variable.
 * \param[in] DOFs          DOFManager to use for constucting the vector
 * \param[in] variable      Variable for the vector
 * \param[in] split         If we are given a multiDOFManager, do we want to split the vector
 *                              based on the individual DOFManagers to create a MultiVector
 * \param[in] memType       Memory space in which to create vector
 */
template<typename TYPE = double>
AMP::LinearAlgebra::Vector::shared_ptr
createVector( std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
              std::shared_ptr<AMP::LinearAlgebra::Variable> variable,
              bool split,
              AMP::Utilities::MemoryType memType );

/**
 * \brief  This function will create a vector from an arbitrary DOFManager
 * \details  This function is responsible for creating vectors from a DOFManager and variable.
 * \param[in] DOFs          DOFManager to use for constucting the vector
 * \param[in] variable      Variable for the vector
 * \param[in] split         If we are given a multiDOFManager, do we want to split the vector
 *                              based on the individual DOFManagers to create a MultiVector
 * \param[in] memType       Memory space in which to create vector
 * \param[in] backend       Backend for operations
 */
template<typename TYPE = double>
AMP::LinearAlgebra::Vector::shared_ptr
createVector( std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
              std::shared_ptr<AMP::LinearAlgebra::Variable> variable,
              bool split,
              AMP::Utilities::MemoryType memType,
              AMP::Utilities::Backend backend );

/**
 * \brief  This function will create a vector from a vector
 * \details  This function is responsible for creating vectors from an existing vector.
 * \param[in] vector        Vector we want to mimic
 * \param[in] memType       Memory space in which to create vector
 */
template<typename TYPE = double>
AMP::LinearAlgebra::Vector::shared_ptr
createVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vector,
              AMP::Utilities::MemoryType memType );


/**
 * \brief  This function will create a vector from a vector
 * \details  This function is responsible for creating vectors from an existing vector.
 * \param[in] vector        Vector we want to mimic
 * \param[in] memType       Memory space in which to create vector
 */
AMP::LinearAlgebra::Vector::shared_ptr
createVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> vector,
              AMP::Utilities::MemoryType memType );

/**
 * \brief  This function will create a vector from a vector
 * \details  This function is responsible for creating vectors from an existing vector.
 * \param[in] vector        Vector we want to mimic
 * \param[in] memType       Memory space in which to create vector
 * \param[in] backend       Backend for operations
 */
template<typename TYPE = double>
AMP::LinearAlgebra::Vector::shared_ptr
createVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vector,
              AMP::Utilities::MemoryType memType,
              AMP::Utilities::Backend backend );


/**
 * \brief  This function will create a vector from a vector
 * \details  This function is responsible for creating vectors from an existing vector.
 * \param[in] vector        Vector we want to mimic
 * \param[in] memType       Memory space in which to create vector
 * \param[in] backend       Backend for operations
 */
AMP::LinearAlgebra::Vector::shared_ptr
createVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> vector,
              AMP::Utilities::MemoryType memType,
              AMP::Utilities::Backend backend );

/**
 * \brief  Create a vector from an arbitrary PETSc Vec
 * \details  This function creates a vector from an arbitrary PETSc Vec
 * \param[in] v             PETSc Vec
 * \param[in] deleteable    If true, ~Vector() will call VecDestroy()
 * \param[in] comm          The communicator associated with the Vec (optional)
 * \param[in] var           The variable to use with the vector (optional)
 */
std::shared_ptr<Vector> createVector( Vec v,
                                      bool deleteable,
                                      AMP_MPI comm                  = AMP_MPI(),
                                      std::shared_ptr<Variable> var = nullptr );


/**
 * \brief  Create an epetra vector
 * \param[in] commList      Communication list
 * \param[in] DOFs          DOF manager
 * \param[in] p             Optional VectorData object
 */
std::shared_ptr<Vector> createEpetraVector( std::shared_ptr<CommunicationList> commList,
                                            std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
                                            std::shared_ptr<VectorData> p = nullptr );


/**
 * \brief  Create a Tpetra vector
 * \param[in] commList      Communication list
 * \param[in] DOFs          DOF manager
 * \param[in] p             Optional VectorData object
 */
std::shared_ptr<Vector> createTpetraVector( std::shared_ptr<CommunicationList> commList,
                                            std::shared_ptr<AMP::Discretization::DOFManager> DOFs );

#if defined( AMP_USE_TRILINOS ) && defined( AMP_USE_TRILINOS_THYRA )

/**
 * \brief  Create a vector from an arbitrary Thyra Vector
 * \details  This function creates a vector from an arbitrary Thyra Vector
 * \param[in] vec           PETSc Vec
 * \param[in] local         The local size of the vector
 * \param[in] comm          The communicator associated with the Vec
 * \param[in] var           The variable to use with the vector (optional)
 */
std::shared_ptr<Vector> createVector( Teuchos::RCP<Thyra::VectorBase<double>> vec,
                                      size_t local,
                                      AMP_MPI comm,
                                      std::shared_ptr<Variable> var = nullptr );
#endif

/** \brief   Create a simple AMP vector
 * \details  This is a factory method to create a simple AMP vector.
 * \param    localSize  The number of elements in the vector on this processor
 * \param    var The variable associated with the new vector
 */
template<typename TYPE,
         typename VecOps  = VectorOperationsDefault<TYPE>,
         typename VecData = VectorDataDefault<TYPE>>
Vector::shared_ptr createSimpleVector( size_t localSize, const std::string &var );


/** \brief   Create a simple AMP vector
 * \details  This is a factory method to create a simple AMP vector.
 * \param    localSize  The number of elements in the vector on this processor
 * \param    var The variable associated with the new vector
 */
template<typename TYPE,
         typename VecOps  = VectorOperationsDefault<TYPE>,
         typename VecData = VectorDataDefault<TYPE>>
Vector::shared_ptr createSimpleVector( size_t localSize, std::shared_ptr<Variable> var );


/** \brief   Create a simple AMP vector
 * \details  This is a factory method to create a simple AMP vector.
 * \param    localSize  The number of elements in the vector on this processor
 * \param    var The variable associated with the new vector
 * \param    comm The variable associated with the new vector
 */
template<typename TYPE,
         typename VecOps  = VectorOperationsDefault<TYPE>,
         typename VecData = VectorDataDefault<TYPE>>
Vector::shared_ptr
createSimpleVector( size_t localSize, std::shared_ptr<Variable> var, AMP_MPI comm );


/** \brief   Create a simple AMP vector
 * \details  This is a factory method to create a simple AMP vector.
 *           It spans a comm and contains ghost values.
 * \param    var The variable associated with the new vector
 * \param    DOFs The DOFManager
 * \param    commlist The communication list
 */
template<typename TYPE,
         typename VecOps  = VectorOperationsDefault<TYPE>,
         typename VecData = VectorDataDefault<TYPE>>
Vector::shared_ptr createSimpleVector( std::shared_ptr<Variable> var,
                                       std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
                                       std::shared_ptr<CommunicationList> commlist );


/** \brief    Create a ArrayVector
 * \details  This is the factory method for the ArrayVector.
 * \param    localSize  The number of elements in the vector on this processor
 * \param    var The variable name for the new vector
 */
template<typename T, typename FUN = FunctionTable<T>, typename Allocator = AMP::HostAllocator<void>>
Vector::shared_ptr createArrayVector( const ArraySize &localSize, const std::string &var );

/** \brief    Cre
ate a ArrayVector
 * \details  This is the factory method for the ArrayVector.
 * \param    localSize  The number of elements in the vector on this processor
 * \param    var The variable associated with the new vector
 */
template<typename T, typename FUN = FunctionTable<T>, typename Allocator = AMP::HostAllocator<void>>
Vector::shared_ptr createArrayVector( const ArraySize &localSize, std::shared_ptr<Variable> var );


/** \brief    Create a ArrayVector
 * \details  This is the factory method for the ArrayVector.
 * \param    localSize  The number of elements in the vector on this processor
 * \param    blockIndex  The block index for the current rank
 * \param    comm  The communicator for the vector
 * \param    var The variable associated with the new vector
 */
template<typename T, typename FUN = FunctionTable<T>, typename Allocator = AMP::HostAllocator<void>>
Vector::shared_ptr createArrayVector( const ArraySize &localSize,
                                      const ArraySize &blockIndex,
                                      const AMP_MPI &comm,
                                      std::shared_ptr<Variable> var );

/** \brief    Create a view to raw vector data
 * \details  This is the factory method for the raw vector view.
 */
template<typename T>
Vector::shared_ptr createVectorAdaptor( const std::string &name,
                                        std::shared_ptr<AMP::Discretization::DOFManager> DOFs,
                                        T *data );

} // namespace AMP::LinearAlgebra

#endif
