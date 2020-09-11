#ifdef USE_AMP_DISCRETIZATION
#ifndef included_AMP_VectorBuider
#define included_AMP_VectorBuider

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/VectorDataCPU.h"
#include "AMP/vectors/operations/VectorOperationsDefault.h"

#include <string>


extern "C" {
typedef struct _p_Vec *Vec;
}


namespace AMP {
namespace LinearAlgebra {


/**
 * \brief  This function will create a vector from an arbitrary DOFManager
 * \details  This function is responsible for creating vectors from a DOFManager and variable.
 * \param[in] DOFs          DOFManager to use for constucting the vector
 * \param[in] variable      Variable for the vector
 * \param[in] split         If we are given a multiDOFManager, do we want to split the vector
 *                              based on the individual DOFManagers to create a MultiVector
 */
AMP::LinearAlgebra::Vector::shared_ptr
createVector( AMP::Discretization::DOFManager::shared_ptr DOFs,
              AMP::LinearAlgebra::Variable::shared_ptr variable,
              bool split = true );


#if defined( USE_EXT_PETSC )
/**
 * \brief  Create a vector from an arbitrary PETSc Vec
 * \details  This function creates a vector from an arbitrary PETSc Vec
 * \param[in] v             PETSc Vec
 * \param[in] deleteable    If true, ~Vector() will call VecDestroy()
 * \param[in] comm          The communicator associated with the Vec (optional)
 */
std::shared_ptr<Vector> createVector( Vec v, bool deleteable, AMP_MPI comm = AMP_MPI() );
#endif


/** \brief   Create a simple AMP vector
 * \details  This is a factory method to create a simple AMP vector.
 * \param    localSize  The number of elements in the vector on this processor
 * \param    var The variable associated with the new vector
 */
template<typename TYPE,
         typename VecOps  = VectorOperationsDefault<TYPE>,
         typename VecData = VectorDataCPU<TYPE>>
Vector::shared_ptr createSimpleVector( size_t localSize, const std::string &var );


/** \brief   Create a simple AMP vector
 * \details  This is a factory method to create a simple AMP vector.
 * \param    localSize  The number of elements in the vector on this processor
 * \param    var The variable associated with the new vector
 */
template<typename TYPE,
         typename VecOps  = VectorOperationsDefault<TYPE>,
         typename VecData = VectorDataCPU<TYPE>>
Vector::shared_ptr createSimpleVector( size_t localSize, Variable::shared_ptr var );


/** \brief   Create a simple AMP vector
 * \details  This is a factory method to create a simple AMP vector.
 * \param    localSize  The number of elements in the vector on this processor
 * \param    var The variable associated with the new vector
 * \param    comm The variable associated with the new vector
 */
template<typename TYPE,
         typename VecOps  = VectorOperationsDefault<TYPE>,
         typename VecData = VectorDataCPU<TYPE>>
Vector::shared_ptr createSimpleVector( size_t localSize, Variable::shared_ptr var, AMP_MPI comm );


/** \brief   Create a simple AMP vector
 * \details  This is a factory method to create a simple AMP vector.
 *           It spans a comm and contains ghost values.
 * \param    var The variable associated with the new vector
 * \param    DOFs The DOFManager
 * \param    commlist The communication list
 */
template<typename TYPE,
         typename VecOps  = VectorOperationsDefault<TYPE>,
         typename VecData = VectorDataCPU<TYPE>>
Vector::shared_ptr createSimpleVector( Variable::shared_ptr var,
                                       AMP::Discretization::DOFManager::shared_ptr DOFs,
                                       AMP::LinearAlgebra::CommunicationList::shared_ptr commlist );


} // namespace LinearAlgebra
} // namespace AMP

#endif
#endif


#include "VectorBuilder.hpp"
