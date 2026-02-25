#ifndef included_AMP_OperatorBuilder
#define included_AMP_OperatorBuilder

#include "AMP/mesh/Mesh.h"
#include "AMP/operators/ElementPhysicsModel.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/boundary/BoundaryOperator.h"
#include "AMP/utils/Database.h"


/**
 * \namespace OperatorBuilder
 * \brief A class used to create operators
 * \details  This class provides routines for creating operators.
 *  This is a helper class that simplifies operator creation in tests.
 */
namespace AMP::Operator::OperatorBuilder {


/**
 * \brief Create operator from database
 * \details  This function will create a new operator given mesh and input database,
 * \param mesh                  Mesh for the operator
 * \param operatorName          Name of the operator to create
 * \param input_db              Input database
 */
std::shared_ptr<Operator> createOperator( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                          const std::string &operatorName,
                                          std::shared_ptr<AMP::Database> input_db );


} // namespace AMP::Operator::OperatorBuilder


#endif
