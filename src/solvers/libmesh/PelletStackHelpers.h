#ifndef included_AMP_PelletStackHelpers
#define included_AMP_PelletStackHelpers

#include "AMP/mesh/Mesh.h"
#include "AMP/operators/CoupledOperator.h"
#include "AMP/operators/libmesh/PelletStackOperator.h"
#include "AMP/operators/map/AsyncMapColumnOperator.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/libmesh/PelletStackMechanicsSolver.h"
#include "AMP/vectors/Vector.h"


namespace AMP::Operator::PelletMechanics {


std::shared_ptr<AsyncMapColumnOperator>
createMaps( std::shared_ptr<AMP::Mesh::Mesh> manager,
            std::shared_ptr<AMP::Database> global_input_db );


std::shared_ptr<AMP::Operator::PelletStackOperator>
createStackOperator( std::shared_ptr<AMP::Mesh::Mesh> manager,
                     std::shared_ptr<AsyncMapColumnOperator> n2nmaps,
                     std::shared_ptr<AMP::Database> global_input_db );


std::shared_ptr<AMP::Operator::ColumnOperator>
createNonlinearColumnOperator( std::shared_ptr<AMP::Operator::PelletStackOperator> pelletStackOp,
                               std::shared_ptr<AMP::Database> global_input_db );


std::shared_ptr<ColumnOperator>
createLinearColumnOperator( std::shared_ptr<ColumnOperator> nonlinearColumnOperator );


std::shared_ptr<AMP::Operator::CoupledOperator>
createCoupledOperator( std::shared_ptr<AMP::Operator::AsyncMapColumnOperator> n2nmaps,
                       std::shared_ptr<AMP::Operator::ColumnOperator> nonlinearColumnOperator );


void setFrozenVectorForMaps( std::shared_ptr<AMP::Mesh::Mesh> manager,
                             std::shared_ptr<AMP::Operator::CoupledOperator> coupledOp );


void createVectors( std::shared_ptr<AMP::Mesh::Mesh> manager,
                    std::shared_ptr<AMP::Operator::CoupledOperator> coupledOp,
                    AMP::LinearAlgebra::Vector::shared_ptr &solVec,
                    AMP::LinearAlgebra::Vector::shared_ptr &rhsVec,
                    AMP::LinearAlgebra::Vector::shared_ptr &scaledRhsVec );


void buildPointLoadRHS( std::shared_ptr<AMP::Database> global_input_db,
                        std::shared_ptr<AMP::Operator::CoupledOperator> coupledOp,
                        AMP::LinearAlgebra::Vector::shared_ptr rhsVec );


void applyBoundaryCorrections( std::shared_ptr<AMP::Operator::CoupledOperator> coupledOp,
                               AMP::LinearAlgebra::Vector::shared_ptr solVec,
                               AMP::LinearAlgebra::Vector::shared_ptr rhsVec );


void createTemperatureVectors( std::shared_ptr<AMP::Mesh::Mesh> manager,
                               AMP::LinearAlgebra::Vector::shared_ptr &initialTemperatureVec,
                               AMP::LinearAlgebra::Vector::shared_ptr &finalTemperatureVec );


void setReferenceTemperature( std::shared_ptr<AMP::Operator::CoupledOperator> coupledOp,
                              AMP::LinearAlgebra::Vector::shared_ptr initialTemperatureVec );


void setFinalTemperature( std::shared_ptr<AMP::Operator::CoupledOperator> coupledOp,
                          AMP::LinearAlgebra::Vector::shared_ptr finalTemperatureVec );


void buildColumnSolver( std::shared_ptr<AMP::Database> columnSolver_db,
                        std::shared_ptr<AMP::Operator::ColumnOperator> linearColumnOperator,
                        std::shared_ptr<AMP::Solver::ColumnSolver> &columnSolver );


std::shared_ptr<AMP::Solver::SolverStrategy>
buildStackSolver( std::shared_ptr<AMP::Database> pelletStackSolver_db,
                  std::shared_ptr<AMP::Operator::PelletStackOperator> pelletStackOp,
                  std::shared_ptr<AMP::Operator::ColumnOperator> linearColumnOperator );


void resetNonlinearOperator( std::shared_ptr<AMP::Operator::CoupledOperator> coupledOp );


} // namespace AMP::Operator::PelletMechanics


#endif
