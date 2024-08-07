#ifndef included_AMP_ConstructLinearMechanicsRHSVector
#define included_AMP_ConstructLinearMechanicsRHSVector


#include "AMP/mesh/Mesh.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/Vector.h"
#include <memory>

void computeTemperatureRhsVector( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                  std::shared_ptr<AMP::Database> input_db,
                                  std::shared_ptr<AMP::LinearAlgebra::Variable> temperatureVar,
                                  std::shared_ptr<AMP::LinearAlgebra::Variable> displacementVar,
                                  std::shared_ptr<AMP::LinearAlgebra::Vector> currTemperatureVec,
                                  std::shared_ptr<AMP::LinearAlgebra::Vector> prevTemperatureVec,
                                  AMP::LinearAlgebra::Vector::shared_ptr rhsVec );


#endif
