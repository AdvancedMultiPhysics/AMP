
#include "operators/MeshBasedOperator.h"
#include "vectors/VectorSelector.h"

namespace AMP {
  namespace Operator {

    AMP::LinearAlgebra::Vector::shared_ptr MeshBasedOperator :: subsetOutputVector(
        AMP::LinearAlgebra::Vector::shared_ptr vec) {
      AMP::LinearAlgebra::Vector::shared_ptr varSubsetVec = vec->subsetVectorForVariable(getOutputVariable());
      if(varSubsetVec == NULL) {
        return varSubsetVec;
      } else {
        AMP::LinearAlgebra::VS_Mesh meshSelector("meshSelector", d_Mesh);
        return varSubsetVec->select(meshSelector, getOutputVariable()->getName());
      }
    }

    AMP::LinearAlgebra::Vector::shared_ptr MeshBasedOperator :: subsetInputVector(
        AMP::LinearAlgebra::Vector::shared_ptr vec) {
      AMP::LinearAlgebra::Vector::shared_ptr varSubsetVec = vec->subsetVectorForVariable(getInputVariable());
      if(varSubsetVec == NULL) {
        return varSubsetVec;
      } else {
        AMP::LinearAlgebra::VS_Mesh meshSelector("meshSelector", d_Mesh);
        return varSubsetVec->select(meshSelector, getInputVariable()->getName());
      }
    }

  }
}


