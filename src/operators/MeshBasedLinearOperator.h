
#ifndef included_AMP_MeshBasedLinearOperator
#define included_AMP_MeshBasedLinearOperator

#include "operators/LinearOperator.h"
#include "operators/MeshBasedOperatorParameters.h"

namespace AMP {
  namespace Operator {

    class MeshBasedLinearOperator : public LinearOperator {
      public :
        MeshBasedLinearOperator(const boost::shared_ptr<MeshBasedOperatorParameters> & params)
          : LinearOperator(params) {
            d_Mesh = params->d_Mesh;
          }

        virtual ~MeshBasedLinearOperator() { }

        AMP::Mesh::Mesh::shared_ptr getMesh() {
          return d_Mesh;
        }

      protected:
        AMP::Mesh::Mesh::shared_ptr d_Mesh;
    };

  }
}

#endif


