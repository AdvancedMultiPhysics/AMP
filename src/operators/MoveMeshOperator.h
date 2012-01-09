
#ifndef included_AMP_MoveMeshOperator
#define included_AMP_MoveMeshOperator

#include "operators/Operator.h"

namespace AMP {
  namespace Operator {

    class MoveMeshOperator : public Operator {
      public :
        MoveMeshOperator(const boost::shared_ptr<OperatorParameters>& params);

        virtual ~MoveMeshOperator() { }

        void setVariable(AMP::LinearAlgebra::Variable::shared_ptr var);

        AMP::LinearAlgebra::Variable::shared_ptr getInputVariable(int varId = -1);

        virtual void apply(const AMP::LinearAlgebra::Vector::shared_ptr &f, 
            const AMP::LinearAlgebra::Vector::shared_ptr &u, AMP::LinearAlgebra::Vector::shared_ptr &r,
            const double a = -1.0, const double b = 1.0);

      protected :
        AMP::LinearAlgebra::Variable::shared_ptr d_var;
        AMP::LinearAlgebra::Vector::shared_ptr d_prevDisp;

    };

  }
}

#endif




