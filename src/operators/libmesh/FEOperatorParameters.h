
#ifndef included_AMP_FEOperatorParameters
#define included_AMP_FEOperatorParameters

#include "operators/OperatorParameters.h"
#include "operators/ElementOperation.h"

namespace AMP {
  namespace Operator {

    /**
     * This class encapsulates parameters used to initialize or reset operators using a
     finite element discretization (FEOperators). It is an abstract base class.    
     @see LinearFEOperator
     @see NonlinearFEOperator
     */
    class FEOperatorParameters : public OperatorParameters {
      public :

        /**
          Constructor.
          */
        explicit FEOperatorParameters(const AMP::shared_ptr<AMP::Database> &db)
          : OperatorParameters(db) {  }

        /**
          Destructor.
          */
        virtual ~FEOperatorParameters() {}

        AMP::shared_ptr<ElementOperation> d_elemOp; /**< Shared pointer to an element operation */

      protected :

      private :

    };

  }
}

#endif

