#ifndef included_AMP_AsynchronousColumnOperatorParameters
#define included_AMP_AsynchronousColumnOperatorParameters

#include "operators/OperatorParameters.h"

namespace AMP {
namespace Operator {

  class AsynchronousColumnOperatorParameters : public OperatorParameters
  {
    public:

      AsynchronousColumnOperatorParameters ( const AMP::shared_ptr<AMP::Database> &db )
        : OperatorParameters ( db )
      {
      }



  };


}
}

#endif
