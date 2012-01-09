
#include "LinearBVPOperator.h"
#include "LinearBoundaryOperatorParameters.h"
#include "ColumnBoundaryOperator.h"
#include "utils/Utilities.h"

#include <stdexcept>

namespace AMP {
namespace Operator {

  LinearBVPOperator :: LinearBVPOperator(const boost::shared_ptr<BVPOperatorParameters>& params)
    : LinearOperator (params) 
  {
    AMP_ERROR("NonlinearBVPOperator is not converted yet");
/*
    d_volumeOperator = boost::dynamic_pointer_cast<LinearOperator>(params->d_volumeOperator);
    d_boundaryOperator = params->d_boundaryOperator;
    d_MeshAdapter = d_volumeOperator->getMeshAdapter();
    d_matrix = d_volumeOperator->getMatrix();
*/
  }

  void
    LinearBVPOperator :: reset(const boost::shared_ptr<OperatorParameters>& params)
    {
      boost::shared_ptr<BVPOperatorParameters> inParams = 
        boost::dynamic_pointer_cast<BVPOperatorParameters>(params);

      AMP_INSIST( (inParams.get() != NULL), "LinearBVPOperator :: reset Null parameter" );

      d_volumeOperator->reset(inParams->d_volumeOperatorParams);
      
#if 0
      boost::shared_ptr<LinearBoundaryOperatorParameters> boundaryParams =
        boost::dynamic_pointer_cast<LinearBoundaryOperatorParameters>(inParams->d_boundaryOperatorParams);

      AMP_INSIST( ((boundaryParams.get()) != NULL), "NULL boundary parameter" );

      boundaryParams->d_inputMatrix = d_volumeOperator->getMatrix();

      d_matrix = d_volumeOperator->getMatrix();
#else
      // first case - single linear boundary operator parameter object
      boost::shared_ptr<LinearBoundaryOperatorParameters> linearBoundaryParams =
        boost::dynamic_pointer_cast<LinearBoundaryOperatorParameters>(inParams->d_boundaryOperatorParams);

      if(linearBoundaryParams.get()!=NULL)
    {
      linearBoundaryParams->d_inputMatrix = d_volumeOperator->getMatrix();
      d_boundaryOperator->reset(linearBoundaryParams);
    }
      else
    {
      boost::shared_ptr<ColumnBoundaryOperatorParameters> columnBoundaryParams =
        boost::dynamic_pointer_cast<ColumnBoundaryOperatorParameters>(inParams->d_boundaryOperatorParams);

      if(columnBoundaryParams.get()!=NULL)
        {
          for(unsigned int i=0; i<columnBoundaryParams->d_OperatorParameters.size(); i++)
        {
          boost::shared_ptr< OperatorParameters > cparams = columnBoundaryParams->d_OperatorParameters[i];
          boost::shared_ptr<LinearBoundaryOperatorParameters> linearBoundaryParams =
            boost::dynamic_pointer_cast<LinearBoundaryOperatorParameters>(cparams);
          if(linearBoundaryParams.get()!=NULL)
            {
              linearBoundaryParams->d_inputMatrix = d_volumeOperator->getMatrix();
            }          
        }
          d_boundaryOperator->reset(columnBoundaryParams);
        }
      else
        {
          // being here should throw an error
        }
      
    }
#endif
      

      d_matrix = d_volumeOperator->getMatrix();

    }

  void LinearBVPOperator :: modifyRHSvector(AMP::LinearAlgebra::Vector::shared_ptr rhs) {
    (this->getBoundaryOperator())->addRHScorrection(rhs);
    (this->getBoundaryOperator())->setRHScorrection(rhs);
  }

}
}

