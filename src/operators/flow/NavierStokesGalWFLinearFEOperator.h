
#ifndef included_AMP_NavierStokesGalWFLinearFEOperator
#define included_AMP_NavierStokesGalWFLinearFEOperator

/* AMP files */
#include "LinearFEOperator.h"
#include "NavierStokesConstants.h"
#include "NavierStokesGalWFLinearFEOperatorParameters.h"
#include "NavierStokesGalWFLinearElement.h"

#include <vector>

#if 0
//This file has not been converted!

namespace AMP {
  namespace Operator {

    class NavierStokesGalWFLinearFEOperator : public LinearFEOperator 
    {
      public :

        NavierStokesGalWFLinearFEOperator(const boost::shared_ptr<NavierStokesGalWFLinearFEOperatorParameters>& params);

        ~NavierStokesGalWFLinearFEOperator() { }

        void preAssembly(const boost::shared_ptr<OperatorParameters>& params);

        void postAssembly();

        void preElementOperation(const AMP::Mesh::MeshManager::Adapter::Element &);

        void postElementOperation();

        void setInputVariableName(const std::string & name, int varId = -1) {
          (void) varId;      
          d_inpVariable->setName(name);
        }

        void setOutputVariableName(const std::string & name, int varId = -1) {
          (void) varId;      
          d_outVariable->setName(name);
        }

        AMP::LinearAlgebra::Variable::shared_ptr getInputVariable(int varId = -1) {
          return d_inpVariable;
        }

        AMP::LinearAlgebra::Variable::shared_ptr getOutputVariable() {
          return d_outVariable;
        }

        unsigned int numberOfDOFMaps() {
          return 1;
        }

        AMP::LinearAlgebra::Variable::shared_ptr getVariableForDOFMap(unsigned int ) {
          return d_inpVariable;
        }

      protected :

        std::vector<unsigned int> d_dofIndices[3]; 
        std::vector<unsigned int> d_dofIndices1; 

        std::vector<std::vector<double> > d_elementStiffnessMatrix; 

        boost::shared_ptr< NavierStokesGalWFLinearElement > d_flowGalWFLinElem; 

        boost::shared_ptr<FlowTransportModel> d_transportModel; 

        std::vector<AMP::LinearAlgebra::Vector::shared_ptr> d_inVec;

        unsigned int d_numNodesForCurrentElement; /**< Number of nodes in the current element. */

      private :

        boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> d_inpVariable; /**< Input variables. */

        boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> d_outVariable; /**< Output variables. */

    };

  }
}

#endif

#endif


