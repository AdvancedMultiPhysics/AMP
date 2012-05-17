#include "operators/ColumnOperator.h"
#include "operators/ColumnOperatorParameters.h"
#include "utils/Utilities.h"
#include "vectors/MultiVariable.h"
#include "utils/ProfilerApp.h"

namespace AMP {
  namespace Operator {

    void
      ColumnOperator :: apply(const AMP::LinearAlgebra::Vector::shared_ptr & f,
          const AMP::LinearAlgebra::Vector::shared_ptr & u, AMP::LinearAlgebra::Vector::shared_ptr & r,
          const double a, const double b)
      {
        PROFILE_START("apply");
        for(unsigned int i = 0; i < d_Operators.size(); i++)
        {
          d_Operators[i]->apply(f, u, r, a, b);
        }
        PROFILE_STOP("apply");
      }

    boost::shared_ptr<OperatorParameters>
      ColumnOperator :: getJacobianParameters(const AMP::LinearAlgebra::Vector::shared_ptr & u)
      {
        PROFILE_START("getJacobianParameters");
        boost::shared_ptr<AMP::Database> db;
        boost::shared_ptr<ColumnOperatorParameters> opParameters(new ColumnOperatorParameters(db));

        (opParameters->d_OperatorParameters).resize(d_Operators.size());

        for(unsigned int i = 0; i < d_Operators.size(); i++)
        {
          (opParameters->d_OperatorParameters)[i] = (d_Operators[i]->getJacobianParameters(u));
        }
        PROFILE_STOP("getJacobianParameters");
        return opParameters;
      }

    void
      ColumnOperator :: reset(const boost::shared_ptr<OperatorParameters>& params)
      {
        PROFILE_START("reset");
        boost::shared_ptr<ColumnOperatorParameters> columnParameters =
          boost::dynamic_pointer_cast<ColumnOperatorParameters>(params);

        AMP_INSIST( (columnParameters.get() != NULL), "ColumnOperator::reset parameter object is NULL" );

        AMP_INSIST( ( ((columnParameters->d_OperatorParameters).size()) == (d_Operators.size()) ), " std::vector sizes do not match! " );

        for(unsigned int i = 0; i < d_Operators.size(); i++)
        {
          d_Operators[i]->reset((columnParameters->d_OperatorParameters)[i]);
        }
        PROFILE_STOP("reset");
      }

    void
      ColumnOperator :: append(boost::shared_ptr< Operator > op)
      {
        AMP_INSIST( (op.get() != NULL), "AMP::ColumnOperator::appendRow input argument is a NULL operator");

        d_Operators.push_back(op);
      }

    AMP::LinearAlgebra::Variable::shared_ptr ColumnOperator::getInputVariable()
    {
      boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> retVariable( new AMP::LinearAlgebra::MultiVariable("ColumnVariable"));

      for(unsigned int i = 0; i < d_Operators.size(); i++)
      {
        AMP::LinearAlgebra::Variable::shared_ptr opVar = d_Operators[i]->getInputVariable();
        if(opVar.get()!=NULL)
        {
          retVariable->add(opVar);
        }
      }
      retVariable->removeDuplicateVariables();

      return retVariable;
    }

    AMP::LinearAlgebra::Variable::shared_ptr ColumnOperator::getOutputVariable()
    {
      boost::shared_ptr<AMP::LinearAlgebra::MultiVariable> retVariable( new AMP::LinearAlgebra::MultiVariable("ColumnVariable"));

      for(unsigned int i = 0; i < d_Operators.size(); i++)
      {
        AMP::LinearAlgebra::Variable::shared_ptr opVar = d_Operators[i]->getOutputVariable();
        if(opVar.get()!=NULL)
        {
          retVariable->add(opVar);
        }
      }
      retVariable->removeDuplicateVariables();

      return retVariable;
    }

    bool
      ColumnOperator::isValidInput(boost::shared_ptr<AMP::LinearAlgebra::Vector> &u)
      {
        bool bRetVal=true;

        for(unsigned int i = 0; i < d_Operators.size(); i++)
        {
          bRetVal = bRetVal && d_Operators[i]->isValidInput(u);
        }

        return bRetVal;
      }

  }
}

