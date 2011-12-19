
#include "ColumnSolver.h"
#include "operators/ColumnOperatorParameters.h"
#include "operators/LinearOperator.h"

namespace AMP {
namespace Solver {

  ColumnSolver::ColumnSolver(boost::shared_ptr<SolverStrategyParameters> parameters):SolverStrategy(parameters)
  {
    assert(parameters.get()!=NULL);
    const boost::shared_ptr<AMP::Database> &db = parameters->d_db;
    d_IterationType = db->getStringWithDefault("IterationType", "GaussSeidel");
    d_resetColumnOperator = db->getBoolWithDefault("ResetColumnOperator", false);
  }

  void
    ColumnSolver::solve(boost::shared_ptr<AMP::LinearAlgebra::Vector>  f, boost::shared_ptr<AMP::LinearAlgebra::Vector>  u)
    {
      u->zero();

      if(d_IterationType=="GaussSeidel")
      {
        GaussSeidel(f,u);
      }
      else if(d_IterationType=="SymmetricGaussSeidel")
      {
        SymmetricGaussSeidel(f,u);
      }
      else
      {
        AMP::pout << "ERROR: Invalid iteration type specified " << std::endl;
      }
    }

  void
    ColumnSolver::GaussSeidel(boost::shared_ptr<AMP::LinearAlgebra::Vector>  &f, boost::shared_ptr<AMP::LinearAlgebra::Vector>  &u)
    {
      for(int it=0;it<d_iMaxIterations; it++)
      {
        for(unsigned int i=0;i<d_Solvers.size(); i++)
        {
          boost::shared_ptr<AMP::Operator::Operator> op = d_Solvers[i]->getOperator();
          AMP_INSIST(op.get()!=NULL, "EROR: NULL Operator returned by SolverStrategy::getOperator");

          //boost::shared_ptr<AMP::Operator::LinearOperator> linearOperator = boost::dynamic_pointer_cast<LinearOperator>(op);
          //AMP_INSIST(AMP::Operator::linearOperator.get()!=NULL, "ERROR: NULL LinearOperator returned by cast in ColumnSolver");
          //Variable::shared_ptr inputVar = linearOperator->getInputVariable();
          //Variable::shared_ptr outputVar = linearOperator->getOutputVariable();

          AMP::LinearAlgebra::Variable::shared_ptr inputVar = op->getInputVariable();
          AMP::LinearAlgebra::Variable::shared_ptr outputVar = op->getOutputVariable();

          AMP_INSIST(inputVar.get()!=NULL, "ERROR: Null input variable for linear operator");
          AMP_INSIST(outputVar.get()!=NULL, "ERROR: Null output variable for linear operator");

          boost::shared_ptr<AMP::LinearAlgebra::Vector> sf = f->subsetVectorForVariable(outputVar);
          AMP_INSIST(sf.get()!=NULL, "ERROR: subset on rhs f yields NULL vector in ColumnSolver::solve");
          boost::shared_ptr<AMP::LinearAlgebra::Vector> su = u->subsetVectorForVariable(inputVar);
          AMP_INSIST(su.get()!=NULL, "ERROR: subset on solution u yields NULL vector in ColumnSolver::solve");

          d_Solvers[i]->solve(sf, su);
        }	
      }
    }

  void
    ColumnSolver::SymmetricGaussSeidel(boost::shared_ptr<AMP::LinearAlgebra::Vector>  &f, boost::shared_ptr<AMP::LinearAlgebra::Vector>  &u)
    {
      for(int it=0;it<d_iMaxIterations; it++)
      {
        for(unsigned int i=0;i<d_Solvers.size(); i++)
        {
          boost::shared_ptr<AMP::Operator::Operator> op = d_Solvers[i]->getOperator();
          AMP_INSIST(op.get()!=NULL, "EROR: NULL Operator returned by SolverStrategy::getOperator");

          //boost::shared_ptr<AMP::Operator::LinearOperator> linearOperator = boost::dynamic_pointer_cast<LinearOperator>(op);
          //AMP_INSIST(AMP::Operator::linearOperator.get()!=NULL, "ERROR: NULL LinearOperator returned by cast in ColumnSolver");
          //Variable::shared_ptr inputVar = linearOperator->getInputVariable();
          //Variable::shared_ptr outputVar = linearOperator->getOutputVariable();

          AMP::LinearAlgebra::Variable::shared_ptr inputVar = op->getInputVariable();
          AMP::LinearAlgebra::Variable::shared_ptr outputVar = op->getOutputVariable();

          AMP_INSIST(inputVar.get()!=NULL, "ERROR: Null input variable for linear operator");
          AMP_INSIST(outputVar.get()!=NULL, "ERROR: Null output variable for linear operator");

          boost::shared_ptr<AMP::LinearAlgebra::Vector> sf = f->subsetVectorForVariable(outputVar);
          AMP_INSIST(sf.get()!=NULL, "ERROR: subset on rhs f yields NULL vector in ColumnSolver::solve");
          boost::shared_ptr<AMP::LinearAlgebra::Vector> su = u->subsetVectorForVariable(inputVar);
          AMP_INSIST(su.get()!=NULL, "ERROR: subset on solution u yields NULL vector in ColumnSolver::solve");

          d_Solvers[i]->solve(sf, su);
        }	

        for(int i=(int)d_Solvers.size()-1; i>=0;i--)
        {
          boost::shared_ptr<AMP::Operator::Operator> op = d_Solvers[i]->getOperator();
          AMP_INSIST(op.get()!=NULL, "EROR: NULL Operator returned by SolverStrategy::getOperator");

          //boost::shared_ptr<AMP::Operator::LinearOperator> linearOperator = boost::dynamic_pointer_cast<LinearOperator>(op);
          //AMP_INSIST(AMP::Operator::linearOperator.get()!=NULL, "ERROR: NULL LinearOperator returned by cast in ColumnSolver");
          //Variable::shared_ptr inputVar = linearOperator->getInputVariable();
          //Variable::shared_ptr outputVar = linearOperator->getOutputVariable();

          AMP::LinearAlgebra::Variable::shared_ptr inputVar = op->getInputVariable();
          AMP::LinearAlgebra::Variable::shared_ptr outputVar = op->getOutputVariable();

          AMP_INSIST(inputVar.get()!=NULL, "ERROR: Null input variable for linear operator");
          AMP_INSIST(outputVar.get()!=NULL, "ERROR: Null output variable for linear operator");

          boost::shared_ptr<AMP::LinearAlgebra::Vector> sf = f->subsetVectorForVariable(outputVar);
          AMP_INSIST(sf.get()!=NULL, "ERROR: subset on rhs f yields NULL vector in ColumnSolver::solve");
          boost::shared_ptr<AMP::LinearAlgebra::Vector> su = u->subsetVectorForVariable(inputVar);
          AMP_INSIST(su.get()!=NULL, "ERROR: subset on solution u yields NULL vector in ColumnSolver::solve");

          d_Solvers[i]->solve(sf, su);
        }
      }
    }

  void
    ColumnSolver::setInitialGuess( boost::shared_ptr<AMP::LinearAlgebra::Vector> initialGuess )
    {
      for(unsigned int i=0;i<d_Solvers.size(); i++)
      {
        d_Solvers[i]->setInitialGuess(initialGuess);
      }
    }

  void
    ColumnSolver::append(boost::shared_ptr<AMP::Solver::SolverStrategy> solver)
    {
      AMP_INSIST( (solver.get() != NULL), "AMP::Solver::ColumnSolver::append input argument is a NULL solver");

      d_Solvers.push_back(solver);
    }

  void
    ColumnSolver::resetOperator(const boost::shared_ptr<AMP::Operator::OperatorParameters> params)
    {

      if(d_resetColumnOperator) {
        d_pOperator->reset(params);

        boost::shared_ptr<SolverStrategyParameters> solverParams;

        for(unsigned int i = 0; i < d_Solvers.size(); i++) {
          d_Solvers[i]->reset(solverParams);
        }
      } else {
        boost::shared_ptr<AMP::Operator::ColumnOperatorParameters> columnParams = boost::dynamic_pointer_cast<AMP::Operator::ColumnOperatorParameters>(params);
        AMP_INSIST(columnParams.get() != NULL, "Dynamic cast failed!");

        for(unsigned int i = 0; i < d_Solvers.size(); i++) {
          d_Solvers[i]->resetOperator((columnParams->d_OperatorParameters)[i]);
        }
      }

    }

}
}

