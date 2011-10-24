#ifndef included_LinearTimeOperator
#define included_LinearTimeOperator

#ifndef included_AMP_config

#endif

#include "operators/LinearOperator.h"

#include "boost/shared_ptr.hpp"
#include "vectors/Vector.h"
#include "utils/Utilities.h"
#include "operators/OperatorParameters.h"
#include "operators/VolumeIntegralOperator.h"

namespace AMP{
namespace TimeIntegrator{

/*!
  @brief base class for operator class associated with ImplicitTimeIntegrator

  Class ImplicitLinearTimeOperator is a base class derived from Operator. It
  is the operator class associated with a ImplicitTimeIntegrator. The solver associated
  with the ImplicitTimeIntegrator will register this object.

  @see ImplicitTimeIntegrator
  @see Operator
  @see SolverStrategy
*/
class LinearTimeOperator: public AMP::Operator::LinearOperator
{
 public:

  LinearTimeOperator(boost::shared_ptr<AMP::Operator::OperatorParameters > params);
  virtual ~LinearTimeOperator();
     
  /**
   * This function is useful for re-initializing an operator
   * \param params
   *        parameter object containing parameters to change
   */
  virtual void reset(const boost::shared_ptr<AMP::Operator::OperatorParameters>& params);

  void registerRhsOperator(boost::shared_ptr< AMP::Operator::LinearOperator > op) {d_pRhsOperator = op; }
  void registerMassOperator(boost::shared_ptr< AMP::Operator::LinearOperator > op) {d_pMassOperator = op; }

  boost::shared_ptr< Operator > getRhsOperator(void){ return d_pRhsOperator; }
  boost::shared_ptr< Operator > getMassOperator(void){ return d_pMassOperator; }

  void setPreviousSolution(boost::shared_ptr<AMP::LinearAlgebra::Vector> previousSolution){ d_pPreviousTimeSolution = previousSolution; }
  
  void setDt(double dt) {d_dCurrentDt = dt; }

  // added by JL
  void setScalingFactor(double scalingFactor) {d_dScalingFactor = scalingFactor;}

  boost::shared_ptr<AMP::Operator::OperatorParameters> getJacobianParameters(const boost::shared_ptr<AMP::LinearAlgebra::Vector>& u); 

  // added by JL //correction by RS
  AMP::LinearAlgebra::Variable::shared_ptr getInputVariable(int varId = -1){ return d_pRhsOperator->getInputVariable(varId); }

  /**
   * returns a Variable object corresponding to the rhs operator
   */
  AMP::LinearAlgebra::Variable::shared_ptr getOutputVariable(){return d_pRhsOperator->getOutputVariable(); }
  //JL
  void registerCurrentTime( double currentTime ) {d_current_time = currentTime;}
 protected:
  LinearTimeOperator();

  void getFromInput(const boost::shared_ptr<AMP::Database> &db);

  bool d_bModifyRhsOperatorMatrix;

  /**
   * set to true if this operator corresponds to an algebraic component
   */
  bool d_bAlgebraicComponent;
  
  double d_dScalingFactor;
  double d_dCurrentDt;
  
  boost::shared_ptr< AMP::Operator::LinearOperator > d_pRhsOperator;

  boost::shared_ptr< AMP::Operator::LinearOperator > d_pMassOperator;

  boost::shared_ptr<AMP::LinearAlgebra::Vector>  d_pPreviousTimeSolution;
  
  boost::shared_ptr<AMP::LinearAlgebra::Vector>  d_pScratchVector;

  double d_current_time;
  double d_beta;

 private:

  
};

}
}

#endif
