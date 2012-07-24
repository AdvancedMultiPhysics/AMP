
#ifndef included_AMP_SubchannelTwoEqLinearOperator
#define included_AMP_SubchannelTwoEqLinearOperator

#include "operators/Operator.h"
#include "SubchannelOperatorParameters.h"

namespace AMP {
namespace Operator {

  /**
    Linear operator class for the 2-equation {enthalpy, pressure} formulation of the subchannel equations:
    see /AMPFuel-Docs/technicalInfo/flow/SubchannelFlow.pdf for details
    */
  class SubchannelTwoEqLinearOperator : public Operator
  {
    public :

      /**
        Constructor
        */
      SubchannelTwoEqLinearOperator(const boost::shared_ptr<SubchannelOperatorParameters> & params)
        : Operator (params)
      {
        AMP_INSIST( params->d_db->keyExists("InputVariable"), "Key 'InputVariable' does not exist");
        std::string inpVar = params->d_db->getString("InputVariable");
        d_inpVariable.reset(new AMP::LinearAlgebra::Variable(inpVar));

        AMP_INSIST( params->d_db->keyExists("OutputVariable"), "Key 'OutputVariable' does not exist");
        std::string outVar = params->d_db->getString("OutputVariable");
        d_outVariable.reset(new AMP::LinearAlgebra::Variable(outVar));

        reset(params);
      }

      /**
        Destructor
        */
      ~SubchannelTwoEqLinearOperator() { }

      /**
        For this operator we have an in-place apply.
        @param [in]  f auxillary/rhs vector. 
        @param [in]  u input vector. 
        @param [out] r residual/output vector. 
        @param [in]  a first constant used in the expression: r = a*A(u) + b*f. The default value is -1.
        @param [in]  b second constant used in the expression: r = a*A(u) + b*f. The default value is 1.
        */
      void apply(const AMP::LinearAlgebra::Vector::shared_ptr &f, const AMP::LinearAlgebra::Vector::shared_ptr &u,
          AMP::LinearAlgebra::Vector::shared_ptr  &r, const double a = -1.0, const double b = 1.0);

      void reset(const boost::shared_ptr<OperatorParameters>& params);

      AMP::LinearAlgebra::Variable::shared_ptr createInputVariable(const std::string & name, int varId = -1) {
          (void) varId;
          return d_inpVariable->cloneVariable(name);
      }

      AMP::LinearAlgebra::Variable::shared_ptr createOutputVariable(const std::string & name, int varId = -1) {
          (void) varId;
          return d_outVariable->cloneVariable(name);
      }

      AMP::LinearAlgebra::Variable::shared_ptr getInputVariable() {
        return d_inpVariable;
      }

      AMP::LinearAlgebra::Variable::shared_ptr getOutputVariable() {
        return d_outVariable;
      }

      virtual AMP::LinearAlgebra::Vector::shared_ptr subsetOutputVector(AMP::LinearAlgebra::Vector::shared_ptr vec);

      virtual AMP::LinearAlgebra::Vector::shared_ptr subsetInputVector(AMP::LinearAlgebra::Vector::shared_ptr vec);

      /**
        Sets frozen vector
        */
      void setFrozenVector(AMP::LinearAlgebra::Vector::shared_ptr &frozenVec) {
        d_frozenVec = frozenVec;
      }

      // frozen vector
      AMP::LinearAlgebra::Vector::shared_ptr d_frozenVec;

      double **d_Jacobian; // Jacobian matrix for printing

    protected:

      // subchannel physics model
      boost::shared_ptr<SubchannelPhysicsModel> d_subchannelPhysicsModel;

    private :

      /**
        Function used in reset to get double parameter or use default if missing
        */
      double getDoubleParameter(boost::shared_ptr<SubchannelOperatorParameters>, std::string, double);

      /**
        Function used in reset to get double parameter or use default if missing
        */
      std::string getStringParameter(boost::shared_ptr<SubchannelOperatorParameters>, std::string, std::string);

      boost::shared_ptr<AMP::LinearAlgebra::Variable> d_inpVariable;

      boost::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable;

      double d_Pout;     // exit pressure [Pa]
      double d_Tin;      // inlet temperature [K]
      double d_m;        // inlet mass flow rate [kg/s]
      double d_gamma;    // fission heating coefficient
      double d_theta;    // channel angle [rad]
      double d_friction; // friction factor
      double d_pitch;    // lattice pitch [m]
      double d_diameter; // fuel rod diameter [m]
      double d_K;        // form loss coefficient
      double d_Q;        // rod power
      std::string d_source; // heat source type
      double d_printJacobian; // option to print Jacobian

      unsigned int d_solutionSize; // size of solution vector

      static const double d_machinePrecision = 1.0e-15; // machine precision; used in perturbation for derivatives

      /**
        Derivative of enthalpy with respect to pressure
        */
      double dhdp(double,double);

      /**
        Derivative of specific volume with respect to enthalpy
        */
      double dvdh(double,double);

      /**
        Derivative of specific volume with respect to pressure
        */
      double dvdp(double,double);

  };

}
}

#endif

