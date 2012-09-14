
#ifndef included_AMP_SubchannelTwoEqLinearOperator
#define included_AMP_SubchannelTwoEqLinearOperator

#include "operators/LinearOperator.h"
#include "operators/subchannel/SubchannelOperatorParameters.h"

namespace AMP {
namespace Operator {

  /**
    Linear operator class for the 2-equation {enthalpy, pressure} formulation of the subchannel equations:
    see /AMPFuel-Docs/technicalInfo/flow/SubchannelFlow.pdf for details
    */
  class SubchannelTwoEqLinearOperator : public LinearOperator
  {
    public :

      /**
        Constructor
        */
      SubchannelTwoEqLinearOperator(const boost::shared_ptr<SubchannelOperatorParameters> & params)
        : LinearOperator (params)
      {
        AMP_INSIST( params->d_db->keyExists("InputVariable"), "Key 'InputVariable' does not exist");
        std::string inpVar = params->d_db->getString("InputVariable");
        d_inpVariable.reset(new AMP::LinearAlgebra::Variable(inpVar));

        AMP_INSIST( params->d_db->keyExists("OutputVariable"), "Key 'OutputVariable' does not exist");
        std::string outVar = params->d_db->getString("OutputVariable");
        d_outVariable.reset(new AMP::LinearAlgebra::Variable(outVar));

        d_dofMap = (params->d_dofMap);

        d_atConstruction = true ;  
        d_nullFrozenvector = true; 
        
        reset(params);
      }

      /**
        Destructor
        */
      ~SubchannelTwoEqLinearOperator() { }

      void reset(const boost::shared_ptr<OperatorParameters>& params);

      AMP::LinearAlgebra::Variable::shared_ptr getInputVariable() {
        return d_inpVariable;
      }

      AMP::LinearAlgebra::Variable::shared_ptr getOutputVariable() {
        return d_outVariable;
      }

      virtual AMP::LinearAlgebra::Vector::shared_ptr subsetOutputVector(AMP::LinearAlgebra::Vector::shared_ptr vec);
      virtual AMP::LinearAlgebra::Vector::const_shared_ptr subsetOutputVector(AMP::LinearAlgebra::Vector::const_shared_ptr vec);

      virtual AMP::LinearAlgebra::Vector::shared_ptr subsetInputVector(AMP::LinearAlgebra::Vector::shared_ptr vec);
      virtual AMP::LinearAlgebra::Vector::const_shared_ptr subsetInputVector(AMP::LinearAlgebra::Vector::const_shared_ptr vec);

      /**
        Sets frozen vector
        */
      void setFrozenVector(AMP::LinearAlgebra::Vector::shared_ptr &frozenVec) {
        d_frozenVec = frozenVec;
      }

      // frozen vector
      AMP::LinearAlgebra::Vector::shared_ptr d_frozenVec;

    protected:

      boost::shared_ptr<AMP::Discretization::DOFManager> d_dofMap;

      // subchannel physics model
      boost::shared_ptr<SubchannelPhysicsModel> d_subchannelPhysicsModel;

    private :

      bool d_atConstruction, d_nullFrozenvector; 
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

      std::vector<double> d_channelFractions;     // channel fraction to calculate effective radius 

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
      double d_channelDia; // Channel diameter
      double d_reynolds  ;
      double d_prandtl   ;
      std::string d_source; // heat source type
      std::string d_heatShape; // heat shape used if heat source type is "totalHeatGeneration"

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

      std::vector<double> d_x, d_y, d_z;
      std::vector<bool> d_ownSubChannel;                      // Which subchannels do I own (multple procs my own a subchannel)
      int getSubchannelIndex( double x, double y );
      void fillSubchannelGrid(AMP::Mesh::Mesh::shared_ptr);   // Function to fill the subchannel data for all processors
      size_t d_numSubchannels; 

  };

}
}

#endif

