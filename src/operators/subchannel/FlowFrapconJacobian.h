
#ifndef included_AMP_FlowFrapconJacobian
#define included_AMP_FlowFrapconJacobian

#include "AMP/operators/Operator.h"
#include "AMP/operators/subchannel/FlowFrapconJacobianParameters.h"

// Libmesh files
DISABLE_WARNINGS
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/quadrature.h"
ENABLE_WARNINGS

namespace AMP::Operator {

/**
  A class to represent Frapcon Flow operator. This operator acts as a heat sink and
  should be used to compute the 1D flow temperature in the axial direction of the pellet/clad.
  */
class FlowFrapconJacobian : public Operator
{
public:
    /**
      Constructor creates a simpleVariables for Input and Output. The reset is called to
      read the flow parameters.
      */
    explicit FlowFrapconJacobian( std::shared_ptr<const OperatorParameters> params );

    /**
      Destructor
      */
    virtual ~FlowFrapconJacobian() {}

    //! Return the name of the operator
    std::string type() const override { return "FlowFrapconJacobian"; }

    /**
      For this operator we have an in-place apply.
      @param [in]  u input vector.
      @param [out] r residual/output vector.
      */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    /**
      This function reads the entries of the database for the flow operator
      and can also be used to change the parameters if required.
      */
    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    /*
          static bool sort_nodes_in_z ( const libMesh::Node *one , const libMesh::Node *two ) {
            return (*one)(2) < (*two)(2);
          }
    */
    std::shared_ptr<AMP::LinearAlgebra::Variable> createInputVariable( const std::string &name,
                                                                       int varId = -1 );

    std::shared_ptr<AMP::LinearAlgebra::Variable> createOutputVariable( const std::string &name,
                                                                        int varId = -1 );

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override;

    void setInputVariableName( const std::string &name, int varId = -1 );

    void setOutputVariableName( const std::string &name, int varId = -1 );

    std::shared_ptr<AMP::LinearAlgebra::VectorSelector> selectOutputVector() const override;

    std::shared_ptr<AMP::LinearAlgebra::VectorSelector> selectInputVector() const override;

    /**
      @param [in] zloc is the location vector in z direction.
      */
    void setZLocations( const std::vector<double> &zloc ) { zPoints = zloc; }

    const auto &getZLocations() { return zPoints; }

    double getCp() { return Cp; }

    double getdCp() { return dCp; }

    void setVector( AMP::LinearAlgebra::Vector::shared_ptr frozenVec ) { d_cladVec = frozenVec; }

    void setFrozenVector( AMP::LinearAlgebra::Vector::shared_ptr frozenVec )
    {
        d_frozenVec = frozenVec;
    }

    double getHeatCapacity( double T_b )
    {
        double cp = 0.0;
        if ( T_b < 544 ) {
            cp = 2.4e5;
        } else if ( 544 <= T_b && T_b < 588 ) {
            cp = 2.4e5 * ( 1.0 + 2.9e-3 * ( 1.8 * T_b - 1031.0 ) );
        } else {
            cp = 2.4e5 * ( 1.0 + 2.9e-3 * ( 1.8 * T_b - 979.4 ) );
        }
        return cp;
    }

    double getHeatCapacityGradient( double T_b )
    {
        double dcp = 0.0;
        if ( T_b >= 544 ) {
            dcp = 2.4e5 * ( 2.9e-3 * 1.8 * T_b );
        }
        return dcp;
    }

    /*
       std::shared_ptr< std::vector<double>  > getFlowSolution() {
       return flowSolutionVector;
       }
       */


public: // Data members (this should be private)
    AMP::LinearAlgebra::Vector::shared_ptr d_cladVec;
    AMP::LinearAlgebra::Vector::shared_ptr d_frozenVec;

    AMP::LinearAlgebra::Vector::shared_ptr d_localCladVec;
    AMP::LinearAlgebra::Vector::shared_ptr d_localFrozenVec;

    int d_numpoints = 0; //! Number of points in z direction
    std::vector<unsigned int> d_dofIds;
    std::vector<double> zPoints; //! vector to hold z locations

    double d_De  = 0; //! Channel Diameter
    double Cp    = 0; //! Heat Capacity of Coolant
    double dCp   = 0; //! Gradient of Heat Capacity
    double d_G   = 0; //! Coolant Mass Flux
    double d_Tin = 0; //! Coolant Temp Tin
    double d_K   = 0; //! Coolant conductivity
    double d_Re  = 0; //! Reynolds Number
    double d_Pr  = 0; //! Prandtl Number

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_inpVariable;    //! Input Variable
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable;    //! Output Variable
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_SimpleVariable; //! Simple Input Variable

    AMP::LinearAlgebra::Vector::shared_ptr flowInput;
    AMP::LinearAlgebra::Vector::shared_ptr flowOutput;

protected:
private:
};
} // namespace AMP::Operator

#endif
