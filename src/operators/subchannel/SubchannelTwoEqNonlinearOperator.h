
#ifndef included_AMP_SubchannelTwoEqNonlinearOperator
#define included_AMP_SubchannelTwoEqNonlinearOperator

#include "AMP/operators/Operator.h"
#include "AMP/operators/subchannel/SubchannelOperatorParameters.h"
#include "AMP/operators/subchannel/SubchannelPhysicsModel.h"


namespace AMP::Operator {

/**
  Nonlinear operator class for the 2-equation {enthalpy, pressure} formulation of the subchannel
  equations:
  see /AMPFuel-Docs/technicalInfo/flow/SubchannelFlow.pdf for details
  */
class SubchannelTwoEqNonlinearOperator : public Operator
{
public:
    typedef std::unique_ptr<AMP::Mesh::MeshElement> ElementPtr;

    //! Constructor
    explicit SubchannelTwoEqNonlinearOperator( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~SubchannelTwoEqNonlinearOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "SubchannelTwoEqNonlinearOperator"; }

    /**
      For this operator we have an in-place apply.
      @param [in]  u input vector.
      @param [out] f output vector.
     */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_inpVariable;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_outVariable;
    }

    void setVector( AMP::LinearAlgebra::Vector::shared_ptr frozenVec )
    {
        d_cladTemperature = frozenVec;
    }

    //! Get the element physics model
    std::shared_ptr<SubchannelPhysicsModel> getSubchannelPhysicsModel()
    {
        return d_subchannelPhysicsModel;
    }

    //! Get the Inlet Temperature [K]
    double getInletTemperature() { return d_Tin; }

    //! Get the Outlet Pressure [Pa]
    double getOutletPressure() { return d_Pout; }

    //! Get the current operator parameters
    auto getParams() { return d_params; }

protected:
    //! Gets parameters from nonlinear operator for use in linear operator
    std::shared_ptr<OperatorParameters>
    getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u ) override;

    std::shared_ptr<const SubchannelOperatorParameters> d_params;

    std::shared_ptr<SubchannelPhysicsModel> d_subchannelPhysicsModel;

private:
    bool d_initialized;

    // Function used in reset to get double parameter or use default if missing
    double
    getDoubleParameter( std::shared_ptr<const SubchannelOperatorParameters>, std::string, double );

    // Function used in reset to get integer parameter or use default if missing
    int
    getIntegerParameter( std::shared_ptr<const SubchannelOperatorParameters>, std::string, int );

    // Function used in reset to get double parameter or use default if missing
    std::string getStringParameter( std::shared_ptr<const SubchannelOperatorParameters>,
                                    std::string,
                                    std::string );

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_inpVariable;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_cladTemperature;

    double d_Pout;     // exit pressure [Pa]
    double d_Tin;      // inlet temperature [K]
    double d_mass;     // inlet global mass flow rate [kg/s]
    double d_gamma;    // fission heating coefficient
    double d_theta;    // channel angle [rad]
    double d_Q;        // rod power
    double d_reynolds; // reynolds number
    double d_prandtl;  // prandtl number

    std::string d_frictionModel; // friction model
    double d_friction;           // friction factor
    double d_roughness;          // surface roughness [m]

    std::vector<double> d_channelDiam; // Channel hydraulic diameter using the wetted perimeter
    std::vector<double> d_channelArea; // Channel flow area
    std::vector<double> d_rodDiameter; // Average rod diameter for each subchannel
    std::vector<double> d_rodFraction; // Fraction of a rod in each subchannel
    std::vector<double> d_channelMass; // Mass flow rate for each subchannel [kg/s]

    size_t d_NGrid;                 // number of grid spacers
    std::vector<double> d_zMinGrid; // z min positions of each grid spacer
    std::vector<double> d_zMaxGrid; // z max positions of each grid spacer
    std::vector<double> d_lossGrid; // loss coefficients for each grid spacer

    std::string d_source;    // heat source type
    std::string d_heatShape; // heat shape used if heat source type is "totalHeatGeneration"

    std::vector<double> d_x, d_y, d_z;
    std::vector<bool> d_ownSubChannel;                     // Which subchannels do I own
    std::vector<std::vector<ElementPtr>> d_subchannelElem; // List of elements in each subchannel
    std::vector<std::vector<ElementPtr>>
        d_subchannelFace; // List of z-face elements in each subchannel
    int getSubchannelIndex( double x, double y );
    size_t d_numSubchannels;
};
} // namespace AMP::Operator

#endif
