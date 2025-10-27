
#ifndef included_AMP_SubchannelTwoEqLinearOperator
#define included_AMP_SubchannelTwoEqLinearOperator

#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/subchannel/SubchannelOperatorParameters.h"
#include "AMP/operators/subchannel/SubchannelPhysicsModel.h"


namespace AMP::Operator {


/**
  Linear operator class for the 2-equation {enthalpy, pressure} formulation of the subchannel
  equations:
  see /AMPFuel-Docs/technicalInfo/flow/SubchannelFlow.pdf for details
*/
class SubchannelTwoEqLinearOperator : public LinearOperator
{
public:
    typedef std::unique_ptr<AMP::Mesh::MeshElement> ElementPtr;

    //! Constructor
    explicit SubchannelTwoEqLinearOperator( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~SubchannelTwoEqLinearOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "SubchannelTwoEqLinearOperator"; }

    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    //! Sets frozen vector
    void setFrozenVector( AMP::LinearAlgebra::Vector::shared_ptr frozenVec )
    {
        d_frozenVec = frozenVec;
    }

    //! Get the current operator parameters
    auto getParams() { return d_params; }

protected:
    std::shared_ptr<const SubchannelOperatorParameters> d_params;

    // subchannel physics model
    std::shared_ptr<SubchannelPhysicsModel> d_subchannelPhysicsModel;

    // frozen vector
    AMP::LinearAlgebra::Vector::shared_ptr d_frozenVec;

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

    //    static const double d_machinePrecision = 1.0e-15; // machine precision; used in
    //    perturbation for derivatives
    const double d_machinePrecision; // static const double is not allowed in iso c++11
                                     // either remove static or use std::numeric_limits<T>
                                     // by qdi june 14

    // Derivative of enthalpy with respect to pressure
    double dhdp( double, double );

    // Derivative of specific volume with respect to enthalpy
    double dvdh( double, double );

    // Derivative of specific volume with respect to pressure
    double dvdp( double, double );

    // Friction function
    double friction( double h_minus,
                     double p_minus,
                     double h_plus,
                     double p_plus,
                     double mass,
                     double A,
                     double D );

    // Derivatives of friction with respect to lower and upper enthalpy and pressure
    double dfdh_lower( double h_minus,
                       double p_minus,
                       double h_plus,
                       double p_plus,
                       double mass,
                       double A,
                       double D );
    double dfdh_upper( double h_minus,
                       double p_minus,
                       double h_plus,
                       double p_plus,
                       double mass,
                       double A,
                       double D );
    double dfdp_lower( double h_minus,
                       double p_minus,
                       double h_plus,
                       double p_plus,
                       double mass,
                       double A,
                       double D );
    double dfdp_upper( double h_minus,
                       double p_minus,
                       double h_plus,
                       double p_plus,
                       double mass,
                       double A,
                       double D );

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
