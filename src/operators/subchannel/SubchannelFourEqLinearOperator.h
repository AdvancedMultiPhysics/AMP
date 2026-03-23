
#ifndef included_AMP_SubchannelFourEqLinearOperator
#define included_AMP_SubchannelFourEqLinearOperator

#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/subchannel/SubchannelOperatorParameters.h"


namespace AMP::Operator {

/**
  Lnear operator class for the 4-equation formulation of the subchannel equations:
  see /AMPFuel-Docs/technicalInfo/flow/SubchannelFlow.pdf for details
  */
class SubchannelFourEqLinearOperator : public LinearOperator
{
public:
    typedef std::unique_ptr<AMP::Mesh::MeshElement> ElementPtr;

    //! Constructor
    explicit SubchannelFourEqLinearOperator( std::shared_ptr<const OperatorParameters> params );

    //! Destructor
    virtual ~SubchannelFourEqLinearOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "SubchannelFourEqLinearOperator"; }

    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    std::shared_ptr<AMP::LinearAlgebra::VectorSelector> selectOutputVector() const override;

    std::shared_ptr<AMP::LinearAlgebra::VectorSelector> selectInputVector() const override;

    //! Sets frozen solution vector
    void setFrozenVector( AMP::LinearAlgebra::Vector::shared_ptr frozenVec )
    {
        d_frozenVec = frozenVec;
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

    //! Makes map of lateral gaps to their centroids
    void getLateralFaces( std::shared_ptr<AMP::Mesh::Mesh>,
                          std::map<AMP::Mesh::Point, ElementPtr> &,
                          std::map<AMP::Mesh::Point, ElementPtr> & );

    //! Makes map of gap widths to their xy positions
    std::map<AMP::Mesh::Point, double> getGapWidths( std::shared_ptr<AMP::Mesh::Mesh>,
                                                     const std::vector<double> &,
                                                     const std::vector<double> &,
                                                     const std::vector<double> & );

    void getAxialFaces( const AMP::Mesh::MeshElement &, ElementPtr &, ElementPtr & );

    void fillSubchannelGrid( std::shared_ptr<AMP::Mesh::Mesh> ); // function to fill the subchannel
                                                                 // data for all processors

    int getSubchannelIndex( double x,
                            double y ); // function to give unique index for each subchannel

protected:
    //! Gets parameters from nonlinear operator for use in linear operator
    std::shared_ptr<OperatorParameters>
    getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u ) override;

    std::shared_ptr<const SubchannelOperatorParameters> d_params;

    std::shared_ptr<SubchannelPhysicsModel> d_subchannelPhysicsModel;

private:
    bool d_initialized;

    std::shared_ptr<AMP::LinearAlgebra::Vector> d_cladTemperature;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_frozenVec;

    bool d_forceNoConduction; // option to force conduction terms to zero; used for testing
    bool d_forceNoTurbulence; // option to force turbulence terms to zero; used for testing
    bool d_forceNoHeatSource; // option to force heat source terms to zero; used for testing
    bool
        d_forceNoFriction; // option to force friction and form loss terms to zero; used for testing

    double d_Pout;           // exit pressure [Pa]
    double d_Tin;            // inlet temperature [K]
    double d_mass;           // inlet global mass flow rate [kg/s]
    double d_win;            // inlet lateral mass flow rate [kg/s]
    double d_gamma;          // fission heating coefficient
    double d_theta;          // channel angle [rad]
    double d_turbulenceCoef; // proportionality constant relating turbulent momentum to turbulent
                             // energy transport
    double d_reynolds;       // reynolds number
    double d_prandtl;        // prandtl number
    double d_KG;             // lateral form loss coefficient

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

    std::string d_source;            // heat source type
    std::string d_heatShape;         // heat shape used if heat source type is "totalHeatGeneration"
    double d_Q;                      // (sum of rod powers)/4 for each subchannel
    std::vector<double> d_QFraction; // fraction of max rod power in each subchannel

    std::vector<double> d_x, d_y, d_z;
    std::vector<bool> d_ownSubChannel; // does this processor own this subchannel (multiple
                                       // processors may own a subchannel)?
    std::vector<std::vector<ElementPtr>> d_subchannelElem; // List of elements in each subchannel
    std::vector<std::vector<ElementPtr>>
        d_subchannelFace;    // List of z-face elements in each subchannel
    size_t d_numSubchannels; // number of subchannels

    double Volume( double, double );           // evaluates specific volume
    double Temperature( double, double );      // evaluates temperature
    double DynamicViscosity( double, double ); // evaluates dynamic viscosity

    ElementPtr getAxiallyAdjacentLateralFace( const AMP::Mesh::MeshElement *,
                                              const AMP::Mesh::MeshElement &,
                                              const std::map<AMP::Mesh::Point, ElementPtr> & );
};
} // namespace AMP::Operator

#endif
