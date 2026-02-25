
#ifndef included_AMP_SubchannelParameters
#define included_AMP_SubchannelParameters

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/boundary/libmesh/RobinPhysicsModel.h"
#include "AMP/operators/subchannel/SubchannelPhysicsModel.h"
#include "AMP/vectors/Vector.h"

namespace AMP::Operator {

//! Parameter class to provide parameters to all subchannel classes
class SubchannelOperatorParameters : public OperatorParameters
{
public:
    //! Constructor
    explicit SubchannelOperatorParameters( std::shared_ptr<AMP::Database> db )
        : OperatorParameters( db ), d_initialize( false )
    {
    }

    //! Destructor
    virtual ~SubchannelOperatorParameters() {}

    // pointer to subchannel physics model
    std::shared_ptr<SubchannelPhysicsModel> d_subchannelPhysicsModel;

    std::shared_ptr<RobinPhysicsModel> d_dittusBoelterCoefficient;

    AMP::LinearAlgebra::Vector::shared_ptr d_frozenSolution;

    bool d_initialize; // Do we want to initialize the matrix

    // Clad properties
    std::vector<double> clad_x, clad_y, clad_d;
};
} // namespace AMP::Operator

#endif
