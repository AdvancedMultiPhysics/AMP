#ifndef included_AMP_SubchannelToPointMapMap
#define included_AMP_SubchannelToPointMapMap

#include "AMP/mesh/Mesh.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/subchannel/SubchannelPhysicsModel.h"
#include "AMP/operators/subchannel/SubchannelToPointMapParameters.h"
#include "AMP/utils/AMP_MPI.h"

namespace AMP::Operator {


/**
 * \class SubchannelToPointMap
 * \brief A class used to map subchannel properties to points
 * \details  This class provides routines for mapping the subchannel flow
 *     properties (Density, Temperature) to a set of points provided.
 */
class SubchannelToPointMap : public AMP::Operator::Operator
{
public:
    //! Default constructor
    explicit SubchannelToPointMap( std::shared_ptr<const SubchannelToPointMapParameters> params );

    //! Deconstructor
    virtual ~SubchannelToPointMap() {}

    std::string type() const override { return "SubchannelToPointMap"; }

    /**
     * \brief Perform the map
     * \details  This performs the map of the output propertiy (Density or Temperature) to the given
     * points.
     * \param u     Vector containing the subchannel solution (may be a multivector containing other
     * variables)
     * \param f     Vector to fill the densities.  The local size must match the number of points
     * set
     *              by the call to setPoints.
     */
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return std::shared_ptr<AMP::LinearAlgebra::Variable>(
            new AMP::LinearAlgebra::Variable( "Flow" ) );
    }


    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_outputVar;
    }


private:
    AMP_MPI d_comm;
    std::vector<double> d_point_x, d_point_y, d_point_z;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outputVar;

    // Create the subchannel grid for all processors
    size_t N_subchannels;
    std::vector<double> d_subchannel_x; // x-coordinates of the center of the subchannel
    std::vector<double> d_subchannel_y; // y-coordinates of the center of the subchannel
    std::vector<double> d_subchannel_z; // z-coordinates of the z-faces of the subchannel
    void createGrid();

    std::shared_ptr<SubchannelPhysicsModel> d_subchannelPhysicsModel;
};
} // namespace AMP::Operator

#endif
