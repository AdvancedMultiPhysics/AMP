#ifndef included_AMP_TractionBoundaryOperatorParameters
#define included_AMP_TractionBoundaryOperatorParameters

#include "AMP/mesh/Mesh.h"
#include "AMP/operators/boundary/BoundaryOperatorParameters.h"

namespace AMP::Operator {

class TractionBoundaryOperatorParameters : public BoundaryOperatorParameters
{
public:
    explicit TractionBoundaryOperatorParameters( std::shared_ptr<AMP::Database> db )
        : BoundaryOperatorParameters( db )
    {
    }

    virtual ~TractionBoundaryOperatorParameters() {}

    std::vector<double> d_traction;
    std::vector<double> d_volumeElements;
    std::vector<unsigned int> d_sideNumbers;
    std::vector<AMP::Mesh::MeshElementID> d_nodeID;
};
} // namespace AMP::Operator

#endif
