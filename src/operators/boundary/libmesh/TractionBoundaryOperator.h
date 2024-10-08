
#ifndef included_AMP_TractionBoundaryOperator
#define included_AMP_TractionBoundaryOperator

#include "AMP/operators/boundary/BoundaryOperator.h"
#include "AMP/operators/boundary/libmesh/TractionBoundaryOperatorParameters.h"

namespace AMP::Operator {

class TractionBoundaryOperator : public BoundaryOperator
{
public:
    explicit TractionBoundaryOperator(
        std::shared_ptr<const TractionBoundaryOperatorParameters> params );

    virtual ~TractionBoundaryOperator() {}

    std::string type() const override { return "TractionBoundaryOperator"; }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    void addRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs ) override;

protected:
    AMP::LinearAlgebra::Vector::shared_ptr
    mySubsetVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                    std::shared_ptr<AMP::LinearAlgebra::Variable> var );

    void computeCorrection();

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_var;
    AMP::LinearAlgebra::Vector::shared_ptr d_correction;
    std::vector<double> d_traction;
    std::vector<double> d_volumeElements;
    std::vector<unsigned int> d_sideNumbers;
    std::vector<AMP::Mesh::MeshElementID> d_nodeID;
    bool d_residualMode;
};
} // namespace AMP::Operator

#endif
