
#ifndef included_AMP_GaussPointToGaussPointMap
#define included_AMP_GaussPointToGaussPointMap

#include "AMP/operators/map/NodeToNodeMap.h"

#include <vector>

namespace AMP::Operator {

class GaussPointToGaussPointMap : public NodeToNodeMap
{
public:
    explicit GaussPointToGaussPointMap(
        std::shared_ptr<const AMP::Operator::OperatorParameters> params );

    virtual void applyStart( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    virtual void applyFinish( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                              AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    static bool validMapType( const std::string &t );

    void setFrozenInputVector( AMP::LinearAlgebra::Vector::shared_ptr u ) { d_frozenInputVec = u; }

    virtual ~GaussPointToGaussPointMap() {}

    std::string type() const override { return "GaussPointToGaussPointMap"; }

protected:
    void createIdxMap( std::shared_ptr<const AMP::Operator::OperatorParameters> params );

    void correctLocalOrdering();

    bool d_useFrozenInputVec;

    AMP::LinearAlgebra::Vector::shared_ptr d_frozenInputVec;

    std::vector<std::vector<unsigned int>> d_idxMap;
};
} // namespace AMP::Operator

#endif
