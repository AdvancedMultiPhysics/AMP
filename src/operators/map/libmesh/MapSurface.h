#ifndef included_AMP_MapSurface
#define included_AMP_MapSurface

#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/map/libmesh/Map1Dto3D.h"
#include "AMP/operators/map/libmesh/Map3Dto1D.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"

#include <memory>
#include <string>


namespace AMP::Operator {

class MapSurface : public MapOperator
{
public:
    explicit MapSurface( std::shared_ptr<const OperatorParameters> params );
    virtual ~MapSurface() {}

    std::string type() const override { return "MapSurface"; }

    virtual void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                        AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    std::shared_ptr<AMP::LinearAlgebra::Vector>
    getBoundaryVector( const AMP::LinearAlgebra::Vector::shared_ptr &u )
    {
        return ( u->subsetVectorForVariable( d_outVariable ) );
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_inpVariable;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_outVariable;
    }

    void setVector( AMP::LinearAlgebra::Vector::shared_ptr scratchVec )
    {
        outVec = scratchVec;
        mapTarget->setVector( outVec );
    }

protected:
    std::shared_ptr<AMP::LinearAlgebra::Vector> gap1DVec;
    std::shared_ptr<AMP::LinearAlgebra::Variable> gapVariable;

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_inpVariable;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable;

    AMP::LinearAlgebra::Vector::const_shared_ptr inpVec;
    AMP::LinearAlgebra::Vector::shared_ptr outVec;

private:
    std::shared_ptr<Map3Dto1D> mapMaster;
    std::shared_ptr<MapOperatorParameters> mapMasterParams;
    std::shared_ptr<Map1Dto3D> mapTarget;
    std::shared_ptr<MapOperatorParameters> mapTargetParams;
};
} // namespace AMP::Operator

#endif
