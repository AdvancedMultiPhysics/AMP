#ifndef included_AMP_MapSurface
#define included_AMP_MapSurface

#include "utils/shared_ptr.h"

#include "operators/Operator.h"
#include "operators/OperatorParameters.h"
#include "operators/map/Map3Dto1D.h"
#include "operators/map/Map1Dto3D.h"
#include "vectors/Vector.h"
#include "vectors/Variable.h"
#include "vectors/SimpleVector.h"

#include <string>

#ifdef DEBUG_CHECK_ASSERTIONS
#include <cassert>
#endif


namespace AMP {
namespace Operator {
 
class MapSurface : public MapOperator
{
public :

    MapSurface(const AMP::shared_ptr<OperatorParameters> & params);
    virtual ~MapSurface() { }

    virtual void apply(AMP::LinearAlgebra::Vector::const_shared_ptr f, AMP::LinearAlgebra::Vector::const_shared_ptr u,
            AMP::LinearAlgebra::Vector::shared_ptr r, const double a = -1.0, const double b = 1.0);

    AMP::shared_ptr<AMP::LinearAlgebra::Vector> getBoundaryVector(const AMP::LinearAlgebra::Vector::shared_ptr &u) {
        return (u->subsetVectorForVariable(d_outVariable)) ;
    }

    AMP::LinearAlgebra::Variable::shared_ptr getInputVariable() {
        return d_inpVariable;
    }

    AMP::LinearAlgebra::Variable::shared_ptr getOutputVariable() {
        return d_outVariable;
    }

    void setVector(AMP::LinearAlgebra::Vector::shared_ptr scratchVec)
    {
        outVec = scratchVec;
        mapTarget->setVector(outVec);
    }

    /*
    AMP::shared_ptr<OperatorParameters> getJacobianParameters(const AMP::shared_ptr<AMP::LinearAlgebra::Vector>& )
    {
        AMP::shared_ptr<AMP::InputDatabase> tmp_db (new AMP::InputDatabase("Dummy"));

        AMP::shared_ptr<MapOperatorParameters> outParams(new MapOperatorParameters(tmp_db));

        outParams->d_BoundaryId = (mapMasterParams->d_db)->getInteger("BoundaryId");

        return outParams;

    } */

protected :

    AMP::shared_ptr<AMP::LinearAlgebra::Vector> gap1DVec; 
    AMP::shared_ptr<AMP::LinearAlgebra::Variable> gapVariable;

    AMP::shared_ptr<AMP::LinearAlgebra::Variable> d_inpVariable;
    AMP::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable;

    AMP::LinearAlgebra::Vector::const_shared_ptr inpVec; 
    AMP::LinearAlgebra::Vector::shared_ptr outVec;

private :

    AMP::shared_ptr<Map3Dto1D> mapMaster;
    AMP::shared_ptr<MapOperatorParameters> mapMasterParams;
    AMP::shared_ptr<Map1Dto3D> mapTarget;
    AMP::shared_ptr<MapOperatorParameters> mapTargetParams;
};


}
}

#endif
