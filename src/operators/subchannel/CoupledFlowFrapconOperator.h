#ifndef included_AMP_CoupledFlowFrapconOperator
#define included_AMP_CoupledFlowFrapconOperator

#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/map/libmesh/Map1Dto3D.h"
#include "AMP/operators/map/libmesh/Map3Dto1D.h"
#include "AMP/operators/subchannel/CoupledFlowFrapconOperatorParameters.h"
#include "AMP/operators/subchannel/FlowFrapconOperator.h"
#include "AMP/vectors/Vector.h"

#include <vector>

namespace AMP::Operator {

class CoupledFlowFrapconOperator : public ColumnOperator
{
public:
    explicit CoupledFlowFrapconOperator( std::shared_ptr<const OperatorParameters> params );

    void reset( std::shared_ptr<const OperatorParameters> params ) override
    {
        AMP_ASSERT( params );
        d_operators[2]->reset( params );
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_operators[4]->getOutputVariable();
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_operators[4]->getOutputVariable();
    }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    void append( std::shared_ptr<Operator> op ) override
    {
        AMP_ASSERT( d_operators.size() < 3 );
        AMP_ASSERT( op.get() != NULL );
        d_operators.push_back( op );
    }

    std::shared_ptr<OperatorParameters>
    getParameters( const std::string &type,
                   AMP::LinearAlgebra::Vector::const_shared_ptr u,
                   std::shared_ptr<OperatorParameters> params = NULL ) override
    {
        return ( d_operators[2]->getParameters( type, u, params ) );
    }

    virtual ~CoupledFlowFrapconOperator() {}

    std::string type() const override { return "CoupledFlowFrapconOperator"; }

protected:
private:
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_SimpleVariable; /**< Simple Input Variable */

    int d_numpoints; /**< Number of points in z direction */

    std::vector<double> d_zPoints; /**< vector to hold z locations */
    AMP::LinearAlgebra::Vector::shared_ptr d_flowInput;
    AMP::LinearAlgebra::Vector::shared_ptr d_flowOutput;
    AMP::LinearAlgebra::Vector::shared_ptr d_localCladVec;

    AMP::Operator::MapOperator::shared_ptr d_flowInternal3to1;
    AMP::Operator::MapOperator::shared_ptr d_flowInternal1to3;
};
} // namespace AMP::Operator

#endif
