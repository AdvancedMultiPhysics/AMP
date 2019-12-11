#ifndef included_AMP_LinearCoupledFlowOperator
#define included_AMP_LinearCoupledFlowOperator

#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "LinearCoupledFlowOperatorParameters.h"
#include <vector>

namespace AMP {
namespace Operator {

class LinearCoupledFlowOperator : public Operator
{
public:
    explicit LinearCoupledFlowOperator( const std::shared_ptr<OperatorParameters> &params )
        : Operator( params )
    {
        (void) params;
    }

    virtual ~LinearCoupledFlowOperator() {}

    virtual void apply( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                        AMP::LinearAlgebra::Vector::const_shared_ptr u,
                        AMP::LinearAlgebra::Vector::shared_ptr r,
                        const double a = -1.0,
                        const double b = 1.0 );

    virtual void reset( const std::shared_ptr<OperatorParameters> &params );

    virtual void append( std::shared_ptr<Operator> op );

    virtual AMP::LinearAlgebra::Variable::shared_ptr getOutputVariable();

    virtual AMP::LinearAlgebra::Variable::shared_ptr getInputVariable( int varId = -1 );

protected:
    std::vector<std::shared_ptr<Operator>> d_Operators;

private:
};
} // namespace Operator
} // namespace AMP

#endif
