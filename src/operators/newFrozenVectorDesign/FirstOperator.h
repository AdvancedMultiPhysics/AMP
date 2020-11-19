
#ifndef included_AMP_FirstOperator
#define included_AMP_FirstOperator

#include "AMP/operators/newFrozenVectorDesign/OnePointOperator.h"

namespace AMP {
namespace Operator {

class FirstOperator : public OnePointOperator
{
public:
    explicit FirstOperator( const std::shared_ptr<OperatorParameters> &params )
        : OnePointOperator( params )
    {
        d_constant = 2.0;
        d_var.reset( new AMP::LinearAlgebra::Variable( params->d_db->getString( "Variable" ) ) );
    }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override
    {
        auto in  = u->constSubsetVectorForVariable( d_var );
        auto out = r->subsetVectorForVariable( d_var );
        out->scale( d_constant, *in );
    }

    std::string type() const override { return "FirstOperator"; }

    AMP::LinearAlgebra::Variable::shared_ptr getInputVariable() override { return d_var; }

    AMP::LinearAlgebra::Variable::shared_ptr getOutputVariable() override { return d_var; }

protected:
    AMP::LinearAlgebra::Variable::shared_ptr d_var;

private:
};
} // namespace Operator
} // namespace AMP

#endif
