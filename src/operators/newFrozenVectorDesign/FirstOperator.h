
#ifndef included_AMP_FirstOperator
#define included_AMP_FirstOperator

#include "AMP/operators/newFrozenVectorDesign/OnePointOperator.h"

namespace AMP::Operator {

class FirstOperator : public OnePointOperator
{
public:
    explicit FirstOperator( std::shared_ptr<const OperatorParameters> params )
        : OnePointOperator( params )
    {
        d_constant = 2.0;
        d_var.reset( new AMP::LinearAlgebra::Variable( params->d_db->getString( "Variable" ) ) );
    }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override
    {
        auto in  = u->subsetVectorForVariable( d_var );
        auto out = r->subsetVectorForVariable( d_var );
        out->scale( d_constant, *in );
    }

    std::string type() const override { return "FirstOperator"; }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_var;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_var;
    }

protected:
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_var;

private:
};
} // namespace AMP::Operator

#endif
