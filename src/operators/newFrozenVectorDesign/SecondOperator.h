#ifndef included_AMP_SecondOperator
#define included_AMP_SecondOperator

#include "AMP/operators/newFrozenVectorDesign/OnePointOperator.h"
#include "AMP/vectors/MultiVariable.h"

namespace AMP::Operator {

class SecondOperator : public OnePointOperator
{
public:
    explicit SecondOperator( std::shared_ptr<const OperatorParameters> params )
        : OnePointOperator( params )
    {
        d_constant = 3.0;
        d_primaryVar.reset(
            new AMP::LinearAlgebra::Variable( params->d_db->getString( "PrimaryVariable" ) ) );
        d_secondaryVar.reset(
            new AMP::LinearAlgebra::Variable( params->d_db->getString( "SecondaryVariable" ) ) );
    }

    std::string type() const override { return "SecondOperator"; }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override
    {
        auto inP = u->subsetVectorForVariable( d_primaryVar );
        auto inS = u->subsetVectorForVariable( d_secondaryVar );
        auto out = r->subsetVectorForVariable( d_primaryVar );
        out->linearSum( d_constant, *inP, 1.0, *inS );
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        auto retVariable = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "MultiVariable" );
        retVariable->add( d_primaryVar );
        retVariable->add( d_secondaryVar );
        return retVariable;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_primaryVar;
    }

protected:
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_primaryVar;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_secondaryVar;

private:
};
} // namespace AMP::Operator

#endif
