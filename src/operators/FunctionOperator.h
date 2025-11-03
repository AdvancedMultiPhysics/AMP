#ifndef included_AMP_FunctionOperator
#define included_AMP_FunctionOperator


#include "AMP/operators/Operator.h"


namespace AMP::Operator {


class FunctionOperator : public AMP::Operator::Operator
{
public:
    FunctionOperator( std::function<double( double )> f ) : d_f( f ) {}
    std::string type() const override { return "FunctionOperator"; }
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> r ) override
    {
        auto it_f = f->begin();
        auto it_r = r->begin();
        for ( size_t i = 0; i < it_f.size(); ++i, ++it_f, ++it_r )
            *it_r = d_f( *it_f );
    }

private:
    std::function<double( double )> d_f;
};


} // namespace AMP::Operator
#endif
