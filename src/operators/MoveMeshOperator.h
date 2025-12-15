#ifndef included_AMP_MoveMeshOperator
#define included_AMP_MoveMeshOperator

#include "AMP/operators/Operator.h"

namespace AMP::Operator {

class MoveMeshOperator : public Operator
{
public:
    explicit MoveMeshOperator( std::shared_ptr<const OperatorParameters> params );

    virtual ~MoveMeshOperator() {}

    std::string type() const override { return "MoveMeshOperator"; }

    void setVariable( std::shared_ptr<AMP::LinearAlgebra::Variable> var );

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override;

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    void reset( std::shared_ptr<const OperatorParameters> params ) override;

protected:
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_var;
    AMP::LinearAlgebra::Vector::shared_ptr d_prevDisp;
};
} // namespace AMP::Operator

#endif
