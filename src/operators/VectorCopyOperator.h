#ifndef included_AMP_VectorCopyOperator
#define included_AMP_VectorCopyOperator

#include "AMP/operators/Operator.h"
#include "AMP/operators/VectorCopyOperatorParameters.h"
#include "AMP/vectors/Vector.h"
#include <memory>

namespace AMP::Operator {

class VectorCopyOperator : public Operator
{
public:
    explicit VectorCopyOperator( std::shared_ptr<const VectorCopyOperatorParameters> params );

    virtual ~VectorCopyOperator() {}

    std::string type() const override { return "VectorCopyOperator"; }

    virtual void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                        AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override;

private:
    // vector to copy into
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_copyVector;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_copyVariable;
};

} // namespace AMP::Operator

#endif
