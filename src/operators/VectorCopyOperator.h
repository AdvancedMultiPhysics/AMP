#ifndef included_AMP_VectorCopyOperator
#define included_AMP_VectorCopyOperator

#include "AMP/operators/Operator.h"
#include "AMP/operators/VectorCopyOperatorParameters.h"
#include "AMP/vectors/Vector.h"
#include <memory>

namespace AMP {
namespace Operator {

class VectorCopyOperator : public Operator
{
public:
    explicit VectorCopyOperator( const std::shared_ptr<VectorCopyOperatorParameters> &params );

    virtual ~VectorCopyOperator() {}

    virtual void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                        AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    AMP::LinearAlgebra::Variable::shared_ptr getOutputVariable() override;

    AMP::LinearAlgebra::Variable::shared_ptr getInputVariable() override;

private:
    // vector to copy into
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_copyVector;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_copyVariable;
};

} // namespace Operator
} // namespace AMP

#endif
