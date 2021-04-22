#include "AMP/operators/VectorCopyOperator.h"

namespace AMP {
namespace Operator {

VectorCopyOperator::VectorCopyOperator(
    const std::shared_ptr<VectorCopyOperatorParameters> &params )
    : AMP::Operator::Operator( params )
{
    auto copyParams =
        std::dynamic_pointer_cast<const AMP::Operator::VectorCopyOperatorParameters>( params );
    d_copyVariable = copyParams->d_copyVariable;
    d_copyVector   = copyParams->d_copyVector;

    AMP_INSIST( d_copyVector, "must have non NULL CopyVector" );
    AMP_INSIST( d_copyVariable, "must have non NULL CopyVeriable" );
}

void VectorCopyOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                AMP::LinearAlgebra::Vector::shared_ptr )
{
    AMP::LinearAlgebra::Vector::const_shared_ptr vecToCopy = subsetOutputVector( u );
    d_copyVector->copyVector( vecToCopy );
}

AMP::LinearAlgebra::Variable::shared_ptr VectorCopyOperator::getOutputVariable()
{
    return d_copyVariable;
}

AMP::LinearAlgebra::Variable::shared_ptr VectorCopyOperator::getInputVariable()
{
    return d_copyVariable;
}
} // namespace Operator
} // namespace AMP
