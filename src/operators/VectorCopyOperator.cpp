#include "AMP/operators/VectorCopyOperator.h"

namespace AMP::Operator {

VectorCopyOperator::VectorCopyOperator( std::shared_ptr<const VectorCopyOperatorParameters> params )
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
    auto vecToCopy = subsetOutputVector( u );
    d_copyVector->copyVector( vecToCopy );
}

std::shared_ptr<AMP::LinearAlgebra::Variable> VectorCopyOperator::getOutputVariable() const
{
    return d_copyVariable;
}

std::shared_ptr<AMP::LinearAlgebra::Variable> VectorCopyOperator::getInputVariable() const
{
    return d_copyVariable;
}
} // namespace AMP::Operator
