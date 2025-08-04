#include "AMP/operators/LinearOperator.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Operator {


LinearOperator::LinearOperator( std::shared_ptr<const OperatorParameters> params )
    : Operator( params )
{
    d_matrix.reset();
}


LinearOperator::LinearOperator() : Operator() { d_matrix.reset(); }


std::shared_ptr<AMP::LinearAlgebra::Matrix> LinearOperator::getMatrix() { return d_matrix; }


void LinearOperator::setMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> in_mat )
{
    d_matrix = in_mat;
}

std::shared_ptr<AMP::LinearAlgebra::Vector> LinearOperator::getInputVector() const
{
    if ( d_matrix )
        return d_matrix->getInputVector();
    return nullptr;
}

std::shared_ptr<AMP::LinearAlgebra::Vector> LinearOperator::getOutputVector() const
{
    if ( d_matrix )
        return d_matrix->getOutputVector();
    return nullptr;
}

void LinearOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                            AMP::LinearAlgebra::Vector::shared_ptr f )
{
    AMP_INSIST( u, "NULL Solution Vector" );
    AMP_INSIST( f, "NULL Residual Vector" );
    AMP_INSIST( d_matrix, "NULL Matrix" );

    AMP_INSIST( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED,
                "Input vector is in an inconsistent state" );

    auto uInternal = subsetInputVector( u );
    auto fInternal = subsetOutputVector( f );

    AMP_INSIST( uInternal, "uInternal is NULL" );
    AMP_INSIST( fInternal, "fInternal is NULL" );

    d_matrix->mult( uInternal, fInternal );

    fInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}
} // namespace AMP::Operator
