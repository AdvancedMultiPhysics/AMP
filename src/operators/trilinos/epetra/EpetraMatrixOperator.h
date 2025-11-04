#ifndef included_AMP_EpetraMatrixOperator_h
#define included_AMP_EpetraMatrixOperator_h

#include "AMP/matrices/trilinos/ManagedEpetraMatrix.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/trilinos/EpetraMatrixOperatorParameters.h"

namespace AMP::Operator {

class EpetraMatrixOperator : public LinearOperator
{
private:
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_Input, d_Output;

public:
    explicit EpetraMatrixOperator( std::shared_ptr<EpetraMatrixOperatorParameters> params )
        : LinearOperator( params )
    {
        std::shared_ptr<AMP::LinearAlgebra::Matrix> t(
            new AMP::LinearAlgebra::ManagedEpetraMatrix( params->d_Matrix ) );
        setMatrix( t );
    }

    virtual ~EpetraMatrixOperator() {}
};
} // namespace AMP::Operator

#endif
