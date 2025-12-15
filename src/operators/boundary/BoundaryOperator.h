#ifndef included_AMP_BoundaryOperator
#define included_AMP_BoundaryOperator

#include "AMP/operators/Operator.h"

namespace AMP::Operator {


//  An abstract base class for representing an operator on a boundary.
class BoundaryOperator : public Operator
{

public:
    explicit BoundaryOperator( std::shared_ptr<const OperatorParameters> params )
        : Operator( params )
    {
    }

    virtual ~BoundaryOperator() {}

    virtual void addRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr ) {}

    virtual void setRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr ) {}

    virtual void modifyInitialSolutionVector( AMP::LinearAlgebra::Vector::shared_ptr ) {}

protected:
private:
};


} // namespace AMP::Operator

#endif
