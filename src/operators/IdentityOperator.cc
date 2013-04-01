#include "IdentityOperator.h"
#include "utils/Utilities.h"


namespace AMP {
namespace Operator {


IdentityOperator :: IdentityOperator () :
    LinearOperator () 
{
}

IdentityOperator :: IdentityOperator (const boost::shared_ptr<OperatorParameters> & params) :
    LinearOperator (params) 
{
}


void IdentityOperator :: setMatrix(const boost::shared_ptr<AMP::LinearAlgebra::Matrix> & in_mat) 
{
    AMP_ERROR("setMatrix is invalid for the Identity operator");
}


void IdentityOperator :: apply(AMP::LinearAlgebra::Vector::const_shared_ptr f, 
    AMP::LinearAlgebra::Vector::const_shared_ptr u,
    AMP::LinearAlgebra::Vector::shared_ptr r, const double a, const double b)
{
    AMP_INSIST( ((u.get()) != NULL), "NULL Solution Vector" );
    AMP_INSIST( ((r.get()) != NULL), "NULL Residual Vector" );

    AMP::LinearAlgebra::Vector::const_shared_ptr uInternal = subsetInputVector(u);
    AMP::LinearAlgebra::Vector::shared_ptr rInternal = subsetOutputVector(r);

    AMP_INSIST( (uInternal.get() != NULL), "uInternal is NULL" );
    AMP_INSIST( (rInternal.get() != NULL), "rInternal is NULL" );

    rInternal->copyVector(uInternal);

    if(f.get() == NULL) {
        rInternal->scale(a);
    } else {
        AMP::LinearAlgebra::Vector::const_shared_ptr fInternal = subsetOutputVector(f);
        if(fInternal.get() == NULL) {
            rInternal->scale(a);
        } else {
            rInternal->axpby(b, a, fInternal);
        }
    }
    rInternal->makeConsistent(AMP::LinearAlgebra::Vector::CONSISTENT_SET);
}


}
}

