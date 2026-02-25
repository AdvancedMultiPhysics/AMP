#ifndef included_AMP_SundialsVectorFactory
#define included_AMP_SundialsVectorFactory

#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/testHelpers/VectorTests.h"


namespace AMP::LinearAlgebra {


class NativeSundialsVectorFactory : public VectorFactory
{
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        AMP::LinearAlgebra::Vector::shared_ptr newVec;
        AMP_ERROR( "Not implemented" );
        return newVec;
    }
    std::string name() const override { return "NativeSundialsVectorFactory"; }
};


} // namespace AMP::LinearAlgebra


#endif
