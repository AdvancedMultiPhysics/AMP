#ifndef included_AMP_test_StridedVectorFactory
#define included_AMP_test_StridedVectorFactory

#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorSelector.h"
#include "AMP/vectors/testHelpers/VectorTests.h"


namespace AMP::LinearAlgebra {


class StridedVectorFactory : public VectorFactory
{
public:
    explicit StridedVectorFactory( std::shared_ptr<const VectorFactory> factory )
        : d_factory( factory )
    {
    }
    AMP::LinearAlgebra::Vector::shared_ptr getVector() const override
    {
        auto vec        = d_factory->getVector();
        auto criterion  = AMP::LinearAlgebra::VS_Stride( 1, 3 );
        auto vec_select = vec->select( criterion );
        size_t N1       = vec->getGlobalSize();
        size_t N2       = vec_select->getGlobalSize();
        AMP_ASSERT( N1 / 3 == N2 );
        return vec_select;
    }
    std::string name() const override { return "StridedVectorFactory<" + d_factory->name() + ">"; }

private:
    std::shared_ptr<const VectorFactory> d_factory;
};


} // namespace AMP::LinearAlgebra

#endif
