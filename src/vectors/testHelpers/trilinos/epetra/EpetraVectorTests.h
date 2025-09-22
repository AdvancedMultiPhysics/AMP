#ifndef included_AMP_test_EpetraVectorTests
#define included_AMP_test_EpetraVectorTests

#include "string"
#include <algorithm>

#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/testHelpers/VectorTests.h"

namespace AMP::LinearAlgebra {

class EpetraVectorFactory;

/**
 * \class EpetraVectorTests
 * \brief A helper class to store/run tests for a vector
 */
class EpetraVectorTests
{
public:
    explicit EpetraVectorTests( std::shared_ptr<const VectorFactory> factory )
        : d_factory( factory )
    {
    }

    void testEpetraVector( AMP::UnitTest *ut );

    void VerifyNorms( AMP::UnitTest *ut );

private:
    std::shared_ptr<const VectorFactory> d_factory;
};


} // namespace AMP::LinearAlgebra


#endif
