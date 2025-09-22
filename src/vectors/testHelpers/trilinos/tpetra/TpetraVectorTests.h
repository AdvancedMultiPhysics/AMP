#ifndef included_AMP_test_TpetraVectorTests
#define included_AMP_test_TpetraVectorTests

#include "string"
#include <algorithm>

#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/testHelpers/VectorTests.h"

namespace AMP::LinearAlgebra {

class TpetraVectorFactory;

/**
 * \class TpetraVectorTests
 * \brief A helper class to store/run tests for a vector
 */
class TpetraVectorTests
{
public:
    explicit TpetraVectorTests( std::shared_ptr<const VectorFactory> factory )
        : d_factory( factory )
    {
    }

    void testTpetraVector( AMP::UnitTest *ut );

    void VerifyNorms( AMP::UnitTest *ut );

private:
    std::shared_ptr<const VectorFactory> d_factory;
};


} // namespace AMP::LinearAlgebra


#endif
