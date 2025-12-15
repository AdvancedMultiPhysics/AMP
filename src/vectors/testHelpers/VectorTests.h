#ifndef included_AMP_test_VectorTests
#define included_AMP_test_VectorTests

#include "AMP/utils/UnitTest.h"
#include <memory>

#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"

#include <string>


namespace AMP::LinearAlgebra {


/**
 * \class VectorFactory
 * \brief A helper class to generate vectors
 */
class VectorFactory
{
public:
    virtual ~VectorFactory() {}
    virtual AMP::LinearAlgebra::Vector::shared_ptr getVector() const = 0;
    virtual std::string name() const                                 = 0;

protected:
    VectorFactory() {}
    VectorFactory( const VectorFactory & );
};


/**
 * \class vectorTests
 * \brief A helper class to store/run tests for a vector
 */
class VectorTests
{
public:
    explicit VectorTests( std::shared_ptr<const VectorFactory> factory ) : d_factory( factory ) {}

public:
    void testBasicVector( AMP::UnitTest *ut );

    void testManagedVector( AMP::UnitTest *ut );

    void testNullVector( AMP::UnitTest *ut );

    void testParallelVectors( AMP::UnitTest *ut );

    void testVectorSelector( AMP::UnitTest *ut );

    void testPetsc( AMP::UnitTest *ut );

    void testSundials( AMP::UnitTest *ut );

    void testEpetra( AMP::UnitTest *ut );

    void testTpetra( AMP::UnitTest *ut );


public:
    void InstantiateVector( AMP::UnitTest *utils );

    void CopyVectorConsistency( AMP::UnitTest *utils );

    template<typename VIEWER>
    void DeepCloneOfView( AMP::UnitTest *utils );

    void Bug_728( AMP::UnitTest *utils );

    void SetToScalarVector( AMP::UnitTest *utils );

    void CloneVector( AMP::UnitTest *utils );

    void DotProductVector( AMP::UnitTest *utils );

    void L2NormVector( AMP::UnitTest *utils );

    void AbsVector( AMP::UnitTest *utils );

    void L1NormVector( AMP::UnitTest *utils );

    void MaxNormVector( AMP::UnitTest *utils );

    void ScaleVector( AMP::UnitTest *utils );

    void Bug_491( AMP::UnitTest *utils );

    void AddVector( AMP::UnitTest *utils );

    void SubtractVector( AMP::UnitTest *utils );

    void MultiplyVector( AMP::UnitTest *utils );

    void DivideVector( AMP::UnitTest *utils );

    void VectorIteratorLengthTest( AMP::UnitTest *utils );

    template<typename T, typename ITERATOR>
    void both_VectorIteratorTests( AMP::LinearAlgebra::Vector::shared_ptr p, AMP::UnitTest *utils );

    void VectorIteratorTests( AMP::UnitTest *utils );

    void VerifyVectorMin( AMP::UnitTest *utils );

    void VerifyVectorMax( AMP::UnitTest *utils );

    void VerifyVectorMaxMin( AMP::UnitTest *utils );

    void SetRandomValuesVector( AMP::UnitTest *utils );

    void ReciprocalVector( AMP::UnitTest *utils );

    void LinearSumVector( AMP::UnitTest *utils );

    void AxpyVector( AMP::UnitTest *utils );

    void AxpbyVector( AMP::UnitTest *utils );

    void CopyVector( AMP::UnitTest *utils );

    void CopyRawDataBlockVector( AMP::UnitTest *utils );

    void VerifyVectorGhostCreate( AMP::UnitTest *utils );

    void VerifyVectorMakeConsistentAdd( AMP::UnitTest *utils );

    void VerifyVectorMakeConsistentSet( AMP::UnitTest *utils );

    // Test creating a multivector with multiple copies of the data
    // This should always return one copy of the superset of the data
    void TestMultivectorDuplicate( AMP::UnitTest *utils );

    void TestContainsGlobalElement( AMP::UnitTest *utils );

    void VerifyVectorSetZeroGhosts( AMP::UnitTest *utils );

public: // Vector selector tests
    // Test to check that Vector::select, Vector::select, VectorSelector::subset,
    // and VectorSelector::constSubset return the same vectors
    void testAllSelectors( AMP::UnitTest *ut );

    // Test the behavior of VS_ByVariableName
    void test_VS_ByVariableName( AMP::UnitTest *ut );

    // Test the behavior of VS_Comm
    void test_VS_Comm( AMP::UnitTest *ut );

    // Test the behavior of VS_Comm
    void test_VS_Component( AMP::UnitTest *ut );


private:
    std::shared_ptr<const VectorFactory> d_factory;
};


} // namespace AMP::LinearAlgebra


// Extra includes
#include "AMP/vectors/testHelpers/VectorTests.inline.h"


#endif
