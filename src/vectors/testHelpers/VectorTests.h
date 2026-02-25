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
    void testBasicVector( AMP::UnitTest * );
    void testManagedVector( AMP::UnitTest * );
    void testNullVector( AMP::UnitTest * );
    void testParallelVectors( AMP::UnitTest * );
    void testVectorSelector( AMP::UnitTest * );
    void testPetsc( AMP::UnitTest * );
    void testSundials( AMP::UnitTest * );
    void testEpetra( AMP::UnitTest * );
    void testTpetra( AMP::UnitTest * );


public:
    void InstantiateVector( AMP::UnitTest * );
    void CopyVectorConsistency( AMP::UnitTest * );
    template<typename VIEWER>
    void DeepCloneOfView( AMP::UnitTest * );
    void Bug_728( AMP::UnitTest * );
    void SetToScalarVector( AMP::UnitTest * );
    void CloneVector( AMP::UnitTest * );
    void DotProductVector( AMP::UnitTest * );
    void L2NormVector( AMP::UnitTest * );
    void AbsVector( AMP::UnitTest * );
    void L1NormVector( AMP::UnitTest * );
    void MaxNormVector( AMP::UnitTest * );
    void ScaleVector( AMP::UnitTest * );
    void Bug_491( AMP::UnitTest * );
    void AddVector( AMP::UnitTest * );
    void SubtractVector( AMP::UnitTest * );
    void MultiplyVector( AMP::UnitTest * );
    void DivideVector( AMP::UnitTest * );
    void VectorIteratorLengthTest( AMP::UnitTest * );
    void VectorIteratorTests( AMP::UnitTest * );
    void VerifyVectorSum( AMP::UnitTest * );
    void VerifyVectorMin( AMP::UnitTest * );
    void VerifyVectorMax( AMP::UnitTest * );
    void VerifyVectorMaxMin( AMP::UnitTest * );
    void SetRandomValuesVector( AMP::UnitTest * );
    void ReciprocalVector( AMP::UnitTest * );
    void LinearSumVector( AMP::UnitTest * );
    void AxpyVector( AMP::UnitTest * );
    void AxpbyVector( AMP::UnitTest * );
    void CopyVector( AMP::UnitTest * );
    void CopyRawDataBlockVector( AMP::UnitTest * );
    void VerifyVectorGhostCreate( AMP::UnitTest * );
    void VerifyVectorMakeConsistentAdd( AMP::UnitTest * );
    void VerifyVectorMakeConsistentSet( AMP::UnitTest * );
    // Test creating a multivector with multiple copies of the data
    // This should always return one copy of the superset of the data
    void TestMultivectorDuplicate( AMP::UnitTest * );
    void TestContainsGlobalElement( AMP::UnitTest * );
    void VerifyVectorSetZeroGhosts( AMP::UnitTest * );

public: // Vector selector tests
    // Test to check that Vector::select, Vector::select, VectorSelector::subset,
    // and VectorSelector::constSubset return the same vectors
    void testAllSelectors( AMP::UnitTest * );

    // Test the behavior of VS_ByVariableName
    void test_VS_ByVariableName( AMP::UnitTest * );

    // Test the behavior of VS_Comm
    void test_VS_Comm( AMP::UnitTest * );

    // Test the behavior of VS_Comm
    void test_VS_Component( AMP::UnitTest * );


private:
    std::shared_ptr<const VectorFactory> d_factory;
};


} // namespace AMP::LinearAlgebra


// Extra includes
#include "AMP/vectors/testHelpers/VectorTests.inline.h"


#endif
