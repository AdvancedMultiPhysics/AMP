#include "AMP/AMP_TPLs.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/testHelpers/VectorFactory.h"
#include "AMP/vectors/testHelpers/VectorTests.h"

#ifdef AMP_USE_SUNDIALS
    #include "AMP/vectors/sundials/SundialsVector.h"
    #include "AMP/vectors/testHelpers/sundials/SundialsVectorTests.h"
#endif
#ifdef AMP_USE_PETSC
    #include "AMP/vectors/petsc/PetscVector.h"
    #include "AMP/vectors/testHelpers/petsc/PetscVectorFactory.h"
    #include "AMP/vectors/testHelpers/petsc/PetscVectorTests.h"
#endif
#ifdef AMP_USE_TRILINOS_EPETRA
    #include "AMP/vectors/testHelpers/trilinos/epetra/EpetraVectorFactory.h"
    #include "AMP/vectors/testHelpers/trilinos/epetra/EpetraVectorTests.h"
#endif
#ifdef AMP_USE_TRILINOS_TPETRA
    #include "AMP/vectors/testHelpers/trilinos/tpetra/TpetraVectorFactory.h"
    #include "AMP/vectors/testHelpers/trilinos/tpetra/TpetraVectorTests.h"
#endif

#include "ProfilerApp.h"


namespace AMP::LinearAlgebra {


void VectorTests::testBasicVector( AMP::UnitTest *ut )
{
    PROFILE( "testBasicVector" );
    try {
        InstantiateVector( ut );
        SetToScalarVector( ut );
        SetRandomValuesVector( ut );
        CloneVector( ut );
        DotProductVector( ut );
        AbsVector( ut );
        L1NormVector( ut );
        L2NormVector( ut );
        MaxNormVector( ut );
        ScaleVector( ut );
        AddVector( ut );
        SubtractVector( ut );
        MultiplyVector( ut );
        DivideVector( ut );
        ReciprocalVector( ut );
        LinearSumVector( ut );
        AxpyVector( ut );
        AxpbyVector( ut );
        CopyVector( ut );
        CopyRawDataBlockVector( ut );
        VerifyVectorMin( ut );
        VerifyVectorMax( ut );
        VerifyVectorMaxMin( ut );
#ifdef AMP_USE_PETSC
        DeepCloneOfView<AMP::LinearAlgebra::PetscVector>( ut );
        Bug_491( ut );
#endif
#ifdef AMP_USE_SUNDIALS
        DeepCloneOfView<AMP::LinearAlgebra::SundialsVector>( ut );
#endif
        VectorIteratorLengthTest( ut );
        Bug_728( ut );
        VectorIteratorTests( ut );
        TestMultivectorDuplicate( ut );
        TestContainsGlobalElement( ut );
        VerifyVectorSetZeroGhosts( ut );
    } catch ( const std::exception &err ) {
        ut->failure( "Caught std::exception testing " + d_factory->name() + ":\n" + err.what() );
    } catch ( ... ) {
        ut->failure( "Caught unhandled error testing " + d_factory->name() );
    }
}


void VectorTests::testManagedVector( AMP::UnitTest * ) {}


void VectorTests::testPetsc( [[maybe_unused]] AMP::UnitTest *ut )
{
#ifdef AMP_USE_PETSC
    PROFILE( "testPetsc" );
    auto petscViewFactory  = std::make_shared<PetscViewFactory>( d_factory );
    auto petscCloneFactory = std::make_shared<PetscCloneFactory>( petscViewFactory );
    PetscVectorTests test1( petscViewFactory );
    PetscVectorTests test2( petscCloneFactory );
    test1.testPetscVector( ut );
    test2.testPetscVector( ut );
    if ( std::dynamic_pointer_cast<const PetscVectorFactory>( d_factory ) ) {
        PetscVectorTests test3( petscCloneFactory );
        test3.testPetscVector( ut );
    }
#endif
}

void VectorTests::testEpetra( [[maybe_unused]] AMP::UnitTest *ut )
{
#ifdef AMP_USE_TRILINOS_EPETRA
    PROFILE( "testEpetra" );
    if ( d_factory->getVector()->numberOfDataBlocks() <= 1 ) {
        // Epetra currently only supports one data block
        EpetraVectorTests test( d_factory );
        test.testEpetraVector( ut );
    }
#endif
}


void VectorTests::testTpetra( [[maybe_unused]] AMP::UnitTest *ut )
{
#ifdef AMP_USE_TRILINOS_TPETRA
    PROFILE( "testTpetra" );
    if ( d_factory->getVector()->numberOfDataBlocks() <= 1 ) {
        // Epetra currently only supports one data block
        TpetraVectorTests test( d_factory );
        test.testTpetraVector( ut );
    }
#endif
}


void VectorTests::testSundials( [[maybe_unused]] AMP::UnitTest *ut )
{
#ifdef AMP_USE_SUNDIALS
    PROFILE( "testSundials" );
    auto viewFactory =
        std::make_shared<ViewFactory<AMP::LinearAlgebra::SundialsVector>>( d_factory );
    auto viewVec = viewFactory->getVector();
    if ( viewVec->getVectorData()->isType<double>() ) {
        VectorTests test1( viewFactory );
        test1.testBasicVector( ut );
    }
    auto cloneFactory = std::make_shared<CloneFactory>( viewFactory );
    auto cloneVec     = cloneFactory->getVector();
    if ( cloneVec->getVectorData()->isType<double>() ) {
        VectorTests test2( cloneFactory );
        test2.testBasicVector( ut );
    }

    SundialsVectorTests test3( viewFactory );
    SundialsVectorTests test4( cloneFactory );
    test3.testSundialsVector( ut );
    test4.testSundialsVector( ut );
#endif
}


void VectorTests::testNullVector( AMP::UnitTest *ut )
{
    PROFILE( "testNullVector" );
    VectorTests test1( std::make_shared<NullVectorFactory>() );
    VectorTests test2( std::make_shared<NullVectorDataFactory>() );
    test1.InstantiateVector( ut );
    test2.InstantiateVector( ut );
    test2.testBasicVector( ut );
}


void VectorTests::testParallelVectors( AMP::UnitTest *ut )
{
    PROFILE( "testParallelVectors" );
    InstantiateVector( ut );
    // VerifyVectorGhostCreate( ut );
    VerifyVectorMakeConsistentSet( ut );
    VerifyVectorMakeConsistentAdd( ut );
    CopyVectorConsistency( ut );
}


void VectorTests::testVectorSelector( AMP::UnitTest *ut )
{
    PROFILE( "testVectorSelector" );
    testAllSelectors( ut );
    test_VS_ByVariableName( ut );
    test_VS_Comm( ut );
    test_VS_Component( ut );
}


} // namespace AMP::LinearAlgebra
