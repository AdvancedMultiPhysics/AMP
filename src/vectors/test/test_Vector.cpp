#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/testHelpers/VectorTests.h"
#include "AMP/vectors/testHelpers/generateVectorFactories.h"

#include "test_ArrayVector.h"

#ifdef AMP_USE_TRILINOS_THYRA
    #include "AMP/vectors/testHelpers/trilinos/thyra/ThyraVectorFactory.h"
#endif

#include "ProfilerApp.h"

#include <iostream>

using namespace AMP::unit_test;
using namespace AMP::LinearAlgebra;
using AMP::LinearAlgebra::generateVectorFactory;
using AMP::LinearAlgebra::VectorTests;

std::string SimpleFactory1 = "SimpleVectorFactory<15,false,double>";
std::string SimpleFactory2 = "SimpleVectorFactory<45,true,double>";
std::string SNPVFactory    = "NativePetscVectorFactory";
std::string SNEVFactory    = "NativeEpetraFactory";
#ifdef AMP_USE_PETSC
std::string MVFactory1 = "MultiVectorFactory<" + SNEVFactory + ", 1, " + SNPVFactory + ", 1>";
#else
std::string MVFactory1 =
    "MultiVectorFactory<SimpleVectorFactory<15,false>,1," + SNEVFactory + ",1>";
#endif


int testByVariableName( std::shared_ptr<AMP::LinearAlgebra::Vector> vec )
{
    PROFILE( "testByVariableName" );
    int N_it  = 50;
    auto name = vec->getName();
    auto t1   = AMP::Utilities::time();
    for ( int i = 0; i < N_it; i++ ) {
        VS_ByVariableName variable( name );
        AMP_ASSERT( variable.subset( vec ) );
    }
    auto t2 = AMP::Utilities::time();
    return 1e9 * ( t2 - t1 ) / N_it;
}
int testByStride( std::shared_ptr<AMP::LinearAlgebra::Vector> vec, size_t offset, size_t length )
{
    PROFILE( "testByStride" );
    int N_it  = 20;
    auto name = vec->getName();
    auto t1   = AMP::Utilities::time();
    for ( int i = 0; i < N_it; i++ ) {
        VS_Stride variable( offset, length );
        AMP_ASSERT( variable.subset( vec ) );
    }
    auto t2 = AMP::Utilities::time();
    return 1e9 * ( t2 - t1 ) / N_it;
}
int testByComm( std::shared_ptr<AMP::LinearAlgebra::Vector> vec, const AMP::AMP_MPI &comm )
{
    PROFILE( "testByComm" );
    int N_it  = 50;
    auto name = vec->getName();
    auto t1   = AMP::Utilities::time();
    for ( int i = 0; i < N_it; i++ ) {
        VS_Comm variable( comm );
        AMP_ASSERT( variable.subset( vec ) );
    }
    auto t2 = AMP::Utilities::time();
    return 1e9 * ( t2 - t1 ) / N_it;
}

std::shared_ptr<const AMP::Mesh::Mesh>
getMesh( std::shared_ptr<AMP::Discretization::DOFManager> dof )
{
    auto mesh = dof->getMesh();
    if ( mesh )
        return mesh;
    auto multiDof = std::dynamic_pointer_cast<AMP::Discretization::multiDOFManager>( dof );
    if ( multiDof ) {
        std::vector<std::shared_ptr<const AMP::Mesh::Mesh>> meshes;
        for ( auto dof2 : multiDof->getDOFManagers() ) {
            auto mesh2 = getMesh( dof2 );
            if ( mesh2 )
                meshes.push_back( mesh2 );
        }
        if ( meshes.empty() )
            return {};
        return meshes[0];
    }
    return {};
}
int testByMesh( std::shared_ptr<AMP::LinearAlgebra::Vector> vec )
{
    PROFILE( "testByMesh" );
    auto mesh = getMesh( vec->getDOFManager() );
    if ( !mesh )
        return 0;
    int N_it  = 100;
    auto name = vec->getName();
    auto t1   = AMP::Utilities::time();
    for ( int i = 0; i < N_it; i++ ) {
        AMP::LinearAlgebra::VS_Mesh meshSelector( mesh );
        AMP_ASSERT( vec->select( meshSelector ) );
    }
    auto t2 = AMP::Utilities::time();
    return 1e9 * ( t2 - t1 ) / N_it;
}
void testVectorSelectorPerformance()
{
    PROFILE( "testVectorSelectorPerformance" );
    AMP::AMP_MPI worldComm( AMP_COMM_WORLD );
    AMP::AMP_MPI selfComm( AMP_COMM_SELF );
    auto splitComm = worldComm.split( worldComm.getRank() % 2 );
    if ( worldComm.getRank() == 0 )
        printf( "       subset performance (ns):              "
                " Variable  Stride  VecComm    World    Self    Split     Mesh 1\n" );
    for ( auto name : getAllFactories() ) {
        auto factory = generateVectorFactory( name );
        auto vec     = factory->getVector();
        auto vecComm = vec->getComm();
        worldComm.barrier();
        auto t1 = testByVariableName( vec );
        auto t2 = testByStride( vec, 0, 1 );
        auto t3 = testByComm( vec, vecComm );
        auto t4 = testByComm( vec, worldComm );
        auto t5 = testByComm( vec, selfComm );
        auto t6 = testByComm( vec, splitComm );
        auto t7 = testByMesh( vec );
        if ( worldComm.getRank() == 0 ) {
            auto name2 = name.substr( 0, 40 );
            printf(
                "  %40s  %8i %8i %8i %8i %8i %8i %8i\n", name2.data(), t1, t2, t3, t4, t5, t6, t7 );
        }
    }
}


int main( int argc, char **argv )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    PROFILE_ENABLE( 3 );

#if 1

    // Test ArrayVector dimensions
    std::vector<size_t> dims{ 3, 3, 3, 3 };
    testArrayVectorDimensions<double>( dims, ut );

    // Print the total nummber of vector factories to test
    AMP::pout << "Testing " << getAllFactories().size() << " vector factories\n";
    for ( auto name : getManagedVectorFactories() ) {
        auto factory = generateVectorFactory( name );
        auto name2   = factory->name();
        if ( name != name2 ) {
            ut.failure( "Factory name " + name2 + " does not match input name " + name );
        }
    }

    // Run null vector tests
    AMP::pout << std::endl << "Testing NullVector" << std::endl;
    {
        auto factory = generateVectorFactory( SimpleFactory1 );
        VectorTests tests( factory );
        tests.testNullVector( &ut );
    }

    // Run the basic vector tests
    AMP::pout << std::endl << "Running basic vector tests:" << std::endl;
    for ( auto name : getAllFactories() ) {
        auto factory = generateVectorFactory( name );
        VectorTests tests( factory );
        tests.testBasicVector( &ut );
    }

    // Run the managed vector tests
    AMP::pout << std::endl << "Running managed vector tests:" << std::endl;
    for ( auto name : getManagedVectorFactories() ) {
        auto factory = generateVectorFactory( name );
        VectorTests tests( factory );
        tests.testManagedVector( &ut );
    }

    // Run the petsc vector tests
    AMP::pout << std::endl << "Running petsc vector tests:" << std::endl;
    for ( auto name : getAllFactories() ) {
        auto factory = generateVectorFactory( name );
        VectorTests tests( factory );
        tests.testPetsc( &ut );
    }

    // Run the epetra vector tests
    AMP::pout << std::endl << "Running epetra vector tests:" << std::endl;
    for ( auto name : getAllFactories() ) {
        auto factory = generateVectorFactory( name );
        VectorTests tests( factory );
        tests.testEpetra( &ut );
    }

    // Run the sundials vector tests
    AMP::pout << std::endl << "Running sundials vector tests:" << std::endl;
    for ( auto name : getAllFactories() ) {
        auto factory = generateVectorFactory( name );
        VectorTests tests( factory );
        tests.testSundials( &ut );
    }

    // Run Belos tests of thyra vectors
    #if defined( AMP_USE_TRILINOS_THYRA ) && defined( AMP_USE_TRILINOS_BELOS )
    AMP::pout << std::endl << "Testing Belos interface to Thyra vectors" << std::endl;
    testBelosThyraVector( ut, NativeThyraFactory() );
    testBelosThyraVector( ut, ManagedThyraFactory( generateVectorFactory( SimpleFactory2 ) ) );
    testBelosThyraVector( ut,
                          ManagedNativeThyraFactory( generateVectorFactory( SimpleFactory2 ) ) );
    testBelosThyraVector( ut, ManagedNativeThyraFactory( generateVectorFactory( MVFactory1 ) ) );
    #endif

    // Run the vector selector tests
    AMP::pout << std::endl << "Running vector selector tests:" << std::endl;
    for ( auto name : getAllFactories() ) {
        auto factory = generateVectorFactory( name );
        VectorTests tests( factory );
        tests.testVectorSelector( &ut );
    }

#endif

    // Test vector selector performance
    testVectorSelectorPerformance();

    PROFILE_SAVE( "testVector" );
    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
