#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/ManagedVector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorSelector.h"
#include "AMP/vectors/testHelpers/VectorTests.h"


inline bool compareVecSubset( AMP::LinearAlgebra::Vector::const_shared_ptr vec1,
                              AMP::LinearAlgebra::Vector::const_shared_ptr vec2 )
{
    return vec1->getLocalSize() == vec2->getLocalSize() &&
           vec1->getGlobalSize() == vec2->getGlobalSize() &&
           vec1->getComm().compare( vec2->getComm() ) > 0;
}


// Test to check that Vector::select, Vector::constSelect, VectorSelector::subset,
// and VectorSelector::constSubset return the same vectors
inline void testSelector( AMP::UnitTest *ut,
                          const std::string &test_name,
                          const AMP::LinearAlgebra::VectorSelector &selector,
                          AMP::LinearAlgebra::Vector::shared_ptr vec )
{
    auto vec1 = selector.subset( vec );
    auto vec2 = selector.subset( AMP::LinearAlgebra::Vector::const_shared_ptr( vec ) );
    auto vec3 = vec->select( selector, vec->getVariable()->getName() );
    auto vec4 = vec->constSelect( selector, vec->getVariable()->getName() );
    if ( !vec1 || !vec2 || !vec3 || !vec4 ) {
        ut->failure( "Failed to select (" + test_name + ")" );
        return;
    }
    bool equal = compareVecSubset( vec1, vec2 ) && compareVecSubset( vec1, vec3 ) &&
                 compareVecSubset( vec1, vec4 );
    if ( equal )
        ut->passes( "select matches constSelect and subset (" + test_name + ")" );
    else
        ut->failure( "select matches constSelect and subset (" + test_name + ")" );
}
void AMP::LinearAlgebra::VectorTests::testAllSelectors( AMP::UnitTest *ut )
{
    auto vec = d_factory->getVector();
    vec->setVariable( std::make_shared<AMP::LinearAlgebra::Variable>( "test_selector" ) );
    AMP::AMP_MPI vec_comm = vec->getComm();
    AMP::AMP_MPI world_comm( AMP_COMM_WORLD );
    AMP::AMP_MPI self_comm( AMP_COMM_SELF );
    testSelector( ut,
                  "VS_ByVariableName",
                  AMP::LinearAlgebra::VS_ByVariableName( vec->getVariable()->getName() ),
                  vec );
    testSelector( ut, "VS_Stride", AMP::LinearAlgebra::VS_Stride( 0, 1 ), vec );
    testSelector( ut, "VS_Comm(vec)", AMP::LinearAlgebra::VS_Comm( vec_comm ), vec );
    testSelector( ut, "VS_Comm(world)", AMP::LinearAlgebra::VS_Comm( world_comm ), vec );
    for ( int i = 0; i < vec_comm.getRank(); i++ )
        vec_comm.barrier();
    testSelector( ut, "VS_Comm(self)", AMP::LinearAlgebra::VS_Comm( self_comm ), vec );
    for ( int i = vec_comm.getRank(); i < vec_comm.getSize(); i++ )
        vec_comm.barrier();
    // testSelector( ut, "VS_Mesh", AMP::LinearAlgebra::VS_Mesh(), vec );
    // testSelector( ut, "VS_MeshIterator", AMP::LinearAlgebra::VS_MeshIterator(), vec );
}


// Test the behavior of VS_ByVariableName
void AMP::LinearAlgebra::VectorTests::test_VS_ByVariableName( AMP::UnitTest *ut )
{
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto vec1  = d_factory->getVector();
    auto vec2  = vec1->cloneVector( "vec2" );
    auto vec3a = vec1->cloneVector( "vec3" );
    auto vec3b = vec1->cloneVector( "vec3" );
    auto vec3  = AMP::LinearAlgebra::MultiVector::create( "multivec", globalComm );
    vec3->addVector( vec2 );
    vec3->addVector( vec3a );
    vec3->addVector( vec3b );

    bool pass       = true;
    auto selection1 = vec2->select( AMP::LinearAlgebra::VS_ByVariableName( "None" ), "None" );
    if ( selection1 ) {
        ut->failure( "Found vector where there should be none" );
        pass = false;
    }

    selection1 = vec2->select( AMP::LinearAlgebra::VS_ByVariableName( "vec2" ), "subset" );
    if ( selection1 ) {
        if ( !compareVecSubset( vec2, selection1 ) ) {
            ut->failure( "Could not find vector" );
            pass = false;
        }
    } else {
        ut->failure( "Did not find a vector" );
    }

    selection1    = vec3->select( AMP::LinearAlgebra::VS_ByVariableName( "vec3" ), "subset" );
    auto vec3_sub = AMP::LinearAlgebra::MultiVector::create( "multivec", globalComm );
    vec3_sub->addVector( vec3a );
    vec3_sub->addVector( vec3b );
    if ( selection1 ) {
        if ( !compareVecSubset( vec3_sub, selection1 ) ) {
            ut->failure( "Could not find vector" );
            pass = false;
        }
    } else {
        if ( std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( vec2 ) ) {
            ut->expected_failure(
                "Subsetting a multivector of multivectors by name is not functional yet" );
        } else if ( std::dynamic_pointer_cast<AMP::LinearAlgebra::ManagedVector>( vec2 ) ) {
            ut->expected_failure(
                "Subsetting a multivector of multivectors by name is not functional yet" );
        } else {
            ut->failure( "Did not find a vector" );
            pass = false;
        }
    }
    if ( pass )
        ut->passes( "passed subset by name" );
}


// Test the behavior of VS_Comm
void AMP::LinearAlgebra::VectorTests::test_VS_Comm( AMP::UnitTest *ut )
{
    auto vec1             = d_factory->getVector();
    AMP::AMP_MPI vec_comm = vec1->getComm();
    AMP::AMP_MPI world_comm( AMP_COMM_WORLD );
    AMP::AMP_MPI self_comm( AMP_COMM_SELF );
    bool pass = true;
    auto vec2 = AMP::LinearAlgebra::VS_Comm( vec_comm ).subset( vec1 );
    if ( !compareVecSubset( vec1, vec2 ) ) {
        ut->failure( "Subset for vec comm" );
        pass = false;
    }
    vec2 = AMP::LinearAlgebra::VS_Comm( world_comm ).subset( vec1 );
    if ( !compareVecSubset( vec1, vec2 ) ) {
        ut->failure( "Subset for AMP_COMM_WORLD" );
        pass = false;
    }
    // Test comm subset for self without any other processors involved
    for ( int i = 0; i < vec_comm.getRank(); i++ )
        vec_comm.barrier();
    vec2 = AMP::LinearAlgebra::VS_Comm( self_comm ).subset( vec1 );
    if ( vec1 != nullptr ) {
        if ( vec2->getLocalSize() != vec1->getLocalSize() ||
             vec2->getGlobalSize() != vec1->getLocalSize() || vec2->getComm().getSize() != 1 ) {
            ut->failure( "Subset for AMP_COMM_SELF" );
            pass = false;
        }
    } else {
        if ( vec1->getLocalSize() != 0 ) {
            ut->failure( "Subset for AMP_COMM_SELF" );
            pass = false;
        }
    }
    for ( int i = vec_comm.getRank(); i < vec_comm.getSize(); i++ )
        vec_comm.barrier();
    if ( pass )
        ut->passes( "passed subset by comm" );
}