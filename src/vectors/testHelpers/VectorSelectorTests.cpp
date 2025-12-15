#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorSelector.h"
#include "AMP/vectors/testHelpers/VectorTests.h"


inline bool compareVecSubset( AMP::LinearAlgebra::Vector::const_shared_ptr vec1,
                              AMP::LinearAlgebra::Vector::const_shared_ptr vec2 )
{
    return vec1->getLocalSize() == vec2->getLocalSize() &&
           vec1->getGlobalSize() == vec2->getGlobalSize() &&
           vec1->numberOfDataBlocks() == vec2->numberOfDataBlocks() &&
           vec1->getComm().compare( vec2->getComm() ) > 0;
}


// Test to check that Vector::select, Vector::select, VectorSelector::subset,
// and VectorSelector::constSubset return the same vectors
inline void testSelector( AMP::UnitTest *ut,
                          const std::string &test_name,
                          const std::string &factory_name,
                          const AMP::LinearAlgebra::VectorSelector &selector,
                          AMP::LinearAlgebra::Vector::shared_ptr vec )
{
    auto constVec = std::const_pointer_cast<const AMP::LinearAlgebra::Vector>( vec );
    auto vec1     = selector.subset( vec );
    auto vec2     = selector.subset( constVec );
    auto vec3     = vec->select( selector );
    auto vec4     = constVec->select( selector );
    if ( !vec1 || !vec2 || !vec3 || !vec4 ) {
        ut->failure( "Failed to select (" + test_name + ")" );
        return;
    }
    bool equal = compareVecSubset( vec1, vec2 ) && compareVecSubset( vec1, vec3 ) &&
                 compareVecSubset( vec1, vec4 );
    if ( equal )
        ut->passes( "select matches select and subset: " + test_name );
    else
        ut->failure( "select matches select and subset: " + test_name + " " + factory_name );
}
void AMP::LinearAlgebra::VectorTests::testAllSelectors( AMP::UnitTest *ut )
{
    auto vec  = d_factory->getVector();
    auto name = d_factory->name();
    vec->setVariable( std::make_shared<Variable>( "test_selector" ) );
    AMP::AMP_MPI vec_comm = vec->getComm();
    AMP::AMP_MPI world_comm( AMP_COMM_WORLD );
    AMP::AMP_MPI self_comm( AMP_COMM_SELF );
    testSelector( ut, "VS_ByVariableName", name, VS_ByVariableName( vec->getName() ), vec );
    testSelector( ut, "VS_Stride", name, VS_Stride( 0, 1 ), vec );
    testSelector( ut, "VS_Comm(vec)", name, VS_Comm( vec_comm ), vec );
    testSelector( ut, "VS_Comm(world)", name, VS_Comm( world_comm ), vec );
    for ( int i = 0; i < vec_comm.getRank(); i++ )
        vec_comm.barrier();
    testSelector( ut, "VS_Comm(self)", name, VS_Comm( self_comm ), vec );
    for ( int i = vec_comm.getRank(); i < vec_comm.getSize(); i++ )
        vec_comm.barrier();
    if ( vec_comm.getSize() > 2 ) {
        auto splitComm = vec_comm.split( vec_comm.getRank() % 2 );
        testSelector( ut, "VS_Comm(split)", name, VS_Comm( splitComm ), vec );
    }
    // testSelector( ut, "VS_Mesh", VS_Mesh(), vec );
    // testSelector( ut, "VS_MeshIterator", VS_MeshIterator(), vec );
}


// Test the behavior of VS_ByVariableName
void AMP::LinearAlgebra::VectorTests::test_VS_ByVariableName( AMP::UnitTest *ut )
{
    AMP::AMP_MPI comm( AMP_COMM_WORLD );
    auto vec1  = d_factory->getVector();
    auto vec2  = vec1->clone( "vec2" );
    auto vec3a = vec1->clone( "vec3" );
    auto vec3b = vec1->clone( "vec3" );
    auto vec3 = AMP::LinearAlgebra::MultiVector::create( "multivec", comm, { vec2, vec3a, vec3b } );

    bool pass       = true;
    auto selection1 = vec2->select( AMP::LinearAlgebra::VS_ByVariableName( "None" ) );
    if ( selection1 ) {
        ut->failure( "Found vector where there should be none" );
        pass = false;
    }

    selection1 = vec2->select( AMP::LinearAlgebra::VS_ByVariableName( "vec2" ) );
    if ( selection1 ) {
        if ( !compareVecSubset( vec2, selection1 ) ) {
            ut->failure( "Could not find vector" );
            pass = false;
        }
    } else {
        ut->failure( "Did not find a vector" );
    }

    selection1    = vec3->select( AMP::LinearAlgebra::VS_ByVariableName( "vec3" ) );
    auto vec3_sub = AMP::LinearAlgebra::MultiVector::create( "multivec", comm );
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
    if ( vec2->getLocalSize() != vec1->getLocalSize() ||
         vec2->getGlobalSize() != vec1->getLocalSize() ||
         vec2->numberOfDataBlocks() != vec1->numberOfDataBlocks() ||
         vec2->getComm().getSize() != 1 ) {
        ut->failure( "Subset for AMP_COMM_SELF" );
        pass = false;
    }
    for ( int i = vec_comm.getRank(); i < vec_comm.getSize(); i++ )
        vec_comm.barrier();
    // Test subset for set of ranks
    if ( vec_comm.getSize() > 2 ) {
        auto splitComm = vec_comm.split( vec_comm.getRank() % 2 );
        vec2           = AMP::LinearAlgebra::VS_Comm( splitComm ).subset( vec1 );
        if ( vec2->getLocalSize() != vec1->getLocalSize() ||
             vec2->numberOfDataBlocks() != vec1->numberOfDataBlocks() ) {
            ut->failure( "Subset for splitComm" );
            pass = false;
        }
    }
    if ( pass )
        ut->passes( "passed subset by comm" );
}

// Test the behavior of VS_Components
void AMP::LinearAlgebra::VectorTests::test_VS_Component( AMP::UnitTest *ut )
{
    auto vec  = d_factory->getVector();
    size_t N  = vec->getNumberOfComponents();
    size_t n1 = vec->getGlobalSize();
    size_t n2 = 0;
    std::vector<size_t> index;
    for ( size_t i = 0; i < N; i++ ) {
        auto vec2 = vec->subsetVectorForComponent( i );
        AMP_ASSERT( vec2 );
        AMP_ASSERT( vec2->getNumberOfComponents() == 1 );
        n2 += vec2->getGlobalSize();
        index.push_back( i );
    }
    auto vec3 = vec->selectInto( VS_Components( index ) );
    AMP_ASSERT( vec3 );
    size_t n3 = vec3->getGlobalSize();
    if ( n1 == n2 && n1 == n3 )
        ut->passes( "Subset by component" );
    if ( n1 != n2 )
        ut->failure( "subsetVectorForComponent" );
    if ( n1 != n3 )
        ut->failure( "VS_Components (all)" );
}
