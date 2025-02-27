#include "AMP/discretization/testHelpers/discretizationTests.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/discretization/structuredFaceDOFManager.h"
#include "AMP/discretization/subsetDOFManager.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/structured/PureLogicalMesh.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/MeshVariable.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"


#define PASS_FAIL( test, MSG ) \
    do {                       \
        if ( test )            \
            ut.passes( MSG );  \
        else                   \
            ut.failure( MSG ); \
    } while ( 0 )


namespace AMP::unit_test {


constexpr auto Vertex = AMP::Mesh::GeomType::Vertex;


// Function to test getting the DOFs for a mesh iterator
void testGetDOFIterator( AMP::UnitTest &ut,
                         const AMP::Mesh::MeshIterator &iterator,
                         std::shared_ptr<AMP::Discretization::DOFManager> DOF )
{
    bool pass1 = true;
    bool pass2 = true;
    auto cur   = iterator.begin();
    std::vector<size_t> dofs;
    for ( size_t i = 0; i < cur.size(); i++ ) {
        const auto id = cur->globalID();
        DOF->getDOFs( id, dofs );
        if ( dofs.empty() )
            pass1 = false;
        for ( size_t dof : dofs ) {
            auto id2 = DOF->getElement( dof ).globalID();
            if ( id2 != id )
                pass2 = false;
        }
        ++cur;
    }
    PASS_FAIL( pass1, "Got the DOFs for every element in iterator" );
    PASS_FAIL( pass2, "getElementID" );
}


// Function to test some very basic properites
void testBasics( std::shared_ptr<AMP::Discretization::DOFManager> DOF, AMP::UnitTest &ut )
{
    bool passAll = true;

    // Get the comm
    AMP::AMP_MPI comm = DOF->getComm();
    if ( comm.isNull() ) {
        passAll = false;
        ut.failure( "Comm is not valid" );
    }

    // Get the dof ranges
    size_t start    = DOF->beginDOF();
    size_t end      = DOF->endDOF();
    size_t N_local  = DOF->numLocalDOF();
    size_t N_global = DOF->numGlobalDOF();
    if ( N_local != end - start ) {
        passAll = false;
        ut.failure( "N_local is the wrong size" );
    }
    if ( N_global != comm.sumReduce( N_local ) ) {
        passAll = false;
        ut.failure( "N_global is the wrong size" );
    }

    // Trivial check of == and !=
    if ( *DOF != *DOF || !( *DOF == *DOF ) ) {
        passAll = false;
        ut.failure( "Failed operator==" );
    }

    // Test getting the DOFs for the DOF iterator
    AMP::UnitTest ut2;
    testGetDOFIterator( ut2, DOF->getIterator(), DOF );
    if ( ut2.NumFailLocal() != 0 ) {
        passAll = false;
        ut.failure( "Failed checking local iterator" );
    }

    // Check the results of the basic tests
    PASS_FAIL( passAll, "Basic DOF tests" );
}


// Test correct accounting of remote DOFs
// made to mimic how matrices are constructed
void testRemoteDOFs( std::shared_ptr<AMP::Discretization::DOFManager> DOF, AMP::UnitTest &ut )
{
    // first get all remote dofs that are actually referenced
    std::set<size_t> remote_DOFs;
    for ( size_t row = DOF->beginDOF(); row < DOF->endDOF(); ++row ) {
        auto row_dofs = DOF->getRowDOFs( DOF->getElementID( row ) );
        for ( auto &&rdof : row_dofs ) {
            remote_DOFs.insert( rdof );
        }
    }

    const auto num_remote_used = remote_DOFs.size();

    // then check that stored remote dofs are the same
    // Inserting any *should* fail, so if one does
    // enter the std::set above then it is an
    // unnecessary remote dof
    bool pass = true;
    for ( auto &&rdof : DOF->getRemoteDOFs() ) {
        auto it = remote_DOFs.insert( rdof );
        if ( it.second ) {
            pass = false;
        }
    }

    if ( !pass ) {
        std::cout << "testRemoteDOFs on rank " << DOF->getComm().getRank() << ": DOFManager has "
                  << remote_DOFs.size() << " remote DOFs, but should only have " << num_remote_used
                  << ". Global size is " << DOF->numGlobalDOF() << ". Type is " << DOF->className()
                  << "." << std::endl;
        ut.expected_failure( "Unnecessary remote DOFs present" );
        return;
    }

    ut.passes( "Test remote DOFs" );
}

// Test subsetting for different comms
void testSubsetComm( std::shared_ptr<AMP::Discretization::DOFManager> DOF, AMP::UnitTest &ut )
{
    auto subsetDOF = DOF->subset( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    PASS_FAIL( *DOF == *subsetDOF, "Subset DOF on COMM_WORLD" );
    subsetDOF = DOF->subset( DOF->getComm() );
    PASS_FAIL( *DOF == *subsetDOF, "Subset DOF on DOF comm" );
    subsetDOF = DOF->subset( AMP::AMP_MPI( AMP_COMM_SELF ) );
    PASS_FAIL( subsetDOF->numLocalDOF() == DOF->numLocalDOF() &&
                   subsetDOF->numGlobalDOF() == DOF->numLocalDOF(),
               "Subset DOF on COMM_SELF" );
}


// Test subsetting for different meshes
void testSubsetMesh( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                     std::shared_ptr<AMP::Discretization::DOFManager> DOF,
                     bool is_nodal,
                     int DOFsPerNode,
                     int gcw,
                     AMP::UnitTest &ut )
{
    auto subsetDOF = DOF->subset( mesh );
    PASS_FAIL( *DOF == *subsetDOF, "Subset DOF on full mesh" );
    auto meshIDs = mesh->getBaseMeshIDs();
    if ( meshIDs.size() > 1 && is_nodal ) {
        bool passes = true;
        for ( size_t i = 0; i < meshIDs.size(); i++ ) {
            auto subsetMesh = mesh->Subset( meshIDs[i] );
            if ( subsetMesh ) {
                subsetDOF = DOF->subset( subsetMesh );
                testGetDOFIterator( ut, subsetMesh->getIterator( Vertex, gcw ), subsetDOF );
                auto mesh_DOF = AMP::Discretization::simpleDOFManager::create(
                    subsetMesh, Vertex, gcw, DOFsPerNode, false );
                if ( *mesh_DOF != *subsetDOF )
                    passes = false;
            }
        }
        PASS_FAIL( passes, "Subset DOF for each mesh" );
    }
}


// Function to test the index conversion given a multDOFManager
void testMultiDOFMap( AMP::UnitTest &ut,
                      std::shared_ptr<AMP::Discretization::multiDOFManager> multiDOF )
{
    // First create a global DOF list
    size_t N_global = multiDOF->numGlobalDOF();
    std::vector<size_t> globalDOFList( N_global );
    for ( size_t i = 0; i < N_global; i++ )
        globalDOFList[i] = i;

    // Loop through the DOFManagers
    auto managers = multiDOF->getDOFManagers();
    for ( size_t i = 0; i < managers.size(); i++ ) {
        AMP::AMP_MPI comm = managers[i]->getComm();
        comm.barrier();
        // Convert the global list to a local list
        std::vector<size_t> tmp = multiDOF->getSubDOF( (int) i, globalDOFList );
        std::vector<size_t> localDOFList, localToGlobal;
        localDOFList.reserve( N_global );
        localToGlobal.reserve( N_global );
        size_t neg_one = ~( (size_t) 0 );
        for ( size_t j = 0; j < N_global; j++ ) {
            if ( tmp[j] != neg_one ) {
                localDOFList.push_back( tmp[j] );
                localToGlobal.push_back( j );
            }
        }
        // Check that we created the proper list
        bool passes = localDOFList.size() == managers[i]->numGlobalDOF();
        for ( size_t j = 0; j < localDOFList.size(); j++ ) {
            if ( localDOFList[j] != j )
                passes = false;
        }
        PASS_FAIL( passes, "Conversion from global to sub DOFs in multiDOFManager" );
        // Check that we can convert back to the global ids
        auto globalDOFList2 = multiDOF->getGlobalDOF( (int) i, localDOFList );
        passes              = globalDOFList2.size() == localDOFList.size();
        for ( size_t j = 0; j < globalDOFList2.size(); j++ ) {
            if ( globalDOFList2[j] != localToGlobal[j] )
                passes = false;
        }
        PASS_FAIL( passes, "Conversion from sub to global DOFs in multiDOFManager" );
    }
}


// Function to test that a multivector with a DOFManager repeated correctly sets the values
void testMultiDOFVector( AMP::UnitTest &ut, std::shared_ptr<AMP::Discretization::DOFManager> DOF )
{
    // Create the individual vectors
    auto var1 = std::make_shared<AMP::LinearAlgebra::Variable>( "a" );
    auto var2 = std::make_shared<AMP::LinearAlgebra::Variable>( "b" );
    auto var3 = std::make_shared<AMP::LinearAlgebra::Variable>( "c" );
    auto vec1 = AMP::LinearAlgebra::createVector( DOF, var1, true );
    auto vec2 = AMP::LinearAlgebra::createVector( DOF, var2, true );
    // Create the multivector
    auto multiVector = AMP::LinearAlgebra::MultiVector::create( var3, DOF->getComm() );
    multiVector->addVector( { vec1, vec2 } );
    auto multiDOF = multiVector->getDOFManager();
    // Check that we can set each value correctly
    multiVector->zero();
    multiVector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    AMP::Mesh::MeshIterator it = DOF->getIterator();
    std::vector<size_t> dof1, dof2;
    bool uniqueMultiDOFs = true;
    for ( size_t i = 0; i < it.size(); i++ ) {
        DOF->getDOFs( it->globalID(), dof1 );
        multiDOF->getDOFs( it->globalID(), dof2 );
        AMP_ASSERT( dof1.size() == 1 );
        AMP_ASSERT( dof2.size() == 2 );
        if ( dof2[0] == dof2[1] )
            uniqueMultiDOFs = false;
        double val = dof1[0];
        multiVector->setValuesByGlobalID( 1, &dof2[0], &val );
        val = 2 * dof1[0];
        multiVector->setValuesByGlobalID( 1, &dof2[1], &val );
        ++it;
    }
    PASS_FAIL( uniqueMultiDOFs,
               "MultiDOFManger with duplicate subDOFManagers returns unique DOFs" );
    multiVector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    double vec1norm        = static_cast<double>( vec1->L1Norm() );
    double vec2norm        = static_cast<double>( vec2->L1Norm() );
    double multiVectorNorm = static_cast<double>( multiVector->L1Norm() );
    double N_tot           = ( DOF->numGlobalDOF() * ( DOF->numGlobalDOF() - 1 ) ) / 2;
    PASS_FAIL( vec1norm == N_tot && vec2norm == 2 * N_tot && multiVectorNorm == 3 * N_tot,
               "MultiVector with repeated DOFs sets values correctly" );
}


// Test SimpleDOFManager / boxMeshDOFManager with a logical mesh
static std::vector<size_t> createMeshSize( int ndim )
{
    if ( ndim == 1 )
        return { 20 };
    else if ( ndim == 2 )
        return { 15, 10 };
    else if ( ndim == 3 )
        return { 20, 15, 10 };
    else
        AMP_ERROR( "Invalid number of dimensions" );
}
static size_t expectedRowSize( int ndim, AMP::Mesh::GeomType type )
{
    if ( type == AMP::Mesh::GeomType::Vertex )
        return std::pow( 3, ndim );
    else if ( static_cast<int>( type ) == ndim )
        return 2 * ndim + 1;
    else if ( type == AMP::Mesh::GeomType::Edge && ndim == 2 )
        return 7;
    else if ( type == AMP::Mesh::GeomType::Edge && ndim == 3 )
        return 33;
    else if ( type == AMP::Mesh::GeomType::Face && ndim == 3 )
        return 11;
    else
        AMP_ERROR( "Internal error" );
}
void testLogicalDOFMap( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                        AMP::Mesh::GeomType type,
                        int gcw,
                        int DOFsPerElement,
                        AMP::UnitTest &ut )
{
    int commSize = AMP::AMP_MPI( AMP_COMM_WORLD ).getSize();
    auto msg     = AMP::Utilities::stringf( "testLogicalDOFMap (ndim=%i) (%s) (gcw=%i) (dofs=%i)",
                                        mesh->getDim(),
                                        AMP::Mesh::to_string( type ).data(),
                                        gcw,
                                        DOFsPerElement );
    // Create the DOF managers
    auto local = mesh->getIterator( type, 0 );
    auto ghost = mesh->getIterator( type, gcw );
    auto dofs1 =
        AMP::Discretization::simpleDOFManager::create( mesh, type, gcw, DOFsPerElement, true );
    auto dofs2 =
        AMP::Discretization::simpleDOFManager::create( mesh, ghost, local, DOFsPerElement );
    AMP_ASSERT( std::dynamic_pointer_cast<AMP::Discretization::boxMeshDOFManager>( dofs1 ) );
    AMP_ASSERT( !std::dynamic_pointer_cast<AMP::Discretization::boxMeshDOFManager>( dofs2 ) );
    AMP_ASSERT( std::dynamic_pointer_cast<AMP::Discretization::simpleDOFManager>( dofs1 ) );
    AMP_ASSERT( std::dynamic_pointer_cast<AMP::Discretization::simpleDOFManager>( dofs2 ) );
    testRemoteDOFs( dofs1, ut );
    testRemoteDOFs( dofs2, ut );

    // Check the sizes on the DOF managers
    bool pass = true;
    pass      = pass && dofs1->numLocalDOF() == dofs2->numLocalDOF();
    pass      = pass && dofs1->numGlobalDOF() == dofs2->numGlobalDOF();
    pass      = pass && dofs1->getRemoteDOFs().size() == dofs2->getRemoteDOFs().size();
    pass      = pass && dofs1->numLocalDOF() == DOFsPerElement * mesh->numLocalElements( type );
    pass      = pass && dofs1->numGlobalDOF() == DOFsPerElement * mesh->numGlobalElements( type );
    if ( commSize > 1 )
        pass = pass && dofs1->getRemoteDOFs().size() ==
                           DOFsPerElement * mesh->numGhostElements( type, gcw );
    else
        pass = pass && dofs1->getRemoteDOFs().empty();
    if ( !pass ) {
        ut.failure( msg + " basic checks" );
        return;
    }

    // Check getRow
    size_t size       = DOFsPerElement * expectedRowSize( mesh->getDim(), type );
    size_t minRowSize = size;
    size_t maxRowSize = 0;
    for ( size_t dof = dofs1->beginDOF(); dof < dofs1->endDOF(); dof++ ) {
        auto id1   = dofs1->getElementID( dof );
        auto id2   = dofs2->getElementID( dof );
        auto row1  = dofs1->getRowDOFs( id1 );
        auto row2  = dofs2->getRowDOFs( id2 );
        pass       = pass && id1 == id2;
        pass       = pass && row1.size() == row2.size();
        minRowSize = std::min( minRowSize, row1.size() );
        maxRowSize = std::max( maxRowSize, row1.size() );
    }
    if ( commSize == 1 || gcw >= 1 ) {
        pass = pass && minRowSize == size;
        pass = pass && maxRowSize == size;
    } else {
        pass = pass && minRowSize < size;
        pass = pass && maxRowSize == size;
    }
    PASS_FAIL( pass, msg );
}
void testLogicalDOFMap( int ndim, AMP::UnitTest &ut )
{
    // Create the mesh
    auto size = createMeshSize( ndim );
    std::vector<bool> per( size.size(), true );
    auto db =
        AMP::Database::create( "MeshName", "domain", "Size", size, "Periodic", per, "GCW", 2 );
    auto params = std::make_shared<AMP::Mesh::MeshParameters>( std::move( db ) );
    params->setComm( AMP_COMM_WORLD );
    auto mesh = std::make_shared<AMP::Mesh::PureLogicalMesh>( params );

    // Run different tests
    for ( int i = 0; i <= ndim; i++ ) {
        auto type = static_cast<AMP::Mesh::GeomType>( i );
        for ( int gcw = 0; gcw <= 2; gcw++ ) {
            testLogicalDOFMap( mesh, type, gcw, 1, ut );
            testLogicalDOFMap( mesh, type, gcw, 3, ut );
        }
    }
}
void testLogicalDOFMap( AMP::UnitTest &ut )
{
    testLogicalDOFMap( 1, ut );
    testLogicalDOFMap( 2, ut );
    testLogicalDOFMap( 3, ut );
}


} // namespace AMP::unit_test
