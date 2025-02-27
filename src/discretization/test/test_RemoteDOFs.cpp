#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"

#include "ProfilerApp.h"

#include <set>

void testPair( AMP::UnitTest *ut,
               std::shared_ptr<AMP::Discretization::DOFManager> leftDOF,
               std::shared_ptr<AMP::Discretization::DOFManager> rightDOF )
{
    auto comm = leftDOF->getComm();

    // For the remote dofs inside rightDOF to be valid two things must be true
    // 1. Every referenced remote dof must be present
    // 2. Every present remote dof must be referenced
    // By corollary the two sets will be the same size

    const size_t start_row = leftDOF->beginDOF(), end_row = leftDOF->endDOF();
    const size_t start_col = rightDOF->beginDOF(), end_col = rightDOF->endDOF();

    // stick all remote dofs in rightDOF into a set
    std::set<size_t> stored_rDOFs;
    for ( auto &&rdof : rightDOF->getRemoteDOFs() ) {
        stored_rDOFs.insert( rdof );
    }
    const size_t num_stored = stored_rDOFs.size();

    // get all remote dofs that are actually referenced
    bool pass_ref_in_stored = true;
    auto is_remote          = [start_col, end_col]( const size_t rdof ) -> bool {
        return rdof < start_col || rdof >= end_col;
    };
    std::set<size_t> refd_rDOFs;
    for ( size_t row = start_row; row < end_row; ++row ) {
        auto row_dofs = rightDOF->getRowDOFs( leftDOF->getElementID( row ) );
        for ( auto &&rdof : row_dofs ) {
            if ( is_remote( rdof ) ) {
                refd_rDOFs.insert( rdof );
                auto it = stored_rDOFs.insert( rdof );
                if ( it.second ) {
                    // successful insertion means there was a referenced DOF
                    // missing from the stored DOFs
                    pass_ref_in_stored = false;
                }
            }
        }
    }
    const size_t num_refd = refd_rDOFs.size();

    // now attempt inserting all stored dofs into referenced
    bool pass_stored_is_refd = true;
    for ( auto &&rdof : stored_rDOFs ) {
        auto it = refd_rDOFs.insert( rdof );
        if ( it.second ) {
            // successful insertion means that a stored dof was never referenced
            pass_stored_is_refd = false;
        }
    }

    if ( pass_ref_in_stored ) {
        ut->passes( "All referenced remote DOFs present in DOFManager" );
    } else {
        ut->failure( "Some referenced remote DOFs missing from DOFManager" );
    }
    if ( pass_stored_is_refd ) {
        ut->passes( "All remote DOFs in DOFManager are referenced" );
    } else {
        ut->failure( "DOFManager contains un-referenced remote DOFs" );
    }

    comm.barrier();

    if ( !pass_ref_in_stored || !pass_stored_is_refd || true ) {
        std::cout << "Rank " << comm.getRank() << " stored " << num_stored << " DOFs, referenced "
                  << num_refd << " DOFs, with " << rightDOF->numLocalDOF() << " local DOFs and "
                  << rightDOF->numGlobalDOF() << " global DOFs" << std::endl;
    }
    comm.barrier();
}

void remoteDOFTest( AMP::UnitTest *ut, std::string input_file )
{
    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    auto comm     = AMP::AMP_MPI( AMP_COMM_WORLD );
    params->setComm( comm );
    if ( comm.getSize() == 1 ) {
        ut->passes( "Trivial pass for single rank runs" );
        return;
    }

    comm.barrier();
    AMP::pout << "Testing vertex remote DOFs with input: " << input_file << std::endl;
    comm.barrier();

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Create DOFManagers over different entities
    auto vertexDOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );
    auto cellDOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Cell, 1, 1 );

    // test combinations of managers
    testPair( ut, vertexDOFs, vertexDOFs );
    testPair( ut, cellDOFs, cellDOFs );
}

int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    std::vector<std::string> files;
    PROFILE_ENABLE();

    if ( argc > 1 ) {
        files.emplace_back( argv[1] );
    } else {
        files.emplace_back( "input_testRemoteDOFs-boxmesh-1" );
        files.emplace_back( "input_testRemoteDOFs-boxmesh-2" );
#ifdef AMP_USE_LIBMESH
        files.emplace_back( "input_testRemoteDOFs-libmesh-1" );
        files.emplace_back( "input_testRemoteDOFs-libmesh-2" );
#endif
    }

    for ( auto &file : files ) {
        remoteDOFTest( &ut, file );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
