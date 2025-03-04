#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/dendro/DendroSearch.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iomanip>
#include <random>


double dummyFunction( const std::vector<double> &xyz, const int dof )
{
    AMP_ASSERT( xyz.size() == 3 );
    double x = xyz[0], y = xyz[1], z = xyz[2];
    //  return 7.0;
    // return (1.0 + 6.0 * x) * (2.0 - 5.0 * y) * (3.0 + 4.0 * z);
    return ( 1.0 + 6.0 * x ) + ( 2.0 - 5.0 * y ) + ( 3.0 + 4.0 * z );
}

void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    int rank = globalComm.getRank();
    int npes = globalComm.getSize();

    // Load the input file
    globalComm.barrier();
    double inpReadBeginTime = MPI_Wtime();

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );
    globalComm.barrier();
    double inpReadEndTime = MPI_Wtime();
    if ( !rank ) {
        std::cout << "Finished parsing the input file in " << ( inpReadEndTime - inpReadBeginTime )
                  << " seconds." << std::endl;
    }

    // Load the mesh
    globalComm.barrier();
    double meshBeginTime = MPI_Wtime();
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto meshAdapter = AMP::Mesh::MeshFactory::create( meshParams );
    globalComm.barrier();
    double meshEndTime = MPI_Wtime();
    if ( !rank ) {
        std::cout << "Finished reading the mesh in " << ( meshEndTime - meshBeginTime )
                  << " seconds." << std::endl;
    }

    // Create a vector field
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    auto DOFs           = AMP::Discretization::simpleDOFManager::create(
        meshAdapter, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    auto dummyVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "Dummy" );
    auto dummyVector   = createVector( DOFs, dummyVariable, split );

    auto node     = meshAdapter->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto end_node = node.end();
    for ( ; node != end_node; ++node ) {
        std::vector<size_t> globalID;
        DOFs->getDOFs( node->globalID(), globalID );
        for ( size_t d = 0; d < globalID.size(); ++d ) {
            dummyVector->setLocalValueByGlobalID( globalID[d], dummyFunction( node->coord(), d ) );
        } // end d
    }
    dummyVector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    double minCoords[3];
    double maxCoords[3];
    std::vector<double> box = meshAdapter->getBoundingBox();
    for ( int i = 0; i < meshAdapter->getDim(); ++i ) {
        minCoords[i] = box[2 * i + 0];
        maxCoords[i] = box[2 * i + 1];
    }

    int totalNumPts = input_db->getScalar<int>( "TotalNumberOfPoints" );
    int avgNumPts   = totalNumPts / npes;
    int extraNumPts = totalNumPts % npes;

    size_t numLocalPts = avgNumPts;
    if ( rank < extraNumPts ) {
        numLocalPts++;
    }

    // Generate Random points in [min, max]
    std::random_device rd;
    std::mt19937 gen( rd() );
    std::uniform_real_distribution<double> dist_x( minCoords[0], maxCoords[0] );
    std::uniform_real_distribution<double> dist_y( minCoords[1], maxCoords[1] );
    std::uniform_real_distribution<double> dist_z( minCoords[2], maxCoords[2] );
    std::vector<double> pts( 3 * numLocalPts );
    for ( size_t i = 0; i < numLocalPts; ++i ) {
        pts[3 * i]     = dist_x( gen );
        pts[3 * i + 1] = dist_y( gen );
        pts[3 * i + 2] = dist_z( gen );
    }
    globalComm.barrier();
    if ( !rank ) {
        std::cout << "Finished generating " << totalNumPts << " random points for search!"
                  << std::endl;
    }

    bool dendroVerbose = input_db->getScalar<bool>( "DENDRO_VERBOSE" );
    globalComm.barrier();
    double dendroConBeginTime = MPI_Wtime();
    AMP::Mesh::DendroSearch dendroSearch( meshAdapter, dendroVerbose );
    globalComm.barrier();
    double dendroConEndTime = MPI_Wtime();
    if ( !rank ) {
        std::cout << "Finished building the DendroSearch object in "
                  << ( dendroConEndTime - dendroConBeginTime ) << " seconds." << std::endl;
    }

    std::vector<double> interpolatedData;
    std::vector<bool> interpolationWasDone;
    globalComm.barrier();
    double dendroSandIbeginTime = MPI_Wtime();
    dendroSearch.searchAndInterpolate(
        globalComm, dummyVector, DOFsPerNode, pts, interpolatedData, interpolationWasDone );
    globalComm.barrier();
    double dendroSandIendTime = MPI_Wtime();
    if ( !rank ) {
        std::cout << "Finished searching and interpolating in "
                  << ( dendroSandIendTime - dendroSandIbeginTime ) << " seconds." << std::endl;
    }
    AMP_ASSERT( interpolatedData.size() == ( DOFsPerNode * numLocalPts ) );
    AMP_ASSERT( interpolationWasDone.size() == numLocalPts );

    int localNotFound = 0;
    if ( numLocalPts > 0 ) {
        localNotFound = static_cast<int>(
            std::count( interpolationWasDone.begin(), interpolationWasDone.end(), false ) );
    }
    int globalNotFound;
    globalNotFound = globalComm.sumReduce( localNotFound );
    if ( !rank ) {
        std::cout << globalNotFound << " points (total) were not found" << std::endl;
    }

    std::vector<double> interpolationError( ( DOFsPerNode * numLocalPts ), 0.0 );
    for ( unsigned int i = 0; i < numLocalPts; ++i ) {
        if ( interpolationWasDone[i] ) {
            for ( int d = 0; d < DOFsPerNode; ++d ) {
                interpolationError[( i * DOFsPerNode ) + d] =
                    fabs( interpolatedData[( i * DOFsPerNode ) + d] -
                          dummyFunction(
                              std::vector<double>( &( pts[3 * i] ), &( pts[3 * i] ) + 3 ), d ) );
            } // end d
        }
    } // end for i
    double localMaxError = 0;
    if ( numLocalPts > 0 ) {
        localMaxError =
            *( std::max_element( interpolationError.begin(), interpolationError.end() ) );
    }
    if ( !rank ) {
        std::cout << "Finished computing the local squared norm of the interpolation error."
                  << std::endl;
    }
    globalComm.barrier();

    double globalMaxError = globalComm.maxReduce<double>( localMaxError );
    if ( !rank ) {
        std::cout << "Global max error is " << std::setprecision( 15 ) << globalMaxError
                  << std::endl;
    }

    AMP_ASSERT( globalMaxError < 1.0e-12 );

    std::vector<AMP::Mesh::MeshElementID> faceVerticesGlobalIDs;
    std::vector<double> shiftGlobalCoords, projectionLocalCoordsOnGeomType::Face;
    std::vector<int> flags;
    dendroSearch.projectOnBoundaryID( globalComm,
                                      4,
                                      faceVerticesGlobalIDs,
                                      shiftGlobalCoords,
                                      projectionLocalCoordsOnGeomType::Face,
                                      flags );

    unsigned int localPtsNotFound =
        std::count( flags.begin(), flags.end(), AMP::Mesh::DendroSearch::NotFound );
    unsigned int localPtsFoundNotOnBoundary =
        std::count( flags.begin(), flags.end(), AMP::Mesh::DendroSearch::FoundNotOnBoundary );
    unsigned int localPtsFoundOnBoundary =
        std::count( flags.begin(), flags.end(), AMP::Mesh::DendroSearch::FoundOnBoundary );
    unsigned int globalPtsNotFound           = globalComm.sumReduce( localPtsNotFound );
    unsigned int globalPtsFoundNotOnBoundary = globalComm.sumReduce( localPtsFoundNotOnBoundary );
    unsigned int globalPtsFoundOnBoundary    = globalComm.sumReduce( localPtsFoundOnBoundary );
    if ( !rank ) {
        std::cout << "Global number of points not found is " << globalPtsNotFound << std::endl;
        std::cout << "Global number of points found not on boundary is "
                  << globalPtsFoundNotOnBoundary << std::endl;
        std::cout << "Global number of points found on boundary is " << globalPtsFoundOnBoundary
                  << std::endl;
        std::cout << "Total number of points is "
                  << globalPtsNotFound + globalPtsFoundNotOnBoundary + globalPtsFoundOnBoundary
                  << std::endl;
    }

    ut->passes( exeName );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName = "testDendroInterpolation";

    myTest( &ut, exeName );

    ut.report();
    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
