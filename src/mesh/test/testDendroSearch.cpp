#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/dendro/DendroSearch.h"
#include "AMP/mesh/euclidean_geometry_tools.h"
#include "AMP/mesh/latex_visualization_tools.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>


void draw_hex8_element_revisited( hex8_element_t *e_ptr,
                                  double const *point_of_view,
                                  std::ostream &os )
{
    os << "\\tikzset{facestyle/.style={opacity=0.4,line join=round}}\n";
    std::vector<std::string> options( 6, "facestyle," );
    triangle_t **t_ptr = e_ptr->get_bounding_polyhedron();
    for ( unsigned int f = 0; f < 6; ++f ) {
        if ( compute_scalar_product( point_of_view, t_ptr[2 * f]->get_normal() ) > 0.0 ) {
            options[f] += "fill=none";
        } else {
            options[f] += "fill=none,dotted";
        } // end if
        draw_face( e_ptr, f, options[f], os );
    }
    for ( unsigned int p = 0; p < 8; ++p ) {
        draw_point( e_ptr->get_support_point( p ), "", os, "$\\cdot$" );
    } // end for p
}

void drawOctant( double const *octant, double const *point_of_view, std::ostream &os )
{
    double x_min              = octant[0];
    double x_max              = octant[1];
    double y_min              = octant[2];
    double y_max              = octant[3];
    double z_min              = octant[4];
    double z_max              = octant[5];
    double support_points[24] = { x_min, y_min, z_min, x_max, y_min, z_min, x_max, y_max,
                                  z_min, x_min, y_max, z_min, x_min, y_min, z_max, x_max,
                                  y_min, z_max, x_max, y_max, z_max, x_min, y_max, z_max };
    hex8_element_t volume_element( support_points );
    draw_hex8_element_revisited( &volume_element, point_of_view, os );
    double centroid_local_coordinates[3] = { 0.0, 0.0, 0.0 };
    double centroid_global_coordinates[3];
    volume_element.map_local_to_global( centroid_local_coordinates, centroid_global_coordinates );
    draw_point( centroid_global_coordinates, "blue", os, "C" );
}

void buildOctant( double const *space,
                  size_t x,
                  size_t y,
                  size_t z,
                  size_t level,
                  size_t max_depth,
                  double *octant )
{
    AMP_ASSERT( level <= max_depth );
    AMP_ASSERT( x < static_cast<size_t>( 1u << max_depth ) );
    AMP_ASSERT( x % static_cast<size_t>( 1u << ( max_depth - level ) ) == 0 );
    AMP_ASSERT( y < static_cast<size_t>( 1u << max_depth ) );
    AMP_ASSERT( y % static_cast<size_t>( 1u << ( max_depth - level ) ) == 0 );
    AMP_ASSERT( z < static_cast<size_t>( 1u << max_depth ) );
    AMP_ASSERT( z % static_cast<size_t>( 1u << ( max_depth - level ) ) == 0 );
    octant[0] = space[0] + ( space[1] - space[0] ) * static_cast<double>( x ) /
                               static_cast<double>( 2 * max_depth );
    octant[1] = space[0] +
                ( space[1] - space[0] ) *
                    static_cast<double>( x + static_cast<size_t>( 1u << ( max_depth - level ) ) ) /
                    static_cast<double>( 2 * max_depth );
    octant[2] = space[2] + ( space[3] - space[2] ) * static_cast<double>( y ) /
                               static_cast<double>( 2 * max_depth );
    octant[3] = space[2] +
                ( space[3] - space[2] ) *
                    static_cast<double>( y + static_cast<size_t>( 1u << ( max_depth - level ) ) ) /
                    static_cast<double>( 2 * max_depth );
    octant[4] = space[4] + ( space[5] - space[4] ) * static_cast<double>( z ) /
                               static_cast<double>( 2 * max_depth );
    octant[5] = space[4] +
                ( space[5] - space[4] ) *
                    static_cast<double>( z + static_cast<size_t>( 1u << ( max_depth - level ) ) ) /
                    static_cast<double>( 2 * max_depth );
}

void drawOctant( double const *space,
                 size_t x,
                 size_t y,
                 size_t z,
                 size_t level,
                 size_t max_depth,
                 double const *point_of_view,
                 std::ostream &os )
{
    double octant[8];
    buildOctant( space, x, y, z, level, max_depth, octant );
    drawOctant( octant, point_of_view, os );
}

void drawSpacePartition( std::shared_ptr<AMP::Mesh::Mesh> meshAdapter,
                         double const *point_of_view,
                         std::ostream &os )
{
    auto space = meshAdapter->getBoundingBox();
    drawOctant( &( space[0] ), 0, 0, 0, 0, 2, point_of_view, os );

    drawOctant( &( space[0] ), 0, 0, 0, 1, 2, point_of_view, os );
    drawOctant( &( space[0] ), 2, 0, 0, 1, 2, point_of_view, os );
    drawOctant( &( space[0] ), 2, 2, 0, 1, 2, point_of_view, os );
    drawOctant( &( space[0] ), 0, 2, 0, 1, 2, point_of_view, os );
    drawOctant( &( space[0] ), 0, 0, 2, 1, 2, point_of_view, os );
    drawOctant( &( space[0] ), 2, 0, 2, 1, 2, point_of_view, os );
    drawOctant( &( space[0] ), 2, 2, 2, 1, 2, point_of_view, os );
    drawOctant( &( space[0] ), 0, 2, 2, 1, 2, point_of_view, os );

    drawOctant( &( space[0] ), 0, 2, 2, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 1, 2, 2, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 1, 3, 2, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 0, 3, 2, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 0, 2, 3, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 1, 2, 3, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 1, 3, 3, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 0, 3, 3, 2, 2, point_of_view, os );

    drawOctant( &( space[0] ), 2, 0, 0, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 3, 0, 0, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 3, 1, 0, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 2, 1, 0, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 2, 0, 1, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 3, 0, 1, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 3, 1, 1, 2, 2, point_of_view, os );
    drawOctant( &( space[0] ), 2, 1, 1, 2, 2, point_of_view, os );
}

void drawGeomType::FacesOnBoundaryID( std::shared_ptr<AMP::Mesh::Mesh> meshAdapter,
                                      int boundaryID,
                                      std::ostream &os,
                                      double const *point_of_view,
                                      const std::string &option = "" )
{
    auto boundaryIterator =
        meshAdapter->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, boundaryID );
    auto boundaryIterator_begin = boundaryIterator.begin();
    auto boundaryIterator_end   = boundaryIterator.end();
    std::vector<AMP::Mesh::MeshElement> faceVertices;
    std::vector<double> faceGeomType::VertexCoordinates;
    double faceData[12]          = { 0.0 };
    double const *faceDataPtr[4] = { faceData, faceData + 3, faceData + 6, faceData + 9 };

    os << std::setprecision( 6 ) << std::fixed;

    for ( boundaryIterator = boundaryIterator_begin; boundaryIterator != boundaryIterator_end;
          ++boundaryIterator ) {
        faceVertices = boundaryIterator->getElements( AMP::Mesh::GeomType::Vertex );
        AMP_ASSERT( faceVertices.size() == 4 );
        for ( size_t i = 0; i < 4; ++i ) {
            faceGeomType::VertexCoordinates = faceVertices[i].coord();
            AMP_ASSERT( faceGeomType::VertexCoordinates.size() == 3 );
            std::copy( faceGeomType::VertexCoordinates.begin(),
                       faceGeomType::VertexCoordinates.end(),
                       faceData + 3 * i );
        } // end for i
        triangle_t t( faceDataPtr[0], faceDataPtr[1], faceDataPtr[2] );

        if ( compute_scalar_product( point_of_view, t.get_normal() ) > 0.0 ) {
            //    if (true) {
            os << "\\draw[" << option << "]\n";
            write_face( faceDataPtr, os );
        } // end if
    }     // end for
}

double dummyFunction( const std::vector<double> &xyz, const int dof )
{
    AMP_ASSERT( xyz.size() == 3 );
    double x = xyz[0], y = xyz[1], z = xyz[2];
    //  return 7.0;
    // return (1.0 + 6.0 * x) * (2.0 - 5.0 * y) * (3.0 + 4.0 * z);
    return ( 1.0 + 6.0 * x ) + ( 2.0 - 5.0 * y ) + ( 3.0 + 4.0 * z );
}

void genGaussPts( int rank, size_t numOctPtsPerProc, std::vector<double> &pts );
void genUniformPts( int rank, size_t numOctPtsPerProc, std::vector<double> &pts );

void rescalePts( std::vector<double> &pts );

double gaussian( double mean, double std_deviation );

void run( const std::string &meshFileName,
          size_t numRandomPts,
          void ( *randomPtsGenerator )( int, size_t, std::vector<double> & ),
          std::vector<double> &timingMeasurements )
{

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    int rank = globalComm.getRank();
    int npes = globalComm.getSize();

    int size_radius = 16, size_height = 38;

    int numberOfMeshes = npes;
    // Load the mesh
    globalComm.barrier();
    double meshBeginTime = MPI_Wtime();
    auto mesh_db         = std::make_shared<AMP::Database>( "input_db" );
    mesh_db->putScalar( "MeshName", "PelletMeshes" );
    mesh_db->putScalar( "MeshType", "Multimesh" );
    mesh_db->putScalar( "MeshDatabasePrefix", "Mesh_" );
    mesh_db->putScalar( "MeshArrayDatabasePrefix", "MeshArray_" );
    if ( false ) {
        mesh_db->putDatabase( "MeshArray_1" );
        auto meshArray_db = mesh_db->getDatabase( "MeshArray_1" );
        meshArray_db->putScalar( "N", numberOfMeshes );
        meshArray_db->putScalar( "iterator", "%i" );
        std::vector<int> meshIndices;
        std::vector<double> x_offset, y_offset, z_offset;
        for ( int i = 0; i < numberOfMeshes; ++i ) {
            meshIndices.push_back( i );
            x_offset.push_back( 0.0 );
            y_offset.push_back( 0.0 );
            z_offset.push_back( 0.0105 * static_cast<double>( i ) );
        } // end for i
        meshArray_db->putVector<int>( "indicies", meshIndices );
        meshArray_db->putScalar( "MeshName", "Pellet_%i" );
        meshArray_db->putScalar( "MeshType", "libMesh" );
        meshArray_db->putScalar( "FileName", meshFileName );
        meshArray_db->putScalar( "dim", 3 );
        meshArray_db->putVector( "x_offset", x_offset );
        meshArray_db->putVector( "y_offset", y_offset );
        meshArray_db->putVector( "z_offset", z_offset );
    } else {
        // WEAK SCALING
        numRandomPts *= static_cast<size_t>( npes );

        mesh_db->putDatabase( "Mesh_1" );
        auto meshArray_db = mesh_db->getDatabase( "Mesh_1" );
        meshArray_db->putScalar( "MeshName", "Cylinder_1" );
        meshArray_db->putScalar( "MeshType", "AMP" );
        meshArray_db->putScalar( "dim", 3 );
        meshArray_db->putScalar( "x_offset", 0.0 );
        meshArray_db->putScalar( "y_offset", 0.0 );
        meshArray_db->putScalar( "z_offset", 0.0 );
        meshArray_db->putScalar( "Generator", "cylinder" );
        std::vector<int> size;
        // if (true) {
        //  numRandomPts = 3200000;
        //  size.push_back(size_radius+48);
        //  size.push_back(size_height+38);
        //} else
        if ( npes == 1 ) {
            size.push_back( size_radius );
            size.push_back( size_height );
        } else if ( npes == 2 ) {
            size.push_back( size_radius + 5 );
            size.push_back( size_height + 6 );
            //    size.push_back(size_radius);
            //    size.push_back(size_height*2);
        } else if ( npes == 4 ) {
            size.push_back( size_radius + 9 );
            size.push_back( size_height + 24 );
            //    size.push_back(size_radius*2);
            //    size.push_back(size_height);
        } else if ( npes == 8 ) {
            size.push_back( size_radius + 16 );
            size.push_back( size_height + 38 );
            //    size.push_back(size_radius*2);
            //    size.push_back(size_height*2);
        } else if ( npes == 16 ) {
            size.push_back( size_radius + 29 );
            size.push_back( size_height + 39 );
            //    size.push_back(size_radius*2);
            //    size.push_back(size_height*4);
        } else if ( npes == 32 ) {
            size.push_back( size_radius + 48 );
            size.push_back( size_height + 38 );
            //    size.push_back(size_radius*4);
            //    size.push_back(size_height*2);
        } else {
            AMP_ASSERT( false );
        }
        std::vector<double> range;
        range.push_back( 0.004095 );
        range.push_back( 0.0 );
        range.push_back( 0.01 );
        meshArray_db->putVector<int>( "Size", size );
        meshArray_db->putVector( "Range", range );
    }


    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto meshAdapter = AMP::Mesh::MeshFactory::create( meshParams );
    globalComm.barrier();
    double meshEndTime                = MPI_Wtime();
    size_t numberGlobalVolumeElements = meshAdapter->numGlobalElements( AMP::Mesh::GeomType::Cell );
    //  AMP_ASSERT(static_cast<size_t>(npes) * size_radius * size_radius * 4 * size_height ==
    //  numberGlobalVolumeElements);
    if ( !rank ) {
        std::cout << "Number volume elements is " << numberGlobalVolumeElements << "\n";
        std::cout << "Finished reading the mesh in " << ( meshEndTime - meshBeginTime )
                  << " seconds." << std::endl;
    }

    /*  double const point_of_view[3] = { 1.0, 1.0, 1.0 };

      std::fstream fout;
      std::string file_name;
      file_name = "outer_surface_" + meshFileName;
      fout.open(file_name.c_str(), std::fstream::out);
      drawGeomType::FacesOnBoundaryID(meshAdapter, 4, fout, point_of_view);
      fout.close();
      file_name = "top_surface_" + meshFileName;
      fout.open(file_name.c_str(), std::fstream::out);
      drawGeomType::FacesOnBoundaryID(meshAdapter, 1, fout, point_of_view);
      drawGeomType::FacesOnBoundaryID(meshAdapter, 33, fout, point_of_view);
      fout.close();
    */

    //  drawSpacePartition(meshAdapter, point_of_view, std::cout);
    //  std::cout<<std::flush;
    //  char a;
    //  std::cin>>a;

    // Build the vector field
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

    // Generate the random points
    int totalNumPts = numRandomPts;

    int avgNumPts   = totalNumPts / npes;
    int extraNumPts = totalNumPts % npes;

    size_t numLocalPts = avgNumPts;
    if ( rank < extraNumPts ) {
        numLocalPts++;
    }

    std::vector<double> pts;
    randomPtsGenerator( rank, numLocalPts, pts );

    double minCoords[3];
    double maxCoords[3];
    std::vector<double> box = meshAdapter->getBoundingBox();
    for ( unsigned char i = 0; i < meshAdapter->getDim(); ++i ) {
        minCoords[i] = box[2 * i + 0];
        maxCoords[i] = box[2 * i + 1];
    } // end for i

    for ( size_t i = 0; i < numLocalPts; ++i ) {
        pts[3 * i + 0] = ( maxCoords[0] - minCoords[0] ) * pts[3 * i + 0] + minCoords[0];
        pts[3 * i + 1] = ( maxCoords[1] - minCoords[1] ) * pts[3 * i + 1] + minCoords[1];
        pts[3 * i + 2] = ( maxCoords[2] - minCoords[2] ) * pts[3 * i + 2] + minCoords[2];
    } // end for i
    if ( !rank ) {
        std::cout << "Finished generating " << totalNumPts << " random points for search!"
                  << std::endl;
    }

    // Perform the search
    bool dendroVerbose = false;
    AMP::Mesh::DendroSearch dendroSearch( meshAdapter, dendroVerbose );
    dendroSearch.search( globalComm, pts );

    // Interpolate
    std::vector<double> interpolatedData;
    std::vector<bool> interpolationWasDone;
    dendroSearch.interpolate(
        globalComm, dummyVector, DOFsPerNode, interpolatedData, interpolationWasDone );

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
    double localMaxError = 0.0;
    if ( numLocalPts > 0 ) {
        localMaxError =
            *( std::max_element( interpolationError.begin(), interpolationError.end() ) );
    }
    double globalMaxError = globalComm.maxReduce<double>( localMaxError );
    if ( !rank ) {
        std::cout << "Intepolation global max error is " << std::setprecision( 15 )
                  << globalMaxError << std::endl;
    }

    AMP_ASSERT( globalMaxError < 1.0e-12 );

    // Project on boundary
    std::vector<AMP::Mesh::MeshElementID> faceVerticesGlobalIDs;
    std::vector<double> shiftGlobalCoords, projectionLocalCoordsOnGeomType::Face;
    std::vector<int> flags;
    dendroSearch.projectOnBoundaryID( globalComm,
                                      4,
                                      faceVerticesGlobalIDs,
                                      shiftGlobalCoords,
                                      projectionLocalCoordsOnGeomType::Face,
                                      flags );

    size_t localPtsNotFound =
        std::count( flags.begin(), flags.end(), AMP::Mesh::DendroSearch::NotFound );
    size_t localPtsFoundNotOnBoundary =
        std::count( flags.begin(), flags.end(), AMP::Mesh::DendroSearch::FoundNotOnBoundary );
    size_t localPtsFoundOnBoundary =
        std::count( flags.begin(), flags.end(), AMP::Mesh::DendroSearch::FoundOnBoundary );
    size_t globalPtsNotFound           = globalComm.sumReduce( localPtsNotFound );
    size_t globalPtsFoundNotOnBoundary = globalComm.sumReduce( localPtsFoundNotOnBoundary );
    size_t globalPtsFoundOnBoundary    = globalComm.sumReduce( localPtsFoundOnBoundary );
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

    AMP::Mesh::DendroSearch::TimingType timingTypes[5] = {
        AMP::Mesh::DendroSearch::Setup,
        AMP::Mesh::DendroSearch::CoarseSearch,
        AMP::Mesh::DendroSearch::FineSearch,
        AMP::Mesh::DendroSearch::Interpolation,
        AMP::Mesh::DendroSearch::ProjectionOnBoundaryID
    };
    timingMeasurements.resize( 5 );
    dendroSearch.reportTiming( 5, timingTypes, &( timingMeasurements[0] ) );

    timingMeasurements.push_back( timingMeasurements[1] + timingMeasurements[2] );
    timingMeasurements.push_back( timingMeasurements[0] + timingMeasurements[1] +
                                  timingMeasurements[2] );
    //  localTimingMeasurements[4] = std::accumulate(localTimingMeasurements,
    //  localTimingMeasurements+4, 0.0);
    //
    //  globalComm.maxReduce(localTimingMeasurements, globalTimingMeasurements, 5);
    //
    //  if(!rank) {
    //    std::cout<<"Setup time = "<<globalTimingMeasurements[0]<<"\n";
    //    std::cout<<"Coarse Search time = "<<globalTimingMeasurements[1]<<"\n";
    //    std::cout<<"Fine Search time = "<<globalTimingMeasurements[2]<<"\n";
    //    std::cout<<"Interpolation time = "<<globalTimingMeasurements[3]<<"\n";
    //    std::cout<<"Total time = "<<globalTimingMeasurements[4]<<"\n";
    //  }
}

void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    int rank = globalComm.getRank();
    int npes = globalComm.getSize();

    size_t n_k                  = 1;
    std::string meshFileNames[] = { "pellet_1x.e", "pellet_2x.e", "pellet_4x.e" };
    //  size_t meshNumElements[] = { 3705, 26146, 183210 };

    size_t n_j            = 1;
    size_t numRandomPts[] = { 100000, 20000, 40000, 80000, 160000, 320000 };

    size_t n_i                                                            = 1;
    void ( *randomPtsGenerators[] )( int, size_t, std::vector<double> & ) = { &genUniformPts };
    const std::string prefixes[]                                          = { "uniform" };

    std::string suffix = std::to_string( npes );


    for ( size_t i = 0; i < n_i; ++i ) {
        std::fstream fout;
        if ( !rank ) {
            std::string fileName = prefixes[i] + "_" + suffix;
            fout.open( fileName.c_str(), std::fstream::out );
            //      fout<<n_j<<"  "<<n_k<<"\n";
            //      for (size_t j = n_j-1; j < n_j; ++j) {
            //        fout<<numRandomPts[j]<<"  ";
            //      } // end for j
            //      fout<<"\n";
            //      for (size_t k = 0; k < n_k; ++k) {
            //        fout<<meshNumElements[k]<<"  ";
            //      } // end for k
            //      fout<<"\n";
        } // end if
        for ( size_t j = n_j - 1; j < n_j; ++j ) {
            for ( size_t k = n_k - 1; k < n_k; ++k ) {
                std::vector<double> localTimingMeasurements;
                run( meshFileNames[k],
                     numRandomPts[j],
                     randomPtsGenerators[i],
                     localTimingMeasurements );
                size_t numTimingMeasurements = localTimingMeasurements.size();
                AMP_ASSERT( numTimingMeasurements > 0 );
                std::vector<double> globalTimingMeasurements( numTimingMeasurements, -1.0 );
                globalComm.maxReduce( &( localTimingMeasurements[0] ),
                                      &( globalTimingMeasurements[0] ),
                                      numTimingMeasurements );
                if ( !rank ) {
                    fout << globalTimingMeasurements[0] << "  " // setup
                         << globalTimingMeasurements[1] << "  " // coarse
                         << globalTimingMeasurements[2] << "  " // fine
                         << globalTimingMeasurements[3] << "  " // interpolation
                         << globalTimingMeasurements[4] << "  " // project on boundary
                         << globalTimingMeasurements[5] << "  " << std::flush; // coarse+fine
                }                                                              // end if
            }                                                                  // end for k
            if ( !rank ) {
                fout << "\n";
            } // end if
        }     // end for j
        if ( !rank ) {
            fout.close();
        } // end if
    }     // end for i

    ut->passes( exeName );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName = "testDendroSearch";

    myTest( &ut, exeName );

    ut.report();
    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}

void genUniformPts( int rank, size_t numPtsPerProc, std::vector<double> &pts )
{
    static std::random_device rd;
    static std::mt19937 gen( rd() );
    static std::uniform_real_distribution<double> dist( 0, 1 );
    pts.resize( 3 * numPtsPerProc );
    for ( size_t i = 0; i < 3 * numPtsPerProc; ++i )
        pts[i] = dist( gen );
}

void genGaussPts( int rank, size_t numPtsPerProc, std::vector<double> &pts )
{
    pts.resize( 3 * numPtsPerProc );
    for ( size_t i = 0; i < ( 3 * numPtsPerProc ); i++ )
        pts[i] = gaussian( 0.5, 0.16 );
}

double gaussian( double mean, double std_deviation )
{
    static double t1 = 0, t2 = 0;
    double x1, x2, x3, r;

    // reuse previous calculations
    if ( t1 ) {
        const double tmp = t1;
        t1               = 0;
        return mean + std_deviation * tmp;
    }
    if ( t2 ) {
        const double tmp = t2;
        t2               = 0;
        return mean + std_deviation * tmp;
    }

    // pick randomly a point inside the unit disk
    static std::random_device rd;
    static std::mt19937 gen( rd() );
    static std::uniform_real_distribution<double> dist( -1, 1 );
    do {
        x1 = dist( gen );
        x2 = dist( gen );
        x3 = dist( gen );
        r  = x1 * x1 + x2 * x2 + x3 * x3;
    } while ( r >= 1 );

    // Box-Muller transform
    r = std::sqrt( -2.0 * log( r ) / r );

    // save for next call
    t1 = ( r * x2 );
    t2 = ( r * x3 );

    return mean + ( std_deviation * r * x1 );
} // end gaussian

void rescalePts( std::vector<double> &pts )
{
    double minX = pts[0];
    double maxX = pts[0];

    double minY = pts[1];
    double maxY = pts[1];

    double minZ = pts[2];
    double maxZ = pts[2];

    for ( unsigned int i = 0; i < pts.size(); i += 3 ) {
        double xPt = pts[i];
        double yPt = pts[i + 1];
        double zPt = pts[i + 2];

        if ( xPt < minX ) {
            minX = xPt;
        }

        if ( xPt > maxX ) {
            maxX = xPt;
        }

        if ( yPt < minY ) {
            minY = yPt;
        }

        if ( yPt > maxY ) {
            maxY = yPt;
        }

        if ( zPt < minZ ) {
            minZ = zPt;
        }

        if ( zPt > maxZ ) {
            maxZ = zPt;
        }
    } // end for i

    double xRange = ( maxX - minX );
    double yRange = ( maxY - minY );
    double zRange = ( maxZ - minZ );

    for ( unsigned int i = 0; i < pts.size(); i += 3 ) {
        double xPt = pts[i];
        double yPt = pts[i + 1];
        double zPt = pts[i + 2];

        pts[i]     = 0.05 + ( 0.925 * ( xPt - minX ) / xRange );
        pts[i + 1] = 0.05 + ( 0.925 * ( yPt - minY ) / yRange );
        pts[i + 2] = 0.05 + ( 0.925 * ( zPt - minZ ) / zRange );
    } // end for i
}
