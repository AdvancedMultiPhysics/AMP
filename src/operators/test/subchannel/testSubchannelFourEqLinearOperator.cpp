#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/discretization/structuredFaceDOFManager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/operators/subchannel/SubchannelFourEqLinearOperator.h"
#include "AMP/operators/subchannel/SubchannelOperatorParameters.h"
#include "AMP/operators/subchannel/SubchannelPhysicsModel.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>


static constexpr size_t numSubchannels    = 3 * 3;                 // number of subchannels
static constexpr size_t numAxialIntervals = 3;                     // number of axial intervals
static constexpr size_t numAxialFaces     = numAxialIntervals + 1; // number of axial faces
static constexpr size_t numGaps_MATLAB    = 12;                    // number of gaps in MATLAB
// # of DoFs in MATLAB
static constexpr size_t num_dofs_MATLAB =
    3 * numSubchannels * numAxialFaces + numGaps_MATLAB * numAxialIntervals;
// number of gaps in AMP; (AMP automatically puts DOFs on exterior gaps, as opposed to MATLAB)
static constexpr size_t numGaps_AMP = 24;
// total number of DoFs in AMP
static constexpr size_t num_dofs_AMP =
    3 * numSubchannels * numAxialFaces + numGaps_AMP * numAxialIntervals;


// function to get the MATLAB subchannel index
static size_t getMATLABSubchannelIndex( const AMP::Mesh::MeshElement &face )
{
    double pitch = 0.0126; // pitch for test problem [m]
    double x1    = 0.5 * pitch;
    double x2    = 1.5 * pitch;
    double x3    = 2.5 * pitch;

    // get face centroid
    auto centroid = face.centroid();
    // gap MATLAB index
    size_t i = 0;
    // look at location of subchannel to determine subchannel MATLAB index
    if ( ( AMP::Utilities::approx_equal( centroid[0], x1, 1.0e-12 ) ) &&
         ( AMP::Utilities::approx_equal( centroid[1], x3, 1.0e-12 ) ) )
        i = 0;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x2, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x3, 1.0e-12 ) ) )
        i = 1;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x3, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x3, 1.0e-12 ) ) )
        i = 2;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x1, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x2, 1.0e-12 ) ) )
        i = 3;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x2, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x2, 1.0e-12 ) ) )
        i = 4;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x3, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x2, 1.0e-12 ) ) )
        i = 5;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x1, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x1, 1.0e-12 ) ) )
        i = 6;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x2, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x1, 1.0e-12 ) ) )
        i = 7;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x3, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x1, 1.0e-12 ) ) )
        i = 8;
    else
        AMP_ERROR( "GeomType::Face did not match any known location" );

    return i;
}

// function to get the MATLAB gap index
static size_t getMATLABGapIndex( const AMP::Mesh::MeshElement &gapFace )
{
    double pitch = 0.0126; // pitch for test problem [m]
    double x1    = 0.5 * pitch;
    double x2    = 1.0 * pitch;
    double x3    = 1.5 * pitch;
    double x4    = 2.0 * pitch;
    double x5    = 2.5 * pitch;

    // get gap face centroid
    auto centroid = gapFace.centroid();
    // gap MATLAB index
    size_t k = 0;
    // look at location of gap to determine gap MATLAB index
    if ( ( AMP::Utilities::approx_equal( centroid[0], x1, 1.0e-12 ) ) &&
         ( AMP::Utilities::approx_equal( centroid[1], x4, 1.0e-12 ) ) )
        k = 0;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x3, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x4, 1.0e-12 ) ) )
        k = 1;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x5, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x4, 1.0e-12 ) ) )
        k = 2;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x1, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x2, 1.0e-12 ) ) )
        k = 3;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x3, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x2, 1.0e-12 ) ) )
        k = 4;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x5, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x2, 1.0e-12 ) ) )
        k = 5;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x2, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x5, 1.0e-12 ) ) )
        k = 6;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x2, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x3, 1.0e-12 ) ) )
        k = 7;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x2, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x1, 1.0e-12 ) ) )
        k = 8;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x4, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x5, 1.0e-12 ) ) )
        k = 9;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x4, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x3, 1.0e-12 ) ) )
        k = 10;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x4, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x1, 1.0e-12 ) ) )
        k = 11;
    else
        AMP_ERROR( "Gap face did not match any known location" );

    return k;
}

// function to determine the axial level
static size_t getMATLABAxialIndex( const AMP::Mesh::MeshElement &face, bool is_axial_face_quantity )
{
    double height = 3.66;
    double dz     = height / numAxialIntervals;
    auto centroid = face.centroid();
    size_t j      = 0;
    // boolean for if the axial index has been found
    bool foundIndex = false;
    // loop over axial intervals
    if ( is_axial_face_quantity ) {
        for ( size_t k = 0; k < numAxialFaces; ++k ) {
            if ( AMP::Utilities::approx_equal( centroid[2], k * dz, 1.0e-12 ) ) {
                j          = k;
                foundIndex = true;
                break;
            }
        }
    } else {
        for ( size_t k = 0; k < numAxialIntervals; ++k ) {
            if ( AMP::Utilities::approx_equal( centroid[2], ( k + 0.5 ) * dz, 1.0e-12 ) ) {
                j          = k;
                foundIndex = true;
                break;
            }
        }
    }
    if ( !foundIndex )
        AMP_ERROR( "Axial index was not found" );

    return j;
}


// function used to get all lateral gaps
static std::map<AMP::Mesh::Point, std::unique_ptr<AMP::Mesh::MeshElement>>
getLateralFaces( std::shared_ptr<AMP::Mesh::Mesh> mesh, bool )
{
    // map of lateral gaps to their centroids
    std::map<AMP::Mesh::Point, std::unique_ptr<AMP::Mesh::MeshElement>> lateralFaceMap;
    // get iterator over all faces of mesh
    AMP::Mesh::MeshIterator face = mesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    // loop over faces
    for ( ; face != face.end(); ++face ) {
        // check that face is vertical
        // ---------------------------
        // get centroid of current face
        auto faceCentroid = face->centroid();
        // get vertices of current face
        auto vertices = face->getElements( AMP::Mesh::GeomType::Vertex );

        bool perpindicular_to_x = true; // is the current face perpindicular to x-axis?
        bool perpindicular_to_y = true; // is the current face perpindicular to y-axis?
        // loop over vertices of current face
        for ( auto &vertice : vertices ) {
            // get coordinates of current vertex
            auto vertexCoord = vertice->coord();
            // if any vertex does not have the same x-coordinate as the face centroid,
            if ( !AMP::Utilities::approx_equal( vertexCoord[0], faceCentroid[0], 1.0e-6 ) )
                // then the face is not perpindicular to x-axis
                perpindicular_to_x = false;
            // if any vertex does not have the same y-coordinate as the face centroid,
            if ( !AMP::Utilities::approx_equal( vertexCoord[1], faceCentroid[1], 1.0e-6 ) )
                // then the face is not perpindicular to y-axis
                perpindicular_to_y = false;
        }
        // check that face is in the interior of the mesh; it must have two adjacent cells
        // -------------------------------------------------------------------------------
        // if the face is vertical
        if ( perpindicular_to_x || perpindicular_to_y ) {
            // if the face has more than 1 adjacent cell
            if ( ( mesh->getElementParents( *face, AMP::Mesh::GeomType::Cell ) ).size() > 1 ) {
                // insert face into map with centroid
                lateralFaceMap.insert(
                    decltype( lateralFaceMap )::value_type( faceCentroid, face->clone() ) );
            }
        }
    } // end loop over faces
    return lateralFaceMap;
}

// find the MATLAB index for a variable on a given face
static size_t AMP_to_MATLAB( const AMP::Mesh::MeshElement &face, size_t variable_id )
{
    // determine if variable is axial face quantity or lateral face quantity
    bool is_axial_face_quantity = false;
    switch ( variable_id ) {
    case 0: {
        is_axial_face_quantity = true;
        break;
    }
    case 1: {
        is_axial_face_quantity = true;
        break;
    }
    case 2: {
        is_axial_face_quantity = true;
        break;
    }
    case 3: {
        is_axial_face_quantity = false;
        break;
    }
    default: {
        AMP_ERROR( "Invalid variable id" );
        break;
    }
    }

    // determine axial level
    size_t j = getMATLABAxialIndex( face, is_axial_face_quantity );

    // determine subchannel or gap
    size_t i;
    if ( is_axial_face_quantity )
        i = getMATLABSubchannelIndex( face );
    else
        i = getMATLABGapIndex( face );

    // compute index
    size_t index = 0;
    switch ( variable_id ) {
    case 0: {
        index = i * numAxialFaces + j;
        break;
    }
    case 1: {
        index = i * numAxialFaces + j + numGaps_MATLAB * numAxialIntervals +
                numSubchannels * numAxialFaces;
        break;
    }
    case 2: {
        index = i * numAxialFaces + j + numGaps_MATLAB * numAxialIntervals +
                2 * numSubchannels * numAxialFaces;
        break;
    }
    case 3: {
        index = i * numAxialIntervals + j + numSubchannels * numAxialFaces;
        break;
    }
    default: {
        AMP_ERROR( "Invalid variable id" );
        break;
    }
    }

    return index;
}

// function to create map of global IDs to elements and variables
static void
createGlobalIDMaps( std::shared_ptr<AMP::Discretization::DOFManager> dof_manager,
                    std::shared_ptr<AMP::Mesh::Mesh> mesh,
                    std::map<size_t, std::unique_ptr<AMP::Mesh::MeshElement>> &elements_by_globalID,
                    std::map<size_t, size_t> &variables_by_globalID )
{
    // loop over axial faces
    auto xyMesh     = mesh->Subset( AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( mesh, 0 ) );
    auto axial_face = xyMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    for ( ; axial_face != axial_face.end(); ++axial_face ) {
        std::vector<size_t> dofs;
        dof_manager->getDOFs( axial_face->globalID(), dofs );
        elements_by_globalID.insert( std::make_pair( dofs[0], axial_face->clone() ) );
        elements_by_globalID.insert( std::make_pair( dofs[1], axial_face->clone() ) );
        elements_by_globalID.insert( std::make_pair( dofs[2], axial_face->clone() ) );
        variables_by_globalID.insert( std::make_pair( dofs[0], 0 ) );
        variables_by_globalID.insert( std::make_pair( dofs[1], 1 ) );
        variables_by_globalID.insert( std::make_pair( dofs[2], 2 ) );
    }

    // loop over lateral faces
    auto lateral_face_map = getLateralFaces( mesh, true );
    auto face             = mesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    for ( ; face != face.end(); ++face ) {
        auto centroid              = face->centroid();
        auto lateral_face_iterator = lateral_face_map.find( centroid );
        if ( lateral_face_iterator != lateral_face_map.end() ) {
            auto &lateral_face = lateral_face_iterator->second;
            std::vector<size_t> dofs;
            dof_manager->getDOFs( lateral_face->globalID(), dofs );
            elements_by_globalID.insert( std::make_pair( dofs[0], lateral_face->clone() ) );
            variables_by_globalID.insert( std::make_pair( dofs[0], 3 ) );
        }
    }
}

// function to check that Jacobian matches known values
static bool JacobianIsCorrect(
    std::shared_ptr<AMP::LinearAlgebra::Matrix> J_test_AMP,
    double J_reference[num_dofs_MATLAB][num_dofs_MATLAB],
    std::shared_ptr<AMP::Discretization::DOFManager> dof_manager,
    std::shared_ptr<AMP::Mesh::Mesh> mesh,
    const std::map<AMP::Mesh::Point, std::unique_ptr<AMP::Mesh::MeshElement>> &lateral_face_map,
    const std::map<size_t, std::unique_ptr<AMP::Mesh::MeshElement>> &elements_by_globalID,
    const std::map<size_t, size_t> &variables_by_globalID )
{
    double J_test_MATLAB[num_dofs_MATLAB][num_dofs_MATLAB] = { { 0.0 } };
    // loop over axial faces
    auto xyMesh     = mesh->Subset( AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( mesh, 0 ) );
    auto axial_face = xyMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    for ( ; axial_face != axial_face.end(); ++axial_face ) {
        std::vector<size_t> dofs;
        dof_manager->getDOFs( axial_face->globalID(), dofs );
        size_t i_m_AMP    = dofs[0];
        size_t i_h_AMP    = dofs[1];
        size_t i_p_AMP    = dofs[2];
        size_t i_m_MATLAB = AMP_to_MATLAB( *axial_face, 0 );
        size_t i_h_MATLAB = AMP_to_MATLAB( *axial_face, 1 );
        size_t i_p_MATLAB = AMP_to_MATLAB( *axial_face, 2 );
        std::vector<double> val_m, val_h, val_p; // nonzero values in Jacobian row
        std::vector<size_t> ind_m, ind_h, ind_p; // indices of nonzeros in Jacobian row
        J_test_AMP->getRowByGlobalID( i_m_AMP, ind_m, val_m );
        J_test_AMP->getRowByGlobalID( i_h_AMP, ind_h, val_h );
        J_test_AMP->getRowByGlobalID( i_p_AMP, ind_p, val_p );
        // loop over all DOFs
        for ( size_t j_AMP = 0; j_AMP < num_dofs_AMP; j_AMP++ ) {
            auto face_iterator = elements_by_globalID.find( j_AMP );
            if ( face_iterator != elements_by_globalID.end() ) {
                auto &face                = face_iterator->second;
                auto variable_id_iterator = variables_by_globalID.find( j_AMP );
                if ( variable_id_iterator == variables_by_globalID.end() )
                    AMP_ERROR( "index exists only for face" );
                size_t variable_id = variable_id_iterator->second;
                size_t j_MATLAB    = AMP_to_MATLAB( *face, variable_id );
                auto nonzero_ind_m = std::find( ind_m.begin(), ind_m.end(), j_AMP );
                auto nonzero_ind_h = std::find( ind_h.begin(), ind_h.end(), j_AMP );
                auto nonzero_ind_p = std::find( ind_p.begin(), ind_p.end(), j_AMP );
                if ( nonzero_ind_m != ind_m.end() ) {
                    size_t m_ind = std::distance( ind_m.begin(), nonzero_ind_m );
                    J_test_MATLAB[i_m_MATLAB][j_MATLAB] = val_m[m_ind];
                }
                if ( nonzero_ind_h != ind_h.end() ) {
                    size_t h_ind = std::distance( ind_h.begin(), nonzero_ind_h );
                    J_test_MATLAB[i_h_MATLAB][j_MATLAB] = val_h[h_ind];
                }
                if ( nonzero_ind_p != ind_p.end() ) {
                    size_t p_ind = std::distance( ind_p.begin(), nonzero_ind_p );
                    J_test_MATLAB[i_p_MATLAB][j_MATLAB] = val_p[p_ind];
                }
            }
        }
    }

    // loop over lateral faces
    AMP::Mesh::MeshIterator lateral_face = mesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    for ( ; lateral_face != lateral_face.end(); ++lateral_face ) {
        auto centroid              = lateral_face->centroid();
        auto lateral_face_iterator = lateral_face_map.find( centroid );
        if ( lateral_face_iterator != lateral_face_map.end() ) {
            std::vector<size_t> dofs;
            dof_manager->getDOFs( lateral_face->globalID(), dofs );
            size_t i_w_AMP    = dofs[0];
            size_t i_w_MATLAB = AMP_to_MATLAB( *lateral_face, 3 );
            std::vector<double> val_w; // nonzero values in Jacobian row
            std::vector<size_t> ind_w; // indices of nonzeros in Jacobian row
            J_test_AMP->getRowByGlobalID( i_w_AMP, ind_w, val_w );
            // loop over all DOFs
            for ( size_t j_AMP = 0; j_AMP < num_dofs_AMP; j_AMP++ ) {
                auto face_iterator = elements_by_globalID.find( j_AMP );
                if ( face_iterator != elements_by_globalID.end() ) {
                    auto &face                = face_iterator->second;
                    auto variable_id_iterator = variables_by_globalID.find( j_AMP );
                    if ( variable_id_iterator == variables_by_globalID.end() )
                        AMP_ERROR( "index exists only for face" );
                    size_t variable_id = variable_id_iterator->second;
                    size_t j_MATLAB    = AMP_to_MATLAB( *face, variable_id );
                    auto nonzero_ind_w = std::find( ind_w.begin(), ind_w.end(), j_AMP );
                    if ( nonzero_ind_w != ind_w.end() ) {
                        size_t w_ind = std::distance( ind_w.begin(), nonzero_ind_w );
                        J_test_MATLAB[i_w_MATLAB][j_MATLAB] = val_w[w_ind];
                    }
                }
            }
        }
    }

    // compare each matrix entry
    bool passed = true;
    for ( size_t i = 0; i < num_dofs_MATLAB; i++ ) {
        for ( size_t j = 0; j < num_dofs_MATLAB; j++ ) {
            if ( AMP::Utilities::approx_equal_abs( J_reference[i][j], 0.0, 1.0e-12 ) ) {
                if ( !AMP::Utilities::approx_equal( J_test_MATLAB[i][j], 0.0, 1.0e-12 ) ) {
                    passed = false;
                    std::cout << "Values are inconsistent for entry (" << i << "," << j
                              << ") (MATLAB indices): " << J_reference[i][j] << " (MATLAB) vs. "
                              << J_test_MATLAB[i][j] << " (AMP)" << std::endl;
                }
            } else if ( !AMP::Utilities::approx_equal(
                            J_reference[i][j], J_test_MATLAB[i][j], 1.0e-7 ) ) {
                passed = false;
                std::cout << "Values are inconsistent for entry (" << i << "," << j
                          << ") (MATLAB indices): " << J_reference[i][j] << " (MATLAB) vs. "
                          << J_test_MATLAB[i][j] << " (AMP)" << std::endl;
            }
        }
    }

    return passed;
}

static void Test( AMP::UnitTest *ut, const std::string &exeName )
{
    // create input and output file names
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;
    AMP::logOnlyNodeZero( log_file );

    // get input database from input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // create mesh
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto subchannelMesh = AMP::Mesh::MeshFactory::create( meshParams );

    // check number of cells
    size_t expected_number_of_cells = numSubchannels * numAxialIntervals;
    size_t numCells = subchannelMesh->numGlobalElements( AMP::Mesh::GeomType::Cell );
    if ( numCells == expected_number_of_cells )
        ut->passes( exeName + ": number of cells" );
    else {
        ut->failure( exeName + ": number of cells" );
        std::cout << "number of cells in mesh: " << numCells << std::endl;
        std::cout << "expected number of cells: " << expected_number_of_cells << std::endl;
    }

    // get dof manager
    std::shared_ptr<AMP::Discretization::DOFManager> subchannelDOFManager;
    int DOFsPerFace[3]   = { 1, 1, 3 };
    subchannelDOFManager = std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
        subchannelMesh, DOFsPerFace, 1 );
    // check number of DOFs
    if ( subchannelDOFManager->numGlobalDOF() == num_dofs_AMP )
        ut->passes( exeName + ": number of DOFs" );
    else {
        ut->failure( exeName + ": number of DOFs" );
        std::cout << "number of DOFs in DOF Manager: " << subchannelDOFManager->numGlobalDOF()
                  << std::endl;
        std::cout << "expected number of DOFs: " << num_dofs_AMP << std::endl;
    }

    // get input and output variables
    auto inputVariable  = std::make_shared<AMP::LinearAlgebra::Variable>( "flow" );
    auto outputVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "flow" );

    // create solution, rhs, and residual vectors
    auto FrozenVec = AMP::LinearAlgebra::createVector( subchannelDOFManager, inputVariable, false );
    auto SolVec    = AMP::LinearAlgebra::createVector( subchannelDOFManager, inputVariable, false );
    auto ResVec = AMP::LinearAlgebra::createVector( subchannelDOFManager, outputVariable, false );

    // create subchannel physics model
    auto subchannelPhysics_db = input_db->getDatabase( "SubchannelPhysicsModel" );
    auto params =
        std::make_shared<AMP::Operator::ElementPhysicsModelParameters>( subchannelPhysics_db );
    auto subchannelPhysicsModel = std::make_shared<AMP::Operator::SubchannelPhysicsModel>( params );

    // get linear operator database
    auto subchannelOperator_db = input_db->getDatabase( "SubchannelFourEqLinearOperator" );
    // set operator parameters
    auto subchannelOpParams =
        std::make_shared<AMP::Operator::SubchannelOperatorParameters>( subchannelOperator_db );
    subchannelOpParams->d_Mesh                   = subchannelMesh;
    subchannelOpParams->d_subchannelPhysicsModel = subchannelPhysicsModel;
    subchannelOpParams->d_frozenSolution         = FrozenVec;
    subchannelOpParams->clad_x =
        input_db->getDatabase( "CladProperties" )->getVector<double>( "x" );
    subchannelOpParams->clad_y =
        input_db->getDatabase( "CladProperties" )->getVector<double>( "y" );
    subchannelOpParams->clad_d =
        input_db->getDatabase( "CladProperties" )->getVector<double>( "d" );
    // create linear operator
    auto subchannelOperator =
        std::make_shared<AMP::Operator::SubchannelFourEqLinearOperator>( subchannelOpParams );

    // report successful creation
    ut->passes( exeName + ": linear operator creation" );
    std::cout.flush();

    // check number of lateral gaps
    auto interiorLateralFaceMap = getLateralFaces( subchannelMesh, false );
    size_t Ngaps                = interiorLateralFaceMap.size();
    if ( Ngaps == 36 ) { // for 3x3 subchannel array with 3 axial intervals, there are 12x3=36 gaps
        ut->passes( exeName + ": number of lateral gaps" );
    } else {
        std::cout << "Incorrent number of lateral gaps. Found: " << Ngaps << ". Expected: 36."
                  << std::endl;
        ut->failure( exeName + ": number of lateral gaps" );
    }

    // compute height of subchannels
    auto box            = subchannelMesh->getBoundingBox();
    const double height = box[5] - box[4];
    // height of each axial interval
    const double dz = height / numAxialIntervals;
    // get all of the unique x,y,z points in subchannel mesh
    subchannelOperator->fillSubchannelGrid( subchannelMesh );

    // put all cells in an array by subchannel
    std::unique_ptr<AMP::Mesh::MeshElement>
        d_elem[numSubchannels][numAxialIntervals]; // array of array of elements for each subchannel
    auto cell =
        subchannelMesh->getIterator( AMP::Mesh::GeomType::Cell, 0 ); // iterator for cells of mesh
    for ( ; cell != cell.end(); ++cell ) {                           // loop over all cells
        auto center = cell->centroid();
        // get the index of the subchannel
        int isub               = subchannelOperator->getSubchannelIndex( center[0], center[1] );
        bool found_axial_index = false;
        // get the axial index of the cell
        for ( unsigned int j = 0; j < numAxialIntervals; ++j ) {
            // center of interval j
            double center_interval_j = ( j + 0.5 ) * dz;
            // check if center of cell is equal to center of axial interval j
            if ( AMP::Utilities::approx_equal( center[2], center_interval_j, 1.0e-12 ) ) {
                d_elem[isub][j]   = cell->clone();
                found_axial_index = true;
                break;
            }
        }
        // ensure that a valid axial index was found
        if ( !found_axial_index )
            AMP_ERROR( "A cell was not in center of any axial interval" );
    } // end for cell

    // get scale factors for axial mass flow rate, enthalpy, and pressure
    const double m_scale = AMP::Operator::Subchannel::scaleAxialMassFlowRate;
    const double h_scale = AMP::Operator::Subchannel::scaleEnthalpy;
    const double p_scale = AMP::Operator::Subchannel::scalePressure;

    // set axial face quantities of input vector
    // loop over subchannels
    for ( size_t isub = 0; isub < numSubchannels; ++isub ) {
        size_t ii = isub + 1; // corresponding MATLAB index for this subchannel
        // loop over axial intervals
        for ( unsigned int j = 0; j < numAxialIntervals; ++j ) {
            // get axial faces
            std::unique_ptr<AMP::Mesh::MeshElement> plusFace;  // upper axial face for current cell
            std::unique_ptr<AMP::Mesh::MeshElement> minusFace; // lower axial face for current cell
            subchannelOperator->getAxialFaces( *d_elem[isub][j], plusFace, minusFace );

            double val;
            if ( j == 0 ) // for first axial interval only, set the quantities for the lower face
            {
                size_t jj = j + 1; // corresponding MATLAB index for this axial face
                // get dofs on minus face
                std::vector<size_t> minusDofs;
                subchannelDOFManager->getDOFs( minusFace->globalID(), minusDofs );
                // set values of minus face
                val = m_scale * 0.35 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
                FrozenVec->setValuesByGlobalID( 1, &minusDofs[0], &val );
                val = h_scale * 1000.0e3 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
                FrozenVec->setValuesByGlobalID( 1, &minusDofs[1], &val );
                val = p_scale * 15.5e6 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
                FrozenVec->setValuesByGlobalID( 1, &minusDofs[2], &val );
                val = m_scale * 1.0;
                SolVec->setValuesByGlobalID( 1, &minusDofs[0], &val );
                val = h_scale * 1.0;
                SolVec->setValuesByGlobalID( 1, &minusDofs[1], &val );
                val = p_scale * 1.0;
                SolVec->setValuesByGlobalID( 1, &minusDofs[2], &val );
            }

            size_t jj = j + 2; // corresponding MATLAB index for this axial face
            // get dofs on plus face
            std::vector<size_t> plusDofs;
            subchannelDOFManager->getDOFs( plusFace->globalID(), plusDofs );
            // set values of plus face
            val = m_scale * 0.35 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
            FrozenVec->setValuesByGlobalID( 1, &plusDofs[0], &val );
            val = h_scale * 1000.0e3 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
            FrozenVec->setValuesByGlobalID( 1, &plusDofs[1], &val );
            val = p_scale * 15.5e6 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
            FrozenVec->setValuesByGlobalID( 1, &plusDofs[2], &val );
            val = m_scale * 1.0;
            SolVec->setValuesByGlobalID( 1, &plusDofs[0], &val );
            val = h_scale * 1.0;
            SolVec->setValuesByGlobalID( 1, &plusDofs[1], &val );
            val = p_scale * 1.0;
            SolVec->setValuesByGlobalID( 1, &plusDofs[2], &val );
        }
    }

    // get scale factors for lateral mass flow rate
    const double w_scale = AMP::Operator::Subchannel::scaleLateralMassFlowRate;

    // set lateral face quantities (lateral mass flow rates) of input vector
    // array of gap faces
    [[maybe_unused]] std::unique_ptr<AMP::Mesh::MeshElement>
        gapFaces[numGaps_MATLAB][numAxialIntervals]; // gap faces
    // loop over all faces in mesh
    auto face = subchannelMesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
    for ( ; face != face.end(); ++face ) { // loop over all faces in mesh
        auto faceCentroid = face->centroid();
        // try to find face in lateral face map
        auto lateralFaceIterator = interiorLateralFaceMap.find( faceCentroid );
        if ( lateralFaceIterator != interiorLateralFaceMap.end() ) { // if face in lateral face map,
            // get lateral face
            auto &lateralFace = lateralFaceIterator->second;
            // get MATLAB index for gap
            unsigned int k = getMATLABGapIndex( *lateralFace );
            // get MATLAB axial index
            unsigned int j = getMATLABAxialIndex( *lateralFace, false );
            // put gap face in array
            gapFaces[k][j] = lateralFace->clone();
            // get crossflow from solution vector
            std::vector<size_t> gapDofs;
            subchannelDOFManager->getDOFs( lateralFace->globalID(), gapDofs );
            // set test value for crossflow
            double val =
                w_scale * 0.001 * ( 1.0 + 1.0 / 100.0 * cos( k + 1 ) * cos( 17.3 * ( j + 1 ) ) );
            FrozenVec->setValuesByGlobalID( 1, &gapDofs[0], &val );
            val = w_scale * 1.0;
            SolVec->setValuesByGlobalID( 1, &gapDofs[0], &val );
        }
    }
    SolVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // apply the operator
    subchannelOperator->setFrozenVector( FrozenVec );
    subchannelOpParams->d_initialize = true;
    subchannelOperator->reset( subchannelOpParams );
    subchannelOperator->apply( SolVec, ResVec );

    // get the AMP Jacobian matrix to be tested against the MATLAB Jacobian matrix
    auto testJacobian = subchannelOperator->getMatrix();

    // get the MATLAB Jacobian matrix
    // clang-format off
    double knownJacobian[num_dofs_MATLAB][num_dofs_MATLAB] = { 
        { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, -0.00677668901258425, 0.00674966917997032, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00680134179247133, 0.00675123052041697, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75450738336373, 4.7585455885653, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, -0.00674966917997032, 0.00679583334404178, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00675123052041697, 0.00677129943182299, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75113294598402, 4.75494869704572, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, -0.00679502285863625, 0.00681578079453035, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00678057423704236, 0.00681485684336233, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75504157219458, 4.75865451523295, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, -0.00681578079453035, 0.00678027402447358, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00681485684336233, 0.00679466062218694, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75117047528547, 4.75430442314381, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00680615306651423, 0.00685545819708779, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0067659218851098, 0.00685286883884361, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75207026445087, 4.75849451551659, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00685545819708779, 0.00677105865990493, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00685286883884361, 0.00681119707306303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75096878063051, 4.7576813791005, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00679961730689246, 0.00683220093186155, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00677060673094952, 0.006830338373976, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.74793016128515, 4.75821082957797, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00683220093186155, 0.00677644930146954, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.006830338373976, 0.00680544001532756, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75071169563154, 4.76158255561702, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00678158213030341, 0.00676740617221729, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00679042650059103, 0.00676796877440433, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.74664515776453, 4.75806532596245, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00676740617221729, 0.0067916335529895, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00676796877440433, 0.00678279735427316, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75063646484676, 4.76263668616477, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00676872486067267, 0.00672065503545349, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00680732255528395, 0.00672309165630821, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.74961025253147, 4.7581927669453, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00672065503545349, 0.00680274407584255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00672309165630821, 0.00676425457951943, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75081312459266, 4.76008803129863, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.30015873015873, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, -0.00677699640914768, 0.00673518671850398, 0, 0, -0.0067967116015305, 0.00673643465839852, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.74712413718295, 4.75805832943047, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, -0.00673518671850398, 0.0067949086811636, 0, 0, -0.00673643465839852, 0.00677522361754851, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75066428209911, 4.76224741386004, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00680035124824462, 0.00679773991114397, 0, 0, -0.00678096403351543, 0.00679650559033574, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.76242481837535, 4.7591057405026, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00679773991114397, 0.00677425751608255, 0, 0, -0.00679650559033574, 0.00679348992193275, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75163913917039, 4.74700340865238, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00677023641473485, 0.00684779571028473, 0, 0, -0.0067887034679692, 0.00684898861682152, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.73215050067433, 4.75722148107336, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00684779571028473, 0.00680655538876296, 0, 0, -0.00684898861682152, 0.00678795878231737, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.74970902130792, 4.77721909105585, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, -0.00679441244743652, 0.0068444611640022, 0, 0, -0.00680637127556524, 0.00684522244914879, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.79457858836221, 4.76121375642899, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, -0.0068444611640022, 0.00678209953915717, 0, 0, -0.00684522244914879, 0.00677040834669981, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.7536836042023, 4.71510171169436, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00678117996261892, 0.00678633985029537, 0, 0, -0.00676735116574352, 0.00678545097362945, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.71253324828324, 4.75589285140811, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00678633985029537, 0.00679284140848072, 0, 0, -0.00678545097362945, 0.00680688675891539, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.74845565171368, 4.79688137932266, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00679123436366546, 0.00672998171707532, 0, 0, -0.00680715442010169, 0.00673098033097224, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.78372216201276, 4.76039458044818, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00672998171707532, 0.00678037736113117, 0, 0, -0.00673098033097224, 0.00676475590104976, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.75299393772574, 4.72584521458958, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.30015873015873, 0, 0, 0, 0.30015873015873, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { -1000094.19706307, 994622.408115175, 0, 0, 9.14366044264196, 9.14366044264196, 0, 0, 0, 0, 0, 0, 11.413523529005, 11.413523529005, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1220139.99959977, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1220139.99959977, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.207569754782073, 0.348110647992782, 0, 0, -0.0699966637489306, 0, 0, 0, 0, 0, 0, 0, -0.0700334099427809, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, -995569.394593763, 998688.402559005, 0, 0, -429.944638472618, -429.944638472618, 0, 0, 0, 0, 0, 0, -536.55066884904, -536.55066884904, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1213414.2587177, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1213414.2587177, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.205656288118707, 0.349879581010687, 0, 0, -0.0699989268532521, 0, 0, 0, 0, 0, 0, 0, -0.0700312080717666, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, -999717.581831606, 1005321.89790061, 0, 0, -27.4118739980478, -27.4118739980478, 0, 0, 0, 0, 0, 0, -34.2199460998972, -34.2199460998972, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1219580.25380868, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1219580.25380868, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.207407221377212, 0.351884236895589, 0, 0, -0.0700343676693115, 0, 0, 0, 0, 0, 0, 0, -0.0699989973965461, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { -9.14347408824019, -9.14347408824019, 0, 0, -999921.964878841, 1004147.36448996, 0, 0, 5.48397791432918, 5.48397791432918, 0, 0, 0, 0, 0, 0, -6.68965780523463, -6.68965780523463, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1219892.17075343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1220139.99959977, 0, 0, 0, 0, 0, 0, 0, 0, 1219892.17075343, 0, 0, 0, 0, 0, 0, 0, 0, -0.071216859095152, 0, 0, 0, -0.137262961971054, 0.351455199898998, 0, 0, -0.0702333947721765, 0, 0, 0, 0, 0, 0, 0, -0.0700363701318951, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 430.354641697736, 430.354641697736, 0, 0, -1003671.07843432, 1000751.6299712, 0, 0, -257.725050945294, -257.725050945294, 0, 0, 0, 0, 0, 0, 314.78117189742, 314.78117189742, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1225072.41107651, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1213414.2587177, 0, 0, 0, 0, 0, 0, 0, 0, 1225072.41107651, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712097375435909, 0, 0, 0, -0.138759940549918, 0.350092748042935, 0, 0, -0.0702071651132457, 0, 0, 0, 0, 0, 0, 0, -0.0700338675341108, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 27.4135418549702, 27.4135418549702, 0, 0, -1000233.97716082, 995884.559581242, 0, 0, -16.4518000525577, -16.4518000525577, 0, 0, 0, 0, 0, 0, 20.0586665873575, 20.0586665873575, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1220323.29317823, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1219580.25380868, 0, 0, 0, 0, 0, 0, 0, 0, 1220323.29317823, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712537819831198, 0, 0, 0, -0.137814845410307, 0.348548739816786, 0, 0, -0.0698064182258784, 0, 0, 0, 0, 0, 0, 0, -0.0699961415923722, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, -5.48391083553225, -5.48391083553225, 0, 0, -999813.864463332, 1009866.86644102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -18.6423179641248, -18.6423179641248, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1219743.47961911, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1219892.17075343, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0714531773574496, 0, 0, 0, -0.208452777700326, 0.353461847729074, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.070020492135328, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 257.871135015198, 257.871135015198, 0, 0, -1008757.15657443, 1001764.24537217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 877.814667261428, 877.814667261428, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1232067.0120842, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1225072.41107651, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.071437392550641, 0, 0, 0, -0.212002909847406, 0.350220642952239, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0700197057563376, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 16.4524001963693, 16.4524001963693, 0, 0, -1000558.06489177, 990208.128279203, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55.8986767779937, 55.8986767779937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1220769.09829066, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1220323.29317823, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0710270700777671, 0, 0, 0, -0.209182007217241, 0.346547524657601, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0700114484108286, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { -11.413233138775, -11.413233138775, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -999895.003154797, 1006496.70993662, 0, 0, -8.95942964008051, -8.95942964008051, 0, 0, 0, 0, 0, 0, -13.4557338450741, -13.4557338450741, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1220139.99959977, 0, 0, 0, 0, 0, 0, 0, 0, 1219830.63213998, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1219830.63213998, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712535499423807, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.137404731042822, 0.352285688722168, 0, 0, -0.0700731140154348, 0, 0, 0, 0, 0, 0, 0, -0.0700003629276686, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 537.188544116988, 537.188544116988, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1004940.22343719, 1002006.54362272, 0, 0, 421.587830120486, 421.587830120486, 0, 0, 0, 0, 0, 0, 633.457524213495, 633.457524213495, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1213414.2587177, 0, 0, 0, 0, 0, 0, 0, 0, 1227967.25783156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1227967.25783156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712446223304843, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.139736467929449, 0.350145679748801, 0, 0, -0.0700661564728274, 0, 0, 0, 0, 0, 0, 0, -0.0700021154913298, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 34.222544931372, 34.222544931372, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1000314.8020055, 993588.567009108, 0, 0, 26.8675595838737, 26.8675595838737, 0, 0, 0, 0, 0, 0, 40.3435561774616, 40.3435561774616, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1219580.25380868, 0, 0, 0, 0, 0, 0, 0, 0, 1220507.79798153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1220507.79798153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712185776503548, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.137714478596706, 0.347720499406311, 0, 0, -0.0699607692121066, 0, 0, 0, 0, 0, 0, 0, -0.070030813710435, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 6.68975756866653, 6.68975756866653, 0, 0, 0, 0, 0, 0, 8.95960860656955, 8.95960860656955, 0, 0, -1000046.96374013, 997179.219999, 0, 0, -6.46901364403112, -6.46901364403112, 0, 0, 0, 0, 0, 0, 4.10278425519037, 4.10278425519037, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1219892.17075343, 0, 0, 0, 0, 0, 0, 0, 0, 1220073.50069027, 0, 0, 0, 0, 0, 0, 0, 0, -1219830.63213998, 0, 0, 0, 0, 0, 0, 0, 0, 1220073.50069027, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712562623026485, 0, 0, 0, 0, 0, 0, 0, -0.0712930763144177, 0, 0, 0, -0.0676536310672783, 0.349008078045094, 0, 0, -0.0698234010431359, 0, 0, 0, 0, 0, 0, 0, -0.0699944953584088, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, -314.562059357811, -314.562059357811, 0, 0, 0, 0, 0, 0, -421.19538722176, -421.19538722176, 0, 0, -997790.968403646, 999194.338171343, 0, 0, 304.597421520438, 304.597421520438, 0, 0, 0, 0, 0, 0, -192.970297104984, -192.970297104984, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1225072.41107651, 0, 0, 0, 0, 0, 0, 0, 0, 1216542.44347147, 0, 0, 0, 0, 0, 0, 0, 0, -1227967.25783156, 0, 0, 0, 0, 0, 0, 0, 0, 1216542.44347147, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712589399451873, 0, 0, 0, 0, 0, 0, 0, -0.0712879299717906, 0, 0, 0, -0.0666278711532075, 0.349936779256152, 0, 0, -0.0698466303323418, 0, 0, 0, 0, 0, 0, 0, -0.0699970640539391, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, -20.0577737642986, -20.0577737642986, 0, 0, 0, 0, 0, 0, -26.8659580257674, -26.8659580257674, 0, 0, -999859.211352149, 1002786.54791801, 0, 0, 19.3853267273472, 19.3853267273472, 0, 0, 0, 0, 0, 0, -12.2999789623031, -12.2999789623031, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1220323.29317823, 0, 0, 0, 0, 0, 0, 0, 0, 1219779.63055002, 0, 0, 0, 0, 0, 0, 0, 0, -1220507.79798153, 0, 0, 0, 0, 0, 0, 0, 0, 1219779.63055002, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712164648855504, 0, 0, 0, 0, 0, 0, 0, -0.0711808822471323, 0, 0, 0, -0.067302530849995, 0.350989236488403, 0, 0, -0.0702011111721388, 0, 0, 0, 0, 0, 0, 0, -0.0700364502371939, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 18.6430927038996, 18.6430927038996, 0, 0, 0, 0, 0, 0, 6.46910687871318, 6.46910687871318, 0, 0, -1000160.92860224, 990449.959662738, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17.8895048266275, 17.8895048266275, 0, 0, 0, 0, 0, 0, 0, 0, -1219743.47961911, 0, 0, 0, 0, 0, 0, 0, 0, 1220248.79304489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1220073.50069027, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712402356149471, 0, 0, 0, 0, 0, 0, 0, -0.0710434021898938, 0, 0, 0, -0.138998942974057, 0.346642435838878, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0700082901082964, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, -876.111886261751, -876.111886261751, 0, 0, 0, 0, 0, 0, -304.390722688526, -304.390722688526, 0, 0, -992431.703164106, 997363.838157946, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -840.762852169367, -840.762852169367, 0, 0, 0, 0, 0, 0, 0, 0, -1232067.0120842, 0, 0, 0, 0, 0, 0, 0, 0, 1208296.49063838, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1216542.44347147, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712517727684218, 0, 0, 0, 0, 0, 0, 0, -0.0710665763875378, 0, 0, 0, -0.135558808770639, 0.34978600362384, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0700089944889218, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -55.8917431473313, -55.8917431473313, 0, 0, 0, 0, 0, 0, -19.3844920746244, -19.3844920746244, 0, 0, -999517.502987933, 1009438.14766634, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -53.6308177074652, -53.6308177074652, 0, 0, 0, 0, 0, 0, 0, 0, -1220769.09829066, 0, 0, 0, 0, 0, 0, 0, 0, 1219254.06977453, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1219779.63055002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712322175091192, 0, 0, 0, 0, 0, 0, 0, -0.0714211077339342, 0, 0, 0, -0.138330997295328, 0.353348474105155, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0700231926757699, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13.4561374354576, 13.4561374354576, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1000138.06501875, 992489.833801141, 0, 0, 8.59926580788568, 8.59926580788568, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1219830.63213998, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1220195.34622141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712201935598085, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.208914179899625, 0.347363722638197, 0, 0, -0.0699217352075871, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -632.569712929349, -632.569712929349, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -993506.052050475, 998481.656266873, 0, 0, -404.451809485839, -404.451809485839, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1227967.25783156, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1210810.69033886, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712300827491613, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.206197447958818, 0.349831975272874, 0, 0, -0.0699330534272303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -40.3399440931136, -40.3399440931136, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -999586.048786897, 1007445.70934105, 0, 0, -25.774404633235, -25.774404633235, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1220507.79798153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1219414.31380831, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712513215084166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.20847396725155, 0.352629140071907, 0, 0, -0.0701064864785274, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.10274673552472, -4.10274673552472, 0, 0, 0, 0, 0, 0, -8.59910101799077, -8.59910101799077, 0, 0, -999974.482308221, 1001448.3028934, 0, 0, 7.31738817846815, 7.31738817846815, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1220073.50069027, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1220195.34622141, 0, 0, 0, 0, 0, 0, 0, 0, 1219962.29898285, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712145688590991, 0, 0, 0, 0, 0, 0, 0, -0.0711414991214726, 0, 0, 0, -0.138673400593227, 0.350508790686142, 0, 0, -0.0701793343212826, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 193.052852618787, 193.052852618787, 0, 0, 0, 0, 0, 0, 404.815559983562, 404.815559983562, 0, 0, -1001200.43787511, 1000345.90145999, 0, 0, -343.911657271068, -343.911657271068, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1216542.44347147, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1210810.69033886, 0, 0, 0, 0, 0, 0, 0, 0, 1221773.49896312, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712136064974105, 0, 0, 0, 0, 0, 0, 0, -0.0711641591880499, 0, 0, 0, -0.139209362953551, 0.350032428081143, 0, 0, -0.0701595959818201, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12.3003147612105, 12.3003147612105, 0, 0, 0, 0, 0, 0, 25.7758796498834, 25.7758796498834, 0, 0, -1000076.52678177, 998566.372825627, 0, 0, -21.9484365427487, -21.9484365427487, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1219779.63055002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1219414.31380831, 0, 0, 0, 0, 0, 0, 0, 0, 1220113.0350257, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712562298677439, 0, 0, 0, 0, 0, 0, 0, -0.0713271943108892, 0, 0, 0, -0.138811660844161, 0.349492586781448, 0, 0, -0.0698584860884893, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -17.8887914772881, -17.8887914772881, 0, 0, 0, 0, 0, 0, -7.31726876923409, -7.31726876923409, 0, 0, -999831.693684471, 1009077.87574601, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1220248.79304489, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1219962.29898285, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712285389013413, 0, 0, 0, 0, 0, 0, 0, -0.0713995529752014, 0, 0, 0, -0.209744645947434, 0.353186078923656, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 842.331535671518, 842.331535671518, 0, 0, 0, 0, 0, 0, 344.17230700849, 344.17230700849, 0, 0, -1007918.40812958, 1001764.86496989, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1208296.49063838, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1221773.49896312, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712172909795602, 0, 0, 0, 0, 0, 0, 0, -0.0713693102514209, 0, 0, 0, -0.213017488452914, 0.350203066661158, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 53.6372020286134, 53.6372020286134, 0, 0, 0, 0, 0, 0, 21.9495049627517, 21.9495049627517, 0, 0, -1000504.6112062, 990997.141723021, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1219254.06977453, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1220113.0350257, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.0712424467455444, 0, 0, 0, 0, 0, 0, 0, -0.0710778305212608, 0, 0, 0, -0.210321387896899, 0.34682254686407, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { -5.05190760027393, 13.8864001993601, 0, 0, -0.476042304755734, -0.476042304755734, 0, 0, 0, 0, 0, 0, -0.476529357197454, -0.476529357197454, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.78099920913611, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.78099920913611, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, -3.37621337574046, 15.5564092395998, 0, 0, -0.47739318449351, -0.47739318449351, 0, 0, 0, 0, 0, 0, -0.478383920637558, -0.478383920637558, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.77001438234528, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.77001438234528, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, -5.02451575435734, 14.0417178228723, 0, 0, -0.474448343813498, -0.474448343813498, 0, 0, 0, 0, 0, 0, -0.474003048546973, -0.474003048546973, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.81010562175692, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.81010562175692, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { -0.482379808693443, -0.482379808693443, 0, 0, -4.55330548752196, 14.5008460627009, 0, 0, -0.477236262769714, -0.477236262769714, 0, 0, 0, 0, 0, 0, -0.474624490323972, -0.474624490323972, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.80774594468969, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.78099920913611, 0, 0, 0, 0, 0, 0, 0, 0, 5.80774594468969, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, -0.480802415306281, -0.480802415306281, 0, 0, -2.96777944384538, 16.0907516179426, 0, 0, -0.478971617697908, -0.478971617697908, 0, 0, 0, 0, 0, 0, -0.473660693010736, -0.473660693010736, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.81629596924518, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.77001438234528, 0, 0, 0, 0, 0, 0, 0, 0, 5.81629596924518, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, -0.484463918199257, -0.484463918199257, 0, 0, -4.57610871410561, 14.3795130355621, 0, 0, -0.47331916866445, -0.47331916866445, 0, 0, 0, 0, 0, 0, -0.475707558877716, -0.475707558877716, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.7853266931937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.81010562175692, 0, 0, 0, 0, 0, 0, 0, 0, 5.7853266931937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, -0.484348803668567, -0.484348803668567, 0, 0, -5.02410944025782, 14.0999886014834, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.473247526563853, -0.473247526563853, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.82379176344284, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.80774594468969, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, -0.484430626221117, -0.484430626221117, 0, 0, -3.4943489398375, 15.6401687686325, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.470360090610084, -0.470360090610084, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.84426113458347, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.81629596924518, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, -0.48264931261501, -0.48264931261501, 0, 0, -5.07442025053009, 13.8152731510536, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.476944422689672, -0.476944422689672, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.77045581522745, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.7853266931937, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { -0.482383895197991, -0.482383895197991, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.54724538550988, 14.5358141181201, 0, 0, -0.474628931282051, -0.474628931282051, 0, 0, 0, 0, 0, 0, -0.473664308242654, -0.473664308242654, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.78099920913611, 0, 0, 0, 0, 0, 0, 0, 0, 5.81438694913692, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.81438694913692, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, -0.480612327144841, -0.480612327144841, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.98265276531383, 16.107285936737, 0, 0, -0.473453294484112, -0.473453294484112, 0, 0, 0, 0, 0, 0, -0.471507540526748, -0.471507540526748, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.77001438234528, 0, 0, 0, 0, 0, 0, 0, 0, 5.82785173576436, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.82785173576436, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, -0.484452127343276, -0.484452127343276, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4.58293638890622, 14.345360409225, 0, 0, -0.475694612002936, -0.475694612002936, 0, 0, 0, 0, 0, 0, -0.476576027499387, -0.476576027499387, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.81010562175692, 0, 0, 0, 0, 0, 0, 0, 0, 5.77917247818845, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.77917247818845, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, -0.484326139150934, -0.484326139150934, 0, 0, 0, 0, 0, 0, -0.48481351939682, -0.48481351939682, 0, 0, -4.09663394764801, 14.8726680703392, 0, 0, -0.473223509023094, -0.473223509023094, 0, 0, 0, 0, 0, 0, -0.475491862338708, -0.475491862338708, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.80774594468969, 0, 0, 0, 0, 0, 0, 0, 0, 5.78817637059894, 0, 0, 0, 0, 0, 0, 0, 0, -5.81438694913692, 0, 0, 0, 0, 0, 0, 0, 0, 5.78817637059894, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, -0.485499174908505, -0.485499174908505, 0, 0, 0, 0, 0, 0, -0.486473858755366, -0.486473858755366, 0, 0, -2.44598565227206, 16.5203314758601, 0, 0, -0.471473203467884, -0.471473203467884, 0, 0, 0, 0, 0, 0, -0.476047300618851, -0.476047300618851, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.81629596924518, 0, 0, 0, 0, 0, 0, 0, 0, 5.78239301554426, 0, 0, 0, 0, 0, 0, 0, 0, -5.82785173576436, 0, 0, 0, 0, 0, 0, 0, 0, 5.78239301554426, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, -0.482717113506835, -0.482717113506835, 0, 0, 0, 0, 0, 0, -0.482270742276126, -0.482270742276126, 0, 0, -4.08000025195207, 14.956463187109, 0, 0, -0.47701359338184, -0.47701359338184, 0, 0, 0, 0, 0, 0, -0.474941505265021, -0.474941505265021, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.7853266931937, 0, 0, 0, 0, 0, 0, 0, 0, 5.80345763815278, 0, 0, 0, 0, 0, 0, 0, 0, -5.77917247818845, 0, 0, 0, 0, 0, 0, 0, 0, 5.80345763815278, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, -0.485489492470082, -0.485489492470082, 0, 0, 0, 0, 0, 0, -0.482879542348031, -0.482879542348031, 0, 0, -4.59685910430782, 14.290881770537, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.477049897607695, -0.477049897607695, 0, 0, 0, 0, 0, 0, 0, 0, -5.82379176344284, 0, 0, 0, 0, 0, 0, 0, 0, 5.76925673671385, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.78817637059894, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.488533096636386, -0.488533096636386, 0, 0, 0, 0, 0, 0, -0.483117001892858, -0.483117001892858, 0, 0, -2.8833518680663, 15.9942866037892, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.479815617656994, -0.479815617656994, 0, 0, 0, 0, 0, 0, 0, 0, -5.84426113458347, 0, 0, 0, 0, 0, 0, 0, 0, 5.74982508558181, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.78239301554426, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.481682797805609, -0.481682797805609, 0, 0, 0, 0, 0, 0, -0.484063038140981, -0.484063038140981, 0, 0, -4.54438083303517, 14.5707014019383, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.473543714922803, -0.473543714922803, 0, 0, 0, 0, 0, 0, 0, 0, -5.77045581522745, 0, 0, 0, 0, 0, 0, 0, 0, 5.82098050892845, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.80345763815278, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.484804196560867, -0.484804196560867, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.06523832061017, 13.8473228458755, 0, 0, -0.475483350274719, -0.475483350274719, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.81438694913692, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.77502552498157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.486916087011048, -0.486916087011048, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3.36860597645304, 15.5360227491629, 0, 0, -0.47644871216546, -0.47644871216546, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.82785173576436, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.75973382124875, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.482299618199046, -0.482299618199046, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.02694660335143, 14.0641153619742, 0, 0, -0.474967584057198, -0.474967584057198, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.77917247818845, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.8156382236572, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.482900264990277, -0.482900264990277, 0, 0, 0, 0, 0, 0, -0.481933709165944, -0.481933709165944, 0, 0, -4.56852880011342, 14.4527493972609, 0, 0, -0.477069671157993, -0.477069671157993, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.78817637059894, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.77502552498157, 0, 0, 0, 0, 0, 0, 0, 0, 5.80017773904589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.482145483890945, -0.482145483890945, 0, 0, 0, 0, 0, 0, -0.480287932737162, -0.480287932737162, 0, 0, -2.95740204739159, 16.0654073149214, 0, 0, -0.478874387788298, -0.478874387788298, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.78239301554426, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.75973382124875, 0, 0, 0, 0, 0, 0, 0, 0, 5.80315824135793, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.484000565975519, -0.484000565975519, 0, 0, 0, 0, 0, 0, -0.484887409334384, -0.484887409334384, 0, 0, -4.57649713271395, 14.4103316912153, 0, 0, -0.473482065401585, -0.473482065401585, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.80345763815278, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.8156382236572, 0, 0, 0, 0, 0, 0, 0, 0, 5.79233927099339, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.481528377499255, -0.481528377499255, 0, 0, 0, 0, 0, 0, -0.483795475125457, -0.483795475125457, 0, 0, -5.03425305296178, 14.080212471678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.76925673671385, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.80017773904589, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.478606756622184, -0.478606756622184, 0, 0, 0, 0, 0, 0, -0.483225055987423, -0.483225055987423, 0, 0, -3.49672665982835, 15.6273283768126, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.74982508558181, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.80315824135793, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05, 0 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.485222227303247, -0.485222227303247, 0, 0, 0, 0, 0, 0, -0.483151876148865, -0.483151876148865, 0, 0, -5.08054114762413, 13.8181934804834, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.82098050892845, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -5.79233927099339, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.78778157533803e-05, 8.78778157533803e-05 },
       { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 } };
    // clang-format on

    auto left_DOFManager         = testJacobian->getLeftDOFManager();
    auto right_DOFManager        = testJacobian->getRightDOFManager();
    bool equal_to_leftDOFManager = false, equal_to_rightDOFManager = false;
    if ( *left_DOFManager == *subchannelDOFManager )
        equal_to_leftDOFManager = true;
    if ( *right_DOFManager == *subchannelDOFManager )
        equal_to_rightDOFManager = true;
    AMP_ASSERT( equal_to_leftDOFManager );
    AMP_ASSERT( equal_to_rightDOFManager );

    // create global ID maps
    std::map<size_t, std::unique_ptr<AMP::Mesh::MeshElement>> elements_by_globalID;
    std::map<size_t, size_t> variables_by_globalID;
    createGlobalIDMaps(
        subchannelDOFManager, subchannelMesh, elements_by_globalID, variables_by_globalID );

    // initialize success boolean for known residual comparison test
    bool passed_known_test = JacobianIsCorrect( testJacobian,
                                                knownJacobian,
                                                subchannelDOFManager,
                                                subchannelMesh,
                                                interiorLateralFaceMap,
                                                elements_by_globalID,
                                                variables_by_globalID );

    if ( passed_known_test )
        ut->passes( exeName + ": known value test" );
    else
        ut->failure( exeName + ": known residual test" );
}

int testSubchannelFourEqLinearOperator( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );

    AMP::UnitTest ut;

    // create list of input files
    std::string files[] = { "testSubchannelFourEqLinearOperator" };

    // execute unit test for each input file
    for ( auto &file : files )
        Test( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
