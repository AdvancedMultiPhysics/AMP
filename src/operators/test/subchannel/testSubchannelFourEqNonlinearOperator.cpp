#include "AMP/IO/PIO.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/operators/subchannel/SubchannelFourEqNonlinearOperator.h"
#include "AMP/operators/subchannel/SubchannelOperatorParameters.h"
#include "AMP/operators/subchannel/SubchannelPhysicsModel.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>


static void compare_face( const std::string &variable,
                          unsigned int i,
                          unsigned int j,
                          double value_array[][10],
                          double known_value,
                          bool &pass )
{
    if ( !AMP::Utilities::approx_equal( value_array[i][j], known_value, 1.0e-6 ) ) {
        std::cout << "Residual value for " << variable << "[" << i << "][" << j << "]"
                  << " does not match known MATLAB value" << std::endl;
        pass = false;
    }
}

static void compare_gap( const std::string &variable,
                         unsigned int i,
                         unsigned int j,
                         double value_array[][9],
                         double known_value,
                         bool &pass )
{
    if ( !AMP::Utilities::approx_equal( value_array[i][j], known_value, 1.0e-6 ) ) {
        std::cout << "Residual value for " << variable << "[" << i << "][" << j << "]"
                  << " does not match known MATLAB value" << std::endl;
        pass = false;
    }
}

unsigned int getMATLABGapIndex( const AMP::Mesh::MeshElement &gapFace )
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
    unsigned int k = 0;
    // look at location of gap to determine gap MATLAB index
    if ( ( AMP::Utilities::approx_equal( centroid[0], x1, 1.0e-12 ) ) &&
         ( AMP::Utilities::approx_equal( centroid[1], x4, 1.0e-12 ) ) )
        k = 1;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x3, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x4, 1.0e-12 ) ) )
        k = 2;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x5, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x4, 1.0e-12 ) ) )
        k = 3;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x1, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x2, 1.0e-12 ) ) )
        k = 4;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x3, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x2, 1.0e-12 ) ) )
        k = 5;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x5, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x2, 1.0e-12 ) ) )
        k = 6;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x2, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x5, 1.0e-12 ) ) )
        k = 7;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x2, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x3, 1.0e-12 ) ) )
        k = 8;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x2, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x1, 1.0e-12 ) ) )
        k = 9;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x4, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x5, 1.0e-12 ) ) )
        k = 10;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x4, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x3, 1.0e-12 ) ) )
        k = 11;
    else if ( ( AMP::Utilities::approx_equal( centroid[0], x4, 1.0e-12 ) ) &&
              ( AMP::Utilities::approx_equal( centroid[1], x1, 1.0e-12 ) ) )
        k = 12;
    else
        AMP_ERROR( "Gap face did not match any known location" );

    return k;
}

unsigned int getMATLABAxialIndex( const AMP::Mesh::MeshElement &gapFace )
{
    double height   = 3.66;
    unsigned int Nz = 9;
    double dz       = height / Nz;

    // get gap face centroid
    auto centroid = gapFace.centroid();
    // axial interval MATLAB index
    unsigned int j = 0;
    // boolean for if the axial index has been found
    bool foundIndex = false;
    // loop over axial intervals
    for ( unsigned int i = 0; i < Nz; ++i ) {
        if ( AMP::Utilities::approx_equal( centroid[2], ( i + 0.5 ) * dz, 1.0e-12 ) ) {
            j          = i + 1;
            foundIndex = true;
            break;
        }
    }
    if ( !foundIndex )
        AMP_ERROR( "Axial index was not found for gap face" );

    return j;
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

    // get dof manager
    std::shared_ptr<AMP::Discretization::DOFManager> subchannelDOFManager;
    if ( subchannelMesh ) {
        auto axialFaces0 = AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( subchannelMesh, 0 );
        auto axialFaces1 = AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( subchannelMesh, 1 );
        auto gapFaces0   = AMP::Mesh::Mesh::getIterator(
            AMP::Mesh::SetOP::Union,
            AMP::Mesh::StructuredMeshHelper::getXZFaceIterator( subchannelMesh, 0 ),
            AMP::Mesh::StructuredMeshHelper::getYZFaceIterator( subchannelMesh, 0 ) );
        auto gapFaces1 = AMP::Mesh::Mesh::getIterator(
            AMP::Mesh::SetOP::Union,
            AMP::Mesh::StructuredMeshHelper::getXZFaceIterator( subchannelMesh, 1 ),
            AMP::Mesh::StructuredMeshHelper::getYZFaceIterator( subchannelMesh, 1 ) );
        std::vector<std::shared_ptr<AMP::Discretization::DOFManager>> subchannelChildrenDOFManagers(
            2 );
        subchannelChildrenDOFManagers[0] = AMP::Discretization::simpleDOFManager::create(
            subchannelMesh, axialFaces1, axialFaces0, 3 );
        subchannelChildrenDOFManagers[1] = AMP::Discretization::simpleDOFManager::create(
            subchannelMesh, gapFaces1, gapFaces0, 1 );
        subchannelDOFManager = std::make_shared<AMP::Discretization::multiDOFManager>(
            subchannelMesh->getComm(), subchannelChildrenDOFManagers );
    }

    // get input and output variables
    auto inputVariable  = std::make_shared<AMP::LinearAlgebra::Variable>( "flow" );
    auto outputVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "flow" );

    // create solution, rhs, and residual vectors
    auto SolVec = AMP::LinearAlgebra::createVector( subchannelDOFManager, inputVariable, true );
    auto ResVec = AMP::LinearAlgebra::createVector( subchannelDOFManager, outputVariable, true );

    // create subchannel physics model
    auto subchannelPhysics_db = input_db->getDatabase( "SubchannelPhysicsModel" );
    auto params =
        std::make_shared<AMP::Operator::ElementPhysicsModelParameters>( subchannelPhysics_db );
    auto subchannelPhysicsModel = std::make_shared<AMP::Operator::SubchannelPhysicsModel>( params );

    // get nonlinear operator database
    auto subchannelOperator_db = input_db->getDatabase( "SubchannelFourEqNonlinearOperator" );
    // set operator parameters
    auto subchannelOpParams =
        std::make_shared<AMP::Operator::SubchannelOperatorParameters>( subchannelOperator_db );
    subchannelOpParams->d_Mesh                   = subchannelMesh;
    subchannelOpParams->d_subchannelPhysicsModel = subchannelPhysicsModel;
    subchannelOpParams->clad_x =
        input_db->getDatabase( "CladProperties" )->getVector<double>( "x" );
    subchannelOpParams->clad_y =
        input_db->getDatabase( "CladProperties" )->getVector<double>( "y" );
    subchannelOpParams->clad_d =
        input_db->getDatabase( "CladProperties" )->getVector<double>( "d" );
    // create nonlinear operator
    auto subchannelOperator =
        std::make_shared<AMP::Operator::SubchannelFourEqNonlinearOperator>( subchannelOpParams );
    // reset the nonlinear operator
    subchannelOperator->reset( subchannelOpParams );

    // report successful creation
    ut->passes( exeName + ": creation" );
    std::cout.flush();

    // check number of lateral gaps
    std::map<AMP::Mesh::Point, std::unique_ptr<AMP::Mesh::MeshElement>> interiorLateralFaceMap;
    std::map<AMP::Mesh::Point, std::unique_ptr<AMP::Mesh::MeshElement>> exteriorLateralFaceMap;
    subchannelOperator->getLateralFaces(
        subchannelOpParams->d_Mesh, interiorLateralFaceMap, exteriorLateralFaceMap );
    size_t Ngaps = interiorLateralFaceMap.size();
    if ( Ngaps ==
         108 ) { // for 3x3 subchannel array with 9 axial intervals, there are 12x9=108 gaps
        ut->passes( exeName + ": number of lateral gaps" );
    } else {
        std::cout << "Incorrent number of lateral gaps. Found: " << Ngaps << ". Expected: 108."
                  << std::endl;
        ut->failure( exeName + ": number of lateral gaps" );
    }

    // number of subchannels
    const size_t numSubchannels = 3 * 3; // 3x3 subchannel array
    // number of axial intervals
    const size_t numAxialIntervals = 9;
    // number of gaps
    const size_t numGaps = 12;
    // compute height of subchannels
    auto box            = subchannelOpParams->d_Mesh->getBoundingBox();
    const double height = box[5] - box[4];
    // height of each axial interval
    const double dz = height / numAxialIntervals;
    // get all of the unique x,y,z points in subchannel mesh
    subchannelOperator->fillSubchannelGrid( subchannelOpParams->d_Mesh );

    // put all cells in an array by subchannel
    std::unique_ptr<AMP::Mesh::MeshElement>
        d_elem[numSubchannels][numAxialIntervals]; // array of array of elements for each subchannel
    auto cell = subchannelOpParams->d_Mesh->getIterator( AMP::Mesh::GeomType::Cell,
                                                         0 ); // iterator for cells of mesh
    for ( ; cell != cell.end(); ++cell ) {                    // loop over all cells
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

    // set axial face quantities of input vector for the nonlinear residual
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
                SolVec->setValuesByGlobalID( 1, &minusDofs[0], &val );
                val = h_scale * 1000.0e3 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
                SolVec->setValuesByGlobalID( 1, &minusDofs[1], &val );
                val = p_scale * 15.5e6 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
                SolVec->setValuesByGlobalID( 1, &minusDofs[2], &val );
            }

            size_t jj = j + 2; // corresponding MATLAB index for this axial face
            // get dofs on plus face
            std::vector<size_t> plusDofs;
            subchannelDOFManager->getDOFs( plusFace->globalID(), plusDofs );
            // set values of plus face
            val = m_scale * 0.35 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
            SolVec->setValuesByGlobalID( 1, &plusDofs[0], &val );
            val = h_scale * 1000.0e3 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
            SolVec->setValuesByGlobalID( 1, &plusDofs[1], &val );
            val = p_scale * 15.5e6 * ( 1.0 + 1.0 / 100.0 * cos( ii ) * cos( 17.3 * jj ) );
            SolVec->setValuesByGlobalID( 1, &plusDofs[2], &val );
        }
    }

    // get scale factors for lateral mass flow rate
    const double w_scale = AMP::Operator::Subchannel::scaleLateralMassFlowRate;

    // set lateral face quantities (lateral mass flow rates) of input vector for nonlinear residual
    // array of gap faces
    std::unique_ptr<AMP::Mesh::MeshElement> gapFaces[numGaps][numAxialIntervals]; // gap faces
    // loop over all faces in mesh
    AMP::Mesh::MeshIterator face =
        subchannelOpParams->d_Mesh->getIterator( AMP::Mesh::GeomType::Face, 0 );
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
            unsigned int j = getMATLABAxialIndex( *lateralFace );
            // put gap face in array
            gapFaces[k - 1][j - 1] = lateralFace->clone();
            // get crossflow from solution vector
            std::vector<size_t> gapDofs;
            subchannelDOFManager->getDOFs( lateralFace->globalID(), gapDofs );
            // set test value for crossflow
            double val = w_scale * 0.001 * ( 1.0 + 1.0 / 100.0 * cos( k ) * cos( 17.3 * j ) );
            SolVec->setValuesByGlobalID( 1, &gapDofs[0], &val );
        }
    }
    SolVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // apply the operator
    subchannelOperator->apply( SolVec, ResVec );

    // initialize success boolean for known residual comparison test
    bool pass = true;

    std::cout << std::setprecision( 6 ) << std::scientific;

    double m_res[numSubchannels][numAxialIntervals + 1];
    double h_res[numSubchannels][numAxialIntervals + 1];
    double p_res[numSubchannels][numAxialIntervals + 1];
    // loop over subchannels
    for ( size_t isub = 0; isub < numSubchannels; ++isub ) {
        size_t ii = isub + 1; // corresponding MATLAB index for this subchannel
        std::cout << "\nSubchannel " << ii << ":" << std::endl;
        // loop over axial intervals
        for ( size_t j = 0; j < numAxialIntervals; ++j ) {
            // get axial faces
            std::unique_ptr<AMP::Mesh::MeshElement> plusFace;  // upper axial face for current cell
            std::unique_ptr<AMP::Mesh::MeshElement> minusFace; // lower axial face for current cell
            subchannelOperator->getAxialFaces( *d_elem[isub][j], plusFace, minusFace );

            // get unknowns of first face
            if ( j == 0 ) {
                // get dofs on minus face
                std::vector<size_t> minusDofs;
                subchannelDOFManager->getDOFs( minusFace->globalID(), minusDofs );
                // get values of minus face
                m_res[ii - 1][0] = ResVec->getValueByGlobalID( minusDofs[0] ) / m_scale;
                h_res[ii - 1][0] = ResVec->getValueByGlobalID( minusDofs[1] ) / h_scale;
                p_res[ii - 1][0] = ResVec->getValueByGlobalID( minusDofs[2] ) / p_scale;
                std::cout << "GeomType::Face 0:\t" << m_res[ii - 1][0] << "\t" << h_res[ii - 1][0]
                          << "\t" << p_res[ii - 1][0] << std::endl;
            }

            // get dofs on plus face
            std::vector<size_t> plusDofs;
            subchannelDOFManager->getDOFs( plusFace->globalID(), plusDofs );
            // get values of plus face
            m_res[ii - 1][j + 1] = ResVec->getValueByGlobalID( plusDofs[0] ) / m_scale;
            h_res[ii - 1][j + 1] = ResVec->getValueByGlobalID( plusDofs[1] ) / h_scale;
            p_res[ii - 1][j + 1] = ResVec->getValueByGlobalID( plusDofs[2] ) / p_scale;
            std::cout << "GeomType::Face " << j + 1 << ":\t" << m_res[ii - 1][j + 1] << "\t"
                      << h_res[ii - 1][j + 1] << "\t" << p_res[ii - 1][j + 1] << std::endl;
        }
    }

    double w_res[numGaps][numAxialIntervals];
    // loop over gaps
    for ( size_t k = 0; k < numGaps; ++k ) {
        std::cout << "\nGap " << k + 1 << ":" << std::endl;
        for ( size_t j = 0; j < numAxialIntervals; ++j ) {
            std::vector<size_t> gapDofs;
            subchannelDOFManager->getDOFs( gapFaces[k][j]->globalID(), gapDofs );
            w_res[k][j] = ResVec->getValueByGlobalID( gapDofs[0] ) / w_scale;
            std::cout << "Interval " << j + 1 << ":\t" << w_res[k][j] << std::endl;
        }
    }

    // compare residual with MATLAB residual values
    compare_face( "Axial mass flow rate", 0, 0, m_res, 0.0380401638196053, pass );
    compare_face( "Axial mass flow rate", 0, 1, m_res, -0.00111607071154988, pass );
    compare_face( "Axial mass flow rate", 0, 2, m_res, 0.00257700800092406, pass );
    compare_face( "Axial mass flow rate", 0, 3, m_res, 0.002817654074108, pass );
    compare_face( "Axial mass flow rate", 0, 4, m_res, -0.000865202566793987, pass );
    compare_face( "Axial mass flow rate", 0, 5, m_res, -0.001262287611432, pass );
    compare_face( "Axial mass flow rate", 0, 6, m_res, 0.00240370180328643, pass );
    compare_face( "Axial mass flow rate", 0, 7, m_res, 0.00295650933981623, pass );
    compare_face( "Axial mass flow rate", 0, 8, m_res, -0.000685998128451605, pass );
    compare_face( "Axial mass flow rate", 0, 9, m_res, -0.00139353069976071, pass );
    compare_face( "Axial mass flow rate", 1, 0, m_res, 0.0379690653800828, pass );
    compare_face( "Axial mass flow rate", 1, 1, m_res, 0.00189262765551727, pass );
    compare_face( "Axial mass flow rate", 1, 2, m_res, -0.000947622136685143, pass );
    compare_face( "Axial mass flow rate", 1, 3, m_res, -0.00113682128239614, pass );
    compare_face( "Axial mass flow rate", 1, 4, m_res, 0.00169539178108148, pass );
    compare_face( "Axial mass flow rate", 1, 5, m_res, 0.00200489658671673, pass );
    compare_face( "Axial mass flow rate", 1, 6, m_res, -0.000814169450369545, pass );
    compare_face( "Axial mass flow rate", 1, 7, m_res, -0.00124342146158573, pass );
    compare_face( "Axial mass flow rate", 1, 8, m_res, 0.00155741097325171, pass );
    compare_face( "Axial mass flow rate", 1, 9, m_res, 0.00210563567008801, pass );
    compare_face( "Axial mass flow rate", 2, 0, m_res, 0.0379264080874497, pass );
    compare_face( "Axial mass flow rate", 2, 1, m_res, 0.00353542660640621, pass );
    compare_face( "Axial mass flow rate", 2, 2, m_res, -0.00324059158527243, pass );
    compare_face( "Axial mass flow rate", 2, 3, m_res, -0.00367307921250371, pass );
    compare_face( "Axial mass flow rate", 2, 4, m_res, 0.00308456793567051, pass );
    compare_face( "Axial mass flow rate", 2, 5, m_res, 0.0038041043058722, pass );
    compare_face( "Axial mass flow rate", 2, 6, m_res, -0.00292297865242048, pass );
    compare_face( "Axial mass flow rate", 2, 7, m_res, -0.00392826547166117, pass );
    compare_face( "Axial mass flow rate", 2, 8, m_res, 0.00275611529880499, pass );
    compare_face( "Axial mass flow rate", 2, 9, m_res, 0.00404533867993266, pass );
    compare_face( "Axial mass flow rate", 3, 0, m_res, 0.0379514108598289, pass );
    compare_face( "Axial mass flow rate", 3, 1, m_res, 0.00274082883951322, pass );
    compare_face( "Axial mass flow rate", 3, 2, m_res, -0.00172790014067486, pass );
    compare_face( "Axial mass flow rate", 3, 3, m_res, -0.00201816681608982, pass );
    compare_face( "Axial mass flow rate", 3, 4, m_res, 0.00243823232762159, pass );
    compare_face( "Axial mass flow rate", 3, 5, m_res, 0.00291779621615355, pass );
    compare_face( "Axial mass flow rate", 3, 6, m_res, -0.00151823219668687, pass );
    compare_face( "Axial mass flow rate", 3, 7, m_res, -0.00218622799824502, pass );
    compare_face( "Axial mass flow rate", 3, 8, m_res, 0.00222142554521501, pass );
    compare_face( "Axial mass flow rate", 3, 9, m_res, 0.00307664796298427, pass );
    compare_face( "Axial mass flow rate", 4, 0, m_res, 0.0380210862636011, pass );
    compare_face( "Axial mass flow rate", 4, 1, m_res, -0.00101293482593628, pass );
    compare_face( "Axial mass flow rate", 4, 2, m_res, 0.000925248740600401, pass );
    compare_face( "Axial mass flow rate", 4, 3, m_res, 0.00105223718696533, pass );
    compare_face( "Axial mass flow rate", 4, 4, m_res, -0.000880552213066112, pass );
    compare_face( "Axial mass flow rate", 4, 5, m_res, -0.00108964094604573, pass );
    compare_face( "Axial mass flow rate", 4, 6, m_res, 0.000834266863003721, pass );
    compare_face( "Axial mass flow rate", 4, 7, m_res, 0.00112507861378369, pass );
    compare_face( "Axial mass flow rate", 4, 8, m_res, -0.000786476205286927, pass );
    compare_face( "Axial mass flow rate", 4, 9, m_res, -0.00115848624829812, pass );
    compare_face( "Axial mass flow rate", 5, 0, m_res, 0.0380713750538627, pass );
    compare_face( "Axial mass flow rate", 5, 1, m_res, -0.00383543782609493, pass );
    compare_face( "Axial mass flow rate", 5, 2, m_res, 0.00272899559274731, pass );
    compare_face( "Axial mass flow rate", 5, 3, m_res, 0.00315529995121174, pass );
    compare_face( "Axial mass flow rate", 5, 4, m_res, -0.00339102507464462, pass );
    compare_face( "Axial mass flow rate", 5, 5, m_res, -0.00409540171604186, pass );
    compare_face( "Axial mass flow rate", 5, 6, m_res, 0.00242100306907703, pass );
    compare_face( "Axial mass flow rate", 5, 7, m_res, 0.00340218105290036, pass );
    compare_face( "Axial mass flow rate", 5, 8, m_res, -0.00307254563025331, pass );
    compare_face( "Axial mass flow rate", 5, 9, m_res, -0.00432875457007729, pass );
    compare_face( "Axial mass flow rate", 6, 0, m_res, 0.0380560419487658, pass );
    compare_face( "Axial mass flow rate", 6, 1, m_res, -0.00269234154998653, pass );
    compare_face( "Axial mass flow rate", 6, 2, m_res, 0.00246929880233966, pass );
    compare_face( "Axial mass flow rate", 6, 3, m_res, 0.00279723147715915, pass );
    compare_face( "Axial mass flow rate", 6, 4, m_res, -0.00235047907681835, pass );
    compare_face( "Axial mass flow rate", 6, 5, m_res, -0.00289707422561653, pass );
    compare_face( "Axial mass flow rate", 6, 6, m_res, 0.0022274182693329, pass );
    compare_face( "Axial mass flow rate", 6, 7, m_res, 0.00299168964430632, pass );
    compare_face( "Axial mass flow rate", 6, 8, m_res, -0.0021003384243919, pass );
    compare_face( "Axial mass flow rate", 6, 9, m_res, -0.00308090701409712, pass );
    compare_face( "Axial mass flow rate", 7, 0, m_res, 0.0379891841344244, pass );
    compare_face( "Axial mass flow rate", 7, 1, m_res, 0.00011306696483164, pass );
    compare_face( "Axial mass flow rate", 7, 2, m_res, -0.000889007249895837, pass );
    compare_face( "Axial mass flow rate", 7, 3, m_res, -0.000946888976408175, pass );
    compare_face( "Axial mass flow rate", 7, 4, m_res, 5.27265605470775e-05, pass );
    compare_face( "Axial mass flow rate", 7, 5, m_res, 0.000153069572332974, pass );
    compare_face( "Axial mass flow rate", 7, 6, m_res, -0.000842283632664553, pass );
    compare_face( "Axial mass flow rate", 7, 7, m_res, -0.000984906876022336, pass );
    compare_face( "Axial mass flow rate", 7, 8, m_res, 4.38803356166644e-06, pass );
    compare_face( "Axial mass flow rate", 7, 9, m_res, 0.000189034166543037, pass );
    compare_face( "Axial mass flow rate", 8, 0, m_res, 0.0379322703770131, pass );
    compare_face( "Axial mass flow rate", 8, 1, m_res, 0.00244031939765477, pass );
    compare_face( "Axial mass flow rate", 8, 2, m_res, -0.00378901584924407, pass );
    compare_face( "Axial mass flow rate", 8, 3, m_res, -0.00419338596460365, pass );
    compare_face( "Axial mass flow rate", 8, 4, m_res, 0.00201877260367274, pass );
    compare_face( "Axial mass flow rate", 8, 5, m_res, 0.0026870204073839, pass );
    compare_face( "Axial mass flow rate", 8, 6, m_res, -0.00349675258706133, pass );
    compare_face( "Axial mass flow rate", 8, 7, m_res, -0.00442767232760942, pass );
    compare_face( "Axial mass flow rate", 8, 8, m_res, 0.00171655741524677, pass );
    compare_face( "Axial mass flow rate", 8, 9, m_res, 0.00290846938956146, pass );
    compare_gap( "Lateral mass flow rate", 0, 0, w_res, 0.0010001147537703, pass );
    compare_gap( "Lateral mass flow rate", 0, 1, w_res, 18499.2488452321, pass );
    compare_gap( "Lateral mass flow rate", 0, 2, w_res, 1179.06080219325, pass );
    compare_gap( "Lateral mass flow rate", 0, 3, w_res, -18449.1651505823, pass );
    compare_gap( "Lateral mass flow rate", 0, 4, w_res, -1962.73735876877, pass );
    compare_gap( "Lateral mass flow rate", 0, 5, w_res, 18365.7927483099, pass );
    compare_gap( "Lateral mass flow rate", 0, 6, w_res, 2742.87245966353, pass );
    compare_gap( "Lateral mass flow rate", 0, 7, w_res, -18249.2820620335, pass );
    compare_gap( "Lateral mass flow rate", 0, 8, w_res, -3518.05845909234, pass );
    compare_gap( "Lateral mass flow rate", 1, 0, w_res, 0.000999911615371665, pass );
    compare_gap( "Lateral mass flow rate", 1, 1, w_res, -10842.9879036625, pass );
    compare_gap( "Lateral mass flow rate", 1, 2, w_res, -691.0843942142, pass );
    compare_gap( "Lateral mass flow rate", 1, 3, w_res, 10813.6322882727, pass );
    compare_gap( "Lateral mass flow rate", 1, 4, w_res, 1150.42172036487, pass );
    compare_gap( "Lateral mass flow rate", 1, 5, w_res, -10764.7651134222, pass );
    compare_gap( "Lateral mass flow rate", 1, 6, w_res, -1607.68328043691, pass );
    compare_gap( "Lateral mass flow rate", 1, 7, w_res, 10696.4745622075, pass );
    compare_gap( "Lateral mass flow rate", 1, 8, w_res, 2062.04402563898, pass );
    compare_gap( "Lateral mass flow rate", 2, 0, w_res, 0.000999789737392714, pass );
    compare_gap( "Lateral mass flow rate", 2, 1, w_res, -30216.2315780837, pass );
    compare_gap( "Lateral mass flow rate", 2, 2, w_res, -1925.84978424341, pass );
    compare_gap( "Lateral mass flow rate", 2, 3, w_res, 30134.4260717582, pass );
    compare_gap( "Lateral mass flow rate", 2, 4, w_res, 3205.88837667081, pass );
    compare_gap( "Lateral mass flow rate", 2, 5, w_res, -29998.2475732798, pass );
    compare_gap( "Lateral mass flow rate", 2, 6, w_res, -4480.14242528513, pass );
    compare_gap( "Lateral mass flow rate", 2, 7, w_res, 29807.9418041319, pass );
    compare_gap( "Lateral mass flow rate", 2, 8, w_res, 5746.31274422399, pass );
    compare_gap( "Lateral mass flow rate", 3, 0, w_res, 0.000999861173885226, pass );
    compare_gap( "Lateral mass flow rate", 3, 1, w_res, -21808.811287665, pass );
    compare_gap( "Lateral mass flow rate", 3, 2, w_res, -1389.99776310714, pass );
    compare_gap( "Lateral mass flow rate", 3, 3, w_res, 21749.7674981342, pass );
    compare_gap( "Lateral mass flow rate", 3, 4, w_res, 2313.87604522359, pass );
    compare_gap( "Lateral mass flow rate", 3, 5, w_res, -21651.4795570349, pass );
    compare_gap( "Lateral mass flow rate", 3, 6, w_res, -3233.579284512, pass );
    compare_gap( "Lateral mass flow rate", 3, 7, w_res, 21514.1248189362, pass );
    compare_gap( "Lateral mass flow rate", 3, 8, w_res, 4147.44802730216, pass );
    compare_gap( "Lateral mass flow rate", 4, 0, w_res, 0.00100006024646743, pass );
    compare_gap( "Lateral mass flow rate", 4, 1, w_res, 6649.52952517638, pass );
    compare_gap( "Lateral mass flow rate", 4, 2, w_res, 423.811792378254, pass );
    compare_gap( "Lateral mass flow rate", 4, 3, w_res, -6631.52700804774, pass );
    compare_gap( "Lateral mass flow rate", 4, 4, w_res, -705.503249970196, pass );
    compare_gap( "Lateral mass flow rate", 4, 5, w_res, 6601.55891407478, pass );
    compare_gap( "Lateral mass flow rate", 4, 6, w_res, 985.921739266044, pass );
    compare_gap( "Lateral mass flow rate", 4, 7, w_res, -6559.67930627218, pass );
    compare_gap( "Lateral mass flow rate", 4, 8, w_res, -1264.561277752, pass );
    compare_gap( "Lateral mass flow rate", 5, 0, w_res, 0.00100020392872532, pass );
    compare_gap( "Lateral mass flow rate", 5, 1, w_res, 28994.3235592589, pass );
    compare_gap( "Lateral mass flow rate", 5, 2, w_res, 1847.97074193061, pass );
    compare_gap( "Lateral mass flow rate", 5, 3, w_res, -28915.8261650702, pass );
    compare_gap( "Lateral mass flow rate", 5, 4, w_res, -3076.24610926885, pass );
    compare_gap( "Lateral mass flow rate", 5, 5, w_res, 28785.1545650523, pass );
    compare_gap( "Lateral mass flow rate", 5, 6, w_res, 4298.97086423552, pass );
    compare_gap( "Lateral mass flow rate", 5, 7, w_res, -28602.5445279784, pass );
    compare_gap( "Lateral mass flow rate", 5, 8, w_res, -5513.93877441103, pass );
    compare_gap( "Lateral mass flow rate", 6, 0, w_res, 0.00100016011985362, pass );
    compare_gap( "Lateral mass flow rate", 6, 1, w_res, 14819.4237999902, pass );
    compare_gap( "Lateral mass flow rate", 6, 2, w_res, 944.524951759385, pass );
    compare_gap( "Lateral mass flow rate", 6, 3, w_res, -14779.302629138, pass );
    compare_gap( "Lateral mass flow rate", 6, 4, w_res, -1572.31449618199, pass );
    compare_gap( "Lateral mass flow rate", 6, 5, w_res, 14712.5144595199, pass );
    compare_gap( "Lateral mass flow rate", 6, 6, w_res, 2197.26704342857, pass );
    compare_gap( "Lateral mass flow rate", 6, 7, w_res, -14619.1797907007, pass );
    compare_gap( "Lateral mass flow rate", 6, 8, w_res, -2818.25495136741, pass );
    compare_gap( "Lateral mass flow rate", 7, 0, w_res, 0.000999969097526927, pass );
    compare_gap( "Lateral mass flow rate", 7, 1, w_res, -14522.8129655036, pass );
    compare_gap( "Lateral mass flow rate", 7, 2, w_res, -925.62025764387, pass );
    compare_gap( "Lateral mass flow rate", 7, 3, w_res, 14483.4948256229, pass );
    compare_gap( "Lateral mass flow rate", 7, 4, w_res, 1540.84459676743, pass );
    compare_gap( "Lateral mass flow rate", 7, 5, w_res, -14418.0434176786, pass );
    compare_gap( "Lateral mass flow rate", 7, 6, w_res, -2153.28871099462, pass );
    compare_gap( "Lateral mass flow rate", 7, 7, w_res, 14326.5768482459, pass );
    compare_gap( "Lateral mass flow rate", 7, 8, w_res, 2761.84754846584, pass );
    compare_gap( "Lateral mass flow rate", 8, 0, w_res, 0.000999806486791466, pass );
    compare_gap( "Lateral mass flow rate", 8, 1, w_res, 13935.5279570443, pass );
    compare_gap( "Lateral mass flow rate", 8, 2, w_res, 888.189253201211, pass );
    compare_gap( "Lateral mass flow rate", 8, 3, w_res, -13897.7997924659, pass );
    compare_gap( "Lateral mass flow rate", 8, 4, w_res, -1478.53465821826, pass );
    compare_gap( "Lateral mass flow rate", 8, 5, w_res, 13834.9951667083, pass );
    compare_gap( "Lateral mass flow rate", 8, 6, w_res, 2066.2122777409, pass );
    compare_gap( "Lateral mass flow rate", 8, 7, w_res, -13747.2273920973, pass );
    compare_gap( "Lateral mass flow rate", 8, 8, w_res, -2650.16172605263, pass );
    compare_gap( "Lateral mass flow rate", 9, 0, w_res, 0.0009998217912075, pass );
    compare_gap( "Lateral mass flow rate", 9, 1, w_res, 8891.28514445208, pass );
    compare_gap( "Lateral mass flow rate", 9, 2, w_res, 566.69133723886, pass );
    compare_gap( "Lateral mass flow rate", 9, 3, w_res, -8867.21344500693, pass );
    compare_gap( "Lateral mass flow rate", 9, 4, w_res, -943.34942543078, pass );
    compare_gap( "Lateral mass flow rate", 9, 5, w_res, 8827.14221233361, pass );
    compare_gap( "Lateral mass flow rate", 9, 6, w_res, 1318.30538829636, pass );
    compare_gap( "Lateral mass flow rate", 9, 7, w_res, -8771.14373945173, pass );
    compare_gap( "Lateral mass flow rate", 9, 8, w_res, -1690.88266452952, pass );
    compare_gap( "Lateral mass flow rate", 10, 0, w_res, 0.00100000093996551, pass );
    compare_gap( "Lateral mass flow rate", 10, 1, w_res, -10481.958524698, pass );
    compare_gap( "Lateral mass flow rate", 10, 2, w_res, -668.0739063223, pass );
    compare_gap( "Lateral mass flow rate", 10, 3, w_res, 10453.5803397917, pass );
    compare_gap( "Lateral mass flow rate", 10, 4, w_res, 1112.1170841339, pass );
    compare_gap( "Lateral mass flow rate", 10, 5, w_res, -10406.3402547761, pass );
    compare_gap( "Lateral mass flow rate", 10, 6, w_res, -1554.15361037873, pass );
    compare_gap( "Lateral mass flow rate", 10, 7, w_res, 10340.323516158, pass );
    compare_gap( "Lateral mass flow rate", 10, 8, w_res, 1993.38590827519, pass );
    compare_gap( "Lateral mass flow rate", 11, 0, w_res, 0.00100017922452356, pass );
    compare_gap( "Lateral mass flow rate", 11, 1, w_res, 11862.835437601, pass );
    compare_gap( "Lateral mass flow rate", 11, 2, w_res, 756.084977656127, pass );
    compare_gap( "Lateral mass flow rate", 11, 3, w_res, -11830.7187478497, pass );
    compare_gap( "Lateral mass flow rate", 11, 4, w_res, -1258.62570699965, pass );
    compare_gap( "Lateral mass flow rate", 11, 5, w_res, 11777.2553300435, pass );
    compare_gap( "Lateral mass flow rate", 11, 6, w_res, 1758.89544331628, pass );
    compare_gap( "Lateral mass flow rate", 11, 7, w_res, -11702.5416421479, pass );
    compare_gap( "Lateral mass flow rate", 11, 8, w_res, -2255.99151465695, pass );
    compare_face( "Specific Enthalpy", 0, 0, h_res, -262962.972855096, pass );
    compare_face( "Specific Enthalpy", 0, 1, h_res, -4497.24092396573, pass );
    compare_face( "Specific Enthalpy", 0, 2, h_res, -410.518717108628, pass );
    compare_face( "Specific Enthalpy", 0, 3, h_res, -1696.21755851735, pass );
    compare_face( "Specific Enthalpy", 0, 4, h_res, -10023.2254975965, pass );
    compare_face( "Specific Enthalpy", 0, 5, h_res, -11755.2417670484, pass );
    compare_face( "Specific Enthalpy", 0, 6, h_res, -4481.96414056463, pass );
    compare_face( "Specific Enthalpy", 0, 7, h_res, -1461.246842005, pass );
    compare_face( "Specific Enthalpy", 0, 8, h_res, -5940.98220748848, pass );
    compare_face( "Specific Enthalpy", 0, 9, h_res, -4968.35594336474, pass );
    compare_face( "Specific Enthalpy", 1, 0, h_res, -263168.515839132, pass );
    compare_face( "Specific Enthalpy", 1, 1, h_res, 1727.01535103654, pass );
    compare_face( "Specific Enthalpy", 1, 2, h_res, -6816.00368335511, pass );
    compare_face( "Specific Enthalpy", 1, 3, h_res, -9947.50883515348, pass );
    compare_face( "Specific Enthalpy", 1, 4, h_res, -6212.9718613018, pass );
    compare_face( "Specific Enthalpy", 1, 5, h_res, -5932.76672396386, pass );
    compare_face( "Specific Enthalpy", 1, 6, h_res, -10734.5492681999, pass );
    compare_face( "Specific Enthalpy", 1, 7, h_res, -10139.7872510909, pass );
    compare_face( "Specific Enthalpy", 1, 8, h_res, -2302.16898057053, pass );
    compare_face( "Specific Enthalpy", 1, 9, h_res, 2110.39440238724, pass );
    compare_face( "Specific Enthalpy", 2, 0, h_res, -263291.836531722, pass );
    compare_face( "Specific Enthalpy", 2, 1, h_res, 5574.97874220185, pass );
    compare_face( "Specific Enthalpy", 2, 2, h_res, -10293.821727646, pass );
    compare_face( "Specific Enthalpy", 2, 3, h_res, -13969.1925327425, pass );
    compare_face( "Specific Enthalpy", 2, 4, h_res, -2665.21836142746, pass );
    compare_face( "Specific Enthalpy", 2, 5, h_res, -1160.27165031973, pass );
    compare_face( "Specific Enthalpy", 2, 6, h_res, -13504.2526063368, pass );
    compare_face( "Specific Enthalpy", 2, 7, h_res, -14431.0161424561, pass );
    compare_face( "Specific Enthalpy", 2, 8, h_res, 528.407674478783, pass );
    compare_face( "Specific Enthalpy", 2, 9, h_res, 6492.29252056397, pass );
    compare_face( "Specific Enthalpy", 3, 0, h_res, -263219.554417629, pass );
    compare_face( "Specific Enthalpy", 3, 1, h_res, 3419.99492580448, pass );
    compare_face( "Specific Enthalpy", 3, 2, h_res, -7807.09404303347, pass );
    compare_face( "Specific Enthalpy", 3, 3, h_res, -11664.6198542485, pass );
    compare_face( "Specific Enthalpy", 3, 4, h_res, -5311.66571794595, pass );
    compare_face( "Specific Enthalpy", 3, 5, h_res, -4159.67836696387, pass );
    compare_face( "Specific Enthalpy", 3, 6, h_res, -11577.0464493042, pass );
    compare_face( "Specific Enthalpy", 3, 7, h_res, -11931.3704349631, pass );
    compare_face( "Specific Enthalpy", 3, 8, h_res, -1551.82084135415, pass );
    compare_face( "Specific Enthalpy", 3, 9, h_res, 3950.62212483903, pass );
    compare_face( "Specific Enthalpy", 4, 0, h_res, -263018.125366054, pass );
    compare_face( "Specific Enthalpy", 4, 1, h_res, -3992.87443114835, pass );
    compare_face( "Specific Enthalpy", 4, 2, h_res, -4172.25831839824, pass );
    compare_face( "Specific Enthalpy", 4, 3, h_res, -6634.1278977701, pass );
    compare_face( "Specific Enthalpy", 4, 4, h_res, -12135.2261524175, pass );
    compare_face( "Specific Enthalpy", 4, 5, h_res, -13525.613922812, pass );
    compare_face( "Specific Enthalpy", 4, 6, h_res, -9357.2210087679, pass );
    compare_face( "Specific Enthalpy", 4, 7, h_res, -6516.19329111857, pass );
    compare_face( "Specific Enthalpy", 4, 8, h_res, -6945.1171800448, pass );
    compare_face( "Specific Enthalpy", 4, 9, h_res, -4228.79819519211, pass );
    compare_face( "Specific Enthalpy", 5, 0, h_res, -262872.742334926, pass );
    compare_face( "Specific Enthalpy", 5, 1, h_res, -8882.02213628856, pass );
    compare_face( "Specific Enthalpy", 5, 2, h_res, -30.498708470193, pass );
    compare_face( "Specific Enthalpy", 5, 3, h_res, -694.909653946385, pass );
    compare_face( "Specific Enthalpy", 5, 4, h_res, -14410.6814779411, pass );
    compare_face( "Specific Enthalpy", 5, 5, h_res, -17271.8703391642, pass );
    compare_face( "Specific Enthalpy", 5, 6, h_res, -4872.21864080329, pass );
    compare_face( "Specific Enthalpy", 5, 7, h_res, -291.065461246878, pass );
    compare_face( "Specific Enthalpy", 5, 8, h_res, -9549.15625013573, pass );
    compare_face( "Specific Enthalpy", 5, 9, h_res, -9694.16629333427, pass );
    compare_face( "Specific Enthalpy", 6, 0, h_res, -262917.06977172, pass );
    compare_face( "Specific Enthalpy", 6, 1, h_res, -6855.81596407678, pass );
    compare_face( "Specific Enthalpy", 6, 2, h_res, 47.6219931502584, pass );
    compare_face( "Specific Enthalpy", 6, 3, h_res, -1049.12960607768, pass );
    compare_face( "Specific Enthalpy", 6, 4, h_res, -12307.7235533979, pass );
    compare_face( "Specific Enthalpy", 6, 5, h_res, -14369.9053251914, pass );
    compare_face( "Specific Enthalpy", 6, 6, h_res, -4236.15340133692, pass );
    compare_face( "Specific Enthalpy", 6, 7, h_res, -707.092755988836, pass );
    compare_face( "Specific Enthalpy", 6, 8, h_res, -8007.77246186547, pass );
    compare_face( "Specific Enthalpy", 6, 9, h_res, -7542.49314215733, pass );
    compare_face( "Specific Enthalpy", 7, 0, h_res, -263110.353252648, pass );
    compare_face( "Specific Enthalpy", 7, 1, h_res, -1026.77413904948, pass );
    compare_face( "Specific Enthalpy", 7, 2, h_res, -6002.60654338878, pass );
    compare_face( "Specific Enthalpy", 7, 3, h_res, -8788.44723563834, pass );
    compare_face( "Specific Enthalpy", 7, 4, h_res, -8599.4798007274, pass );
    compare_face( "Specific Enthalpy", 7, 5, h_res, -8845.53863260117, pass );
    compare_face( "Specific Enthalpy", 7, 6, h_res, -10106.6659794468, pass );
    compare_face( "Specific Enthalpy", 7, 7, h_res, -8852.44646337933, pass );
    compare_face( "Specific Enthalpy", 7, 8, h_res, -4498.18374408486, pass );
    compare_face( "Specific Enthalpy", 7, 9, h_res, -898.803459953632, pass );
    compare_face( "Specific Enthalpy", 8, 0, h_res, -263274.888863299, pass );
    compare_face( "Specific Enthalpy", 8, 1, h_res, 4217.38499620078, pass );
    compare_face( "Specific Enthalpy", 8, 2, h_res, -10479.7684028723, pass );
    compare_face( "Specific Enthalpy", 8, 3, h_res, -14090.2351042221, pass );
    compare_face( "Specific Enthalpy", 8, 4, h_res, -3863.93823014883, pass );
    compare_face( "Specific Enthalpy", 8, 5, h_res, -2444.65376888404, pass );
    compare_face( "Specific Enthalpy", 8, 6, h_res, -13678.021625005, pass );
    compare_face( "Specific Enthalpy", 8, 7, h_res, -14508.3063808984, pass );
    compare_face( "Specific Enthalpy", 8, 8, h_res, -681.060722901632, pass );
    compare_face( "Specific Enthalpy", 8, 9, h_res, 5048.11277610616, pass );
    compare_face( "Pressure", 0, 0, p_res, -4.37809754537724, pass );
    compare_face( "Pressure", 0, 1, p_res, 10.6136179100588, pass );
    compare_face( "Pressure", 0, 2, p_res, 11.5787095565292, pass );
    compare_face( "Pressure", 0, 3, p_res, -2.80654141589653, pass );
    compare_face( "Pressure", 0, 4, p_res, -4.38245970487631, pass );
    compare_face( "Pressure", 0, 5, p_res, 9.93563461747059, pass );
    compare_face( "Pressure", 0, 6, p_res, 12.1199573028697, pass );
    compare_face( "Pressure", 0, 7, p_res, -2.1055481921653, pass );
    compare_face( "Pressure", 0, 8, p_res, -5.46039428371658, pass );
    compare_face( "Pressure", 0, 9, p_res, -81864.8092918955, pass );
    compare_face( "Pressure", 1, 0, p_res, 8.97855298145408, pass );
    compare_face( "Pressure", 1, 1, p_res, -1.56190346468237, pass );
    compare_face( "Pressure", 1, 2, p_res, -2.3042832934749, pass );
    compare_face( "Pressure", 1, 3, p_res, 8.77547650798638, pass );
    compare_face( "Pressure", 1, 4, p_res, 9.98864032840842, pass );
    compare_face( "Pressure", 1, 5, p_res, -1.03972423870057, pass );
    compare_face( "Pressure", 1, 6, p_res, -2.72121500329431, pass );
    compare_face( "Pressure", 1, 7, p_res, 8.23559777165683, pass );
    compare_face( "Pressure", 1, 8, p_res, 9.81221614436193, pass );
    compare_face( "Pressure", 1, 9, p_res, 63053.185302658, pass );
    compare_face( "Pressure", 2, 0, p_res, 16.9914810764723, pass );
    compare_face( "Pressure", 2, 1, p_res, -8.86826312629966, pass );
    compare_face( "Pressure", 2, 2, p_res, -10.6339184097802, pass );
    compare_face( "Pressure", 2, 3, p_res, 15.7242163704584, pass );
    compare_face( "Pressure", 2, 4, p_res, 18.6102470535056, pass );
    compare_face( "Pressure", 2, 5, p_res, -7.62600975889775, pass );
    compare_face( "Pressure", 2, 6, p_res, -11.6258125764742, pass );
    compare_face( "Pressure", 2, 7, p_res, 14.4398941108048, pass );
    compare_face( "Pressure", 2, 8, p_res, 18.9747207459307, pass );
    compare_face( "Pressure", 2, 9, p_res, 150000.37211461, pass );
    compare_face( "Pressure", 3, 0, p_res, 12.2966261165189, pass );
    compare_face( "Pressure", 3, 1, p_res, -4.58323576660972, pass );
    compare_face( "Pressure", 3, 2, p_res, -5.75181674476704, pass );
    compare_face( "Pressure", 3, 3, p_res, 11.6503690284676, pass );
    compare_face( "Pressure", 3, 4, p_res, 13.5584783690628, pass );
    compare_face( "Pressure", 3, 5, p_res, -3.76296698723317, pass );
    compare_face( "Pressure", 3, 6, p_res, -6.40655255658872, pass );
    compare_face( "Pressure", 3, 7, p_res, 10.8023166869431, pass );
    compare_face( "Pressure", 3, 8, p_res, 13.6057626394502, pass );
    compare_face( "Pressure", 3, 9, p_res, 99037.9085665476, pass );
    compare_face( "Pressure", 4, 0, p_res, -0.797643218074481, pass );
    compare_face( "Pressure", 4, 1, p_res, 7.34313469303979, pass );
    compare_face( "Pressure", 4, 2, p_res, 7.85019207330265, pass );
    compare_face( "Pressure", 4, 3, p_res, 0.297993433449162, pass );
    compare_face( "Pressure", 4, 4, p_res, -0.529788121205712, pass );
    compare_face( "Pressure", 4, 5, p_res, 6.98717521307002, pass );
    compare_face( "Pressure", 4, 6, p_res, 8.13433033370699, pass );
    compare_face( "Pressure", 4, 7, p_res, 0.666026894223469, pass );
    compare_face( "Pressure", 4, 8, p_res, -1.36580511066195, pass );
    compare_face( "Pressure", 4, 9, p_res, -42979.5513808839, pass );
    compare_face( "Pressure", 5, 0, p_res, -10.2484056599371, pass );
    compare_face( "Pressure", 5, 1, p_res, 15.9510132281231, pass );
    compare_face( "Pressure", 5, 2, p_res, 17.6671799563894, pass );
    compare_face( "Pressure", 5, 3, p_res, -7.89667646770721, pass );
    compare_face( "Pressure", 5, 4, p_res, -10.6980346740424, pass );
    compare_face( "Pressure", 5, 5, p_res, 14.7461556668325, pass );
    compare_face( "Pressure", 5, 6, p_res, 18.6289631504753, pass );
    compare_face( "Pressure", 5, 7, p_res, -6.65090403089089, pass );
    compare_face( "Pressure", 5, 8, p_res, -12.1716687785823, pass );
    compare_face( "Pressure", 5, 9, p_res, -145481.809999086, pass );
    compare_face( "Pressure", 6, 0, p_res, -7.36488372805792, pass );
    compare_face( "Pressure", 6, 1, p_res, 13.3290014621954, pass );
    compare_face( "Pressure", 6, 2, p_res, 14.6746611294526, pass );
    compare_face( "Pressure", 6, 3, p_res, -5.39798463564638, pass );
    compare_face( "Pressure", 6, 4, p_res, -7.59585203934206, pass );
    compare_face( "Pressure", 6, 5, p_res, 12.3830211563596, pass );
    compare_face( "Pressure", 6, 6, p_res, 15.429937121664, pass );
    compare_face( "Pressure", 6, 7, p_res, -4.41988455155307, pass );
    compare_face( "Pressure", 6, 8, p_res, -8.87517858424614, pass );
    compare_face( "Pressure", 6, 9, p_res, -114228.76342787, pass );
    compare_face( "Pressure", 7, 0, p_res, 5.1947528271687, pass );
    compare_face( "Pressure", 7, 1, p_res, 1.879322203949, pass );
    compare_face( "Pressure", 7, 2, p_res, 1.61950916990281, pass );
    compare_face( "Pressure", 7, 3, p_res, 5.493359810221, pass );
    compare_face( "Pressure", 7, 4, p_res, 5.91775625306381, pass );
    compare_face( "Pressure", 7, 5, p_res, 2.06190168036136, pass );
    compare_face( "Pressure", 7, 6, p_res, 1.47374876649865, pass );
    compare_face( "Pressure", 7, 7, p_res, 5.30458926447365, pass );
    compare_face( "Pressure", 7, 8, p_res, 5.48620978945642, pass );
    compare_face( "Pressure", 7, 9, p_res, 22045.6814459972, pass );
    compare_face( "Pressure", 8, 0, p_res, 15.8863064922488, pass );
    compare_face( "Pressure", 8, 1, p_res, -7.86797901764816, pass );
    compare_face( "Pressure", 8, 2, p_res, -9.49349397356854, pass );
    compare_face( "Pressure", 8, 3, p_res, 14.7648148853103, pass );
    compare_face( "Pressure", 8, 4, p_res, 17.4213966107677, pass );
    compare_face( "Pressure", 8, 5, p_res, -6.7246724402292, pass );
    compare_face( "Pressure", 8, 6, p_res, -10.406343450275, pass );
    compare_face( "Pressure", 8, 7, p_res, 13.5827871652755, pass );
    compare_face( "Pressure", 8, 8, p_res, 17.7115093622372, pass );
    compare_face( "Pressure", 8, 9, p_res, 138051.428467283, pass );

    if ( pass )
        ut->passes( exeName + ": known value test" );
    else
        ut->failure( exeName + ": known residual test" );
}

int testSubchannelFourEqNonlinearOperator( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );

    AMP::UnitTest ut;

    // create list of input files
    std::string files[] = { "testSubchannelFourEqNonlinearOperator" };

    // execute unit test for each input file
    for ( auto &file : files )
        Test( &ut, file );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
