#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/discretization/structuredFaceDOFManager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"
#include "AMP/operators/subchannel/SubchannelHelpers.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include <iostream>
#include <string>


static void testSubchannelHelpers( AMP::UnitTest *ut, const std::string &input_file )
{

    const double pi = 3.1415926535897932;

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto database   = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( database );
    meshParams->setComm( globalComm );

    // Get the meshes and clad properties
    auto manager        = AMP::Mesh::MeshFactory::create( meshParams );
    auto cladMesh       = manager->Subset( "clad" );
    auto subchannelMesh = manager->Subset( "subchannel" );
    std::vector<double> clad_x, clad_y, clad_d;
    AMP::Operator::Subchannel::getCladProperties( globalComm, cladMesh, clad_x, clad_y, clad_d );
    // std::vector<double> box = cladMesh->getBoundingBox();
    bool pass = true;
    for ( size_t i = 0; i < clad_x.size(); i++ ) {
        if ( !AMP::Utilities::approx_equal( clad_d[i], clad_d[0] ) )
            pass = false;
        if ( !AMP::Utilities::approx_equal( clad_d[0], 2 * 0.004705 ) )
            pass = false;
    }
    if ( pass )
        ut->passes( "Get clad properties passes basic sanity check" );
    else
        ut->failure( "Get clad properties passes basic sanity check" );

    // Get the subchannel properties
    std::vector<double> x, y, z, area, subchannel_diam, rod_diameter, channel_fraction;
    AMP::Mesh::StructuredMeshHelper::getXYZCoordinates( subchannelMesh, x, y, z );
    AMP::Operator::Subchannel::getSubchannelProperties( subchannelMesh,
                                                        clad_x,
                                                        clad_y,
                                                        clad_d,
                                                        x,
                                                        y,
                                                        area,
                                                        subchannel_diam,
                                                        rod_diameter,
                                                        channel_fraction );
    size_t N_subchannels = ( x.size() - 1 ) * ( y.size() - 1 );
    AMP_ASSERT( area.size() == N_subchannels );
    pass = true;
    for ( size_t i = 0; i < N_subchannels; i++ ) {
        if ( subchannel_diam[i] == 0 )
            pass = false;
        if ( channel_fraction[i] > 1.0 )
            pass = false;
        if ( !AMP::Utilities::approx_equal( rod_diameter[i], clad_d[0] ) )
            pass = false;
    }
    if ( pass )
        ut->passes( "Get subchannel properties passes basic sanity check" );
    else
        ut->failure( "Get subchannel properties passes basic sanity check" );

    // First test getHeatFluxGeneration
    size_t N = z.size() - 1;
    std::vector<double> dz( N );
    for ( size_t i = 0; i < N; i++ )
        dz[i] = z[i + 1] - z[i];
    double diam  = 0.00822;
    double Q_tot = 66.5e3;
    auto flux1   = AMP::Operator::Subchannel::getHeatFluxGeneration( "Sinusoidal", z, diam, Q_tot );
    auto flux2   = AMP::Operator::Subchannel::getHeatFluxGeneration( "Flat", z, diam, Q_tot );
    double Q_tot1  = 0.0;
    double Q_tot2  = 0.0;
    bool pass_sine = true;
    bool pass_flat = true;
    for ( size_t i = 0; i < N; i++ ) {
        Q_tot1 += flux1[i] * pi * diam * dz[i];
        Q_tot2 += flux2[i] * pi * diam * dz[i];
        if ( flux2[i] != flux2[0] )
            pass_flat = false;
    }
    if ( !AMP::Utilities::approx_equal( Q_tot, Q_tot1 ) )
        pass_sine = false;
    if ( !AMP::Utilities::approx_equal( Q_tot, Q_tot2 ) )
        pass_flat = false;
    if ( pass_sine )
        ut->passes( "Sinusoidal shape gives correct flux" );
    else
        ut->failure( "Sinusoidal shape gives correct flux" );
    if ( pass_flat )
        ut->passes( "Flat shape gives correct flux" );
    else
        ut->failure( "Flat shape gives correct flux" );

    // Test getHeatFluxClad
    auto subchannel_db = input_db->getDatabase( "SubchannelPhysicsModel" );
    auto params = std::make_shared<AMP::Operator::ElementPhysicsModelParameters>( subchannel_db );
    auto subchannelPhysicsModel = std::make_shared<AMP::Operator::SubchannelPhysicsModel>( params );
    auto reynolds = subchannel_db->getDatabase( "Defaults" )->getScalar<double>( "reynolds" );
    auto prandtl  = subchannel_db->getDatabase( "Defaults" )->getScalar<double>( "prandtl" );
    AMP::LinearAlgebra::Vector::shared_ptr flowVec, cladTemp;
    if ( subchannelMesh ) {
        int DOFsPerFace[3] = { 0, 0, 2 };
        auto flowDOF       = std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
            subchannelMesh, DOFsPerFace, 1 );
        DOFsPerFace[2] = 1;
        auto cladDOF   = std::make_shared<AMP::Discretization::structuredFaceDOFManager>(
            subchannelMesh, DOFsPerFace, 1 );
        auto flowVariable    = std::make_shared<AMP::LinearAlgebra::Variable>( "Flow" );
        auto thermalVariable = std::make_shared<AMP::LinearAlgebra::Variable>( "Temperature" );
        flowVec              = AMP::LinearAlgebra::createVector( flowDOF, flowVariable );
        cladTemp             = AMP::LinearAlgebra::createVector( cladDOF, thermalVariable );
        double clad_temp     = 632;
        double flow_temp     = 570;
        double pressure      = 1.6e7;
        std::map<std::string, std::shared_ptr<std::vector<double>>> enthalpyArgMap;
        enthalpyArgMap.insert( std::make_pair(
            "temperature", std::make_shared<std::vector<double>>( 1, flow_temp ) ) );
        enthalpyArgMap.insert(
            std::make_pair( "pressure", std::make_shared<std::vector<double>>( 1, pressure ) ) );
        std::vector<double> enthalpyResult( 1 );
        subchannelPhysicsModel->getProperty( "Enthalpy", enthalpyResult, enthalpyArgMap );
        auto subchannelEnthalpy = flowVec->select( AMP::LinearAlgebra::VS_Stride( 0, 2 ) );
        auto subchannelPressure = flowVec->select( AMP::LinearAlgebra::VS_Stride( 1, 2 ) );
        subchannelEnthalpy->setToScalar( AMP::Operator::Subchannel::scaleEnthalpy *
                                         enthalpyResult[0] );
        subchannelPressure->setToScalar( AMP::Operator::Subchannel::scalePressure * pressure );
        cladTemp->setToScalar( clad_temp );
    }
    pass = true;
    for ( size_t sc = 0; sc < N_subchannels; sc++ ) {
        AMP_INSIST( N_subchannels == 1, "This is currently only programmed for one subchannel" );
        auto it = AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( subchannelMesh, 0 );
        std::vector<AMP::Mesh::MeshElementID> face_ids( it.size() );
        std::vector<double> z_face( it.size() );
        for ( size_t j = 0; j < it.size(); j++ ) {
            face_ids[j] = it->globalID();
            z_face[j]   = it->centroid()[2];
            ++it;
        }
        double Q_tot3 = 0.0;
        std::vector<double> flux3 =
            AMP::Operator::Subchannel::getHeatFluxClad( z_face,
                                                        face_ids,
                                                        subchannel_diam[sc],
                                                        reynolds,
                                                        prandtl,
                                                        channel_fraction[sc],
                                                        subchannelPhysicsModel,
                                                        flowVec,
                                                        cladTemp );
        AMP_ASSERT( flux3.size() == N );
        for ( size_t i = 0; i < N; i++ )
            Q_tot3 += flux3[i] * pi * diam * dz[i];
        if ( fabs( Q_tot3 - Q_tot ) / Q_tot > 0.2 )
            pass = false;
    }
    if ( pass )
        ut->passes( "Clad shape gives ~ correct flux" );
    else
        ut->failure( "Clad shape gives ~ correct flux" );
}


int testSubchannelHelpers( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );

    AMP::UnitTest ut;

    std::string input = "input_testSubchannelUtilities";
    testSubchannelHelpers( &ut, input );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
