#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/createLibmeshElements.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/map/AsyncMapColumnOperator.h"
#include "AMP/operators/map/libmesh/ScalarN2GZAxisMap.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorBuilder.h"

// Libmesh files
DISABLE_WARNINGS
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/face_quad4.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/node.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS


static void setBoundary( int id,
                         AMP::LinearAlgebra::Vector::shared_ptr &v1,
                         std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    if ( mesh.get() == nullptr )
        return;

    auto d1 = v1->getDOFManager();

    auto curBnd = mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, id, 0 );
    auto endBnd = curBnd.end();

    std::vector<size_t> ids;
    while ( curBnd != endBnd ) {
        d1->getDOFs( curBnd->globalID(), ids );
        auto x = curBnd->coord();
        v1->setValuesByGlobalID( ids.size(), &ids[0], &x[2] );
        ++curBnd;
    }
}

static void setGpBoundary( int id,
                           AMP::LinearAlgebra::Vector::shared_ptr &v1,
                           std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    if ( mesh.get() == nullptr )
        return;

    auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
    auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );

    auto d_feType = std::make_shared<libMesh::FEType>( feTypeOrder, feFamily );
    std::shared_ptr<libMesh::FEBase> d_fe(
        ( libMesh::FEBase::build( 2, ( *d_feType ) ) ).release() );
    d_fe->get_xyz();

    auto qruleOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "SECOND" );
    std::shared_ptr<libMesh::QBase> d_qrule(
        ( libMesh::QBase::build( "QGAUSS", 2, qruleOrder ) ).release() );

    d_fe->attach_quadrature_rule( d_qrule.get() );

    AMP::Discretization::createLibmeshElements libmeshElements;

    auto d1 = v1->getDOFManager();

    auto curBnd = mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, id, 0 );
    auto endBnd = curBnd.end();

    libmeshElements.reinit( curBnd );

    std::vector<size_t> ids;
    while ( curBnd != endBnd ) {
        d_fe->reinit( libmeshElements.getElement( curBnd->globalID() ) );
        auto coordinates = d_fe->get_xyz();

        d1->getDOFs( curBnd->globalID(), ids );
        for ( unsigned int qp = 0; qp < ids.size(); qp++ ) {
            double pos = coordinates[qp]( 2 );
            v1->setValuesByGlobalID( 1, &ids[qp], &pos );
        }
        ++curBnd;
    }
}

static void runTest( const std::string &fname, AMP::UnitTest *ut )
{
    // Read the input file
    auto input_db = AMP::Database::parseInputFile( fname );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto mesh_db = input_db->getDatabase( "Mesh" );
    auto params  = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Get the database for the node to node maps
    auto map_db = input_db->getDatabase( "MeshToMeshMaps" );

    // Create a simple DOFManager and the vectors
    int DOFsPerObject   = map_db->getScalar<int>( "DOFsPerObject" );
    std::string varName = map_db->getString( "VariableName" );
    auto Variable       = std::make_shared<AMP::LinearAlgebra::Variable>( varName );
    auto DOFparams      = std::make_shared<AMP::Discretization::DOFManagerParameters>( mesh );
    auto DOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );
    auto GpDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Face, 1, DOFsPerObject, true );

    // Test the creation/destruction of ScalarN2GZAxisMap (no apply call)
    try {
        auto gapmaps =
            AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::ScalarN2GZAxisMap>(
                mesh, map_db );
        gapmaps.reset();
        ut->passes( "Created / Destroyed ScalarN2GZAxisMap" );
    } catch ( ... ) {
        ut->failure( "Created / Destroyed ScalarN2GZAxisMap" );
    }

    // Perform a complete test of ScalarN2GZAxisMap
    auto gapmaps = AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::ScalarN2GZAxisMap>(
        mesh, map_db );

    // Create the vectors
    AMP::LinearAlgebra::Vector::shared_ptr dummy;
    auto v1 = AMP::LinearAlgebra::createVector( DOFs, Variable );
    auto v2 = AMP::LinearAlgebra::createVector( GpDofMap, Variable );
    auto v3 = AMP::LinearAlgebra::createVector( GpDofMap, Variable );
    gapmaps->setVector( v2 );

    // Initialize the vectors
    v1->setToScalar( 0.0 );
    v2->setToScalar( 0.0 );
    size_t N_maps = (size_t) map_db->getScalar<int>( "N_maps" );
    auto mesh1    = map_db->getVector<std::string>( "Mesh1" );
    auto mesh2    = map_db->getVector<std::string>( "Mesh2" );
    auto surface1 = map_db->getVector<int>( "Surface1" );
    auto surface2 = map_db->getVector<int>( "Surface2" );
    AMP_ASSERT( mesh1.size() == N_maps || mesh1.size() == 1 );
    AMP_ASSERT( mesh2.size() == N_maps || mesh2.size() == 1 );
    AMP_ASSERT( surface1.size() == N_maps || surface1.size() == 1 );
    AMP_ASSERT( surface2.size() == N_maps || surface2.size() == 1 );
    for ( size_t i = 0; i < N_maps; i++ ) {
        std::string meshname1, meshname2;
        if ( mesh1.size() == N_maps ) {
            meshname1 = mesh1[i];
            meshname2 = mesh2[i];
        } else {
            meshname1 = mesh1[0];
            meshname2 = mesh2[0];
        }
        int surface_id1, surface_id2;
        if ( surface1.size() == N_maps ) {
            surface_id1 = surface1[i];
            surface_id2 = surface2[i];
        } else {
            surface_id1 = surface1[0];
            surface_id2 = surface2[0];
        }
        auto curMesh = mesh->Subset( meshname1 );
        setBoundary( surface_id1, v1, curMesh );
        setGpBoundary( surface_id1, v3, curMesh );
        curMesh = mesh->Subset( meshname2 );
        setBoundary( surface_id2, v1, curMesh );
        setGpBoundary( surface_id2, v3, curMesh );
    }

    // Apply the maps
    globalComm.barrier();
    gapmaps->apply( v1, v2 );
    v2->subtract( *v2, *v3 );
    if ( v2->maxNorm() < 1.e-12 )
        ut->passes( "Node to node map test" );
    else
        ut->failure( "Node to node map test" );

    gapmaps.reset();
}


int testScalarN2GZAxisMap( int argc, char *argv[] )
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup( argc, argv, startup_properties );
    AMP::UnitTest ut;

    runTest( "inputScalarN2GZAxisMap-1", &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
