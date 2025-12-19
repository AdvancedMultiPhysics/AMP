#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

// Libmesh files
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
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

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );


    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    int surfaceId         = input_db->getScalar<int>( "SurfaceId" );
    bool setConstantValue = input_db->getScalar<bool>( "SetConstantValue" );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Create a nodal scalar vector
    auto var            = std::make_shared<AMP::LinearAlgebra::Variable>( "myVar" );
    auto nodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto vec = AMP::LinearAlgebra::createVector( nodalScalarDOF, var, true );
    vec->zero();

    auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
    auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );

    auto bnd = mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, surfaceId, 0 );
    std::cout << "Number of surface elements: " << bnd.size() << std::endl;
    AMP::Mesh::MeshIterator end_bnd = bnd.end();

    bool volume_passes = true;
    std::vector<size_t> dofs;
    while ( bnd != end_bnd ) {

        auto nodes = bnd->getElements( AMP::Mesh::GeomType::Vertex );
        std::vector<size_t> bndGlobalIds;
        for ( auto &node : nodes ) {
            nodalScalarDOF->getDOFs( node->globalID(), dofs );
            for ( auto &dof : dofs )
                bndGlobalIds.push_back( dof );
        }

        // Some basic checks
        AMP_ASSERT( bndGlobalIds.size() == 4 );
        // AMP_ASSERT((bnd->getElem()).default_order() == feTypeOrder);
        AMP_ASSERT( bnd->elementType() == AMP::Mesh::GeomType::Face );

        // Create the libmesh element
        // Note: This must be done inside the loop because libmesh's reinit function doesn't seem to
        // work properly
        std::shared_ptr<libMesh::FEType> feType( new libMesh::FEType( feTypeOrder, feFamily ) );
        std::shared_ptr<libMesh::FEBase> fe(
            ( libMesh::FEBase::build( 2, ( *feType ) ) ).release() );
        const std::vector<std::vector<libMesh::Real>> &phi = fe->get_phi();
        const std::vector<libMesh::Real> &djxw             = fe->get_JxW();
        auto qruleType = libMesh::Utility::string_to_enum<libMeshEnums::QuadratureType>( "QGAUSS" );
        libMeshEnums::Order qruleOrder = feType->default_quadrature_order();
        std::shared_ptr<libMesh::QBase> qrule(
            ( libMesh::QBase::build( qruleType, 2, qruleOrder ) ).release() );
        fe->attach_quadrature_rule( qrule.get() );
        libMesh::Elem *currElemPtr = new libMesh::Quad4;
        for ( size_t i = 0; i < nodes.size(); i++ ) {
            auto pt                    = nodes[i]->coord();
            currElemPtr->set_node( i ) = new libMesh::Node( pt[0], pt[1], pt[2], i );
        }
        fe->reinit( currElemPtr );

        // Check the volume
        double vol1 = 0.0;
        for ( unsigned int qp = 0; qp < qrule->n_points(); qp++ )
            vol1 += djxw[qp];
        double vol2 = bnd->volume();
        if ( fabs( vol1 - vol2 ) > ( 1.0e-8 * vol2 ) ) {
            std::cout << "GeomType::Cell 1 = " << std::setprecision( 15 ) << vol1 << std::endl;
            std::cout << "GeomType::Cell 2 = " << std::setprecision( 15 ) << vol2 << std::endl
                      << std::endl;
            volume_passes = false;
        }

        // Fill the surface vector
        if ( setConstantValue ) {
            std::vector<double> vals( bndGlobalIds.size(), 100.0 );
            vec->addValuesByGlobalID( bndGlobalIds.size(), &( bndGlobalIds[0] ), &( vals[0] ) );
        } else {
            std::vector<double> vals( bndGlobalIds.size(), 0.0 );
            for ( unsigned int i = 0; i < bndGlobalIds.size(); i++ ) {
                for ( unsigned int qp = 0; qp < qrule->n_points(); qp++ ) {
                    AMP_ASSERT( djxw[qp] >= 0.0 );
                    AMP_ASSERT( phi[i][qp] >= 0.0 );
                    vals[i] += ( djxw[qp] * phi[i][qp] * 100.0 );
                } // end qp
            }     // end i
            vec->addValuesByGlobalID( bndGlobalIds.size(), &( bndGlobalIds[0] ), &( vals[0] ) );
        }

        // Destroy the libmesh element
        for ( size_t i = 0; i < nodes.size(); i++ ) {
            delete ( currElemPtr->node_ptr( i ) );
            currElemPtr->set_node( i ) = nullptr;
        } // end for j
        delete currElemPtr;
        currElemPtr = nullptr;

        ++bnd;
    } // end for bnd

    if ( volume_passes == true )
        ut->passes( "GeomType::Cell passes" );
    else
        ut->failure( "GeomType::Cell passes" );

    vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    double l2Norm = static_cast<double>( vec->L2Norm() );
    std::cout << "size = " << vec->getGlobalSize() << std::endl;
    std::cout << "L2 Norm = " << std::setprecision( 15 ) << l2Norm << std::endl;

    if ( AMP::Utilities::approx_equal( l2Norm, 0.00106829009941852 ) )
        ut->passes( "L2 Norm has expected value" );
    else
        ut->failure( "L2 Norm has expected value" );
}

int testSurfaceIteratorBug( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName = "testSurfaceIteratorBug";

    myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
