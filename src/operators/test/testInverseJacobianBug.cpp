#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/libmesh/libmeshMesh.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"
#include "libmesh/boundary_info.h"
#include "libmesh/cell_hex8.h"
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/libmesh.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_communication.h"
#include "libmesh/mesh_generation.h"
#include "libmesh/node.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string log_file = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    AMP_INSIST( globalComm.getSize() == 1, "testInverseJacobianBug is a serial only test" );

    libMesh::Parallel::Communicator comm( globalComm.getCommunicator() );
    auto libmesh = std::make_shared<libMesh::Mesh>( comm, 3 );
    libMesh::MeshTools::Generation::build_cube(
        *libmesh, 1, 1, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, libMesh::HEX8, false );

    libMesh::Elem *elemPtr = libmesh->elem_ptr( 0 );
    elemPtr->point( 4 )( 0 ) -= 0.4;
    elemPtr->point( 5 )( 0 ) -= 0.4;
    elemPtr->point( 6 )( 0 ) -= 0.4;
    elemPtr->point( 7 )( 0 ) -= 0.4;

    auto mesh = std::make_shared<AMP::Mesh::libmeshMesh>( libmesh, "TestMesh" );
    auto Cell = AMP::Mesh::GeomType::Cell;
    AMP_ASSERT( mesh->numGlobalElements( Cell ) == 1 );
    auto nodes = mesh->getIterator( Cell, 0 )->getElements( AMP::Mesh::GeomType::Vertex );
    AMP_ASSERT( nodes.size() == 8 );

    auto myVar   = std::make_shared<AMP::LinearAlgebra::Variable>( "myVar" );
    auto dof_map = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto T = AMP::LinearAlgebra::createVector( dof_map, myVar, true );

    FILE *fp = fopen( "InverseJacobian.txt", "w" );
    for ( int i = 0; i < 8; i++ ) {
        auto pt = nodes[i]->coord();
        fprintf( fp, "nd = %d, x = %.15f, y = %.15f, z = %.15f \n", i, pt[0], pt[1], pt[2] );
    }

    auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
    auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );

    auto feType = std::make_shared<libMesh::FEType>( feTypeOrder, feFamily );
    auto fe     = libMesh::FEBase::build( 3, *feType );

    const auto &dphidxi   = fe->get_dphidxi();
    const auto &dphideta  = fe->get_dphideta();
    const auto &dphidzeta = fe->get_dphidzeta();
    const auto &dphidx    = fe->get_dphidx();
    const auto &dphidy    = fe->get_dphidy();
    const auto &dphidz    = fe->get_dphidz();

    const auto &dxyzdxi   = fe->get_dxyzdxi();
    const auto &dxyzdeta  = fe->get_dxyzdeta();
    const auto &dxyzdzeta = fe->get_dxyzdzeta();

    auto qruleType  = libMesh::Utility::string_to_enum<libMeshEnums::QuadratureType>( "QGAUSS" );
    auto qruleOrder = feType->default_quadrature_order();
    auto qrule      = libMesh::QBase::build( qruleType, 3, qruleOrder );

    fe->attach_quadrature_rule( qrule.get() );

    std::vector<size_t> d_dofIndices;
    std::vector<AMP::Mesh::MeshElementID> globalIDs( 8 );
    for ( int j = 0; j < 8; j++ )
        globalIDs[j] = nodes[j]->globalID();
    dof_map->getDOFs( globalIDs, d_dofIndices );

    libMesh::Hex8 currElem;
    libMesh::Node *libmeshNodes[8];
    for ( int j = 0; j < 8; j++ ) {
        auto pt                = nodes[j]->coord();
        libmeshNodes[j]        = new libMesh::Node( pt[0], pt[1], pt[2], j );
        currElem.set_node( j ) = libmeshNodes[j];
    }

    fe->reinit( &currElem );

    for ( unsigned int i = 0; i < d_dofIndices.size(); i++ ) {
        const double val = 300 * ( i + 1 );
        T->setValuesByGlobalID( 1, &d_dofIndices[i], &val );
    }

    fprintf( fp,
             " dx/dxi = %.15f, dydxi = %.15f, dzdxi = %.15f \n",
             dxyzdxi[0]( 0 ),
             dxyzdxi[0]( 1 ),
             dxyzdxi[0]( 2 ) );
    fprintf( fp,
             " dx/deta = %.15f, dydeta = %.15f, dzdeta = %.15f \n",
             dxyzdeta[0]( 0 ),
             dxyzdeta[0]( 1 ),
             dxyzdeta[0]( 2 ) );
    fprintf( fp,
             " dx/dzeta = %.15f, dydzeta = %.15f, dzdzeta = %.15f \n",
             dxyzdzeta[0]( 0 ),
             dxyzdzeta[0]( 1 ),
             dxyzdzeta[0]( 2 ) );

    std::vector<libMesh::Real> Jinv1( 3 );
    std::vector<libMesh::Real> Jinv2( 3 );
    std::vector<libMesh::Real> Jinv3( 3 );
    Jinv1[0] = 2.;
    Jinv1[1] = 0.;
    Jinv1[2] = 0.;

    Jinv2[0] = 0;
    Jinv2[1] = 2;
    Jinv2[2] = 0;

    Jinv3[0] = 0.8;
    Jinv3[1] = 0.;
    Jinv3[2] = 2.;

    libMesh::Real dTdxi = 0, dTdeta = 0, dTdzeta = 0, dTdx = 0;
    libMesh::Real dTdy = 0, dTdz = 0, lib_dTdx = 0, lib_dTdy = 0, lib_dTdz = 0;

    for ( unsigned int i = 0; i < d_dofIndices.size(); i++ ) {
        dTdxi += dphidxi[i][0] * T->getValueByGlobalID( d_dofIndices[i] );
        dTdeta += dphideta[i][0] * T->getValueByGlobalID( d_dofIndices[i] );
        dTdzeta += dphidzeta[i][0] * T->getValueByGlobalID( d_dofIndices[i] );
    }

    dTdx = Jinv1[0] * dTdxi + Jinv1[1] * dTdeta + Jinv1[2] * dTdzeta;
    dTdy = Jinv2[0] * dTdxi + Jinv2[1] * dTdeta + Jinv2[2] * dTdzeta;
    dTdz = Jinv3[0] * dTdxi + Jinv3[1] * dTdeta + Jinv3[2] * dTdzeta;

    for ( unsigned int i = 0; i < d_dofIndices.size(); i++ ) {
        lib_dTdx += dphidx[i][0] * T->getValueByGlobalID( d_dofIndices[i] );
        lib_dTdy += dphidy[i][0] * T->getValueByGlobalID( d_dofIndices[i] );
        lib_dTdz += dphidz[i][0] * T->getValueByGlobalID( d_dofIndices[i] );
    }

    fprintf( fp, " dT/dx = %.15f, dTdy = %.15f, dTdz = %.15f \n", dTdx, dTdy, dTdz );
    fprintf( fp,
             " lib_dT/dx = %.15f, lib_dTdy = %.15f, lib_dTdz = %.15f \n",
             lib_dTdx,
             lib_dTdy,
             lib_dTdz );
    fclose( fp );
    for ( int j = 0; j < 8; j++ )
        delete libmeshNodes[j];
    ut->passes( "Ran to completion" );
}

int testInverseJacobianBug( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    auto libmeshInit = std::make_shared<AMP::Mesh::initializeLibMesh>( AMP_COMM_WORLD );

    AMP::UnitTest ut;
    myTest( &ut, "testInverseJacobianBug" );
    ut.report();
    int num_failed = ut.NumFailGlobal();

    libmeshInit.reset();
    AMP::AMPManager::shutdown();
    return num_failed;
}
