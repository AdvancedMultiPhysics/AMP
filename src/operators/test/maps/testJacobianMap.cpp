#include "AMP/mesh/libmesh/libmeshMesh.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include <cstdlib>
#include <iostream>

#include "AMP/mesh/testHelpers/meshWriters.h"
#include <memory>

// Libmesh files
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/auto_ptr.h"
#include "libmesh/boundary_info.h"
#include "libmesh/cell_hex8.h"
#include "libmesh/dof_map.h"
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/equation_systems.h"
#include "libmesh/fe.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/linear_implicit_system.h"
#include "libmesh/mesh.h"
#include "libmesh/mesh_communication.h"
#include "libmesh/quadrature.h"
#include "libmesh/quadrature_gauss.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS


// Using mesh and function calls from testLibmeshGeomType::FaceStuff.cc

static void calculateGrad( AMP::UnitTest *ut )
{

    auto libmesh =
        AMP::Mesh::MeshWriters::readTestMeshLibMesh( "distortedElementMesh", AMP_COMM_WORLD );
    auto mesh = libmesh->getlibMesh();

    libMesh::EquationSystems equation_systems( *mesh );

    auto &system = equation_systems.add_system<libMesh::LinearImplicitSystem>( "Poisson" );

    system.add_variable( "V", libMesh::FIRST );
    equation_systems.init();

    const unsigned int V_var = system.variable_number( "V" );

    libMesh::FEType fe_type = system.variable_type( V_var );

    libMesh::QGauss qrule( 3, fe_type.default_quadrature_order() );

    auto fe_3d( libMesh::FEBase::build( 3, fe_type ) );
    fe_3d->attach_quadrature_rule( &qrule );
    // const std::vector<Point>& q_point3d = fe_3d->get_xyz();

    const std::vector<std::vector<libMesh::Real>> &dphi3d = fe_3d->get_dphidx();

    libMesh::MeshBase::const_element_iterator el           = mesh->local_elements_begin();
    const libMesh::MeshBase::const_element_iterator end_el = mesh->local_elements_end();
    std::cout << "Entering Element Iyerator" << std::endl;
    for ( ; el != end_el; ++el ) {
        const libMesh::Elem *elem = *el;

        fe_3d->reinit( elem );
        // std::vector<Point> coordinates = fe_3d->get_xyz();
        std::vector<double> computedAtGauss( qrule.n_points(), 0.0 );
        std::cout << "Entering Gauss Point loop : " << qrule.n_points() << std::endl;
        for ( unsigned int qp = 0; qp < qrule.n_points(); qp++ ) {
            std::cout << "dphidx.size = " << dphi3d.size() << std::endl;
            for ( size_t l = 0; l < dphi3d.size(); l++ ) {
                std::cout << "dphidx[" << l << "][" << qp << "]  = " << dphi3d[l][qp] << std::endl;
            }
        }
    }
    ut->passes( "Ran to completion" );
}

int testJacobianMap( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::cout << "Entering main" << std::endl;
    calculateGrad( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
