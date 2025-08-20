#include "AMP/operators/libmesh/GradientOperator.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/Utilities.h"

// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/auto_ptr.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/node.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS


namespace AMP::Operator {


/****************************************************************
 * Constructor                                                   *
 ****************************************************************/
GradientOperator::GradientOperator( std::shared_ptr<const OperatorParameters> params )
    : Operator( params )
{
    reset( params );
}
void GradientOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    // Initalize basic info
    AMP_ASSERT( params );
    Operator::reset( params );
    d_db = params->d_db;
    AMP_ASSERT( d_db );
    AMP_ASSERT( d_Mesh );

    // Initialize the variables
    auto inputName  = d_db->getString( "InputVariable" );
    auto outputName = d_db->getString( "OutputVariable" );
    d_inputVar      = std::make_shared<AMP::LinearAlgebra::Variable>( inputName );
    d_outputVar     = std::make_shared<AMP::LinearAlgebra::Variable>( outputName );

    // Intialize the shape functions
    auto qtypeName    = d_db->getWithDefault<std::string>( "QRULE_TYPE", "QGAUSS" );
    auto qorderName   = d_db->getWithDefault<std::string>( "QRULE_ORDER", "DEFAULT" );
    auto feOrderName  = d_db->getWithDefault<std::string>( "FE_ORDER", "FIRST" );
    auto feFamilyName = d_db->getWithDefault<std::string>( "FE_FAMILY", "LAGRANGE" );
    auto feOrder      = libMesh::Utility::string_to_enum<libMeshEnums::Order>( feOrderName );
    auto feFamily     = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( feFamilyName );
    auto qtype        = libMesh::Utility::string_to_enum<libMeshEnums::QuadratureType>( qtypeName );
    auto feType       = std::make_shared<libMesh::FEType>( feOrder, feFamily );
    libMeshEnums::Order qorder;
    if ( qorderName == "DEFAULT" ) {
        qorder = feType->default_quadrature_order();
    } else {
        qorder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( qorderName );
    }

    // Initialize the libmesh elements
    d_elem.reinit( d_Mesh->getIterator( d_Mesh->getGeomType() ), qtype, qorder, feType );

    // Get the number of elements attached to each node
    d_nodes.clear();
    AMP::Mesh::MeshElementID ids[32];
    for ( auto &elem : d_Mesh->getIterator( d_Mesh->getGeomType() ) ) {
        int N = elem.getElementsID( AMP::Mesh::GeomType::Vertex, ids );
        for ( int j = 0; j < N; j++ )
            d_nodes.push_back( ids[j] );
    }
    AMP::Utilities::unique( d_nodes );
    d_elementsPerNode = std::vector<int>( d_nodes.size(), 0 );
    for ( auto &elem : d_Mesh->getIterator( d_Mesh->getGeomType() ) ) {
        int N = elem.getElementsID( AMP::Mesh::GeomType::Vertex, ids );
        for ( int j = 0; j < N; j++ ) {
            size_t i = AMP::Utilities::findfirst( d_nodes, ids[j] );
            d_elementsPerNode[i]++;
        }
    }
}


/****************************************************************
 * Apply                                                         *
 ****************************************************************/
void GradientOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                              AMP::LinearAlgebra::Vector::shared_ptr f )
{
    // Subset the input/output vectors
    auto rhs = subsetInputVector( u );
    auto sol = subsetOutputVector( f );
    AMP_ASSERT( rhs );
    AMP_ASSERT( sol );

    // Initialize solution
    sol->zero();

    // Loop through the elements
    auto rhsDOFMap = rhs->getDOFManager();
    auto solDOFMap = sol->getDOFManager();
    std::vector<size_t> dofs;
    std::vector<double> f_x;
    std::vector<std::array<double, 3>> g_x;
    AMP::Mesh::MeshElementID ids[32];
    for ( auto &elem : d_Mesh->getIterator( d_Mesh->getGeomType() ) ) {
        // Get the libmesh element propeties
        auto fe   = d_elem.getFEBase( elem.globalID() );
        auto phi  = fe->get_phi();
        auto dphi = fe->get_dphi();
        // Get the ids of the nodes comprising the element
        int N = elem.getElementsID( AMP::Mesh::GeomType::Vertex, ids );
        AMP_ASSERT( N > 0 && N <= 32 );
        // Get the values of the function at the nodes
        rhsDOFMap->getDOFs( N, ids, dofs );
        AMP_ASSERT( (int) dofs.size() == N );
        f_x.resize( dofs.size() );
        rhs->getValuesByGlobalID( dofs.size(), dofs.data(), f_x.data() );
        // Compute the gradient at the quadrature points
        g_x.resize( dphi.size() );
        for ( size_t qp = 0; qp < dphi.size(); qp++ ) {
            libMesh::RealGradient grad_f = 0.0;
            for ( int n = 0; n < N; n++ ) {
                grad_f += dphi[n][qp] * f_x[n];
            }
            g_x[qp] = { grad_f( 0 ), grad_f( 1 ), grad_f( 2 ) };
        }
        // Integrate the gradient over the quadrature points
        for ( int n = 0; n < N; n++ ) {
            double g[3] = { 0, 0, 0 };
            for ( size_t qp = 0; qp < phi.size(); qp++ ) {
                g[0] += g_x[qp][0] * phi[n][qp];
                g[1] += g_x[qp][1] * phi[n][qp];
                g[2] += g_x[qp][2] * phi[n][qp];
            }
            size_t i = AMP::Utilities::findfirst( d_nodes, ids[n] );
            AMP_ASSERT( d_elementsPerNode[i] > 0 );
            g[0] /= d_elementsPerNode[i];
            g[1] /= d_elementsPerNode[i];
            g[2] /= d_elementsPerNode[i];
            solDOFMap->getDOFs( ids[n], dofs );
            AMP_ASSERT( dofs.size() == 3 );
            sol->addValuesByGlobalID( dofs.size(), dofs.data(), g );
        }
    }
    sol->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


} // namespace AMP::Operator
