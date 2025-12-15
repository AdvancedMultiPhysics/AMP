#include "AMP/operators/map/libmesh/Map1Dto3D.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorSelector.h"

// Libmesh files
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/auto_ptr.h"
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


namespace AMP::Operator {


// Constructor
Map1Dto3D::Map1Dto3D( std::shared_ptr<const OperatorParameters> params ) : MapOperator( params )
{
    auto myparams = std::dynamic_pointer_cast<const MapOperatorParameters>( params );
    d_MapMesh     = myparams->d_MapMesh;
    reset( myparams );
}


void Map1Dto3D::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );

    auto myparams = std::dynamic_pointer_cast<const MapOperatorParameters>( params );

    AMP_INSIST( myparams, "NULL parameter" );
    AMP_INSIST( myparams->d_db, "NULL database" );
    d_Mesh    = myparams->d_Mesh;
    d_MapComm = myparams->d_MapComm;
    d_MapMesh = myparams->d_MapMesh;
    if ( d_MapComm.isNull() && d_MapMesh )
        d_MapComm = d_MapMesh->getComm();
    if ( d_MapComm.isNull() && d_Mesh )
        d_MapComm = d_Mesh->getComm();
    AMP_INSIST( !d_MapComm.isNull(), "NULL communicator" );
    AMP_INSIST( d_MapComm.sumReduce<int>( d_MapMesh ? 1 : 0 ) > 0, "Somebody must own the mesh" );

    d_useGaussVec = myparams->d_db->getWithDefault<bool>( "UseGaussVec", false );

    if ( d_useGaussVec ) {
        AMP::Mesh::MeshIterator iterator =
            d_MapMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, d_boundaryId, 0 );
        libmeshElements.reinit( iterator );
    }

    if ( d_useGaussVec ) {
        computeZGaussLocations();
    } else {
        computeZNodeLocations();
    }

    AMP_INSIST( myparams->d_db->keyExists( "InputVariable" ), "key not found" );
    std::string inpVar = myparams->d_db->getString( "InputVariable" );
    d_inpVariable.reset( new AMP::LinearAlgebra::Variable( inpVar ) );

    AMP_INSIST( myparams->d_db->keyExists( "OutputVariable" ), "key not found" );
    std::string outVar = myparams->d_db->getString( "OutputVariable" );
    d_outVariable.reset( new AMP::LinearAlgebra::Variable( outVar ) );
}

void Map1Dto3D::computeZNodeLocations()
{

    // Check that the mesh exists on some processors
    auto N_mesh = d_MapComm.sumReduce<int>( ( d_MapMesh ? 1 : 0 ) );
    AMP_ASSERT( N_mesh > 0 );

    // Get the local location of nodes on the boundary
    std::vector<double> t_zLocations;
    if ( d_MapMesh ) {
        // Get an iterator over the nodes on the boundary
        auto bnd = d_MapMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_boundaryId, 0 );
        AMP::Mesh::MeshIterator end_bnd = bnd.end();

        double Xx = 0;
        double Yy = 0;
        if ( bnd != end_bnd ) {
            auto x = bnd->coord();
            AMP_ASSERT( x.size() == 3 );
            t_zLocations.push_back( x[2] );
            Xx = x[0];
            Yy = x[1];
            ++bnd;
        }
        for ( ; bnd != end_bnd; ++bnd ) {
            auto x = bnd->coord();
            if ( ( fabs( Xx - x[0] ) <= 1.e-12 ) && ( fabs( Yy - x[1] ) <= 1.e-12 ) ) {
                t_zLocations.push_back( x[2] );
            }
        }
    }

    // Make Z locations consistent across all processors.
    size_t myLen  = t_zLocations.size();
    size_t totLen = d_MapComm.sumReduce( myLen );
    std::vector<double> zLocations( totLen );
    d_MapComm.allGather( t_zLocations.data(), myLen, zLocations.data() );

    // Add the coordinates (internally this will make sure the values are unique and sort)
    setZLocations( zLocations );
}

void Map1Dto3D::computeZGaussLocations()
{

    // Check that the mesh exists on some processors
    auto N_mesh = d_MapComm.sumReduce<int>( ( d_MapMesh ? 1 : 0 ) );
    AMP_ASSERT( N_mesh > 0 );

    // Get the local location of nodes on the boundary
    std::vector<double> t_zLocations;
    if ( d_MapMesh ) {
        // Get an iterator over the nodes on the boundary
        auto bnd = d_MapMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, d_boundaryId, 0 );
        auto end_bnd = bnd.end();

        auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
        auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );

        auto d_feType = std::make_shared<libMesh::FEType>( feTypeOrder, feFamily );
        std::shared_ptr<libMesh::FEBase> d_fe(
            ( libMesh::FEBase::build( 2, ( *d_feType ) ) ).release() );

        auto qruleOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "SECOND" );
        std::shared_ptr<libMesh::QBase> d_qrule(
            ( libMesh::QBase::build( "QGAUSS", 2, qruleOrder ) ).release() );

        d_fe->attach_quadrature_rule( d_qrule.get() );

        d_fe->reinit( libmeshElements.getElement( bnd->globalID() ) );

        double Xx = 0;
        double Yy = 0;
        if ( bnd != end_bnd ) {
            // Get the current position and DOF
            auto coordinates = d_fe->get_xyz();

            for ( auto &coordinate : coordinates ) {
                t_zLocations.push_back( coordinate( 2 ) );
                Xx = coordinate( 0 );
                Yy = coordinate( 1 );
            }
            ++bnd;
        }

        for ( ; bnd != end_bnd; ++bnd ) {
            d_feType.reset( new libMesh::FEType( feTypeOrder, feFamily ) );
            d_fe.reset( ( libMesh::FEBase::build( 2, ( *d_feType ) ) ).release() );
            d_qrule.reset( ( libMesh::QBase::build( "QGAUSS", 2, qruleOrder ) ).release() );
            d_fe->attach_quadrature_rule( d_qrule.get() );
            d_fe->reinit( libmeshElements.getElement( bnd->globalID() ) );

            auto x = d_fe->get_xyz();
            for ( auto &elem : x ) {
                if ( ( fabs( Xx - elem( 0 ) ) <= 1.e-12 ) &&
                     ( fabs( Yy - elem( 1 ) ) <= 1.e-12 ) ) {
                    t_zLocations.push_back( elem( 2 ) );
                }
            }
        }
    }

    // Make Z locations consistent across all processors.
    size_t myLen  = t_zLocations.size();
    size_t totLen = d_MapComm.sumReduce( myLen );
    std::vector<double> zLocations( totLen );
    d_MapComm.allGather( t_zLocations.data(), myLen, zLocations.data() );

    // Add the coordinates (internally this will make sure the values are unique and sort)
    setZLocations( zLocations );
}


// Set the z locations
void Map1Dto3D::setZLocations( const std::vector<double> &z )
{
    const double TOL = 1e-12;
    d_zLocations     = z;
    if ( d_zLocations.size() <= 1 )
        return;
    // Sort the entires
    AMP::Utilities::quicksort( d_zLocations );
    // Remove any duplicate entries
    size_t next = 1;
    for ( size_t i = 1; i < d_zLocations.size(); i++ ) {
        if ( d_zLocations[i] - d_zLocations[next - 1] > TOL ) {
            d_zLocations[next] = d_zLocations[i];
            next++;
        }
    }
    d_zLocations.resize( next );
}


void Map1Dto3D::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                       AMP::LinearAlgebra::Vector::shared_ptr f )
{
    if ( !outputVec )
        setVector( f );
    if ( d_useGaussVec ) {
        apply_Gauss( u, f );
    } else {
        apply_Nodal( u, f );
    }
}

void Map1Dto3D::apply_Gauss( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr )
{

    if ( !d_MapMesh )
        return;

    AMP_ASSERT( u );

    // Subset the input vector, it is a simple vector and we need to subset for the current comm
    // before the variable
    AMP::LinearAlgebra::VS_Comm commSelector( d_MapComm );
    auto commSubsetVec = u->select( commSelector );
    auto inputVec      = commSubsetVec->subsetVectorForVariable( d_inpVariable );

    AMP_ASSERT( inputVec );
    AMP_ASSERT( outputVec );

    // Loop through the points on the surface
    AMP_ASSERT( d_zLocations.size() >= 2 );
    AMP_ASSERT( d_zLocations.size() == inputVec->getLocalSize() );
    AMP_ASSERT( d_zLocations.size() == inputVec->getGlobalSize() );
    const double TOL = 1e-12;
    auto dof_map     = outputVec->getDOFManager();
    auto bnd = d_MapMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, d_boundaryId, 0 );
    const double z1 = d_zLocations[0] - TOL;
    const double z2 = d_zLocations[d_zLocations.size() - 1] + TOL;

    for ( size_t it = 0; it < bnd.size(); it++ ) {
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

        d_fe->reinit( libmeshElements.getElement( bnd->globalID() ) );

        // Get the current position and DOF
        auto coordinates = d_fe->get_xyz();

        std::vector<size_t> ids;
        dof_map->getDOFs( bnd->globalID(), ids );

        for ( size_t i = 0; i < ids.size(); i++ ) {
            // Perform linear interpolation
            double z = coordinates[i]( 2 );
            if ( z < z1 || z > z2 ) {
                // Point is outside interpolant, do nothing
                AMP_ERROR( "Bad interpolant" );
            }
            size_t index = AMP::Utilities::findfirst( d_zLocations, z );
            if ( index == 0 ) {
                index = 1;
            }
            if ( index == d_zLocations.size() ) {
                index = d_zLocations.size() - 1;
            }
            double dz =
                ( z - d_zLocations[index - 1] ) / ( d_zLocations[index] - d_zLocations[index - 1] );
            double f1  = inputVec->getValueByLocalID( index - 1 );
            double f2  = inputVec->getValueByLocalID( index );
            double dz2 = 1.0 - dz;
            double f   = dz2 * f1 + dz * f2;
            outputVec->setValuesByGlobalID( 1, &ids[i], &f );
        }
        ++bnd;
    }

    if ( d_iDebugPrintInfoLevel > 4 ) {
        AMP::pout << "The input to Map1Dto3D " << std::endl;
        AMP::pout << inputVec << std::endl;
    }

    outputVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_iDebugPrintInfoLevel > 5 ) {
        AMP::pout << "The output to Map1Dto3D " << std::endl;
        AMP::pout << outputVec << std::endl;
    }
}

void Map1Dto3D::apply_Nodal( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr )
{

    if ( !d_MapMesh )
        return;

    AMP_ASSERT( u );

    // Subset the input vector, it is a simple vector and we need to subset for the current comm
    // before the variable
    AMP::LinearAlgebra::VS_Comm commSelector( d_MapComm );
    auto commSubsetVec = u->select( commSelector );
    auto inputVec      = commSubsetVec->subsetVectorForVariable( d_inpVariable );

    AMP_ASSERT( inputVec );
    AMP_ASSERT( outputVec );

    // Loop through the points on the surface
    AMP_ASSERT( d_zLocations.size() >= 2 );
    AMP_ASSERT( d_zLocations.size() == inputVec->getLocalSize() );
    AMP_ASSERT( d_zLocations.size() == inputVec->getGlobalSize() );
    const double TOL = 1e-12;
    auto dof_map     = outputVec->getDOFManager();
    std::vector<size_t> dofs( 1 );
    auto bnd = d_MapMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_boundaryId, 0 );
    const double z1 = d_zLocations[0] - TOL;
    const double z2 = d_zLocations[d_zLocations.size() - 1] + TOL;
    for ( size_t i = 0; i < bnd.size(); i++ ) {
        dof_map->getDOFs( bnd->globalID(), dofs );
        auto x = bnd->coord();
        AMP_INSIST( dofs.size() == 1,
                    "Map1Dto3D is currently implemented for scalar quantities only" );
        AMP_INSIST( x.size() == 3, "Map1Dto3D is currently implemented for 3D" );
        // Perform linear interpolation
        double z = x[2];
        if ( z < z1 || z > z2 ) {
            // Point is outside interpolant, do nothing
            AMP_ERROR( "Bad interpolant" );
        }
        size_t index = AMP::Utilities::findfirst( d_zLocations, z );
        if ( index == 0 ) {
            index = 1;
        }
        if ( index == d_zLocations.size() ) {
            index = d_zLocations.size() - 1;
        }
        double dz =
            ( z - d_zLocations[index - 1] ) / ( d_zLocations[index] - d_zLocations[index - 1] );
        double f1  = inputVec->getValueByLocalID( index - 1 );
        double f2  = inputVec->getValueByLocalID( index );
        double dz2 = 1.0 - dz;
        double f   = dz2 * f1 + dz * f2;
        outputVec->setValuesByGlobalID( 1, &dofs[0], &f );
        ++bnd;
    }

    if ( d_iDebugPrintInfoLevel > 4 ) {
        AMP::pout << "The input to Map1Dto3D " << std::endl;
        AMP::pout << inputVec << std::endl;
    }

    outputVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_iDebugPrintInfoLevel > 5 ) {
        AMP::pout << "The output to Map1Dto3D " << std::endl;
        AMP::pout << outputVec << std::endl;
    }
}


void Map1Dto3D::setVector( AMP::LinearAlgebra::Vector::shared_ptr vec )
{
    outputVec = subsetOutputVector( vec );
}
} // namespace AMP::Operator
