#include "AMP/operators/map/libmesh/ScalarN2GZAxisMap.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"


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


/************************************************************************
 *  Default constructor                                                  *
 ************************************************************************/
ScalarN2GZAxisMap::ScalarN2GZAxisMap( std::shared_ptr<const AMP::Operator::OperatorParameters> p )
    : Map3to1to3( p )
{
    auto params = std::dynamic_pointer_cast<const Map3to1to3Parameters>( p );
    AMP_ASSERT( params );

    int DofsPerObj = params->d_db->getScalar<int>( "DOFsPerObject" );
    AMP_INSIST( DofsPerObj == 4, "ScalarZAxis is currently only designed for 4 Gp per elem" );

    // Create the element iterators
    if ( d_mesh1 ) {
        d_srcIterator1 =
            d_mesh1->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, params->d_BoundaryID1, 0 );
        d_dstIterator1 =
            d_mesh1->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, params->d_BoundaryID1, 0 );
    }
    if ( d_mesh2 ) {
        d_srcIterator2 =
            d_mesh2->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, params->d_BoundaryID2, 0 );
        d_dstIterator2 =
            d_mesh2->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, params->d_BoundaryID2, 0 );
    }

    auto iterator =
        AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Union, d_dstIterator1, d_dstIterator2 );
    libmeshElements.reinit( iterator );

    // Build the z-coordinates of the return gauss points
    d_z_coord1 = getGaussPoints( d_dstIterator1 );
    d_z_coord2 = getGaussPoints( d_dstIterator2 );
}


/************************************************************************
 *  De-constructor                                                       *
 ************************************************************************/
ScalarN2GZAxisMap::~ScalarN2GZAxisMap() = default;

/************************************************************************
 *  Check if the map type is "ScalarN2GZAxis"                               *
 ************************************************************************/
bool ScalarN2GZAxisMap::validMapType( const std::string &t )
{
    if ( t == "ScalarN2GZAxis" )
        return true;
    return false;
}


/************************************************************************
 *  buildMap                                                             *
 *  This function constructs the map from the given vector.              *
 *  It loops through all values in "cur", storing the value iv using     *
 *  the z-position as the 1D key.                                        *
 ************************************************************************/
std::multimap<double, double>
ScalarN2GZAxisMap::buildMap( AMP::LinearAlgebra::Vector::const_shared_ptr vec,
                             const std::shared_ptr<AMP::Mesh::Mesh>,
                             const AMP::Mesh::MeshIterator &iterator )
{
    if ( iterator.size() == 0 )
        return std::multimap<double, double>();
    PROFILE( "buildMap" );
    std::multimap<double, double> map;
    auto dof = vec->getDOFManager();
    size_t N = iterator.size();
    std::vector<AMP::Mesh::MeshElementID> ids( N );
    std::vector<double> z( N, 0.0 );
    auto it = iterator.begin();
    for ( size_t i = 0; i < N; ++i, ++it ) {
        ids[i] = it->globalID();
        z[i]   = it->coord( 2 );
        AMP_ASSERT( ids[i].is_local() );
    }
    std::vector<size_t> dofs( N );
    dof->getDOFs( ids, dofs );
    AMP_ASSERT( ids.size() == dofs.size() );
    std::vector<double> vals( N );
    vec->getValuesByGlobalID( N, &dofs[0], &vals[0] );
    addTo1DMap( map, z, vals );
    return map;
}


/************************************************************************
 *  Function to build the z-coordinates of the gauss points              *
 ************************************************************************/
AMP::LinearAlgebra::Vector::const_shared_ptr
ScalarN2GZAxisMap::getGaussPoints( const AMP::Mesh::MeshIterator &iterator )
{
    if ( iterator.size() == 0 )
        return AMP::LinearAlgebra::Vector::const_shared_ptr();
    if ( iterator == d_dstIterator1 && d_z_coord1 )
        return d_z_coord1;
    if ( iterator == d_dstIterator2 && d_z_coord2 )
        return d_z_coord2;
    PROFILE( "getGaussPoints" );
    auto GpDofMap               = AMP::Discretization::simpleDOFManager::create( iterator, 4 );
    auto var                    = std::make_shared<AMP::LinearAlgebra::Variable>( "gauss_z" );
    auto z_pos                  = AMP::LinearAlgebra::createVector( GpDofMap, var );
    AMP::Mesh::MeshIterator cur = iterator.begin();
    std::vector<size_t> ids;
    for ( size_t i = 0; i < cur.size(); i++ ) {
        // Create the libmesh element
        auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
        auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );
        auto d_feType    = std::make_shared<libMesh::FEType>( feTypeOrder, feFamily );
        std::shared_ptr<libMesh::FEBase> d_fe(
            ( libMesh::FEBase::build( 2, ( *d_feType ) ) ).release() );
        d_fe->get_xyz();
        auto qruleOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "SECOND" );
        std::shared_ptr<libMesh::QBase> d_qrule(
            ( libMesh::QBase::build( "QGAUSS", 2, qruleOrder ) ).release() );
        d_fe->attach_quadrature_rule( d_qrule.get() );
        d_fe->reinit( libmeshElements.getElement( cur->globalID() ) );
        // Get the current position and DOF
        const auto &coordinates = d_fe->get_xyz();
        GpDofMap->getDOFs( cur->globalID(), ids );
        for ( unsigned int qp = 0; qp < ids.size(); qp++ ) {
            const double pos = coordinates[qp]( 2 );
            z_pos->setValuesByGlobalID( 1, &ids[qp], &pos );
        }
        ++cur;
    }
    return z_pos;
}


/************************************************************************
 *  buildReturn                                                          *
 ************************************************************************/
void ScalarN2GZAxisMap::buildReturn( AMP::LinearAlgebra::Vector::shared_ptr vec,
                                     const std::shared_ptr<AMP::Mesh::Mesh>,
                                     const AMP::Mesh::MeshIterator &iterator,
                                     const std::map<double, double> &map )
{
    if ( iterator.size() == 0 )
        return;
    PROFILE( "buildReturn" );
    const double TOL = 1e-8;

    // Convert the map to a std::vector
    AMP_ASSERT( map.size() > 1 );
    std::vector<double> z( map.size() ), f( map.size() );
    auto it_map = map.begin();
    for ( size_t i = 0; it_map != map.end(); ++it_map, ++i ) {
        z[i] = it_map->first;
        f[i] = it_map->second;
    }
    for ( size_t i = 1; i < z.size(); i++ )
        AMP_ASSERT( z[i] > ( z[i - 1] + TOL ) );
    double z0 = z[0];
    double z1 = z[z.size() - 1];

    // Get the coordinates of the gauss points
    AMP::LinearAlgebra::Vector::const_shared_ptr z_pos = getGaussPoints( iterator );
    AMP_ASSERT( z_pos );

    // Get the DOF managers
    auto DOFs      = vec->getDOFManager();
    auto gaussDOFs = z_pos->getDOFManager();

    // Loop through the points in the output vector
    size_t N0 = iterator.size();
    std::vector<size_t> dofs;
    std::vector<double> zi;
    dofs.reserve( N0 );
    zi.reserve( N0 );
    std::vector<size_t> id1, id2;
    std::vector<double> zi2;
    auto it_mesh = iterator.begin();
    for ( size_t i = 0; i < N0; ++i, ++it_mesh ) {
        // Get the local DOFs
        DOFs->getDOFs( it_mesh->globalID(), id1 );
        gaussDOFs->getDOFs( it_mesh->globalID(), id2 );
        AMP_ASSERT( id1.size() == id2.size() );
        size_t N2 = id1.size();
        // Get the coordinates of the gauss points
        zi2.resize( N2 );
        z_pos->getValuesByGlobalID( N2, &id2[0], &zi2[0] );
        for ( size_t j = 0; j < N2; j++ ) {
            if ( zi2[j] < z0 - TOL || zi2[j] > z1 + TOL ) {
                // We are outside the bounds of the map
                continue;
            }
            dofs.push_back( id1[j] );
            zi.push_back( zi2[j] );
        }
    }
    std::vector<double> fi( zi.size() );
    for ( size_t i = 0; i < zi.size(); i++ ) {
        // Find the first point > the current position
        size_t k = AMP::Utilities::findfirst( z, zi[i] );
        k        = std::max<size_t>( k, 1 );
        k        = std::min<size_t>( k, z.size() - 1 );
        // Perform linear interpolation
        double wt = ( zi[i] - z[k - 1] ) / ( z[k] - z[k - 1] );
        fi[i]     = ( 1.0 - wt ) * f[k - 1] + wt * f[k];
    }
    vec->setValuesByGlobalID( dofs.size(), &dofs[0], &fi[0] );
}


} // namespace AMP::Operator
