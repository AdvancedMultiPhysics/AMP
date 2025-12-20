#include "AMP/operators/map/libmesh/SubchannelToCladGPMap.h"
#include "AMP/discretization/DOF_Manager.h"

#include "ProfilerApp.h"

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


namespace AMP::Operator {


static double
interp_linear( const std::vector<double> &x, const std::vector<double> &f, double pos )
{
    AMP_ASSERT( x.size() >= 2 && x.size() == f.size() );
    size_t i = AMP::Utilities::findfirst( x, pos );
    if ( i == 0 ) {
        i = 1;
    }
    return f[i - 1] + ( pos - x[i - 1] ) / ( x[i] - x[i - 1] ) * ( f[i] - f[i - 1] );
}

/************************************************************************
 *  Default constructor                                                  *
 ************************************************************************/
SubchannelToCladGPMap::SubchannelToCladGPMap(
    std::shared_ptr<const AMP::Operator::OperatorParameters> p )
    : SubchannelToCladMap( p )
{
    auto params = std::dynamic_pointer_cast<const SubchannelToCladGPMapParameters>( p );
    AMP_ASSERT( params );
    int DofsPerObj = params->d_db->getScalar<int>( "DOFsPerObject" );
    AMP_INSIST( DofsPerObj == 4,
                "SubchannelToCladGPMap is currently only designed for 4 Gp per elem" );
    AMP_INSIST( params->d_db->keyExists( "GeomType" ), "GeomType must exist in database" );
    int type = params->d_db->getScalar<int>( "GeomType" );
    AMP_INSIST( type == 2,
                "SubchannelToCladGPMap is currently only designed for face elements (GeomType)" );
    if ( d_iterator2.size() > 0 ) {
        AMP::Mesh::MeshIterator iterator = d_iterator2;
        libmeshElements.reinit( iterator );
    }
}


/************************************************************************
 *  De-constructor                                                       *
 ************************************************************************/
SubchannelToCladGPMap::~SubchannelToCladGPMap() = default;


/************************************************************************
 *  Check if the map type is "SubchannelToCladMap"                       *
 ************************************************************************/
bool SubchannelToCladGPMap::validMapType( const std::string &t )
{
    if ( t == "SubchannelToCladGPMap" )
        return true;
    return false;
}


/************************************************************************
 *  Fill the return vector for the given subchannel                      *
 ************************************************************************/
void SubchannelToCladGPMap::fillReturnVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                                              double[4],
                                              std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                              const std::vector<AMP::Mesh::MeshElementID> &ids,
                                              const std::vector<double> &z,
                                              const std::vector<double> &f )
{
    PROFILE( "fillReturnVector" );
    std::shared_ptr<AMP::Discretization::DOFManager> DOF = vec->getDOFManager();
    std::vector<gaussPointZCoord> z_gauss                = this->getGaussPoints( mesh, ids );
    std::vector<size_t> dofs( 4 );
    std::vector<double> vals( 4 );
    for ( size_t i = 0; i < ids.size(); i++ ) {
        DOF->getDOFs( ids[i], dofs );
        AMP_ASSERT( dofs.size() == 4 );
        for ( int j = 0; j < 4; j++ )
            vals[j] = interp_linear( z, f, z_gauss[i].z[j] );
        vec->setValuesByGlobalID( 4, &dofs[0], &vals[0] );
    }
}


/************************************************************************
 *  Function to build the z-coordinates of the gauss points              *
 ************************************************************************/
std::vector<SubchannelToCladGPMap::gaussPointZCoord>
SubchannelToCladGPMap::getGaussPoints( std::shared_ptr<AMP::Mesh::Mesh>,
                                       const std::vector<AMP::Mesh::MeshElementID> &ids )
{
    std::vector<SubchannelToCladGPMap::gaussPointZCoord> z_pos;
    z_pos.resize( ids.size() );
    for ( size_t i = 0; i < ids.size(); i++ ) {
        AMP_ASSERT( ids[i].type() == AMP::Mesh::GeomType::Face );
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
        d_fe->reinit( libmeshElements.getElement( ids[i] ) );
        // Get the current position and DOF
        const auto &coordinates = d_fe->get_xyz();
        AMP_ASSERT( coordinates.size() == 4 );
        for ( unsigned int qp = 0; qp < 4; qp++ )
            z_pos[i].z[qp] = coordinates[qp]( 2 );
    }
    return z_pos;
}


} // namespace AMP::Operator
