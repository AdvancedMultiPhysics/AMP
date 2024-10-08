#include "AMP/operators/map/ScalarZAxisMap.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"


namespace AMP::Operator {


/************************************************************************
 *  Default constructor                                                  *
 ************************************************************************/
ScalarZAxisMap::ScalarZAxisMap( std::shared_ptr<const AMP::Operator::OperatorParameters> p )
    : Map3to1to3( p )
{
    auto params = std::dynamic_pointer_cast<const Map3to1to3Parameters>( p );
    AMP_ASSERT( params );

    int DofsPerObj = params->d_db->getScalar<int>( "DOFsPerObject" );
    AMP_INSIST( DofsPerObj == 1, "ScalarZAxis is currently only designed for 1 DOF per node" );

    // Create the element iterators
    if ( d_mesh1 ) {
        d_srcIterator1 =
            d_mesh1->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, params->d_BoundaryID1, 0 );
        d_dstIterator1 =
            d_mesh1->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, params->d_BoundaryID1, 0 );
    }
    if ( d_mesh2 ) {
        d_srcIterator2 =
            d_mesh2->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, params->d_BoundaryID2, 0 );
        d_dstIterator2 =
            d_mesh2->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, params->d_BoundaryID2, 0 );
    }
}


/************************************************************************
 *  De-constructor                                                       *
 ************************************************************************/
ScalarZAxisMap::~ScalarZAxisMap() = default;


/************************************************************************
 *  Check if the map type is "ScalarZAxis"                               *
 ************************************************************************/
bool ScalarZAxisMap::validMapType( const std::string &t ) { return t == "ScalarZAxis"; }


/************************************************************************
 *  buildMap                                                             *
 *  This function constructs the map from the given vector.              *
 *  It loops through all values in "cur", storing the value iv using     *
 *  the z-position as the 1D key.                                        *
 ************************************************************************/
std::multimap<double, double>
ScalarZAxisMap::buildMap( AMP::LinearAlgebra::Vector::const_shared_ptr vec,
                          const std::shared_ptr<AMP::Mesh::Mesh>,
                          const AMP::Mesh::MeshIterator &iterator )
{
    PROFILE( "buildMap" );
    std::multimap<double, double> map;
    auto dof = vec->getDOFManager();
    auto cur = iterator.begin();
    auto end = iterator.end();
    std::vector<size_t> ids;
    while ( cur != end ) {
        dof->getDOFs( cur->globalID(), ids );
        AMP_ASSERT( ids.size() == 1 );
        double val = vec->getValueByGlobalID( ids[0] );
        auto x     = cur->coord();
        addTo1DMap( map, x[2], val );
        ++cur;
    }
    return map;
}


/************************************************************************
 *  buildReturn                                                          *
 ************************************************************************/
void ScalarZAxisMap::buildReturn( const AMP::LinearAlgebra::Vector::shared_ptr vec,
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
    size_t i = 0;
    for ( const auto &elem : map ) {
        z[i] = elem.first;
        f[i] = elem.second;
        i++;
    }
    for ( size_t j = 1; j < z.size(); j++ )
        AMP_ASSERT( z[j] > ( z[j - 1] + TOL ) );
    double z0 = z[0];
    double z1 = z[z.size() - 1];
    double v0 = f[0];
    double v1 = f[z.size() - 1];

    // Loop through the points in the output vector
    auto DOFs = vec->getDOFManager();
    auto cur  = iterator.begin();
    auto end  = iterator.end();
    std::vector<size_t> ids;
    while ( cur != end ) {

        // Get the current position and DOF
        auto x    = cur->coord();
        double zi = x[2];
        DOFs->getDOFs( cur->globalID(), ids );
        AMP_ASSERT( ids.size() == 1 );
        size_t dof = ids[0];

        // Check the endpoints
        if ( fabs( zi - z0 ) <= TOL ) {
            // We are within TOL of the first point
            vec->setValuesByGlobalID( 1, &dof, &v0 );
            ++cur;
            continue;
        } else if ( fabs( zi - z1 ) <= TOL ) {
            // We are within TOL of the last point
            vec->setValuesByGlobalID( 1, &dof, &v1 );
            ++cur;
            continue;
        } else if ( zi < z0 || zi > z1 ) {
            // We are outside the bounds of the map
            ++cur;
            continue;
        }

        // Find the first point > the current position
        size_t k = AMP::Utilities::findfirst( z, zi );
        if ( k == 0 ) {
            k++;
        }
        if ( k == z.size() ) {
            k--;
        }

        // Perform linear interpolation
        double wt = ( zi - z[k - 1] ) / ( z[k] - z[k - 1] );
        double fi = ( 1.0 - wt ) * f[k - 1] + wt * f[k];
        vec->setValuesByGlobalID( 1, &dof, &fi );

        ++cur;
    }
}


} // namespace AMP::Operator
