#include "AMP/utils/NearestPairSearch.h"
#include "AMP/utils/MeshPoint.h"
#include "AMP/utils/NearestPairSearch.hpp"

namespace AMP {


// Calculate the closest pair of points in a list
std::pair<int, int> find_min_dist( const std::vector<AMP::Mesh::MeshPoint<double>> &x )
{
    if ( x.empty() )
        return std::pair<int, int>( 0, 0 );
    ;
    int ndim = x[0].ndim();
    auto x2  = new double[ndim * x.size()];
    for ( size_t i = 0; i < x.size(); i++ ) {
        for ( int d = 0; d < ndim; d++ )
            x2[d + i * ndim] = x[i][d];
    }
    if ( ndim == 1 )
        return find_min_dist<1, double>( x.size(), x2 );
    else if ( ndim == 2 )
        return find_min_dist<2, double>( x.size(), x2 );
    else if ( ndim == 3 )
        return find_min_dist<3, double>( x.size(), x2 );
    AMP_ERROR( "Not programmed" );
}


} // namespace AMP
