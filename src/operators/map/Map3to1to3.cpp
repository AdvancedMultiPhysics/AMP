#include "AMP/operators/map/Map3to1to3.h"
#include "AMP/operators/map/Map3to1to3Parameters.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/VectorSelector.h"

#include "ProfilerApp.h"


namespace AMP::Operator {


/********************************************************
 * Constructor                                           *
 ********************************************************/
Map3to1to3::Map3to1to3( std::shared_ptr<const OperatorParameters> params_in )
    : AsyncMapOperator( params_in )
{
    // Get the input parameters
    auto params = std::dynamic_pointer_cast<const Map3to1to3Parameters>( params_in );
    AMP_ASSERT( params );
    d_commTag = params->d_commTag;

    // Create default iterators (must be overwritten by derived class)
    d_srcIterator1 = AMP::Mesh::MeshIterator();
    d_srcIterator2 = AMP::Mesh::MeshIterator();
    d_dstIterator1 = AMP::Mesh::MeshIterator();
    d_dstIterator2 = AMP::Mesh::MeshIterator();

    // Determine which processors we will be sending/receiving data from
    // 0: No communication, 1: send/recv data
    d_own_mesh1 = std::vector<bool>( d_MapComm.getSize(), false );
    d_own_mesh2 = std::vector<bool>( d_MapComm.getSize(), false );
    std::vector<char> tmp1( d_MapComm.getSize(), false );
    std::vector<char> tmp2( d_MapComm.getSize(), false );
    d_MapComm.allGather<char>( d_mesh1.get() != nullptr, &tmp1[0] );
    d_MapComm.allGather<char>( d_mesh2.get() != nullptr, &tmp2[0] );
    for ( int i = 0; i < d_MapComm.getSize(); i++ ) {
        d_own_mesh1[i] = tmp1[i] != 0;
        d_own_mesh2[i] = tmp2[i] != 0;
    }
    size_t numToSend = 0;
    if ( d_mesh1 ) {
        for ( const auto &elem : d_own_mesh2 ) {
            if ( elem == 1 )
                numToSend++;
        }
    }
    if ( d_mesh2 ) {
        for ( const auto &elem : d_own_mesh1 ) {
            if ( elem == 1 )
                numToSend++;
        }
    }
    reserveRequests( numToSend );
}


/********************************************************
 * De-constructor                                        *
 ********************************************************/
Map3to1to3::~Map3to1to3() { waitForAllRequests(); }


/********************************************************
 * Function to add points to the map                     *
 ********************************************************/
void Map3to1to3::addTo1DMap( std::multimap<double, double> &map, double z, double val )
{
    map.insert( { z, val } );
}
void Map3to1to3::addTo1DMap( std::multimap<double, double> &map,
                             const std::vector<double> &z,
                             const std::vector<double> &val )
{
    for ( size_t i = 0; i < z.size(); i++ )
        map.insert( { z[i], val[i] } );
}


/********************************************************
 * Function to start the communication                   *
 ********************************************************/
void Map3to1to3::applyStart( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr )
{
    const double tol = 1e-8;
    PROFILE( "applyStart" );

    // Subset the vector (we only need to deal with the locally owned portion)
    auto var = getInputVariable();
    AMP::LinearAlgebra::VS_Comm commSelector( AMP_COMM_SELF );
    auto commVec = u->select( commSelector );
    auto vec     = commVec->subsetVectorForVariable( var );

    // Build the local maps
    auto map1 = buildMap( vec, d_mesh1, d_srcIterator1 );
    auto map2 = buildMap( vec, d_mesh2, d_srcIterator2 );

    // Get the local 1D map coordinates
    double z_last = -1e100;
    std::vector<double> z1;
    for ( const auto &tmp : map1 ) {
        if ( fabs( tmp.first - z_last ) > tol ) {
            z_last = tmp.first;
            z1.push_back( z_last );
        }
    }
    z_last = -1e100;
    std::vector<double> z2;
    for ( const auto &tmp : map2 ) {
        if ( fabs( tmp.first - z_last ) > tol ) {
            z_last = tmp.first;
            z2.push_back( z_last );
        }
    }

    // Create the send buffers and sum the local data
    d_SendBuf1.resize( 0 ); // Reset the entries
    d_SendBuf1.resize( z1.size() );
    for ( const auto &tmp : map1 ) {
        double z  = tmp.first;
        size_t i1 = std::min( AMP::Utilities::findfirst( z1, z ), z1.size() - 1 );
        size_t i2 = std::max( i1, (size_t) 1 ) - 1;
        size_t i3 = std::min( i1 + 1, z1.size() - 1 );
        size_t i  = 0;
        if ( fabs( z - z1[i1] ) < tol )
            i = i1;
        else if ( fabs( z - z1[i2] ) < tol )
            i = i2;
        else if ( fabs( z - z1[i3] ) < tol )
            i = i3;
        else
            AMP_ERROR( "Internal error" );
        d_SendBuf1[i].N++;
        d_SendBuf1[i].z = z1[i];
        d_SendBuf1[i].sum += tmp.second;
    }
    d_SendBuf2.resize( 0 ); // Reset the entries
    d_SendBuf2.resize( z2.size() );
    for ( const auto &tmp : map2 ) {
        double z  = tmp.first;
        size_t i1 = std::min( AMP::Utilities::findfirst( z2, z ), z2.size() - 1 );
        size_t i2 = std::max( i1, (size_t) 1 ) - 1;
        size_t i3 = std::min( i1 + 1, z2.size() - 1 );
        size_t i  = 0;
        if ( fabs( z - z2[i1] ) < tol )
            i = i1;
        else if ( fabs( z - z2[i2] ) < tol )
            i = i2;
        else if ( fabs( z - z2[i3] ) < tol )
            i = i3;
        else
            AMP_ERROR( "Internal error" );
        d_SendBuf2[i].N++;
        d_SendBuf2[i].z = z2[i];
        d_SendBuf2[i].sum += tmp.second;
    }


    // Send the data
    auto myRank = (size_t) d_MapComm.getRank();
    auto curReq = beginRequests();
    if ( d_mesh1 ) {
        for ( size_t i = 0; i < d_own_mesh2.size(); i++ ) {
            if ( i == myRank )
                continue; // Don't communicate local data
            if ( d_own_mesh2[i] ) {
                *curReq = d_MapComm.Isend( d_SendBuf1.data(), d_SendBuf1.size(), i, d_commTag + 0 );
                ++curReq;
            }
        }
    }
    if ( d_mesh2 ) {
        for ( size_t i = 0; i < d_own_mesh1.size(); i++ ) {
            if ( i == myRank )
                continue; // Don't communicate local data
            if ( d_own_mesh1[i] ) {
                *curReq = d_MapComm.Isend( d_SendBuf2.data(), d_SendBuf2.size(), i, d_commTag + 1 );
                ++curReq;
            }
        }
    }
}


/********************************************************
 * Function to finish the communication and perform the  *
 * interpolation                                         *
 ********************************************************/
void Map3to1to3::applyFinish( AMP::LinearAlgebra::Vector::const_shared_ptr,
                              AMP::LinearAlgebra::Vector::shared_ptr )
{
    PROFILE( "applyFinish" );

    // Recieve the data and create the maps
    auto myRank = (size_t) d_MapComm.getRank();
    std::map<double, std::pair<int, double>> map1;
    std::map<double, std::pair<int, double>> map2;
    std::vector<comm_data> recvBuf;
    if ( d_mesh1 ) {
        // First get any local data
        if ( d_own_mesh2[myRank] )
            unpackBuffer( d_SendBuf2, map1 );
        // Recieve all remote data
        for ( size_t i = 0; i < d_own_mesh2.size(); i++ ) {
            if ( i == myRank )
                continue; // We already copied the local data
            if ( d_own_mesh2[i] ) {
                // Get the received data
                int num_bytes = std::get<2>( d_MapComm.probe( i, d_commTag + 1 ) );
                int inSize    = num_bytes / sizeof( comm_data );
                recvBuf.resize( inSize );
                d_MapComm.recv( recvBuf.data(), inSize, i, false, d_commTag + 1 );
                // Add it to the map
                unpackBuffer( recvBuf, map1 );
            }
        }
    }
    if ( d_mesh2 ) {
        // First get any local data
        if ( d_own_mesh1[myRank] )
            unpackBuffer( d_SendBuf1, map2 );
        // Recieve all remote data
        for ( size_t i = 0; i < d_own_mesh1.size(); i++ ) {
            if ( i == myRank )
                continue; // We already copied the local data
            if ( d_own_mesh1[i] ) {
                // Get the received data
                int num_bytes = std::get<2>( d_MapComm.probe( i, d_commTag + 0 ) );
                int inSize    = num_bytes / sizeof( comm_data );
                recvBuf.resize( inSize );
                d_MapComm.recv( recvBuf.data(), inSize, i, false, d_commTag + 0 );
                // Add it to the map
                unpackBuffer( recvBuf, map2 );
            }
        }
    }

    // Smear the data to create the final map
    std::map<double, double> final_map1, final_map2;
    for ( const auto &tmp : map1 ) {
        double sum = tmp.second.second;
        double N   = tmp.second.first;
        final_map1.insert( { tmp.first, sum / N } );
    }
    for ( const auto &tmp : map2 ) {
        double sum = tmp.second.second;
        double N   = tmp.second.first;
        final_map2.insert( { tmp.first, sum / N } );
    }

    // Build the return vector
    if ( d_mesh1 )
        buildReturn( d_ResultVector, d_mesh1, d_dstIterator1, final_map1 );
    if ( d_mesh2 )
        buildReturn( d_ResultVector, d_mesh2, d_dstIterator2, final_map2 );

    // Apply make consistent
    d_ResultVector->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


/********************************************************
 * Function to unpack a recv buffer                      *
 ********************************************************/
void Map3to1to3::unpackBuffer( const std::vector<comm_data> &buffer,
                               std::map<double, std::pair<int, double>> &map )
{
    const double tol = 1e-8;
    for ( auto &elem : buffer ) {
        auto iterator = map.end();
        if ( !map.empty() ) {
            auto it1 = map.lower_bound( elem.z );
            if ( it1 == map.end() ) {
                --it1;
            }
            auto it2 = it1;
            auto it3 = it1;
            if ( it1 != map.begin() )
                --it1;
            ++it3;
            if ( it3 == map.end() )
                it3 = it2;
            if ( fabs( it1->first - elem.z ) < tol )
                iterator = it1;
            else if ( fabs( it2->first - elem.z ) < tol )
                iterator = it2;
            else if ( fabs( it3->first - elem.z ) < tol )
                iterator = it3;
        }
        if ( iterator == map.end() ) {
            std::pair<int, double> tmp( elem.N, elem.sum );
            map.insert( std::pair<double, std::pair<int, double>>( elem.z, tmp ) );
        } else {
            iterator->second.first += elem.N;
            iterator->second.second += elem.sum;
        }
    }
}


/********************************************************
 * Other functions                                       *
 ********************************************************/
void Map3to1to3::setVector( AMP::LinearAlgebra::Vector::shared_ptr result )
{
    d_ResultVector = subsetOutputVector( result );
    AMP_ASSERT( d_ResultVector );
}


std::multimap<double, double> Map3to1to3::buildMap( AMP::LinearAlgebra::Vector::const_shared_ptr,
                                                    const std::shared_ptr<AMP::Mesh::Mesh>,
                                                    const AMP::Mesh::MeshIterator & )
{
    AMP_ERROR( "buildMap should never be called for the BaseClass" );
    return std::multimap<double, double>();
}


void Map3to1to3::buildReturn( AMP::LinearAlgebra::Vector::shared_ptr,
                              const std::shared_ptr<AMP::Mesh::Mesh>,
                              const AMP::Mesh::MeshIterator &,
                              const std::map<double, double> & )
{
    AMP_ERROR( "buildReturn should never be called for the BaseClass" );
}
} // namespace AMP::Operator


/********************************************************
 * Explicit instantiations                               *
 ********************************************************/
#include "AMP/utils/AMP_MPI.I"
template void AMP::AMP_MPI::recv<AMP::Operator::Map3to1to3::comm_data>(
    AMP::Operator::Map3to1to3::comm_data *, int &, int, bool, int ) const;
template AMP::AMP_MPI::Request AMP::AMP_MPI::Isend<AMP::Operator::Map3to1to3::comm_data>(
    AMP::Operator::Map3to1to3::comm_data const *, int, int, int ) const;
