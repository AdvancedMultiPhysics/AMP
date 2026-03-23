#include "AMP/mesh/loadBalance/loadBalanceSimulator.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/utils/Utilities.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>


namespace AMP::Mesh {


template<class TYPE>
static inline TYPE min( const std::vector<TYPE> &x )
{
    TYPE y = std::numeric_limits<TYPE>::max();
    for ( auto i : x )
        y = std::min( y, i );
    return y;
}
template<class TYPE>
static inline TYPE max( const std::vector<TYPE> &x )
{
    TYPE y = 0;
    for ( auto i : x )
        y = std::max( y, i );
    return y;
}
template<class TYPE>
static inline TYPE avg( const std::vector<TYPE> &x )
{
    TYPE y = 0;
    for ( auto i : x )
        y += i;
    return y / x.size();
}


template<class TYPE>
static inline int findMin( const std::vector<TYPE> &x )
{
    AMP_ASSERT( !x.empty() );
    int i  = 0;
    TYPE y = x[0];
    for ( int j = 0; j < (int) x.size(); j++ ) {
        if ( x[j] < y ) {
            i = j;
            y = x[j];
        }
    }
    return i;
}
template<class TYPE>
static inline int findMax( const std::vector<TYPE> &x )
{
    AMP_ASSERT( !x.empty() );
    int i  = 0;
    TYPE y = x[0];
    for ( int j = 0; j < (int) x.size(); j++ ) {
        if ( x[j] > y ) {
            i = j;
            y = x[j];
        }
    }
    return i;
}


/********************************************************
 * Divide M items with weights into N groups ( M > N )   *
 ********************************************************/
static std::vector<int> divideGroups( int N, const std::vector<double> &x )
{
    // Divide the entries into N groups minimizing the maximum cost
    AMP_ASSERT( (int) x.size() >= N );
    std::vector<std::pair<double, int>> ids( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        ids[i] = std::make_pair( x[i], (int) i );
    AMP::Utilities::quicksort( ids );
    std::vector<double> cost( N, 0 );
    std::vector<int> groups( x.size(), -1 );
    while ( !ids.empty() ) {
        // Find the entry with the lowest cost
        int i = findMin( cost );
        // Add the largest remaining entry to the lowest cost group
        cost[i] += ids.back().first;
        groups[ids.back().second] = i;
        ids.resize( ids.size() - 1 );
    }
    return groups;
}


/********************************************************
 * Check if all the meshes in a multimesh are the same   *
 ********************************************************/
static bool allMeshesMatch( std::shared_ptr<const AMP::Database> db )
{
    AMP_ASSERT( db->getString( "MeshType" ) == "Multimesh" );
    auto prefix      = db->getString( "MeshDatabasePrefix" );
    auto arrayPrefix = db->getString( "MeshArrayDatabasePrefix" );
    auto keys        = db->getAllKeys();
    int N_unique     = 0;
    for ( auto &key : keys ) {
        if ( key.find( prefix ) == 0 || key.find( arrayPrefix ) == 0 )
            N_unique++;
    }
    return N_unique == 1;
}


/************************************************************
 * Constructors                                              *
 ************************************************************/
static constexpr int MaxProcs = 2000000; // Should be a large number but still allow summation
loadBalanceSimulator::loadBalanceSimulator()
    : d_cost( 0 ),
      d_maxCostRank( 0 ),
      d_max_procs( 0 ),
      d_method( 0 ),
      d_allEqual( true ),
      d_begin( 0 ),
      d_end( 0 )
{
}
loadBalanceSimulator::loadBalanceSimulator( std::shared_ptr<const AMP::Database> db )
    : d_cost( 0 ),
      d_maxCostRank( 0 ),
      d_max_procs( 0 ),
      d_method( 0 ),
      d_allEqual( true ),
      d_begin( 0 ),
      d_end( 0 )
{
    // Get required values from the parameters
    AMP_ASSERT( db );
    AMP_INSIST( db->keyExists( "MeshType" ), "MeshType must exist in input database" );
    AMP_INSIST( db->keyExists( "MeshName" ), "MeshName must exist in input database" );
    d_name    = db->getString( "MeshName" );
    auto type = db->getString( "MeshType" );
    // Simulate the load process
    if ( type == std::string( "Multimesh" ) ) {
        // Create a database for each mesh within the multimesh and generate the load balancers
        auto meshDatabases = MultiMesh::createDatabases( db );
        size_t N           = meshDatabases.size();
        if ( N == 0 )
            return;
        if ( allMeshesMatch( db ) ) {
            // Every submesh is identical
            auto meshDatabases = MultiMesh::createDatabases( db );
            d_submeshes.resize( N );
            d_submeshes[0] = loadBalanceSimulator( meshDatabases[0] );
            for ( size_t i = 1; i < N; i++ ) {
                d_submeshes[i]        = d_submeshes[0];
                d_submeshes[i].d_name = meshDatabases[i]->getString( "MeshName" );
            }
        } else {
            // The submeshes are "potentially" different
            d_submeshes.resize( N );
            for ( size_t i = 0; i < N; i++ )
                d_submeshes[i] = loadBalanceSimulator( meshDatabases[i] );
            for ( size_t i = 1; i < d_submeshes.size(); i++ )
                d_allEqual = d_allEqual && d_submeshes[0].d_cost == d_submeshes[i].d_cost;
        }
        // Compute the total cost / max procs
        d_method = db->getWithDefault<int>( "LoadBalanceMethod", 2 );
        if ( d_allEqual ) {
            d_cost = N * d_submeshes[0].d_cost;
            if ( d_method == 0 )
                d_max_procs = d_submeshes[0].d_max_procs;
            else
                d_max_procs = N * d_submeshes[0].d_max_procs;
        } else {
            d_cost = 0;
            for ( size_t i = 0; i < N; i++ )
                d_cost += d_submeshes[i].d_cost;
            if ( d_method == 0 ) {
                d_max_procs = MaxProcs;
                for ( size_t i = 0; i < N; i++ )
                    d_max_procs = std::min( d_max_procs, d_submeshes[i].d_max_procs );
            } else {
                d_max_procs = 0;
                for ( size_t i = 0; i < N; i++ )
                    d_max_procs += d_submeshes[i].d_max_procs;
            }
        }
        d_max_procs = std::min( d_max_procs, MaxProcs );
    } else {
        d_submeshes.resize( 0 );
        auto params = std::make_shared<MeshParameters>( db->cloneDatabase() );
        d_max_procs = Mesh::maxProcs( params );
        d_cost      = Mesh::estimateMeshSize( params );
    }
    d_cost *= db->getWithDefault<double>( "Weight", 1.0 );
    d_maxCostRank = d_cost;
    AMP_ASSERT( d_max_procs > 0 );
    AMP_ASSERT( d_cost > 0 );
}
loadBalanceSimulator::loadBalanceSimulator( double cost, int maxProc, const std::string &name )
    : d_name( name ),
      d_cost( cost ),
      d_maxCostRank( cost ),
      d_max_procs( maxProc ),
      d_method( 0 ),
      d_allEqual( true ),
      d_begin( 0 ),
      d_end( 0 )
{
    AMP_ASSERT( d_cost > 0 );
    if ( d_max_procs == 0 )
        d_max_procs = MaxProcs;
    AMP_ASSERT( d_max_procs > 0 );
}
loadBalanceSimulator::loadBalanceSimulator( const std::vector<loadBalanceSimulator> &meshes,
                                            int method,
                                            const std::string &name )
    : d_name( name ),
      d_cost( 0 ),
      d_maxCostRank( 0 ),
      d_max_procs( 0 ),
      d_method( method ),
      d_allEqual( true ),
      d_begin( 0 ),
      d_end( 0 ),
      d_submeshes( meshes )
{
    for ( auto &mesh : d_submeshes )
        d_cost += mesh.d_cost;
    d_maxCostRank = d_cost;
    if ( d_method == 0 ) {
        d_max_procs = MaxProcs;
        for ( auto &mesh : d_submeshes )
            d_max_procs = std::min( d_max_procs, mesh.d_max_procs );
    } else {
        d_max_procs = 0;
        for ( auto &mesh : d_submeshes )
            d_max_procs += mesh.d_max_procs;
        d_max_procs = std::min( d_max_procs, MaxProcs );
    }
    AMP_ASSERT( d_max_procs > 0 );
}


/************************************************************
 * Function to return the min, max, and avg # of elements    *
 ************************************************************/
std::vector<double> loadBalanceSimulator::getRankCost() const
{
    std::vector<double> cost( d_end, 0 );
    addRankCost( cost );
    return cost;
}
void loadBalanceSimulator::addRankCost( std::vector<double> &cost ) const
{
    if ( !d_submeshes.empty() ) {
        for ( const auto &mesh : d_submeshes )
            mesh.addRankCost( cost );
    } else {
        double N = d_cost / nRanks();
        for ( int rank = d_begin; rank < d_end; rank++ ) {
            AMP_ASSERT( rank < (int) cost.size() );
            cost[rank] += N;
        }
    }
}


/************************************************************
 * Function to print the load balance                        *
 ************************************************************/
void loadBalanceSimulator::print( uint8_t detail, uint8_t indent_N )
{
    int N_procs = nRanks();
    if ( detail == 0 ) {
        detail = 1;
        if ( N_procs < 1000 )
            detail |= 2;
        if ( getMeshCount() < 100 )
            detail |= 4;
    }
    char indent[256] = { 0 };
    memset( indent, 0x20, indent_N );
    std::vector<double> cost;
    if ( detail & 0x2 ) {
        // Print the global load balance
        if ( cost.empty() )
            cost = getRankCost();
        std::cout << indent << "Rank, Cost:" << std::endl;
        int N_line = 16;
        for ( int i = 0; i < ( N_procs + N_line - 1 ) / N_line; i++ ) {
            std::cout << indent;
            for ( int j = i * N_line; j < std::min( ( i + 1 ) * N_line, N_procs ); j++ )
                std::cout << std::setw( 8 ) << j;
            std::cout << std::endl;
            std::cout << indent;
            for ( int j = i * N_line; j < std::min( ( i + 1 ) * N_line, N_procs ); j++ )
                std::cout << std::setw( 8 ) << static_cast<int>( cost[j] );
            std::cout << std::endl << std::endl;
        }
    }
    if ( detail & 0x4 ) {
        // Print the rank info
        printf( "%s%s (%0.0f): %i", indent, d_name.data(), d_cost, N_procs );
        if ( N_procs > 0 ) {
            printf( " (%i", d_begin );
            for ( int i = d_begin + 1; i < d_end && i < 10; i++ )
                printf( ",%i", i );
            if ( N_procs > 10 )
                printf( ",..." );
            printf( ")" );
        }
        printf( "\n" );
        for ( auto &elem : d_submeshes )
            elem.print( 4, indent_N + 3 );
    }
    if ( detail & 0x1 ) {
        if ( cost.empty() )
            cost = getRankCost();
        std::cout << std::endl;
        std::cout << "Min = " << static_cast<int>( min( cost ) ) << std::endl;
        std::cout << "Max = " << static_cast<int>( max( cost ) ) << std::endl;
        std::cout << "Avg = " << static_cast<int>( avg( cost ) ) << std::endl;
        std::cout << std::endl;
    }
}


/************************************************************
 * Get the number of base meshes                             *
 ************************************************************/
int loadBalanceSimulator::getMeshCount() const
{
    int count = 0;
    for ( const auto &mesh : d_submeshes )
        count += mesh.getMeshCount();
    return std::max( count, 1 );
}


/************************************************************
 * Get the ranks                                             *
 ************************************************************/
std::vector<int> loadBalanceSimulator::getRanks() const
{
    std::vector<int> ranks( d_end - d_begin );
    for ( int i = 0; i < nRanks(); i++ )
        ranks[i] = d_begin + i;
    return ranks;
}


/************************************************************
 * Set the processors                                        *
 ************************************************************/
void loadBalanceSimulator::setProcs( int N_procs ) { setRanks( 0, N_procs ); }
static void setCost0( const std::vector<int> &N,
                      const std::vector<loadBalanceSimulator> &mesh,
                      std::vector<double> &cost )
{
    int cost0 = 0;
    for ( size_t i = 0; i < N.size(); i++ ) {
        if ( N[i] == 0 )
            cost0 += mesh[i].getCost();
    }
    for ( size_t i = 0; i < N.size(); i++ ) {
        if ( N[i] == 0 )
            cost[i] = cost0;
    }
}
void loadBalanceSimulator::loadBalance( int N_proc, std::vector<int> &N_mesh )
{
    // Initialize the cost
    int N  = N_proc;
    int N0 = 0;
    std::vector<double> cost( d_submeshes.size(), 0 );
    for ( size_t i = 0; i < N_mesh.size(); i++ ) {
        if ( N_mesh[i] == 0 ) {
            N0++;
        } else {
            N -= N_mesh[i];
            d_submeshes[i].setProcs( N_mesh[i] );
            cost[i] = d_maxCostRank;
        }
    }
    setCost0( N_mesh, d_submeshes, cost );
    // Add processors until we have used all processors
    while ( N - std::min( N0, 1 ) > 0 ) {
        int i = findMax( cost );
        if ( N_mesh[i] == 0 ) {
            int k    = 0;
            double c = 0;
            for ( size_t j = 0; j < N_mesh.size(); j++ ) {
                if ( N_mesh[j] == 0 && c < d_submeshes[j].d_maxCostRank ) {
                    k = j;
                    c = d_submeshes[j].d_maxCostRank;
                }
            }
            N_mesh[k] = 1;
            d_submeshes[k].setProcs( 1 );
            cost[k] = d_submeshes[k].d_cost;
            setCost0( N_mesh, d_submeshes, cost );
            N0--;
            N--;
        } else {
            d_submeshes[i].addRank();
            cost[i] = d_submeshes[i].d_maxCostRank;
            N_mesh[i]++;
            N--;
        }
    }
    AMP_ASSERT( N == ( ( N0 > 0 ) ? 1 : 0 ) );
}
void loadBalanceSimulator::addRank()
{
    // Add the rank locally
    if ( nRanks() >= d_max_procs )
        return;
    d_end++;
    if ( d_submeshes.size() <= 1 || nRanks() == 1 || d_method == 0 ) {
        for ( auto &mesh : d_submeshes )
            mesh.addRank();
    } else if ( ( d_method == 1 || d_method == 2 ) && ( nRanks() > (int) d_submeshes.size() ) ) {
        // Initialize the cost
        std::vector<int> N( d_submeshes.size(), 0 );
        std::vector<double> cost( d_submeshes.size(), 0 );
        for ( size_t i = 0; i < N.size(); i++ ) {
            N[i] = d_submeshes[i].nRanks();
            if ( N[i] > 0 )
                cost[i] = d_maxCostRank;
        }
        setCost0( N, d_submeshes, cost );
        // Add the processor to the highest cost mesh
        int i = findMax( cost );
        if ( N[i] == 0 ) {
            d_submeshes[i].setRanks( 0, 1 );
        } else {
            d_submeshes[i].addRank();
        }
    } else {
        setRanks( d_begin, d_end );
    }
    if ( d_submeshes.empty() )
        d_maxCostRank = d_cost / nRanks();
    else
        d_maxCostRank = max( getRankCost() );
}
void loadBalanceSimulator::addRanks( int N )
{
    AMP_ASSERT( N >= 0 );
    for ( int i = 0; i < N; i++ )
        addRank();
}
void loadBalanceSimulator::setRanks( int begin, int end )
{
    // Set the processors for this mesh
    d_begin = begin;
    d_end   = end;
    AMP_ASSERT( nRanks() > 0 );
    if ( nRanks() > d_max_procs )
        d_end = d_begin + d_max_procs;
    int N_proc = nRanks();
    int N_mesh = d_submeshes.size();
    // Choose the load balance method
    if ( N_mesh <= 1 || N_proc == 1 || d_method == 0 ) {
        // Everybody is on the same communicator
        for ( auto &mesh : d_submeshes )
            mesh.setRanks( d_begin, d_end );
    } else if ( d_allEqual ) {
        // Special case where all children are the same size
        if ( N_proc <= N_mesh ) {
            for ( int i = 0; i < N_mesh; i++ ) {
                int j = ( i * N_proc ) / N_mesh;
                d_submeshes[i].setRanks( d_begin + j );
            }
        } else {
            int N0     = N_proc / N_mesh;
            auto mesh1 = d_submeshes[0];
            auto mesh2 = d_submeshes[0];
            mesh1.setRanks( 0, N0 );
            mesh2.setRanks( 0, N0 + 1 );
            for ( int i = 0, k = d_begin, Np = N_proc; i < N_mesh; i++ ) {
                int Ng = N_mesh - i;
                int N  = ( Np / Ng );
                if ( N == N0 )
                    d_submeshes[i].copyRanks( mesh1, k );
                else if ( N == N0 + 1 )
                    d_submeshes[i].copyRanks( mesh2, k );
                else
                    AMP_ERROR( "Internal error" );
                Np -= N;
                k += N;
            }
        }
    } else if ( d_method == 1 ) {
        // We want to split the meshes onto independent processors
        if ( N_proc == N_mesh ) {
            // Special case where the # of meshes == # of ranks
            for ( int i = 0; i < N_mesh; i++ )
                d_submeshes[i].setRanks( d_begin + i );
        } else if ( N_proc > N_mesh ) {
            // We have more processors than meshes
            std::vector<int> N( N_mesh, 0 );
            // Perform the initial load balance using 80% of the processors
            int NP = 0.8 * N_proc;
            NP     = std::max( { NP, N_proc - 10 * N_mesh, N_mesh } );
            for ( size_t i = 0; i < N.size(); i++ )
                N[i] = std::max<int>( 1, NP * d_submeshes[i].d_cost / d_cost );
            // Load balance
            loadBalance( N_proc, N );
            // Set the ranks
            for ( int i = 0, k = d_begin; i < N_mesh; i++ ) {
                d_submeshes[i].setRanks( k, k + N[i] );
                k += N[i];
            }
        } else {
            // The number of meshes is <= the number of processors
            std::vector<double> cost( N_mesh, 0 );
            for ( int i = 0; i < N_mesh; i++ )
                cost[i] = d_submeshes[i].d_cost;
            auto groups = divideGroups( N_proc, cost );
            for ( int i = 0; i < N_mesh; i++ )
                d_submeshes[i].setRanks( d_begin + groups[i] );
        }
    } else if ( d_method == 2 ) {
        // Try to achieve the best load balance while splitting the meshes if possible
        if ( N_proc < N_mesh ) {
            // The number of meshes is <= the number of processors
            std::vector<double> cost( N_mesh, 0 );
            for ( int i = 0; i < N_mesh; i++ )
                cost[i] = d_submeshes[i].d_cost;
            auto groups = divideGroups( N_proc, cost );
            for ( int i = 0; i < N_mesh; i++ )
                d_submeshes[i].setRanks( d_begin + groups[i] );
        } else {
            // Perform the initial load balance using 80% of the processors
            std::vector<int> N( N_mesh, 0 );
            int NP = 0.8 * ( N_proc - 1 );
            NP     = std::max( { NP, N_proc - 10 * N_mesh } );
            for ( size_t i = 0; i < N.size(); i++ )
                N[i] = NP * d_submeshes[i].d_cost / d_cost;
            // Load balance
            loadBalance( N_proc, N );
            // Set the ranks
            for ( int i = 0, k = d_begin; i < N_mesh; i++ ) {
                if ( N[i] == 0 ) {
                    d_submeshes[i].setRanks( d_end - 1 );
                } else {
                    d_submeshes[i].setRanks( k, k + N[i] );
                    k += N[i];
                }
            }
        }
    } else {
        AMP_ERROR( "Unknown method" );
    }
    if ( d_submeshes.empty() )
        d_maxCostRank = d_cost / N_proc;
    else
        d_maxCostRank = max( getRankCost() );
}


/************************************************************
 * Copy the processor ranks adjusting the offsets            *
 ************************************************************/
void loadBalanceSimulator::copyRanks( const loadBalanceSimulator &x, int offset )
{
    d_cost        = x.d_cost;
    d_maxCostRank = x.d_maxCostRank;
    d_max_procs   = x.d_max_procs;
    d_begin       = x.d_begin + offset;
    d_end         = x.d_end + offset;
    if ( !d_submeshes.empty() ) {
        AMP_ASSERT( d_method == x.d_method && d_allEqual == x.d_allEqual &&
                    d_submeshes.size() == x.d_submeshes.size() );
        for ( size_t i = 0; i < d_submeshes.size(); i++ )
            d_submeshes[i].copyRanks( x.d_submeshes[i], offset );
    }
}


} // namespace AMP::Mesh


/********************************************************
 * Explicit instantiations                               *
 ********************************************************/
#include "AMP/utils/Utilities.hpp"
template void AMP::Utilities::quicksort<std::pair<double, int>>( size_t, std::pair<double, int> * );
template void
AMP::Utilities::quicksort<std::pair<double, int>>( std::vector<std::pair<double, int>> & );
