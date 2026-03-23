// This program simulates the load balance with a given input file on a given number of processors

#include "AMP/mesh/loadBalance/loadBalanceSimulator.h"
#include "AMP/utils/Database.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>


// Main function
int run( int N_procs, const std::string &filename, double ratio )
{
    // Simulate loading the mesh
    auto input_db = AMP::Database::parseInputFile( filename );
    auto database = input_db->getDatabase( "Mesh" );
    auto t1       = std::chrono::system_clock::now();
    AMP::Mesh::loadBalanceSimulator mesh( database );
    auto t2 = std::chrono::system_clock::now();
    mesh.setProcs( N_procs );
    auto t3      = std::chrono::system_clock::now();
    int create   = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    int setProcs = std::chrono::duration_cast<std::chrono::milliseconds>( t3 - t2 ).count();

    // Print the results of the load balance
    mesh.print();

    // Print the time required
    printf( "time (create) = %i ms\n", create );
    printf( "time (setProcs) = %i ms\n", setProcs );

    // Get the worst and average element count
    auto cost  = mesh.getRankCost();
    double max = 0, min = 1e100, avg = 0;
    for ( auto x : cost ) {
        avg += x;
        max = std::max( max, x );
        min = std::min( min, x );
    }
    avg /= cost.size();

    // Print the errors and return
    int N_errors = 0;
    if ( max > ratio * avg ) {
        N_errors++;
        std::cout << "load balance failed quality limits" << std::endl;
    }
    if ( min == 0 ) {
        N_errors++;
        std::cout << "load balance failed with empty rank" << std::endl;
    }
    if ( create > 15e36 || setProcs > 15e3 ) {
        N_errors++;
        std::cout << "load balance failed run time limits" << std::endl;
    }
    return N_errors;
}


// Main function
int main( int argc, char **argv )
{
    // Load the inputs
    if ( argc < 3 ) {
        std::cout << "Error calling test_LoadBalancer, format should be:" << std::endl;
        std::cout << "   ./test_LoadBalancer  N_procs  input_file" << std::endl;
        return -1;
    }
    int N_procs   = std::atoi( argv[1] );
    auto filename = argv[2];
    double ratio  = 2.0;
    if ( argc > 3 )
        ratio = atof( argv[3] );

    // Run the problem
    int N_errors = run( N_procs, filename, ratio );

    // Shutdown AMP
    return N_errors;
}
