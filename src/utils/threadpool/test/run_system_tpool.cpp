#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/threadpool/ThreadPool.h"

#include <algorithm>
#include <fstream>


// Work item to call system command
class SystemWorkItem final : public AMP::ThreadPool::WorkItemRet<std::string>
{
public:
    explicit SystemWorkItem( std::string cmd ) : d_cmd( std::move( cmd ) ), d_started( false ) {}
    ~SystemWorkItem() override = default;
    void run() override
    {
        d_started = true;
        d_t1      = std::chrono::high_resolution_clock::now();
        d_t2      = d_t1;
        d_result  = d_cmd + "\n";
        int exit  = 0;
        d_result += AMP::Utilities::exec( d_cmd, exit );
    }
    void checkRunTime() const
    {
        if ( !d_started )
            return;
        auto t3 = std::chrono::high_resolution_clock::now();
        int dt  = std::chrono::duration_cast<std::chrono::seconds>( t3 - d_t2 ).count();
        if ( dt >= 3600 ) {
            int dt2     = std::chrono::duration_cast<std::chrono::minutes>( t3 - d_t1 ).count();
            int hours   = dt2 / 60;
            int minutes = dt2 % 60;
            std::cout << "Command running for " << hours << ":" << minutes << "\n";
            std::cout << d_result << std::endl;
            d_t2 = t3;
        }
    }

private:
    std::string d_cmd;
    bool d_started;
    mutable std::chrono::time_point<std::chrono::high_resolution_clock> d_t1, d_t2;
};


// Run the commands across all threads/ranks
void run( int N_threads, const std::string &filename )
{
    // Create the thread pool
    AMP::ThreadPool tpool( N_threads, "none", std::vector<int>(), 100000 );
    tpool.setMaxWaitTimeDebug( 100000 );

    // Read the input file
    std::vector<std::string> commands;
    std::ifstream file( filename );
    std::string cmd;
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    int rank = globalComm.getRank();
    int size = globalComm.getSize();
    for ( int i = 0; std::getline( file, cmd ); i++ ) {
        if ( i % size == rank )
            commands.push_back( cmd );
    }

    // Add the commands to the thread pool
    std::vector<AMP::ThreadPool::WorkItem *> work( commands.size(), nullptr );
    for ( size_t i = 0; i < commands.size(); i++ )
        work[i] = new SystemWorkItem( commands[i] );
    auto ids = tpool.add_work( work );
    work.clear();
    commands.clear();

    // Wait for items to finish printing the results as they finish
    while ( !ids.empty() ) {
        // Wait for some work to finish
        auto index = tpool.wait_some( 1, ids, 900 );
        // Print the output of any work that is complete
        for ( int i : index ) {
            auto result = tpool.getFunctionRet<std::string>( ids[i] );
            std::cout << result << std::endl;
        }
        // Erase the completed work
        for ( auto it = index.rbegin(); it != index.rend(); ++it ) {
            std::swap( ids[*it], ids.back() );
            ids.pop_back();
        }
        // For each work item, check how long it has been running
        for ( const auto &id : ids ) {
            auto item = reinterpret_cast<const SystemWorkItem *>( id.getWork() );
            item->checkRunTime();
        }
        std::cout.flush();
    }
}


// Main program
int main( int argc, char *argv[] )
{
    // Check the inputs
    if ( argc != 3 ) {
        std::cerr << "Invalid usage:\n";
        std::cerr << "   run_system_tpool N_threads filename\n";
        return -1;
    }

    // Initialize
    AMP::AMPManager::startup( argc, argv );

    // Run the commands
    run( std::stoi( argv[1] ), argv[2] );

    // Shutdown
    AMP::AMPManager::shutdown();
    return 0;
}
