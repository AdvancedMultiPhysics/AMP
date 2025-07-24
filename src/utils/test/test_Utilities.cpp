#include "UtilityHelpers.h"

#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"

#include "StackTrace/StackTrace.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <errno.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>


// Run the tests
int main( int argc, char *argv[] )
{
    // Control the behavior of the startup
    AMP::AMPManagerProperties startup_properties;

    // Start AMP
    AMP::AMPManager::startup( argc, argv, startup_properties );
    int num_failed = 0;

    // Limit the scope of variables
    {
        int rank = AMP::AMP_MPI( AMP_COMM_WORLD ).getRank();

        // Create the unit test
        AMP::UnitTest ut;

        // Print the banner
        AMP::Utilities::printBanner();

        // Test enable_shared_from_this
        test_shared_from_this( ut );

        // Check the OS
        constexpr auto OS = AMP::Utilities::getOS();
        if constexpr ( OS == AMP::Utilities::OS::Linux )
            ut.passes( "OS: Linux" );
        else if constexpr ( OS == AMP::Utilities::OS::Windows )
            ut.passes( "OS: Windows" );
        else if constexpr ( OS == AMP::Utilities::OS::macOS )
            ut.passes( "OS: macOS" );
        else
            ut.failure( "Unknown OS" );

        // Test typeid
        testTypeID( ut );

        // Try converting an int to a string
        if ( AMP::Utilities::intToString( 37, 0 ) == "37" &&
             AMP::Utilities::intToString( 37, 3 ) == "037" )
            ut.passes( "Convert int to string" );
        else
            ut.failure( "Convert int to string" );

        // Test approx_equal
        testApproxEqualInt<int>( ut );
        testApproxEqualInt<unsigned int>( ut );
        testApproxEqualInt<size_t>( ut );
        testApproxEqual<float>( ut );
        testApproxEqual<double>( ut );

        // Test interpolations
        test_interp( ut );

        // Test quicksort / quickselect
        testQuickSort( ut );
        testQuickSelect( ut );

        // Test the hash key
        unsigned int key = AMP::Utilities::hash_char( "test" );
        PASS_FAIL( key == 2087956275, "hash 'test'" );

        // Test the factor / primes function
        testPrimes( ut );

        // Test getting the current call stack
        double ts1      = AMP::Utilities::time();
        auto call_stack = get_call_stack();
        double ts2      = AMP::Utilities::time();
        if ( rank == 0 ) {
            std::cout << "Call stack:" << std::endl;
            for ( auto &elem : call_stack )
                std::cout << "   " << elem.print() << std::endl;
            std::cout << "Time to get call stack: " << ts2 - ts1 << std::endl;
        }
        if ( !call_stack.empty() ) {
            ut.passes( "non empty call stack" );
            bool pass = false;
            if ( call_stack.size() > 1 ) {
                if ( call_stack[1].print().find( "get_call_stack" ) != std::string::npos )
                    pass = true;
            }
            if ( pass )
                ut.passes( "call stack decoded function symbols" );
            else
                ut.expected_failure( "call stack was unable to decode function symbols" );
        } else {
            ut.failure( "non empty call stack" );
        }
        ts1        = AMP::Utilities::time();
        auto trace = StackTrace::backtrace();
        ts2        = AMP::Utilities::time();
        std::cout << "Time to get backtrace: " << ts2 - ts1 << std::endl;

        // Test getting the executable
        std::string exe = StackTrace::getExecutable();
        if ( rank == 0 )
            std::cout << "Executable: " << exe << std::endl;
        PASS_FAIL( exe.find( "test_Utilities" ) != std::string::npos, "getExecutable" );

        // Test filesystem routines
        testFileSystem( ut );

        // Test catching an error
        try {
            AMP_ERROR( "test_error" );
            ut.failure( "Failed to catch error" );
        } catch ( const StackTrace::abort_error &err ) {
            PASS_FAIL( err.message == "test_error", "Catch error" );
        } catch ( std::exception &err ) {
            ut.failure( "Caught unknown exception type" );
        }

        // Test precision of different types
        test_precision<int32_t>( ut );
        test_precision<int64_t>( ut );
        test_precision<float>( ut );
        test_precision<double>( ut );
        test_precision<long double>( ut );

        // Test printing a warning
        AMP_WARNING( "Testing warning" );

        // Test errno
        errno         = ETXTBSY;
        auto errorMsg = AMP::Utilities::getLastErrnoString();
        PASS_FAIL( errorMsg == "Text file busy", "errno" );

        // Test demangle
        auto mangled   = "_Z3fooPci";
        auto demangled = AMP::Utilities::demangle( mangled );
        std::cout << "demangled: " << demangled << std::endl;

        // Finished testing, report the results
        ut.report();
        num_failed = ut.NumFailGlobal();
    }

    // Shutdown
    AMP::AMPManager::shutdown();

    // Finished successfully
    return num_failed;
}
