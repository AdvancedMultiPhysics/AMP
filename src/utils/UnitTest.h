#ifndef included_AMP_AMPUnitTest
#define included_AMP_AMPUnitTest

#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>


namespace AMP {


class AMP_MPI;


/*!
 * @brief Class UnitTest is simple utility for running unit tests.
 * It provides basic routines for tracing success or failure of tests,
 * and reporting the results.
 * \par Code Sample:
 * \code
int main(int argc, char *argv[])
{
    AMP::AMPManager::startup(argc, argv);
    AMP::UnitTest ut;
    try {
        std::cout << "Testing tstOne" << std::endl;
        tstOne(&ut);
        ut.passes("Test XXX passed");
    } catch( ... ) {
        ut.failure("An unknown exception was thrown");
    }
    ut.report();
    return ut.NumFail();
}

void tstOne(AMP::UnitTest *ut)
{
    // Run the test code
    if ( problem1 ) {
        ut.failure("Problem 1 detected");
        return
    }
    // Finished running test
    ut.passes("Test XXX passed");
}
 * \endcode

 */
class UnitTest final
{
public:
    //! Constructor
    UnitTest();

    //! Destructor
    ~UnitTest();

    // Copy constructor
    UnitTest( const UnitTest & ) = delete;

    // Assignment operator
    UnitTest &operator=( const UnitTest & ) = delete;

    //! Indicate a passed test (thread-safe)
    void passes( std::string in );

    //! Indicate a failed test (thread-safe)
    void failure( std::string in );

    //! Indicate an expected failed test (thread-safe)
    void expected_failure( std::string in );

    //! Return the number of passed tests locally
    inline size_t NumPassLocal() const { return d_pass.size(); }

    //! Return the number of failed tests locally
    inline size_t NumFailLocal() const { return d_fail.size(); }

    //! Return the number of expected failed tests locally
    inline size_t NumExpectedFailLocal() const { return d_expected.size(); }

    //! Return the number of passed tests locally
    size_t NumPassGlobal() const;

    //! Return the number of failed tests locally
    size_t NumFailGlobal() const;

    //! Return the number of expected failed tests locally
    size_t NumExpectedFailGlobal() const;

    //! Return the tests passed locally
    inline const auto &getPass() const { return d_pass; }

    //! Return the number of failed tests locally
    inline const auto &getFail() const { return d_fail; }

    //! Return the number of expected failed tests locally
    inline const auto &getExpected() const { return d_expected; }

    /*!
     * Print a report of the passed and failed tests.
     * Note: This is a blocking call that all processors must execute together.
     * Note: Only rank 0 will print the messages (this is necessary as other ranks may not be able
     * to print correctly).
     * @param level     Optional integer specifying the level of reporting (default: 1)
     *                  0: Report the number of tests passed, failed, and expected failures.
     *                  1: Report the passed tests (if <=20) or number passed,
     *                     Report all failures,
     *                     Report the expected failed tests (if <=50) or the number passed.
     *                  2: Report the passed tests (if <=50)
     *                     Report all failures,
     *                     Report all expected
     *                  3: Report all passed, failed, and expected failed tests.
     * @param level     Remove duplicate messages.
     *                  If set, the total number of message will be unchanged but if printed
     *                  duplicate messages will be removed
     */
    void report( const int level = 1, bool removeDuplicates = true ) const;

    //! Clear the messages
    void reset();

    //! Make the unit test operator verbose?
    void verbose( bool verbose = true ) { d_verbose = verbose; }

private:
    std::vector<std::string> d_pass;
    std::vector<std::string> d_fail;
    std::vector<std::string> d_expected;
    bool d_verbose;
    mutable std::mutex d_mutex;
    std::unique_ptr<AMP::AMP_MPI> d_comm;
};


} // namespace AMP

#endif
