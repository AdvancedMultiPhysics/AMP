#ifndef included_AMP_StackTrace
#define included_AMP_StackTrace

#include "utils/PIO.h"

#include <functional>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>


namespace AMP {
namespace StackTrace {


struct stack_info {
    void *address;
    void *address2;
    std::string object;
    std::string function;
    std::string filename;
    int line;
    //! Default constructor
    stack_info() : address( nullptr ), address2( nullptr ), line( 0 ) {}
    //! Print the stack info
    std::string print() const;
};


//! Function to return the current call stack
std::vector<stack_info> getCallStack();


//! Function to return the current call stack
std::vector<void *> backtrace();


//! Function to return the stack info for a given address
stack_info getStackInfo( void *address );


//! Function to return the stack info for a given address
std::vector<stack_info> getStackInfo( const std::vector<void *> &address );


/*!
 * Return the symbols from the current executable (not availible for all platforms)
 * @return      Returns 0 if sucessful
 */
int getSymbols( std::vector<void *> &address,
                std::vector<char> &type,
                std::vector<std::string> &obj );


/*!
 * Return the name of the executable
 * @return      Returns the name of the executable (usually the full path)
 */
std::string getExecutable();


/*!
 * Return the search path for the symbols
 * @return      Returns the search path for the symbols
 */
std::string getSymPaths();


//!< Terminate type
enum class terminateType { signal, exception };

/*!
 * Set the error handlers
 * @param[in]   Function to terminate the program: abort(msg,type)
 */
void setErrorHandlers( std::function<void( std::string, terminateType )> abort );


} // namespace StackTrace
} // namespace AMP

#endif
