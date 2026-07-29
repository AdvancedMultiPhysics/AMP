#ifndef included_AMP_AsciiWriter
#define included_AMP_AsciiWriter

#include <map>
#include <set>
#include <sstream>
#include <string.h>
#include <vector>

#include "AMP/IO/Writer.h"


namespace AMP::IO {


/**
 * \class AsciiWriter
 * \brief A class used to abstract away reading/writing files.
 * \details  This class provides routines for reading, accessing and writing vectors
 *    and matrices using a simple ASCII format.
 *    Note: this format only supports a decomposition of 1, and will write all data using rank0
 */
class AsciiWriter : public AMP::IO::Writer
{
public:
    //!  Default constructor
    AsciiWriter( const WriterParameters &properties = WriterParameters() );

    //!  Default destructor
    virtual ~AsciiWriter();

    //! Function to get the writer properties
    WriterProperties getProperties() const override;

    //!  Function to read a file
    void readFile( const std::string &fname ) override;

    //!  Function to write a file
    virtual void
    writeFile( const std::string &fname, size_t iteration_count, double time = 0 ) override;

private:
    static std::shared_ptr<const AMP::LinearAlgebra::Vector>
    sendVecToRoot( std::shared_ptr<const AMP::LinearAlgebra::Vector> src_vec, const AMP_MPI &comm );

    static void sendRowToRoot( std::shared_ptr<const AMP::LinearAlgebra::Matrix> mat,
                               const AMP_MPI &d_comm,
                               int row,
                               std::vector<size_t> &col,
                               std::vector<double> &data );
};

} // namespace AMP::IO

#endif
