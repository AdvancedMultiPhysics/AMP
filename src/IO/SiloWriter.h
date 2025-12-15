#ifndef included_AMP_SiloIO
#define included_AMP_SiloIO

#include "AMP/AMP_TPLs.h"
#include "AMP/IO/Writer.h"
#include "AMP/mesh/Mesh.h"

#include <map>
#include <set>
#include <sstream>
#include <string.h>
#include <vector>

#ifdef AMP_USE_SILO
    #include <silo.h>
#endif


namespace AMP::IO {


/**
 * \class SiloIO
 * \brief A class used to abstract away reading/writing files for visualization
 * \details  This class provides routines for reading, accessing and writing meshes and vectors
 * using silo.
 */
class SiloIO : public AMP::IO::Writer
{
public:
    //!  Default constructor
    SiloIO();

    //!  Default destructor
    virtual ~SiloIO();

    //! Function to get the writer properties
    WriterProperties getProperties() const override;

    //!  Function to read a file
    void readFile( const std::string &fname ) override;

    /**
     * \brief    Function to write a file
     * \details  This function will write a file with all mesh/vector data that
     *    was registered.  If the filename included a relative or absolute path,
     *    the directory structure will be created.
     * \param fname         The filename to use
     * \param iteration     The iteration number
     * \param time          The current simulation time
     */
    void writeFile( const std::string &fname, size_t iteration, double time = 0 ) override;

private:
// Function to write a single mesh
#ifdef AMP_USE_SILO
    void writeMesh( DBfile *file, const baseMeshData &data, int cycle, double time );
#endif

    // Function to write the summary file (the file should already be created, ready to reopen)
    // This function requires global communication
    void writeSummary( const std::string &filename, int cycle, double time );

    // The dimension
    int d_dim;
};

} // namespace AMP::IO

#endif
