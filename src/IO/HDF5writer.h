#ifndef included_AMP_HDF5writer
#define included_AMP_HDF5writer


#include <memory>
#include <string>
#include <vector>

#include "AMP/IO/HDF.h"
#include "AMP/IO/Writer.h"
#include "AMP/IO/Xdmf.h"


// Forward declerations
namespace AMP::Mesh {
class BoxMesh;
}


namespace AMP::IO {


/**
 * \class HDF5writer
 * \brief A class used to abstract away reading/writing files for visualization
 * \details  This class provides routines for reading, accessing and writing meshes and vectors
 * using HDF5.  Note: for visualization an Xdmf file will also be written
 */
class HDF5writer : public AMP::IO::Writer
{
public:
    //!  Default constructor
    HDF5writer();

    //!  Default destructor
    virtual ~HDF5writer();

    //! Delete copy constructor
    HDF5writer( const HDF5writer & ) = delete;

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

    //! Register arbitrary data
    void registerData( std::function<void( hid_t, std::string, Xdmf & )> fun );


private:
    Xdmf::MeshData
    writeMesh( hid_t fid, const std::string &filename, const baseMeshData &mesh ) const;
    Xdmf::MeshData
    writeDefaultMesh( hid_t fid, const std::string &filename, const baseMeshData &mesh ) const;
    Xdmf::MeshData
    writeBoxMesh( hid_t fid, const std::string &filename, const baseMeshData &mesh ) const;

private:
    std::vector<std::function<void( hid_t, std::string, Xdmf & )>> d_fun;
};

} // namespace AMP::IO

#endif
