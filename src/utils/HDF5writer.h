#ifndef included_AMP_HDF5writer
#define included_AMP_HDF5writer


#include <memory>
#include <string>
#include <vector>

#include "AMP/utils/Writer.h"


namespace AMP::Utilities {


/**
 * \class HDF5writer
 * \brief A class used to abstract away reading/writing files for visualization
 * \details  This class provides routines for reading, accessing and writing meshes and vectors
 * using HDF5.  Note: for visualization an Xdmf file will also be written
 */
class HDF5writer : public AMP::Utilities::Writer
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

    /**
     * \brief    Function to register a mesh
     * \details  This function will register a mesh with the writer.
     *           Note: if mesh is a MultiMesh, it will register all sub meshes.
     * \param mesh  The mesh to register
     * \param level How many sub meshes do we want?
     *              0: Only register the local base meshes (advanced users only)
     *              1: Register current mesh only (default)
     *              2: Register all meshes (do not seperate for the ranks)
     *              3: Register all mesh pieces including the individual ranks

     * \param path  The directory path for the mesh.  Default is an empty string.
     */
    void registerMesh( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                       int level               = 1,
                       const std::string &path = std::string() ) override;

    /**
     * \brief    Function to register a vector
     * \details  This function will register a vector with the writer.
     * \param vec   The vector we want to write
     * \param mesh  The mesh we want to write the vector over.
     *              Note: the vector must completely cover the mesh for visualization.
     *              Note: mesh does not have to be previously registered with registerMesh.
     * \param type  The entity type we want to save (vertex, face, cell, etc.)
     *              Note: xdmf only supports writing one entity type.
     *              If the vector spans multiple entity type (eg cell+vertex)  the user should
     *              register the vector multiple times (one for each entity type).
     * \param name  Optional name for the vector.
     */
    void registerVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec,
                         std::shared_ptr<AMP::Mesh::Mesh> mesh,
                         AMP::Mesh::GeomType type,
                         const std::string &name = "" ) override;

    /**
     * \brief    Function to register a vector
     * \details  This function will register a vector with the writer.
     *     This version of registerVector only stores the raw data.
     *     It is not associated with a mesh.
     * \param vec   The vector we want to write
     * \param name  Optional name for the vector.
     */
    void registerVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec,
                         const std::string &name = "" ) override;

    /**
     * \brief    Function to register a matrix
     * \details  This function will register a matrix with the writer.
     *     This version of registerMatrix only stores the raw data..
     *     It is not associated with a mesh.
     * \param mat   The matrix we want to write
     * \param name  Optional name for the vector.
     */
    void registerMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> mat,
                         const std::string &name = "" ) override;


private:
    typedef std::shared_ptr<AMP::Mesh::Mesh> MeshData;

    struct VectorData {
        std::string name;
        std::shared_ptr<AMP::LinearAlgebra::Vector> vec;
        AMP::Mesh::GeomType type;
        MeshData *mesh;
        VectorData() : type( static_cast<AMP::Mesh::GeomType>( 0xFF ) ), mesh( nullptr ) {}
    };

    struct MatrixData {
        std::string name;
        std::shared_ptr<AMP::LinearAlgebra::Matrix> mat;
    };

private:
    std::vector<MeshData> d_mesh;
    std::vector<VectorData> d_vec;
    std::vector<MatrixData> d_mat;
};

} // namespace AMP::Utilities

#endif
