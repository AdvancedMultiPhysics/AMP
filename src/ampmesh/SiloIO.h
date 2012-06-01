#ifndef included_AMP_SiloIO
#define included_AMP_SiloIO

#include <string.h>
#include <sstream>
#include <vector>
#include <map>
#include <set>

#ifdef USE_SILO
    #include <silo.h>
#endif

#include "boost/smart_ptr/shared_ptr.hpp"
#include "ampmesh/Mesh.h"

#ifdef USE_AMP_VECTORS
    #include "vectors/Vector.h"
#endif


namespace AMP { 
namespace Mesh {


/**
 * \class SiloIO
 * \brief A class used to abstract away reading/writing files for visualization
 * \details  This class provides routines for reading, accessing and writing meshes and vectors
 * using silo.
 */
class SiloIO 
{
public:

    //!  Convenience typedef
    typedef boost::shared_ptr<AMP::Mesh::SiloIO>  shared_ptr;

    //!  Default constructor
    SiloIO();

    //!  Function to return the file extension
    std::string getExtension();

    /**
     * \brief   Function to set the file decomposition
     * \details This function will set the method used for file IO.  When writing files, 
     *    there are different decompositions that affect the performance and usability 
     *    of the output files.  By default, this writer will generate a single file.
     * \param decomposition   Decomposition method to use:
     *             1:  This will write all of the data to a single file.  
     *                 Note that this requires a serial write and will have the worst performance
     *             2:  Each processor will write a separate file and a separate 
     *                 summary file will be written.  Note that this will have better performance
     *                 at large scale, but will write many files simultaneously.  
     */
    void setDecomposition( int decomposition );

    //!  Function to read a file
    void  readFile( const std::string &fname );

    //!  Function to write a file
    void  writeFile( const std::string &fname, size_t iteration_count );

    /**
     * \brief    Function to register a mesh
     * \details  This function will register a mesh with the silo writer.  
     *           Note: if mesh is a MultiMesh, it will register all sub meshes.
     * \param mesh  The mesh to register
     * \param path  The directory path for the mesh.  Default is an empty string.
     */
    void registerMesh( AMP::Mesh::Mesh::shared_ptr mesh, std::string path=std::string() );

#ifdef USE_AMP_VECTORS
    /**
     * \brief    Function to register a vector
     * \details  This function will register a vector with the silo writer.  
     * \param vec   The vector we want to write
     * \param mesh  The mesh we want to write the vector over.
     *              Note: the vector must completely cover the mesh (silo limitiation).
     *              Note: mesh does not have to be previously registered with registerMesh.
     * \param type  The entity type we want to save (vertex, face, cell, etc.)
     *              Note: silo only supports writing one entity type.  If the vector
     *              spans multiple entity type (eg cell+vertex) the user should register
     *              the vector multiple times (one for each entity type).
     * \param name  Optional name for the vector.
     */
    void registerVector( AMP::LinearAlgebra::Vector::shared_ptr vec, AMP::Mesh::Mesh::shared_ptr mesh,
        AMP::Mesh::GeomType type, const std::string &name = "" );
#endif

private:

    // Structure used to hold data for the silo meshes
    struct siloBaseMeshData {
        AMP::Mesh::MeshID               id;         // Unique ID to identify the mesh
        AMP::Mesh::Mesh::shared_ptr     mesh;       // Pointer to the mesh
        int                             rank;       // Rank of the current processor on the mesh (used for name mangling)
        std::string                     meshName;   // Name of the mesh in silo
        std::string                     path;       // Path to the mesh in silo
        std::string                     file;       // File that will contain the mesh
        std::vector<std::string>        varName;    // List of the names of variables associated with each mesh
        std::vector<AMP::Mesh::GeomType> varType;   // List of the types of variables associated with each mesh
        std::vector<int> varSize;                   // Number of unknowns per point
        #ifdef USE_AMP_VECTORS
            std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vec; // List of the vectors associated with each mesh
        #endif
        // Function to count the number of bytes needed to pack the data (note: some info may be lost)
        size_t size();
        // Function to pack the data to a byte array (note: some info may be lost)
        void pack( char* );
        // Function to unpack the data from a byte array (note: some info may be lost)
        static siloBaseMeshData unpack( char* );
    };

    // Structure used to hold data for the silo multimeshes
    struct siloMultiMeshData {
        AMP::Mesh::MeshID               id;         // Unique ID to identify the mesh
        AMP::Mesh::Mesh::shared_ptr     mesh;       // Pointer to the mesh
        std::string                     name;       // Name of the multimesh in silo
        std::vector<siloBaseMeshData>   meshes;     // Base mesh info needed to construct the mesh data
        std::vector<std::string>        varName;    // List of the names of variables associated with each mesh
        // Function to count the number of bytes needed to pack the data
        size_t size();
        // Function to pack the data to a byte array
        void pack( char* );
        // Function to unpack the data from a byte array
        static siloMultiMeshData unpack( char* );
    };
    
    // Function to syncronize multimesh data across all processors
    void syncMultiMeshData( std::map<AMP::Mesh::MeshID,siloMultiMeshData> &data ) const;

    // Function to syncronize variable lists across all processors
    void syncVariableList( std::set<std::string> &data ) const;

    // Function to write a single mesh
#ifdef USE_SILO
    void writeMesh( DBfile *file, const siloBaseMeshData &data );
#endif

    // Function to determine which base mesh ids to register a vector with
    std::vector<AMP::Mesh::MeshID> getMeshIDs( AMP::Mesh::Mesh::shared_ptr mesh );

    // Function to write the summary file (the file should already be created, ready to reopen)
    // This function requires global communication
    void writeSummary( std::string filename );

    // The comm of the writer
    AMP_MPI d_comm;

    // The dimension
    int dim;

    // The dimension
    int decomposition;

    // List of all meshes and thier ids
    std::map<AMP::Mesh::MeshID,siloBaseMeshData>  d_baseMeshes;
    std::map<AMP::Mesh::MeshID,siloMultiMeshData> d_multiMeshes;

    // List of all variables
    std::set<std::string>   d_varNames;

    // List of all vectors that have been registered
#ifdef USE_AMP_VECTORS
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> d_vectors;
#endif

};


}
}

#endif


