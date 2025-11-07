#ifndef included_AMP_Writer
#define included_AMP_Writer

#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/Database.h"

#include <memory>
#include <string>
#include <vector>


// Declare some classes
namespace AMP::Mesh {
class Mesh;
struct MeshID;
class MeshIterator;
struct MeshElementID;
enum class GeomType : uint8_t;
} // namespace AMP::Mesh
namespace AMP::LinearAlgebra {
class Vector;
class Matrix;
} // namespace AMP::LinearAlgebra

class dummy;

namespace AMP::IO {


/**
 * \class Writer
 * \brief A class used to abstract away reading/writing files.
 * \details  This class provides routines for reading, accessing and writing meshes and vectors.
 *    The writers can be used to generate files for visualization or interfacing with other codes.
 */
class Writer
{
public:
    struct WriterProperties {
        std::string type;            // Writer type: Silo, HDF5, Ascii
        std::string extension;       // The primary file extension for the writer
        bool registerMesh;           // Does the writer support registering a mesh
        bool registerVector;         // Does the writer support registering a vector
        bool registerVectorWithMesh; // Does the writer support registering a vector with a mesh
        bool registerMatrix;         // Does the writer support registering a matrix
        bool enabled;                // Is the current writer enabled
        bool isNull;                 // Is the current writer a null writer
        WriterProperties();
    };

public:
    /**
     * \brief   Function to build a writer
     * \details This function will build a default writer for use.
     * \param type   Writer type:
     *               "None"  - An empty writer will be created
     *               "Silo"  - A silo writer will be created if silo is configured,
     *                        otherwise an empty writer will be created.
     *               "Ascii" - A simple ascii writer
     *               "HDF5"  - A simple HDF5 writer
     *               "auto"  - Choose the writer based on the comm size and compiled packages
     * \param comm   Communicator to use
     */
    static std::shared_ptr<AMP::IO::Writer> buildWriter( std::string type,
                                                         AMP_MPI comm = AMP_COMM_WORLD );

    /**
     * \brief   Function to build a writer
     * \details This function will build a default writer for use.
     * \param db   Input database for the writer
     */
    static std::shared_ptr<AMP::IO::Writer> buildWriter( std::shared_ptr<AMP::Database> db );


public:
    //!  Default constructor
    Writer();

    //!  Default destructor
    virtual ~Writer();

    //! Delete copy constructor
    Writer( const Writer & ) = delete;

    //! Function to get the writer properties
    virtual WriterProperties getProperties() const = 0;

    //!  Function to return the file extension
    std::string getExtension() const;

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
    virtual void setDecomposition( int decomposition );

    //!  Function to read a file
    virtual void readFile( const std::string &fname ) = 0;

    //!  Function to write a file
    virtual void writeFile( const std::string &fname, size_t iteration, double time = 0 ) = 0;

    /**
     * \brief    Function to register a mesh
     * \details  This function will register a mesh with the writer.
     *           Note: if mesh is a MultiMesh, it will register all sub meshes.
     * \param mesh  The mesh to register
     * \param level How many sub meshes do we want?
     *              0: Only register the local base meshes (advanced users only)
     *              1: Register current mesh only (default)
     *              2: Register all meshes (do not separate for the ranks)
     *              3: Register all mesh pieces including the individual ranks

     * \param path  The directory path for the mesh.  Default is an empty string.
     */
    void registerMesh( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                       int level               = 1,
                       const std::string &path = std::string() );

    /**
     * \brief    Function to register a vector
     * \details  This function will register a vector with the writer and register it with the given
     * mesh.
     *     This version of registerVector allows the data to be "stored" on the mesh for
     * visualization
     *     or mesh-based operations.
     * \param vec   The vector we want to write
     * \param mesh  The mesh we want to write the vector over.
     *              Note: any writers require the vector to completely cover the mesh.
     *              Note: mesh does not have to be previously registered with registerMesh.
     * \param type  The entity type we want to save (vertex, face, cell, etc.)
     *              Note: some writers only supports writing one entity type.
     *              If the vector spans multiple entity type (eg cell+vertex)  the user should
     *              register the vector multiple times (one for each entity type).
     * \param name  Optional name for the vector.
     */
    virtual void registerVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec,
                                 std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                 AMP::Mesh::GeomType type,
                                 const std::string &name = "" );

    /**
     * \brief    Function to register a vector
     * \details  This function will register a vector with the writer.
     *     This version of registerVector only stores the raw data.
     *     It is not associated with a mesh.
     * \param vec   The vector we want to write
     * \param name  Optional name for the vector.
     */
    void registerVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec,
                         const std::string &name = "" );

    /**
     * \brief    Function to register a matrix
     * \details  This function will register a matrix with the writer.
     *     This version of registerMatrix only stores the raw data..
     *     It is not associated with a mesh.
     * \param mat   The matrix we want to write
     * \param name  Optional name for the vector.
     */
    void registerMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> mat,
                         const std::string &name = "" );

protected: // Protected structures
    // Structure to hold id
    struct GlobalID {
        uint64_t objID;     // Object id
        uint32_t ownerRank; // Global rank of the processor that "owns" the data
        GlobalID() : objID( 0 ), ownerRank( 0 ) {}
        GlobalID( uint64_t obj, uint32_t rank ) : objID( obj ), ownerRank( rank ) {}
        bool operator==( const GlobalID &rhs ) const
        {
            return objID == rhs.objID && ownerRank == rhs.ownerRank;
        }
        bool operator!=( const GlobalID &rhs ) const
        {
            return objID != rhs.objID || ownerRank != rhs.ownerRank;
        }
        bool operator<( const GlobalID &rhs ) const
        {
            if ( objID == rhs.objID )
                return ownerRank < rhs.ownerRank;
            return objID < rhs.objID;
        }
        bool operator<=( const GlobalID &rhs ) const
        {
            if ( objID == rhs.objID )
                return ownerRank <= rhs.ownerRank;
            return objID < rhs.objID;
        }
        bool operator>( const GlobalID &rhs ) const
        {
            if ( objID == rhs.objID )
                return ownerRank > rhs.ownerRank;
            return objID > rhs.objID;
        }
        bool operator>=( const GlobalID &rhs ) const
        {
            if ( objID == rhs.objID )
                return ownerRank >= rhs.ownerRank;
            return objID > rhs.objID;
        }
    };

    // Structure to hold vector data
    struct VectorData {
        std::string name;                                // Vector name to store
        int numDOFs;                                     // Number of unknowns per point
        std::shared_ptr<AMP::LinearAlgebra::Vector> vec; // AMP vector
        AMP::Mesh::GeomType type;                        // Types of variables
        VectorData() : numDOFs( 0 ), type( static_cast<AMP::Mesh::GeomType>( 0xFF ) ) {}
        VectorData( std::shared_ptr<AMP::LinearAlgebra::Vector>, const std::string & );
    };

    // Structure to hold matrix data
    struct MatrixData {
        std::string name;                                // Matrix name to store
        std::shared_ptr<AMP::LinearAlgebra::Matrix> mat; // AMP matrix
        MatrixData() = default;
        MatrixData( std::shared_ptr<AMP::LinearAlgebra::Matrix>, const std::string & );
    };

    // Structure used to hold data for a base mesh
    struct baseMeshData {
        GlobalID id;                           // Unique ID to identify the mesh
        std::shared_ptr<AMP::Mesh::Mesh> mesh; // Pointer to the mesh
        int rank;                              // Rank of the current processor on the mesh
        int ownerRank;                         // Global rank of the processor that "owns" the mesh
        std::string meshName;                  // Name of the mesh
        std::string path;                      // Path to the mesh
        std::string file;                      // File that will contain the mesh
        std::vector<VectorData> vectors;       // Vectors for each mesh
        // Function to count the number of bytes needed to pack the data
        size_t size() const;
        // Function to pack the data to a byte array (note: some info may be lost)
        void pack( char * ) const;
        // Function to unpack the data from a byte array (note: some info may be lost)
        static baseMeshData unpack( const char * );
        // Constructor
        baseMeshData() : rank( -1 ), ownerRank( -1 ) {}
    };

    // Structure used to hold data for a multimesh
    struct multiMeshData {
        GlobalID id;                           // Unique ID to identify the mesh
        std::shared_ptr<AMP::Mesh::Mesh> mesh; // Pointer to the mesh
        int ownerRank;                         // Global rank of the processor that "owns" the mesh
                                               // (usually rank 0 on the mesh comm)
        std::string name;                      // Name of the multimesh
        std::vector<GlobalID> meshes;          // Base mesh ids needed to construct the mesh data
        std::vector<std::string> varName;      // Vectors for each mesh
        // Function to count the number of bytes needed to pack the data
        size_t size() const;
        // Function to pack the data to a byte array
        void pack( char * ) const;
        // Function to unpack the data from a byte array
        static multiMeshData unpack( const char * );
        // Constructor
        multiMeshData() : ownerRank( -1 ) {}
    };

protected: // Protected member functions
    // Given a filename, strip the directory information and create the directories if needed
    void createDirectories( const std::string &filename );

    // Function to determine which base mesh ids to register a vector with
    static std::vector<AMP::Mesh::MeshID> getMeshIDs( std::shared_ptr<AMP::Mesh::Mesh> mesh );

    // Get the node coordinates and elements for a mesh
    static void getNodeElemList( std::shared_ptr<const AMP::Mesh::Mesh> mesh,
                                 const AMP::Mesh::MeshIterator &elements,
                                 AMP::Array<double> *x,
                                 AMP::Array<int> &nodelist,
                                 std::vector<AMP::Mesh::MeshElementID> &nodelist_ids );

    // Register the mesh returning the ids of all registered base meshes
    void registerMesh2( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                        int level,
                        const std::string &path,
                        std::set<GlobalID> &base_ids );

    // Function to syncronize multimesh data
    std::tuple<std::vector<multiMeshData>, std::map<GlobalID, baseMeshData>>
    syncMultiMeshData( int root = -1 ) const;

    // Get id from a communicator
    GlobalID getID( const AMP_MPI &comm ) const;

    // Synchronize the vectors (call makeConsistent)
    void syncVectors();

    // Synchronize mesh data
    template<class TYPE>
    void syncData( std::vector<TYPE> &data, int root ) const;

protected: // Internal data
    // The comm of the writer
    AMP_MPI d_comm;

    // The decomposition to use
    int d_decomposition;

    // List of all meshes and their ids
    std::map<GlobalID, baseMeshData> d_baseMeshes;
    std::map<GlobalID, multiMeshData> d_multiMeshes;

    // List of all independent vectors that have been registered
    std::map<GlobalID, VectorData> d_vectors;

    // List of all independent matrices that have been registered
    std::map<GlobalID, MatrixData> d_matrices;

    // List of all vectors that have been registered (work on removing)
    std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>> d_vectorsMesh;
};


} // namespace AMP::IO

#endif
