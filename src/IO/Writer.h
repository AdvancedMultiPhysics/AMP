#ifndef included_AMP_Writer
#define included_AMP_Writer

#include "AMP/mesh/MeshID.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Array.h"
#include "AMP/utils/Database.h"

#include <memory>
#include <string>
#include <vector>


// Declare some classes
namespace AMP::Mesh {
class Mesh;
class MeshIterator;
struct MeshElementID;
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
    enum class VectorType : uint8_t { DOUBLE, SINGLE, INT, UINT8 };
    enum class DecompositionType : uint8_t { SINGLE = 1, MULTIPLE = 2 };
    struct WriterProperties {
        std::string type;            //!< Writer type: Silo, HDF5, Ascii
        std::string extension;       //!< The primary file extension for the writer
        bool registerMesh;           //!< Does the writer support registering a mesh
        bool registerVector;         //!< Does the writer support registering a vector
        bool registerVectorWithMesh; //!< Does the writer support registering a vector with a mesh
        bool registerMatrix;         //!< Does the writer support registering a matrix
        bool enabled;                //!< Is the current writer enabled
        bool isNull;                 //!< Is the current writer a null writer
        DecompositionType decomposition; //!< Decomposition method to use
        WriterProperties();
    };
    struct WriterParameters {
        /**
         * @brief Decomposition method to use.
         *
         * - SINGLE:
         *   Writes all data to a single file.
         *   Requires a serial write and will have the worst performance.
         *
         * - MULTIPLE:
         *   Each processor writes a separate file, and a separate summary file is written.
         *   Typically better performance at large scale, but many files are written simultaneously.
         */
        DecompositionType decomposition = DecompositionType::MULTIPLE;

        /**
         * @brief Enable referencing static variables across timesteps.
         *
         * If true, static data is written once and then referenced for other timesteps.
         * This can significantly reduce the amount of data written, but increases the
         * complexity of the output format.
         */
        bool enableStaticData = false;

        //! Communicator to use
        AMP_MPI comm = AMP_COMM_WORLD;

        //! Default constructor
        WriterParameters() {}
    };


public:
    /**
     * \brief   Function to build a writer
     * \details This function will build a default writer for use.
     * \param[in] type  Writer type:
     *                  "None"  - An empty writer will be created
     *                  "Silo"  - A silo writer will be created if silo is configured,
     *                            otherwise an empty writer will be created.
     *                  "Ascii" - A simple ascii writer
     *                  "HDF5"  - A simple HDF5 writer
     *                  "auto"  - Choose the writer based on the comm size and compiled packages
     * \param[in] properties  Parameters used to initialize the writer
     */
    static std::shared_ptr<AMP::IO::Writer>
    buildWriter( std::string type, const WriterParameters &properties = WriterParameters() );

    /**
     * \brief   Function to build a writer
     * \details This function will build a default writer for use.
     * \param[in] db    Input database for the writer
     */
    static std::shared_ptr<AMP::IO::Writer> buildWriter( std::shared_ptr<AMP::Database> db );


public:
    //!  Default destructor
    virtual ~Writer() = default;

    //! Delete copy constructor
    Writer( const Writer & ) = delete;

    //! Function to get the writer properties
    virtual WriterProperties getProperties() const = 0;

    //!  Function to return the file extension
    std::string getExtension() const;

    //!  Function to read a file
    virtual void readFile( const std::string &fname ) = 0;

    //!  Function to write a file
    virtual void writeFile( const std::string &fname, size_t iteration, double time = 0 ) = 0;

    /**
     * \brief    Function to register a mesh
     * \details  This function will register a mesh with the writer.
     *           Note: if mesh is a MultiMesh, it will register all sub meshes.
     * \param[in] mesh  The mesh to register
     * \param[in] level How many sub meshes do we want?
     *                  0: Only register the local base meshes (advanced users only)
     *                  1: Register current mesh only (default)
     *                  2: Register all meshes (do not separate for the ranks)
     *                  3: Register all mesh pieces including the individual ranks

     * \param[in] path  The directory path for the mesh.  Default is an empty string.
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
     * \param[in] vec   The vector we want to write
     * \param[in] mesh  The mesh we want to write the vector over.
     *                  Note: any writers require the vector to completely cover the mesh.
     *                  Note: mesh does not have to be previously registered with registerMesh.
     * \param[in] type  The entity type we want to save (vertex, face, cell, etc.)
     *                  Note: some writers only supports writing one entity type.
     *                  If the vector spans multiple entity type (eg cell+vertex)  the user should
     *                  register the vector multiple times (one for each entity type).
     * \param[in] name  Optional name for the vector.
     * \param[in] precision  Desired precision in output file.
     *                       Note: not all types are supported by all writers.
     * \param[in] isStatic   Is the vectors static (constant vs time)
     */
    virtual void registerVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec,
                                 std::shared_ptr<AMP::Mesh::Mesh> mesh,
                                 AMP::Mesh::GeomType type,
                                 const std::string &name = "",
                                 VectorType precision    = VectorType::DOUBLE,
                                 bool isStatic           = false );

    /**
     * \brief    Function to register a vector
     * \details  This function will register a vector with the writer.
     *     This version of registerVector only stores the raw data.
     *     It is not associated with a mesh.
     * \param[in] vec   The vector we want to write
     * \param[in] name  Optional name for the vector.
     */
    void registerVector( std::shared_ptr<AMP::LinearAlgebra::Vector> vec,
                         const std::string &name = "" );

    /**
     * \brief    Function to register a matrix
     * \details  This function will register a matrix with the writer.
     *     This version of registerMatrix only stores the raw data..
     *     It is not associated with a mesh.
     * \param[in] mat   The matrix we want to write
     * \param[in] name  Optional name for the vector.
     */
    void registerMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> mat,
                         const std::string &name = "" );

    //! Return the communicator
    inline const AMP_MPI &getComm() const { return d_comm; }


protected: // Internal structures
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
    using GeomType = AMP::Mesh::GeomType;
    struct VectorData {
        bool isStatic       = false;                     // Is the variable static vs time
        VectorType dataType = VectorType::DOUBLE;        // Desired precision (if supported)
        uint8_t numDOFs     = 0;                         // Number of unknowns per point
        GeomType type       = GeomType::Nullity;         // Types of variables
        std::string name;                                // Vector name to store
        std::shared_ptr<AMP::LinearAlgebra::Vector> vec; // AMP vector
        VectorData() = default;
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
    // Default constructor
    Writer( const WriterParameters &properties );

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


protected:
    AMP_MPI d_comm;                                  //!< The comm of the writer
    DecompositionType d_decomposition;               //!< The decomposition to use
    std::map<GlobalID, baseMeshData> d_baseMeshes;   //!< List of all base meshes and their ids
    std::map<GlobalID, multiMeshData> d_multiMeshes; //!< List of all multimeshes and their ids
    std::map<GlobalID, VectorData> d_vectors;        //!< List of independent vectors
    std::map<GlobalID, MatrixData> d_matrices;       //!< List of all independent matrices
    std::vector<std::shared_ptr<AMP::LinearAlgebra::Vector>>
        d_vectorsMesh; //!< List of all vectors (need to remove)
};


} // namespace AMP::IO

#endif
