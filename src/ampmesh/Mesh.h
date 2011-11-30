#ifndef included_AMP_Mesh
#define included_AMP_Mesh

#include "MeshParameters.h"
#include "MeshIterator.h"
#include "MeshElement.h"
#include "utils/AMP_MPI.h"

#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

namespace AMP {
namespace Mesh {


//! Enumeration for basic mesh-based quantities
enum SetOP { Union, Intersection, Complement };


/**
 * \class Mesh
 * \brief A class used to abstract away mesh from an application
 *
 * \details  This class provides routines for reading, accessing and writing meshes.
 */
class Mesh: public boost::enable_shared_from_this<AMP::Mesh::Mesh>
{
public:

    /**
     *\typedef shared_ptr
     *\brief  Name for the shared pointer.
     *\details  Use this typedef for a reference counted pointer to a mesh manager object.
     */
    typedef boost::shared_ptr<AMP::Mesh::Mesh>  shared_ptr;


    /**
     * \brief Read in mesh files, partition domain, and prepare environment for simulation
     * \details  For trivial parallelsim, this method reads in the meshes on each processor.  Each
     * processor contains a piece of each mesh.  For massive parallelism, each mesh is on its own
     * communicator.  As such, some math libraries must be initialized accordingly.
     * \param params  Parameters for constructing a mesh from an input database
     */
    Mesh ( const MeshParameters::shared_ptr &params );


    /**
     * \brief Construct a new mesh from an existing mesh
     * \details  This constructor will construct a new mesh from an existing mesh.
     * This is designed as a path to create a new mesh object of one type from
     * an existing mesh of a different type.  It also allows creating a new single mesh
     * from a subset or superset of other meshes.  Note that instantion of this routine 
     * may not be able to create it's mesh from any arbitrary mesh, and may throw an 
     * error.
     * \param old_mesh  Existing mesh that we will use to construct the new mesh
     */
    Mesh ( const Mesh::shared_ptr &old_mesh );


    /**
     * \brief Construct a new mesh from an existing mesh.
     * \details  This constructor will construct a new mesh from an existing mesh
     * using an iterator over the existing mesh.
     * This is designed as a path to create a new mesh object of one type from
     * an existing mesh of a different type.  It also allows creating a new single mesh
     * from a subset or superset of other meshes.  Note that instantion of this routine 
     * may not be able to create it's mesh from any arbitrary mesh, and may throw an 
     * error.
     * \param old_mesh  Existing mesh that we will use to construct the new mesh
     * \param iterator  Iterator over the existing mesh
     */
    Mesh ( const Mesh::shared_ptr &old_mesh, MeshIterator::shared_ptr &iterator);


    /**
     * \brief   Create a mesh 
     * \details  This function will create a mesh (or series of meshes) based on
     *   the input database.  
     * \param params Parameters for constructing a mesh from an input database
     */
    static boost::shared_ptr<AMP::Mesh::Mesh> buildMesh( const MeshParameters::shared_ptr &params );


    /**
     * \brief   Estimate the number of elements in the mesh 
     * \details  This function will estimate the number of elements in the mesh. 
     *   This is used so that we can properly balance the meshes across multiple processors.
     *   Ideally this should be both an accurate estimate and very fast.  It should not require
     *   any communication and should not have to actually load a mesh.
     * \param params Parameters for constructing a mesh from an input database
     */
    static size_t estimateMeshSize( const MeshParameters::shared_ptr &params );


    //! Deconstructor
     ~Mesh ();


    //! Assignment operator
    virtual Mesh operator=(const Mesh&);


    //! Virtual function to copy the mesh (allows use to proply copy the derived class)
    virtual Mesh copy() const;


    /**
     * \brief    Subset a mesh given a MeshID
     * \details  This function will return the mesh with the given meshID.
     *    Note: for multimeshes, this will return the mesh with the given id.
     *    For a single mesh this will return a pointer to itself if the meshID
     *    matches the meshID of the mesh, and a null pointer otherwise.
     * \param meshID  MeshID of the desired mesh
     */
    virtual boost::shared_ptr<Mesh>  Subset ( size_t meshID );


    /**
     * \brief    Subset a mesh given a mesh name
     * \details  This function will return the mesh with the given name.
     *    Note: for multimeshes, this will return the mesh with the given name.
     *    For a single mesh this will return a pointer to itself if the mesh name
     *    matches the name of the mesh, and a null pointer otherwise.
     *    Note: The mesh name is not gaurenteed to be unique.  If there are multiple
     *    meshes with the same name, the first mesh with the given name will be returned.
     *    It is strongly recommended to use the meshID when possible.
     * \param name  Name of the desired mesh
     */
    virtual boost::shared_ptr<Mesh>  Subset ( std::string name );


    /**
     * \brief    Subset a mesh given a MeshIterator
     * \details  This function will subset a mesh over a given iterator.
     *   This will return a new mesh object.
     * \param iterator  MeshIterator used to subset
     */
    virtual boost::shared_ptr<Mesh>  Subset ( MeshIterator::shared_ptr &iterator );


    /**
     * \brief        Subset a mesh given another mesh
     * \details      This function will subset a mesh given another mesh
     * \param mesh   Mesh used to subset
     */
    virtual boost::shared_ptr<Mesh>  Subset ( Mesh &mesh );


    /* Return the number of local element of the given type
     * \param type   Geometric type
     */
    virtual size_t  numLocalElements( const GeomType type ) const;


    /* Return the global number of elements of the given type
     * \param type   Geometric type
     */
    virtual size_t  numGlobalElements( const GeomType type ) const;


    /* Return the number of ghost elements of the given type on the current processor
     * \param type   Geometric type
     */
    virtual size_t  numGhostElements( const GeomType type, const int gcw ) const;


    /**
     * \brief    Return an MeshIterator over the given geometric objects
     * \details  Return an MeshIterator over the given geometric objects
     * \param type   Geometric type to iterate over
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator getIterator ( const GeomType type, const int gcw=0 );


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the surface
     * \details  Return an MeshIterator over the given geometric objects on the surface
     * \param type   Geometric type to iterate over
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator getSurfaceIterator ( const GeomType type, const int gcw=0 );


    /**
     * \brief    Return the list of all ID sets in the mesh
     * \details  Return the list of all ID sets in the mesh
     */
    virtual std::vector<int> getIDSets ( );


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the given ID set
     * \details  Return an MeshIterator over the given geometric objects on the given ID set
     * \param type   Geometric type to iterate over
     * \param id     id for the elements (example: nodeset id)
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator getIDsetIterator ( const GeomType type, const int id, const int gcw=0 );


    /**
     * \brief    Return an MeshIterator constructed through a set operation of two other MeshIterators.
     * \details  Return an MeshIterator constructed through a set operation of two other MeshIterators.
     * \param OP Set operation to perform.
     *           Union - Perform a union of the iterators ( A U B )
     *           Intersection - Perform an intersection of the iterators ( A n B )
     *           Complement - Perform a compliment of the iterators ( A - B )
     * \param A  Pointer to MeshIterator A
     * \param B  Pointer to MeshIterator B
     */
    virtual MeshIterator getIterator ( SetOP &OP, MeshIterator::shared_ptr &A, MeshIterator::shared_ptr &B);
 

    //! Get the largest geometric type in the mesh
    virtual GeomType getGeomType() const { return GeomDim; } 


    //! Get the largest geometric type in the mesh
    virtual AMP_MPI getComm() const { return comm; }


    //! Get the mesh ID
    virtual inline size_t meshID() const { return d_meshID; }


    //! Get the mesh name
    virtual inline std::string getName() const { return d_name; }


    //! Set the mesh name
    virtual inline void setName(std::string name) { d_name = name; }


protected:

    //!  Empty constructor for a mesh
    Mesh() {}

    //! The mesh parameters
    MeshParameters::shared_ptr params;

    //! The geometric dimension (equivalent to the highest geometric object that could be represented)
    GeomType GeomDim;

    //! The physical dimension
    short int PhysicalDim;

    //! The communicator over which the mesh is stored
    AMP_MPI comm;

    //! A pointer to an AMP database containing the mesh info
    boost::shared_ptr<AMP::Database>  d_db;

    //! A unique id for each mesh
    size_t d_meshID;

    //! A name for the mesh
    std::string d_name;

    /**
     *  A function to create a unique id for the mesh (requires the comm to be set)
     *  Note: this requires a global communication across the mesh communicator.
     *  Note: this function is NOT thread safe, and will need to be modified before threads are used.
     */
    void setMeshID();


};

} // Mesh namespace
} // AMP namespace

#endif

