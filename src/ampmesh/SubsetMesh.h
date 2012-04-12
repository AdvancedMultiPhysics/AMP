#ifndef included_AMP_SubsetMesh
#define included_AMP_SubsetMesh

#include "ampmesh/Mesh.h"

#include <boost/shared_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <map>

namespace AMP {
namespace Mesh {


/**
 * \class SubsetMesh
 * \brief A class used to handle a subset mesh
 * \details  This class provides routines for using subset meshes.
 */
class SubsetMesh: public Mesh
{
public:


    //! Default constructor
    SubsetMesh( boost::shared_ptr<const Mesh> mesh, const AMP::Mesh::MeshIterator iterator );


    //! Deconstructor
     ~SubsetMesh();


    /**
     * \brief    Subset a mesh given a MeshID
     * \details  This function will return the mesh with the given meshID.
     *    Note: for multimeshes, this will return the mesh with the given id.
     *    For a single mesh this will return a pointer to itself if the meshID
     *    matches the meshID of the mesh, and a null pointer otherwise.
     * \param meshID  MeshID of the desired mesh
     */
    virtual boost::shared_ptr<Mesh>  Subset ( MeshID meshID ) const;


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
    virtual boost::shared_ptr<Mesh>  Subset ( std::string name ) const;


    /* Return the number of local element of the given type
     * \param type   Geometric type
     */
    virtual size_t  numLocalElements( const GeomType type ) const;


    /* Return the global number of elements of the given type
     * Note: depending on the mesh this routine may require global communication across the mesh.
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
    virtual MeshIterator getIterator ( const GeomType type, const int gcw=0 ) const;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the surface
     * \details  Return an MeshIterator over the given geometric objects on the surface
     * \param type   Geometric type to iterate over
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator getSurfaceIterator ( const GeomType type, const int gcw=0 ) const;


    /**
     * \brief    Return the list of all ID sets in the mesh
     * \details  Return the list of all ID sets in the mesh
     * Note: depending on the mesh this routine may require global communication across the mesh.
     */
    virtual std::vector<int> getIDSets ( ) const;


    /**
     * \brief    Return an MeshIterator over the given geometric objects on the given ID set
     * \details  Return an MeshIterator over the given geometric objects on the given ID set
     * \param type   Geometric type to iterate over
     * \param id     id for the elements (example: nodeset id)
     * \param gcw    Desired ghost cell width
     */
    virtual MeshIterator getIDsetIterator ( const GeomType type, const int id, const int gcw=0 ) const;


    //! Get the largest geometric type in the mesh
    virtual GeomType getGeomType() const { return GeomDim; } 


    //! Get the largest geometric type in the mesh
    virtual AMP_MPI getComm() const { return d_comm; }


    //! Get the mesh ID
    virtual inline MeshID meshID() const { return d_meshID; }


    //! Is the current mesh a base mesh
    virtual inline bool isBaseMesh() const { return false; }


    /**
     *  Get the meshIDs of all meshes that compose the current mesh (including its self)
     *  Note: This function may require global communication depending on the implimentation
     */
    virtual std::vector<MeshID> getAllMeshIDs() const;


    /**
     *  Get the meshIDs of all the basic meshes that compose the current mesh (excluding multimeshes and subset meshes)
     *  Note: This function may require global communication depending on the implimentation
     */
    virtual std::vector<MeshID> getBaseMeshIDs() const;


    /**
     *  Get the meshIDs of all meshes that compose the current mesh (including its self)
     *  on the current processor.
     */
    virtual std::vector<MeshID> getLocalMeshIDs() const;


    /**
     *  Get the meshIDs of all the basic meshes that compose the current mesh 
     *  (excluding multimeshes and subset meshes) on the current processor.
     */
    virtual std::vector<MeshID> getLocalBaseMeshIDs() const;


    /**
     * \brief    Displace the entire mesh
     * \details  This function will displace the entire mesh by a scalar value.
     *   This function is a blocking call for the mesh communicator, and requires
     *   the same value on all processors.  The displacement vector should be the 
     *   size of the physical dimension.
     * \param x  Displacement vector
     */
    virtual void displaceMesh( std::vector<double> x );


#ifdef USE_AMP_VECTORS
    /**
     * \brief    Displace the entire mesh
     * \details  This function will displace the entire mesh by displacing
     *   each node by the values provided in the vector.  This function is 
     *   a blocking call for the mesh communicator
     * \param x  Displacement vector.  Must have N DOFs per node where N 
     *           is the physical dimension of the mesh.
     */
    virtual void displaceMesh ( boost::shared_ptr<const AMP::LinearAlgebra::Vector> x );
#endif

    // Needed to prevent problems with virtual functions
    using Mesh::Subset;

protected:

    // Parent mesh for the subset
    boost::shared_ptr<const Mesh>  d_parent_mesh;

    // Pointers to store the elements in the subset meshes
    std::vector<size_t> N_global;
    std::vector<std::vector<boost::shared_ptr<std::vector<MeshElement> > > >  d_elements;

    // Pointers to store the elements on the surface
    std::vector<std::vector<boost::shared_ptr<std::vector<MeshElement> > > >  d_surface;

    // Data to store the id sets
    struct map_id_struct{ 
        int id;
        GeomType type;
        int gcw; 
        inline bool operator== (const map_id_struct& rhs ) const { return id==rhs.id && type==rhs.type && gcw==rhs.gcw; }
        inline bool operator!= (const map_id_struct& rhs ) const { return id!=rhs.id && type!=rhs.type && gcw!=rhs.gcw; }
        inline bool operator>= (const map_id_struct& rhs ) const { 
            if ( id != rhs.id ) { return id>rhs.id; }
            if ( type != rhs.type ) { return type>rhs.type; }
            return gcw >= rhs.gcw;
        }
        inline bool operator> (const map_id_struct& rhs ) const { return operator>(rhs) && operator!=(rhs); }
        inline bool operator< (const map_id_struct& rhs ) const { return !operator>=(rhs); }
        inline bool operator<= (const map_id_struct& rhs ) const { return !operator>(rhs); }
    };
    std::vector<int> d_idSets;
    std::map< map_id_struct, boost::shared_ptr<std::vector<MeshElement> > >  d_sets;

};

} // Mesh namespace
} // AMP namespace

#endif

