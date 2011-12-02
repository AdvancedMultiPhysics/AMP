#ifndef included_AMP_libMeshElement
#define included_AMP_libMeshElement

#include <vector>
#include <boost/shared_ptr.hpp>
#include "ampmesh/Mesh.h"
#include "ampmesh/MeshElement.h"
#include "ampmesh/libmesh/libMesh.h"
#include "ampmesh/libmesh/libMeshIterator.h"

// libMesh includes
#include "elem.h"

namespace AMP {
namespace Mesh {


/**
 * \class libMeshElement
 * \brief A derived class used to define a mesh element
 * \details  This class provides routines for accessing and using a mesh element.
 * A mesh element can be thought of as the smallest unit of a mesh.  It is of a type
 * of GeomType.  This class is derived to store a libMesh element.
 */
class libMeshElement: public MeshElement
{
public:

    //! Empty constructor for a MeshElement
    libMeshElement ( );

    //! Copy constructor
    libMeshElement(const libMeshElement&);

    //! Assignment operator
    libMeshElement& operator=(const libMeshElement&);

    //! De-constructor for a MeshElement
    virtual ~libMeshElement ( );

    //! Return the elements composing the current element
    virtual std::vector<MeshElement> getElements(const GeomType type) const;

    //! Return the elements neighboring the current element
    virtual std::vector< MeshElement::shared_ptr > getNeighbors() const;

    //! Return the volume of the current element (does not apply to verticies)
    virtual double volume() const;

    //! Return the coordinates of all verticies composing the element
    virtual std::vector<double> coord() const;

    /**
     * \brief     Check if the current element is on the given boundary
     * \details   Check if the current element is on the boundary specified by the given id
     * \param id  The boundary id to check
     */
    virtual bool isOnBoundary(int id) const;


protected:

    /** Default constructor
     * \param dim       Spatial dimension
     * \param type      Element type
     * \param element   Underlying libmesh element
     * \param mesh      Underlying mesh
     * \param rank      Rank of the current processor (must agree with libmesh->processor_id())
     * \param meshID    ID of the current mesh
     */
    libMeshElement(int dim, GeomType type, void* element, unsigned int rank, MeshID meshID, libMesh* mesh );

    //! Clone the iterator
    virtual MeshElement* clone() const;

    //! The underlying libmesh element
    int d_dim;
    void* ptr_element;

    // The rank of the current processor
    unsigned int d_rank;

    // The ID of the current mesh
    MeshID d_meshID;

    // The pointer to the current mesh
    libMesh* d_mesh;

    friend class AMP::Mesh::libMesh;
    friend class AMP::Mesh::libMeshIterator;

};



}
}

#endif

