#ifndef included_AMP_libMeshElement
#define included_AMP_libMeshElement

#include <vector>
#include <boost/shared_ptr.hpp>
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
    virtual std::vector<MeshElement> getElements(GeomType &type);

    //! Return the elements neighboring the current element
    virtual std::vector<MeshElement> getNeighbors();

    //! Return the volume of the current element (does not apply to verticies)
    virtual double volume();

    //! Return the coordinates of all verticies composing the element
    virtual std::vector<double> coord();


protected:

    /** Default constructor
     * \param dim       Spatial dimension
     * \param type      Entity type:  0: node, 1: element
     * \param element   Underlying libmesh element
     */
    libMeshElement(int dim, int type, void* element);

    //! Clone the iterator
    virtual MeshElement* clone() const;

    //! The underlying libmesh element
    int d_dim;
    int d_type;
    void* ptr_element;

    friend class AMP::Mesh::libMesh;
    friend class AMP::Mesh::libMeshIterator;

};



}
}

#endif

