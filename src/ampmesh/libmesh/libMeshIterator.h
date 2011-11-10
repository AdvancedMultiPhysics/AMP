#ifndef included_AMP_libMeshIterators
#define included_AMP_libMeshIterators

#include <boost/shared_ptr.hpp>
#include "ampmesh/libmesh/libMesh.h"
#include "ampmesh/MeshIterator.h"

#include "mesh.h"

namespace AMP { 
namespace Mesh {


class libMeshIterator: public MeshIterator {
public:

    //! Empty MeshIterator constructor
    libMeshIterator();

    //! Deconstructor
    ~libMeshIterator ();

    //! Copy constructor
    libMeshIterator(const libMeshIterator&);

    //! Assignment operator
    libMeshIterator& operator=(const libMeshIterator&);

    //! Increment
    MeshIterator& operator++();
    
    //! Increment
    MeshIterator operator++(int);

    //! Decrement
    MeshIterator& operator--();
    
    //! Decrement
    MeshIterator operator--(int);

    //! Check if two iterators are equal
    bool operator==(const MeshIterator& rhs);

    //! Check if two iterators are not equal
    bool operator!=(const MeshIterator& rhs);
    
    //! Dereference the iterator
    MeshElement &operator*(void);

    //! Dereference the iterator
    MeshElement *operator->(void);

    //! Return an iterator to the begining
    MeshIterator begin();

    //! Return an iterator to the begining
    MeshIterator end();

protected:
    /** Default constructor
     * \param type      Entity type:  0: node, 1: element
     * \param mesh      Pointer to the libMesh mesh
     * \param gcw       gcw to use
     * \param pos       Pointer to iterator with the current position
     */
    libMeshIterator(int type, AMP::Mesh::libMesh *mesh, int gcw, void *begin, void *end, void *pos);

    //! Clone the iterator
    virtual MeshIterator* clone() const;

friend class AMP::Mesh::libMesh;

private:
    // Data members
    int d_gcw;
    int d_type;
    void *d_begin;
    void *d_end;
    void *d_pos;
    AMP::Mesh::libMesh *d_mesh;
    MeshElement d_cur_element;
};


}
}

#endif

