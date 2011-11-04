#include "ampmesh/MeshElement.h"
#include "utils/Utilities.h"

namespace AMP {
namespace Mesh {


// Create a unique id for this class
static int MeshElementTypeID = 1;


/********************************************************
* Constructors                                          *
********************************************************/
MeshElement::MeshElement()
{
    typeID = MeshElementTypeID;
    element = NULL;
    d_elementType = null;
    d_globalID = 0;
}
MeshElement::MeshElement(const MeshElement& rhs)
{
    typeID = MeshElementTypeID;
    element = NULL;
    if ( rhs.element==NULL && rhs.typeID==MeshElementTypeID ) {
        element = NULL;
    } else if ( rhs.typeID!=MeshElementTypeID ) {
        element = rhs.clone();
    } else {
        element = rhs.element->clone();
    }
    d_elementType = rhs.d_elementType;
    d_globalID = rhs.d_globalID;
}
MeshElement& MeshElement::operator=(const MeshElement& rhs)
{
    if (this == &rhs) // protect against invalid self-assignment
        return *this;
    if ( element != NULL ) {
        // Delete the existing element
        delete element;
        element = NULL;
    }
    typeID = MeshElementTypeID;
    if ( rhs.element==NULL && rhs.typeID==MeshElementTypeID ) {
        element = NULL;
    } else if ( rhs.typeID!=MeshElementTypeID ) {
        element = rhs.clone();
    } else {
        element = rhs.element->clone();
    }
    d_elementType = rhs.d_elementType;
    d_globalID = rhs.d_globalID;
    return *this;
}


/********************************************************
* De-constructor                                        *
********************************************************/
MeshElement::~MeshElement()
{
    if ( element != NULL )
        delete element;
    element = NULL;
}


/********************************************************
* Function to clone the element                         *
********************************************************/
MeshElement* MeshElement::clone() const
{
    if ( element==NULL )
        return new MeshElement();
    else
        AMP_ERROR("clone must instantiated by the derived class");
    return NULL;
}


/********************************************************
* Functions that aren't implimented for the base class  *
********************************************************/
std::vector<MeshElement> MeshElement::getElements(GeomType &type)
{
    if ( element==NULL )
        AMP_ERROR("getElements is not implimented for the base class");
    return element->getElements(type);
}
std::vector<MeshElement> MeshElement::getNeighbors()
{
    if ( element==NULL )
        AMP_ERROR("getNeighbors is not implimented for the base class");
    return element->getNeighbors();
}
double MeshElement::volume()
{
    if ( element==NULL )
        AMP_ERROR("volume is not implimented for the base class");
    return element->volume();
}
std::vector<double> MeshElement::coord()
{
    if ( element==NULL )
        AMP_ERROR("coord is not implimented for the base class");
    return element->coord();
}


} // Mesh namespace
} // AMP namespace

