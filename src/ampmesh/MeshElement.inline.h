#ifndef included_AMP_MeshElement_inline
#define included_AMP_MeshElement_inline

#include "AMP/utils/Utilities.h"

namespace AMP {
namespace Mesh {


/********************************************************
 * Constructors                                          *
 ********************************************************/
MeshElement::MeshElement()
{
    typeID  = getTypeID();
    element = nullptr;
}
MeshElement::MeshElement( const MeshElement &rhs ) : typeID( getTypeID() ), element( nullptr )
{
    if ( rhs.element == nullptr && rhs.typeID == getTypeID() ) {
        element = nullptr;
    } else if ( rhs.typeID != getTypeID() ) {
        element = rhs.clone();
    } else {
        element = rhs.element->clone();
    }
}
MeshElement::MeshElement( MeshElement &&rhs ) : typeID( getTypeID() ), element( rhs.element )
{
    if ( rhs.typeID != getTypeID() )
        element = rhs.clone();
    rhs.element = nullptr;
}
MeshElement &MeshElement::operator=( const MeshElement &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    if ( element != nullptr ) {
        // Delete the existing element
        delete element;
        element = nullptr;
    }
    typeID = getTypeID();
    if ( rhs.element == nullptr && rhs.typeID == getTypeID() ) {
        element = nullptr;
    } else if ( rhs.typeID != getTypeID() ) {
        element = rhs.clone();
    } else {
        element = rhs.element->clone();
    }
    return *this;
}
MeshElement &MeshElement::operator=( MeshElement &&rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    if ( element != nullptr ) {
        // Delete the existing element
        delete element;
        element = nullptr;
    }
    typeID = getTypeID();
    std::swap( element, rhs.element );
    if ( rhs.typeID != getTypeID() )
        element = rhs.clone();
    return *this;
}


/********************************************************
 * Destructor                                            *
 ********************************************************/
MeshElement::~MeshElement()
{
    if ( element != nullptr )
        delete element;
    element = nullptr;
}


/********************************************************
 * Is the element null                                   *
 ********************************************************/
bool MeshElement::isNull() const { return typeID == getTypeID() && element == nullptr; }


/********************************************************
 * Function to clone the element                         *
 ********************************************************/
MeshElement *MeshElement::clone() const
{
    if ( element == nullptr )
        return new MeshElement();
    else
        AMP_ERROR( "clone must instantiated by the derived class" );
    return nullptr;
}


/********************************************************
 * Function to get the raw element                       *
 ********************************************************/
inline MeshElement *MeshElement::getRawElement() { return element == nullptr ? this : element; }
inline const MeshElement *MeshElement::getRawElement() const
{
    return element == nullptr ? this : element;
}


/********************************************************
 * Function to check if a point is within an element     *
 ********************************************************/
inline bool MeshElement::containsPoint( const std::vector<double> &pos, double TOL ) const
{
    return containsPoint( Point( pos.size(), pos.data() ), TOL );
}


/********************************************************
 * Functions that are wrappers to an advanced version    *
 ********************************************************/
inline std::vector<MeshElement> MeshElement::getElements( const GeomType type ) const
{
    std::vector<MeshElement> elements;
    ( element != nullptr ? element : this )->getElements( type, elements );
    return elements;
}
inline std::vector<MeshElement::shared_ptr> MeshElement::getNeighbors() const
{
    std::vector<MeshElement::shared_ptr> neighbors;
    ( element != nullptr ? element : this )->getNeighbors( neighbors );
    return neighbors;
}


} // namespace Mesh
} // namespace AMP

#endif
