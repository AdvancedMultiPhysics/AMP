#ifndef included_AMP_Vector_inline
#define included_AMP_Vector_inline

#include "AMP/vectors/data/VectorDataIterator.h"
#include <algorithm>


namespace AMP::LinearAlgebra {


/****************************************************************
 * Get basic info                                                *
 ****************************************************************/
inline std::shared_ptr<AMP::Discretization::DOFManager> Vector::getDOFManager() const
{
    return d_DOFManager;
}


/****************************************************************
 * getView/hasView                                               *
 ****************************************************************/
template<typename VIEW_TYPE>
std::shared_ptr<VIEW_TYPE> Vector::getView() const
{
    typedef typename std::remove_cv_t<VIEW_TYPE> TYPE;
    for ( size_t i = 0; i != d_Views->size(); i++ ) {
        if ( ( *d_Views )[i].type() == typeid( std::weak_ptr<TYPE> ) ) {
            auto ptr = std::any_cast<std::weak_ptr<TYPE>>( ( *d_Views )[i] );
            auto vec = ptr.lock();
            if ( vec )
                return vec;
        }
    }
    return std::shared_ptr<VIEW_TYPE>();
}
template<typename VIEW_TYPE>
bool Vector::hasView() const
{
    return getView<VIEW_TYPE>() != nullptr;
}
template<typename VIEW_TYPE>
void Vector::registerView( std::shared_ptr<VIEW_TYPE> v ) const
{
    typedef typename std::remove_cv_t<VIEW_TYPE> TYPE;
    for ( size_t i = 0; i != d_Views->size(); i++ ) {
        if ( ( *d_Views )[i].type() == typeid( std::weak_ptr<TYPE> ) ) {
            auto ptr = std::any_cast<std::weak_ptr<TYPE>>( ( *d_Views )[i] );
            auto vec = ptr.lock();
            if ( vec == v )
                return;
        }
    }
    std::weak_ptr<TYPE> ptr = v;
    d_Views->push_back( std::any( ptr ) );
}


/****************************************************************
 * Misc functions                                                *
 ****************************************************************/
inline const std::shared_ptr<Variable> Vector::getVariable() const { return d_Variable; }
inline std::shared_ptr<Variable> Vector::getVariable() { return d_Variable; }
inline void Vector::setVariable( const std::shared_ptr<Variable> name )
{
    AMP_ASSERT( name );
    d_Variable = name;
}


/****************************************************************
 * Wrappers for shared_ptr                                       *
 ****************************************************************/
inline void Vector::swapVectors( shared_ptr other ) { swapVectors( *other ); }
inline std::ostream &operator<<( std::ostream &out, const Vector::shared_ptr p )
{
    return operator<<( out, *p );
}


/****************************************************************
 * Get individual values                                         *
 ****************************************************************/
template<typename TYPE>
TYPE Vector::getValueByGlobalID( size_t i ) const
{
    TYPE ans;
    getValuesByGlobalID( 1, &i, &ans );
    return ans;
}
template<typename TYPE>
TYPE Vector::getLocalValueByGlobalID( size_t i ) const
{
    TYPE ans;
    getValuesByGlobalID( 1, &i, &ans );
    return ans;
}
template<typename TYPE>
TYPE Vector::getGhostValueByGlobalID( size_t i ) const
{
    PROFILE( "Vector::getGhostValueByGlobalID" );
    TYPE ans;
    getGhostValuesByGlobalID( 1, &i, &ans );
    return ans;
}
template<typename TYPE>
TYPE Vector::getValueByLocalID( size_t ndx ) const
{
    TYPE ans;
    getValuesByLocalID( 1, &ndx, &ans );
    return ans;
}
template<typename TYPE>
void Vector::setValueByGlobalID( size_t i, TYPE v )
{
    setValuesByGlobalID<TYPE>( 1, &i, &v );
}

} // namespace AMP::LinearAlgebra

#endif
