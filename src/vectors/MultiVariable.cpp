#include "AMP/vectors/MultiVariable.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorSelector.h"

#include <algorithm>
#include <map>
#include <string>


namespace AMP::LinearAlgebra {


/****************************************************************
 * VectorSelector for multivariable                              *
 ****************************************************************/
VS_MultiVariable::VS_MultiVariable( const std::shared_ptr<MultiVariable> &var ) : d_var( var ) {}
std::string VS_MultiVariable::getName() const { return d_var->getName(); }
bool VS_MultiVariable::isSelected( const Vector & ) const { return true; }
Vector::shared_ptr VS_MultiVariable::subset( Vector::shared_ptr vec ) const
{
    auto var = vec->getVariable();
    if ( var ) {
        if ( var->getName() == d_var->getName() )
            return vec;
        for ( auto var2 : *d_var ) {
            if ( var->getName() == var2->getName() )
                return vec;
        }
    }
    return Vector::shared_ptr();
}
Vector::const_shared_ptr VS_MultiVariable::subset( Vector::const_shared_ptr vec ) const
{
    return subset( std::const_pointer_cast<Vector>( vec ) );
}


/****************************************************************
 * Constructors/Destructors                                      *
 ****************************************************************/
MultiVariable::MultiVariable( const std::string &name,
                              const std::vector<std::shared_ptr<Variable>> &vars )
    : Variable( name ), d_vVariables( vars )
{
}
MultiVariable::~MultiVariable() = default;


/****************************************************************
 * Get/set a variable                                            *
 ****************************************************************/
size_t MultiVariable::numVariables() const { return d_vVariables.size(); }
std::shared_ptr<Variable> MultiVariable::getVariable( size_t which )
{
    AMP_ASSERT( which < d_vVariables.size() );
    return d_vVariables[which];
}
std::shared_ptr<const Variable> MultiVariable::getVariable( size_t which ) const
{
    AMP_ASSERT( which < d_vVariables.size() );
    return d_vVariables[which];
}
void MultiVariable::setVariable( size_t i, std::shared_ptr<Variable> &p )
{
    AMP_ASSERT( i < d_vVariables.size() );
    d_vVariables[i] = p;
}
void MultiVariable::add( std::shared_ptr<Variable> newVar )
{
    auto multivariable = std::dynamic_pointer_cast<MultiVariable>( newVar );
    if ( multivariable ) {
        for ( auto var : *multivariable )
            add( var );
    } else {
        d_vVariables.push_back( newVar );
    }
}


/****************************************************************
 * Comparison operator                                           *
 ****************************************************************/
bool MultiVariable::operator==( const Variable &rhs ) const
{
    const auto *multivariable = dynamic_cast<const MultiVariable *>( &rhs );
    if ( multivariable == nullptr ) {
        // We are comparing a multi variable to another variable
        // Two variables match if the variable equals all sub-variable and the names match
        if ( rhs.getName() != this->getName() )
            return false;
        for ( size_t i = 0; i != d_vVariables.size(); i++ ) {
            if ( *d_vVariables[i] != rhs )
                return false;
        }
    } else {
        // We are dealing with two multivariables, check that the internal variables match
        for ( size_t i = 0; i != d_vVariables.size(); i++ ) {
            if ( i == multivariable->d_vVariables.size() )
                return false;
            if ( ( *d_vVariables[i] ) != ( *( multivariable->d_vVariables[i] ) ) )
                return false;
        }
    }
    return true;
}


/****************************************************************
 * cloneVariable                                                 *
 ****************************************************************/
std::shared_ptr<Variable> MultiVariable::cloneVariable( const std::string &name ) const
{
    std::shared_ptr<MultiVariable> retVal( new MultiVariable( name ) );
    retVal->d_vVariables.resize( d_vVariables.size() );
    std::copy( d_vVariables.begin(), d_vVariables.end(), retVal->d_vVariables.begin() );
    return retVal;
}


/****************************************************************
 * removeDuplicateVariables                                      *
 ****************************************************************/
void MultiVariable::removeDuplicateVariables()
{
    // First remove any NULL pointers
    auto iterator = d_vVariables.begin();
    while ( iterator != d_vVariables.end() ) {
        if ( iterator->get() == nullptr )
            iterator = d_vVariables.erase( iterator );
        else
            ++iterator;
    }
    // Next remove any duplicate entries
    // Note: while it would be faster to sort, then remove duplicate entires,
    // this requires the < operator to be overloaded for the base class
    std::vector<std::shared_ptr<Variable>> unique_list;
    unique_list.reserve( d_vVariables.size() );
    for ( auto &elem : d_vVariables ) {
        bool found = false;
        for ( auto &unique_list_j : unique_list ) {
            if ( elem->operator==( *unique_list_j ) )
                found = true;
        }
        if ( !found )
            unique_list.push_back( elem );
    }
    d_vVariables = unique_list;
}


/****************************************************************
 * setUnits                                                      *
 ****************************************************************/
void MultiVariable::setUnits( const Units &units )
{
    Variable::setUnits( units );
    for ( auto var : *this )
        var->setUnits( units );
}


/****************************************************************
 * createVectorSelector                                          *
 ****************************************************************/
std::shared_ptr<VectorSelector> MultiVariable::createVectorSelector() const
{
    auto multivar = std::dynamic_pointer_cast<MultiVariable>( cloneVariable( getName() ) );
    return std::make_shared<VS_MultiVariable>( multivar );
}


} // namespace AMP::LinearAlgebra
