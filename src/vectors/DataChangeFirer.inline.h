

#include "DataChangeListener.h"

namespace AMP {
namespace LinearAlgebra {

inline DataChangeFirer::DataChangeFirer() {}

inline DataChangeFirer::~DataChangeFirer()
{
    for ( auto &elem : *this )
        ( elem )->deregisterFromFirer( this );
}

inline void DataChangeFirer::registerListener( DataChangeListener *listener )
{
    AMP_ASSERT( listener );
    AMP_ASSERT( std::find( begin(), end(), listener ) == end() );

    listener->registerWithFirer( this );
    push_back( listener );
}

inline void DataChangeFirer::deregisterListener( DataChangeListener *listener )
{
    AMP_ASSERT( listener );
    erase( std::find( begin(), end(), listener ) );
}


inline void DataChangeFirer::fireDataChange()
{
    for ( auto &elem : *this )
        ( elem )->dataChanged();
}
}
}
