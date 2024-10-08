#include "AMP/vectors/data/DataChangeListener.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/vectors/data/DataChangeFirer.h"

#include <algorithm>


namespace AMP::LinearAlgebra {

DataChangeListener::DataChangeListener() = default;

DataChangeListener::~DataChangeListener()
{
    for ( auto &x : d_firers )
        x->deregisterListener( this );
}

void DataChangeListener::registerWithFirer( DataChangeFirer *firer )
{
    AMP_ASSERT( firer );
    AMP_ASSERT( std::find( d_firers.begin(), d_firers.end(), firer ) == d_firers.end() );
    d_firers.push_back( firer );
}

void DataChangeListener::deregisterFromFirer( DataChangeFirer *firer )
{
    AMP_ASSERT( firer );
    AMP_ASSERT( std::find( d_firers.begin(), d_firers.end(), firer ) != d_firers.end() );
    d_firers.erase( std::find( d_firers.begin(), d_firers.end(), firer ) );
}
} // namespace AMP::LinearAlgebra
