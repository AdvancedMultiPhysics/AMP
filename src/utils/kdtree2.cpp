#include "AMP/utils/kdtree2.h"
#include "AMP/utils/Utilities.hpp"
#include "AMP/utils/kdtree2.hpp"


/********************************************************
 *  Explicit instantiations                              *
 ********************************************************/
#define INSANTIATE2( TYPE, NDIM )                                                    \
    template class AMP::kdtree2<NDIM, TYPE>;                                         \
    template void AMP::Utilities::quicksort<double, std::array<double, NDIM>, TYPE>( \
        size_t, double *, std::array<double, NDIM> *, TYPE * )
#define INSANTIATE( TYPE )  \
    INSANTIATE2( TYPE, 1 ); \
    INSANTIATE2( TYPE, 2 ); \
    INSANTIATE2( TYPE, 3 ); \
    INSANTIATE2( TYPE, 4 ); \
    INSANTIATE2( TYPE, 5 )
INSANTIATE( bool );
INSANTIATE( int );
INSANTIATE( float );
INSANTIATE( double );
