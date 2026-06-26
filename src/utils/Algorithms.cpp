#include "AMP/utils/Algorithms.hpp"

#include <complex>

#define INST_SIMPLE( TYPE )                                                 \
    template void AMP::Utilities::Algorithms::fill_n<TYPE>(                 \
        TYPE *, const size_t, const TYPE, const MemoryType );               \
    template void AMP::Utilities::Algorithms::zero_n<TYPE>(                 \
        TYPE *, const size_t, const MemoryType );                           \
    template void AMP::Utilities::Algorithms::exclusive_scan<TYPE>(         \
        const TYPE *, const size_t, TYPE *, const TYPE, const MemoryType ); \
    template void AMP::Utilities::Algorithms::inclusive_scan<TYPE>(         \
        const TYPE *, const size_t, TYPE *, const MemoryType );             \
    template void AMP::Utilities::Algorithms::sort<TYPE>(                   \
        TYPE *, const size_t, const MemoryType );                           \
    template TYPE AMP::Utilities::Algorithms::min_element<TYPE>(            \
        const TYPE *, const size_t, const MemoryType );                     \
    template TYPE AMP::Utilities::Algorithms::max_element<TYPE>(            \
        const TYPE *, const size_t, const MemoryType );                     \
    template TYPE AMP::Utilities::Algorithms::accumulate<TYPE>(             \
        const TYPE *, const size_t, const TYPE, const MemoryType );         \
    template size_t AMP::Utilities::Algorithms::unique<TYPE>(               \
        TYPE *, const size_t, const MemoryType );

#define INST_COPY( TYPE )                                       \
    template void AMP::Utilities::Algorithms::copy_n<TYPE>(     \
        TYPE *, const TYPE *, const size_t, const MemoryType ); \
    template void AMP::Utilities::Algorithms::copy_n<TYPE>(     \
        TYPE *, const MemoryType, const TYPE *, const MemoryType, const size_t );


#define INST_COPYCAST( TDst, TSrc )                                 \
    template void AMP::Utilities::Algorithms::copyCast<TDst, TSrc>( \
        TDst *, const MemoryType, const TSrc *, const MemoryType, size_t );

INST_SIMPLE( int )
INST_SIMPLE( unsigned long )
INST_SIMPLE( long long )
INST_SIMPLE( double )
INST_SIMPLE( float )

INST_COPY( int )
INST_COPY( unsigned long )
INST_COPY( long long )
INST_COPY( double )
INST_COPY( float )
INST_COPY( std::complex<double> )
INST_COPY( std::complex<float> )

INST_COPYCAST( int, int )
INST_COPYCAST( unsigned long, unsigned long )
INST_COPYCAST( long long, long long )
INST_COPYCAST( double, double )
INST_COPYCAST( float, float )

INST_COPYCAST( unsigned long, long long )
INST_COPYCAST( long long, unsigned long )
INST_COPYCAST( double, float )
INST_COPYCAST( float, double )
