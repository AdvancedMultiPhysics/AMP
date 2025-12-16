#include "AMP/vectors/operations/kokkos/VectorOperationsKokkos.hpp"

#ifdef AMP_USE_KOKKOS

// Explicit instantiations
template class AMP::LinearAlgebra::VectorOperationsKokkos<double>;
template class AMP::LinearAlgebra::VectorOperationsKokkos<float>;

#endif
