#include "AMP/vectors/trilinos/tpetra/TpetraVectorOperations.hpp"

namespace AMP::LinearAlgebra {

template class TpetraVectorOperations<double, int32_t, int32_t>;
template class TpetraVectorOperations<double, int32_t, int64_t>;
template class TpetraVectorOperations<float, int32_t, int64_t>;

} // namespace AMP::LinearAlgebra
