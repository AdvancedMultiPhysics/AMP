#include "AMP/vectors/trilinos/tpetra/TpetraVectorOperations.hpp"
#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.hpp"

namespace AMP::LinearAlgebra {

template class TpetraVectorOperations<double, int32_t, int32_t, TpetraNT>;
template class TpetraVectorOperations<double, int32_t, int64_t, TpetraNT>;
template class TpetraVectorOperations<float, int32_t, int32_t, TpetraNT>;
template class TpetraVectorOperations<float, int32_t, int64_t, TpetraNT>;

} // namespace AMP::LinearAlgebra
