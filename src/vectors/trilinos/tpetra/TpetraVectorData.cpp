#include "AMP/vectors/trilinos/tpetra/TpetraVectorData.hpp"

namespace AMP::LinearAlgebra {

template class TpetraVectorData<double, int32_t, int32_t>;
template class TpetraVectorData<double, int32_t, int64_t>;
template class TpetraVectorData<float, int32_t, int64_t>;

} // namespace AMP::LinearAlgebra
