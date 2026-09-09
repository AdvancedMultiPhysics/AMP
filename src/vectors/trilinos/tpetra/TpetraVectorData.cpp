#include "AMP/vectors/trilinos/tpetra/TpetraVectorData.hpp"
#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.hpp"

namespace AMP::LinearAlgebra {

template class TpetraVectorData<double, int32_t, int32_t, TpetraNT>;
template class TpetraVectorData<double, int32_t, int64_t, TpetraNT>;
template class TpetraVectorData<float, int32_t, int32_t, TpetraNT>;
template class TpetraVectorData<float, int32_t, int64_t, TpetraNT>;

} // namespace AMP::LinearAlgebra
