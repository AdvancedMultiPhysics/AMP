#include "AMP/vectors/trilinos/tpetra/TpetraVectorData.hpp"
#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.h"

namespace AMP::LinearAlgebra {

template class TpetraVectorData<double, int32_t, int32_t, Tpetra_NT>;
template class TpetraVectorData<double, int32_t, int64_t, Tpetra_NT>;
template class TpetraVectorData<float, int32_t, int32_t, Tpetra_NT>;
template class TpetraVectorData<float, int32_t, int64_t, Tpetra_NT>;

} // namespace AMP::LinearAlgebra
