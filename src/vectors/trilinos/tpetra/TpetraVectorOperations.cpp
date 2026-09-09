#include "AMP/vectors/trilinos/tpetra/TpetraVectorOperations.hpp"
#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.h"

namespace AMP::LinearAlgebra {

template class TpetraVectorOperations<double, int32_t, int32_t, Tpetra_NT>;
template class TpetraVectorOperations<double, int32_t, int64_t, Tpetra_NT>;
template class TpetraVectorOperations<float, int32_t, int32_t, Tpetra_NT>;
template class TpetraVectorOperations<float, int32_t, int64_t, Tpetra_NT>;

} // namespace AMP::LinearAlgebra
