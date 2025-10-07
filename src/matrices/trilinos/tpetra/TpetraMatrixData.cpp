#include "AMP/matrices/trilinos/tpetra/TpetraMatrixData.hpp"

namespace AMP::LinearAlgebra {

template class TpetraMatrixData<double, int32_t, int64_t>;
template class TpetraMatrixData<float, int32_t, int64_t>;

} // namespace AMP::LinearAlgebra
