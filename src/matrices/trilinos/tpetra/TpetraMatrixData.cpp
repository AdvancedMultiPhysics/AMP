#include "AMP/matrices/trilinos/tpetra/TpetraMatrixData.hpp"
#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.h"

namespace AMP::LinearAlgebra {

template class TpetraMatrixData<Tpetra_ST, Tpetra_LO, Tpetra_GO>;

} // namespace AMP::LinearAlgebra
