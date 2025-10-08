#include "AMP/matrices/trilinos/tpetra/TpetraMatrixOperations.hpp"
#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.h"

namespace AMP::LinearAlgebra {

template class TpetraMatrixOperations<Tpetra_ST, Tpetra_LO, Tpetra_GO>;

} // namespace AMP::LinearAlgebra
