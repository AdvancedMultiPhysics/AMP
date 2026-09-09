#include "AMP/vectors/trilinos/tpetra/TpetraVectorOperations.hpp"
#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.h"

namespace AMP::LinearAlgebra {

template class TpetraVectorOperations<Tpetra_ST, Tpetra_LO, Tpetra_GO, Tpetra_NT>;

} // namespace AMP::LinearAlgebra
