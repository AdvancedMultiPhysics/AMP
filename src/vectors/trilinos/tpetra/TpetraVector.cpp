#include "AMP/vectors/trilinos/tpetra/TpetraVector.hpp"
#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.h"

namespace AMP::LinearAlgebra {

template class TpetraVector<Tpetra_ST, Tpetra_LO, Tpetra_GO, Tpetra_NT>;

} // namespace AMP::LinearAlgebra
