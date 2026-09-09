#include "AMP/vectors/trilinos/tpetra/TpetraVectorData.hpp"
#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.h"

namespace AMP::LinearAlgebra {

template class TpetraVectorData<Tpetra_ST, Tpetra_LO, Tpetra_GO, Tpetra_NT>;

} // namespace AMP::LinearAlgebra
