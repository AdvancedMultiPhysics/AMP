#ifndef included_TpetraMatrixHelpers_h
#define included_TpetraMatrixHelpers_h

#include "AMP/vectors/trilinos/tpetra/TpetraDefaults.h"

#include <memory>

namespace AMP::LinearAlgebra {

class Matrix;
template<typename ST, typename LO, typename GO, typename NT>
class ManagedTpetraMatrix;

template<typename ST = Tpetra_ST,
         typename LO = Tpetra_LO,
         typename GO = Tpetra_GO,
         typename NT = Tpetra::Vector<>::node_type>
std::shared_ptr<ManagedTpetraMatrix<ST, LO, GO, NT>> getTpetraMatrix( std::shared_ptr<Matrix> mat );
} // namespace AMP::LinearAlgebra
#endif
