#ifndef included_EpetraMatrixHelpers_h
#define included_EpetraMatrixHelpers_h

#include <memory>

namespace AMP::LinearAlgebra {

class Matrix;
class ManagedEpetraMatrix;

std::shared_ptr<ManagedEpetraMatrix> getEpetraMatrix( std::shared_ptr<Matrix> mat );
} // namespace AMP::LinearAlgebra
#endif
