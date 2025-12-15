#include "AMP/matrices/operations/device/DeviceMatrixOperations.hpp"

// explicit instantiations
template class AMP::LinearAlgebra::DeviceMatrixOperations<size_t, int, double>;
template class AMP::LinearAlgebra::DeviceMatrixOperations<int, int, double>;
template class AMP::LinearAlgebra::DeviceMatrixOperations<long long int, int, double>;
template class AMP::LinearAlgebra::DeviceMatrixOperations<int, size_t, double>;
template class AMP::LinearAlgebra::DeviceMatrixOperations<size_t, size_t, double>;

template class AMP::LinearAlgebra::DeviceMatrixOperations<size_t, int, float>;
template class AMP::LinearAlgebra::DeviceMatrixOperations<int, int, float>;
template class AMP::LinearAlgebra::DeviceMatrixOperations<long long int, int, float>;
template class AMP::LinearAlgebra::DeviceMatrixOperations<int, size_t, float>;
template class AMP::LinearAlgebra::DeviceMatrixOperations<size_t, size_t, float>;
