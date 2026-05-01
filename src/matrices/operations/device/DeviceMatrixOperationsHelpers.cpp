#include "AMP/matrices/operations/device/DeviceMatrixOperations.hpp"

// explicit instantiations
template struct AMP::LinearAlgebra::DeviceMatrixOperations<size_t, int, double>;
template struct AMP::LinearAlgebra::DeviceMatrixOperations<int, int, double>;
template struct AMP::LinearAlgebra::DeviceMatrixOperations<long long int, int, double>;
template struct AMP::LinearAlgebra::DeviceMatrixOperations<int, size_t, double>;
template struct AMP::LinearAlgebra::DeviceMatrixOperations<size_t, size_t, double>;

template struct AMP::LinearAlgebra::DeviceMatrixOperations<size_t, int, float>;
template struct AMP::LinearAlgebra::DeviceMatrixOperations<int, int, float>;
template struct AMP::LinearAlgebra::DeviceMatrixOperations<long long int, int, float>;
template struct AMP::LinearAlgebra::DeviceMatrixOperations<int, size_t, float>;
template struct AMP::LinearAlgebra::DeviceMatrixOperations<size_t, size_t, float>;
