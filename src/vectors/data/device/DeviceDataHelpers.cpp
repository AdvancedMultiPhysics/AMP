#include "AMP/vectors/data/device/DeviceDataHelpers.h"
#include "AMP/vectors/data/device/DeviceDataHelpers.hpp"


// Explicit instantiations
template class AMP::LinearAlgebra::DeviceDataHelpers<double>;
template class AMP::LinearAlgebra::DeviceDataHelpers<float>;
template class AMP::LinearAlgebra::DeviceDataHelpers<float, double>;
template class AMP::LinearAlgebra::DeviceDataHelpers<double, float>;
