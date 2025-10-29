// this is just meant to experiment with how best to include
// templated kernels
#include "test.hpp"
#include <cuda.h>


__global__ void boo2( void ) {}

void boo( void ) { boo2<<<1, 1>>>(); }
