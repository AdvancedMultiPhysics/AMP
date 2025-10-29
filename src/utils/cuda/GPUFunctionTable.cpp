#include "AMP/utils/cuda/GPUFunctionTable.hpp"

namespace AMP {

template<>
void GPUFunctionTable::rand<int>( size_t n, int *d_x )
{
    curandGenerator_t gen;
    curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT );
    curandSetPseudoRandomGeneratorSeed( gen, time( NULL ) );
    curandGenerate( gen, (unsigned int *) d_x, n );
    curandDestroyGenerator( gen );
}

template<>
void GPUFunctionTable::rand<float>( size_t n, float *d_x )
{
    curandGenerator_t gen;
    curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT );
    curandSetPseudoRandomGeneratorSeed( gen, time( NULL ) );
    curandGenerateUniform( gen, d_x, n );
    curandDestroyGenerator( gen );
}

template<>
void GPUFunctionTable::rand<double>( size_t n, double *d_x )
{
    curandGenerator_t gen;
    curandCreateGenerator( &gen, CURAND_RNG_PSEUDO_DEFAULT );
    curandSetPseudoRandomGeneratorSeed( gen, time( NULL ) );
    curandGenerateUniformDouble( gen, d_x, n );
    curandDestroyGenerator( gen );
}

template double
AMP::GPUFunctionTable::sum<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>>(
    AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> const & );
template void AMP::GPUFunctionTable::
    transformReLU<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>>(
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> const &,
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> & );
template void AMP::GPUFunctionTable::
    transformAbs<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>>(
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> const &,
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> & );
template void AMP::GPUFunctionTable::
    transformHardTanh<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>>(
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> const &,
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> & );
template void AMP::GPUFunctionTable::
    transformTanh<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>>(
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> const &,
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> & );
template void AMP::GPUFunctionTable::
    transformSigmoid<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>>(
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> const &,
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> & );
template void AMP::GPUFunctionTable::
    transformSoftPlus<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>>(
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> const &,
        AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> & );
template bool
AMP::GPUFunctionTable::equals<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>>(
    AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> const &,
    AMP::Array<double, AMP::GPUFunctionTable, AMP::CudaManagedAllocator<double>> const &,
    double );
} // namespace AMP
