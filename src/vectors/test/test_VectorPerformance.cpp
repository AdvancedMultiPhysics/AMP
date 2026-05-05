#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Memory.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/data/VectorDataDefault.h"

#ifdef AMP_USE_OPENMP
    #include "AMP/vectors/operations/OpenMP/VectorOperationsOpenMP.h"
#endif
#ifdef AMP_USE_DEVICE
    #include "AMP/vectors/data/device/VectorDataDevice.h"
    #include "AMP/vectors/operations/device/VectorOperationsDevice.h"
#endif
#ifdef AMP_USE_KOKKOS
    #include "AMP/vectors/operations/kokkos/VectorOperationsKokkos.h"
#endif


#include <chrono>


static inline double speedup( int x, int y )
{
    return static_cast<double>( y ) / static_cast<double>( x );
}


struct test_times {
    int clone       = 0;
    int zero        = 0;
    int setToScalar = 0;
    int setRandom   = 0;
    int L1Norm      = 0;
    int L2Norm      = 0;
    int maxNorm     = 0;
    int axpy        = 0;
    int dot         = 0;
    int min         = 0;
    int max         = 0;
    int multiply    = 0;
    int divide      = 0;
    void print()
    {
        printf( "    clone: %i us\n", clone );
        printf( "    zero: %i us\n", zero );
        printf( "    setToScalar: %i us\n", setToScalar );
        printf( "    setRandom: %i us\n", setRandom );
        printf( "    L1Norm: %i us\n", L1Norm );
        printf( "    L2Norm: %i us\n", L2Norm );
        printf( "    maxNorm: %i us\n", maxNorm );
        printf( "    axpy: %i us\n", axpy );
        printf( "    min: %i us\n", min );
        printf( "    max: %i us\n", max );
        printf( "    dot: %i us\n", dot );
        printf( "    multiply: %i us\n", multiply );
        printf( "    divide: %i us\n", divide );
    }
    void print_speedup( const test_times &time0 )
    {
        printf( "  Speedup: \n" );
        printf( "    clone: %0.2f\n", speedup( clone, time0.clone ) );
        printf( "    zero: %0.2f\n", speedup( zero, time0.zero ) );
        printf( "    setToScalar: %0.2f\n", speedup( setToScalar, time0.setToScalar ) );
        printf( "    setRandom: %0.2f\n", speedup( setRandom, time0.setRandom ) );
        printf( "    L1Norm: %0.2f\n", speedup( L1Norm, time0.L1Norm ) );
        printf( "    L2Norm: %0.2f\n", speedup( L2Norm, time0.L2Norm ) );
        printf( "    maxNorm: %0.2f\n", speedup( maxNorm, time0.maxNorm ) );
        printf( "    axpy: %0.2f\n", speedup( axpy, time0.axpy ) );
        printf( "    min: %0.2f\n", speedup( min, time0.min ) );
        printf( "    max: %0.2f\n", speedup( max, time0.max ) );
        printf( "    dot: %0.2f\n", speedup( dot, time0.dot ) );
        printf( "    multiply: %0.2f\n", speedup( multiply, time0.multiply ) );
        printf( "    divide: %0.2f\n", speedup( divide, time0.divide ) );
    }
};


#define to_us( t1, t2, N ) \
    std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count() / N;

#define runTestN( TEST, ... )                          \
    do {                                               \
        auto t1 = std::chrono::steady_clock::now();    \
        for ( size_t i = 0; i < 20; ++i )              \
            vec->TEST( __VA_ARGS__ );                  \
        auto t2    = std::chrono::steady_clock::now(); \
        times.TEST = to_us( t1, t2, 20 );              \
    } while ( 0 )
#define runTest0( TEST )                               \
    do {                                               \
        auto t1 = std::chrono::steady_clock::now();    \
        for ( size_t i = 0; i < 20; ++i )              \
            vec->TEST();                               \
        auto t2    = std::chrono::steady_clock::now(); \
        times.TEST = to_us( t1, t2, 20 );              \
    } while ( 0 )


test_times testPerformance( AMP::LinearAlgebra::Vector::shared_ptr vec )
{
    test_times times;
    // Test the performance of clone
    auto t1   = std::chrono::steady_clock::now();
    auto vec2 = vec->clone();
    auto vec3 = vec->clone();
    auto t2   = std::chrono::steady_clock::now();
    vec2->setRandomValues();
    vec3->setRandomValues();
    auto t3         = std::chrono::steady_clock::now();
    times.clone     = to_us( t1, t2, 2 );
    times.setRandom = to_us( t2, t3, 2 );
    // Run the tests
    runTest0( zero );
    runTestN( setToScalar, 3.14 );
    runTest0( L1Norm );
    runTest0( L2Norm );
    runTest0( maxNorm );
    runTest0( min );
    runTest0( max );
    runTestN( dot, *vec2 );
    runTestN( axpy, 2.5, *vec2, *vec3 );
    runTestN( multiply, *vec2, *vec3 );
    runTestN( divide, *vec2, *vec3 );
    return times;
}


int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );

    {
        AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
        int rank = globalComm.getRank();

#if ( defined( DEBUG ) || defined( _DEBUG ) ) && !defined( NDEBUG )
        size_t N = 1e6;
#else
        size_t N = 4e6;
#endif

        auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "vec" );

        auto vec   = AMP::LinearAlgebra::createSimpleVector<double>( N, var, globalComm );
        auto time0 = testPerformance( vec );
        if ( rank == 0 ) {
            AMP::pout << "SimpleVector:" << std::endl;
            time0.print();
            AMP::pout << std::endl;
        }

#ifdef AMP_USE_OPENMP
        {
            vec = AMP::LinearAlgebra::
                createSimpleVector<double, AMP::LinearAlgebra::VectorOperationsOpenMP<double>>(
                    N, var, globalComm );
            auto time_openmp = testPerformance( vec );
            if ( rank == 0 ) {
                AMP::pout << "SimpleVector<OpenMP>:" << std::endl;
                time_openmp.print();
                time_openmp.print_speedup( time0 );
                AMP::pout << std::endl;
            }
        }
#endif

#ifdef AMP_USE_KOKKOS
        {
    #ifdef AMP_USE_DEVICE
            using ALLOC = AMP::DeviceAllocator<void>;
            using DATA  = AMP::LinearAlgebra::VectorDataDevice<double, ALLOC>;
    #else
            using ALLOC = AMP::HostAllocator<void>;
            using DATA  = AMP::LinearAlgebra::VectorDataDefault<double, ALLOC>;
    #endif
            using OPS = AMP::LinearAlgebra::VectorOperationsKokkos<double>;
            vec = AMP::LinearAlgebra::createSimpleVector<double, OPS, DATA>( N, var, globalComm );
            auto time_kokkos = testPerformance( vec );
            if ( rank == 0 ) {
                AMP::pout << "SimpleVector<Device>(kokkos):" << std::endl;
                time_kokkos.print();
                time_kokkos.print_speedup( time0 );
                AMP::pout << std::endl;
            }
        }
        {
    #ifdef AMP_USE_DEVICE
            using ALLOC = AMP::ManagedAllocator<void>;
            using DATA  = AMP::LinearAlgebra::VectorDataDevice<double, ALLOC>;
    #else
            using ALLOC = AMP::HostAllocator<void>;
            using DATA  = AMP::LinearAlgebra::VectorDataDefault<double, ALLOC>;
    #endif
            using OPS = AMP::LinearAlgebra::VectorOperationsKokkos<double>;
            vec = AMP::LinearAlgebra::createSimpleVector<double, OPS, DATA>( N, var, globalComm );
            auto time_kokkos = testPerformance( vec );
            if ( rank == 0 ) {
                AMP::pout << "SimpleVector<Managed>(kokkos):" << std::endl;
                time_kokkos.print();
                time_kokkos.print_speedup( time0 );
                AMP::pout << std::endl;
            }
        }
#endif

#ifdef AMP_USE_DEVICE
        {
            using ALLOC = AMP::ManagedAllocator<void>;
            using DATA  = AMP::LinearAlgebra::VectorDataDevice<double, ALLOC>;
            using OPS   = AMP::LinearAlgebra::VectorOperationsDevice<double>;
            vec = AMP::LinearAlgebra::createSimpleVector<double, OPS, DATA>( N, var, globalComm );
            auto time_cuda = testPerformance( vec );
            if ( rank == 0 ) {
                AMP::pout << "SimpleVector<Managed>:" << std::endl;
                time_cuda.print();
                time_cuda.print_speedup( time0 );
                AMP::pout << std::endl;
            }
        }
        {
            using ALLOC = AMP::DeviceAllocator<void>;
            using DATA  = AMP::LinearAlgebra::VectorDataDevice<double, ALLOC>;
            using OPS   = AMP::LinearAlgebra::VectorOperationsDevice<double>;
            vec = AMP::LinearAlgebra::createSimpleVector<double, OPS, DATA>( N, var, globalComm );
            auto time_cuda = testPerformance( vec );
            if ( rank == 0 ) {
                AMP::pout << "SimpleVector<Device>:" << std::endl;
                time_cuda.print();
                time_cuda.print_speedup( time0 );
                AMP::pout << std::endl;
            }
        }
#endif
    }

    AMP::AMPManager::shutdown();
    return 0;
}
