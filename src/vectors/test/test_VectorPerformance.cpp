#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
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

#include "ProfilerApp.h"

#include <chrono>

static inline double speedup( size_t x, size_t y )
{
    return static_cast<double>( y ) / static_cast<double>( x );
}

struct test_times {
    size_t clone;
    size_t zero;
    size_t setToScalar;
    size_t setRandomValues;
    size_t L1Norm;
    size_t L2Norm;
    size_t maxNorm;
    size_t axpy;
    size_t dot;
    size_t min;
    size_t max;
    size_t multiply;
    size_t divide;
    test_times()
        : clone( 0 ),
          zero( 0 ),
          setToScalar( 0 ),
          setRandomValues( 0 ),
          L1Norm( 0 ),
          L2Norm( 0 ),
          maxNorm( 0 ),
          axpy( 0 ),
          dot( 0 ),
          min( 0 ),
          max( 0 ),
          multiply( 0 ),
          divide( 0 )
    {
    }
    void print()
    {
        AMP::pout << "    clone: " << clone / 1000 << " us" << std::endl;
        AMP::pout << "    zero: " << zero / 1000 << " us" << std::endl;
        AMP::pout << "    setToScalar: " << setToScalar / 1000 << " us" << std::endl;
        AMP::pout << "    setRandom: " << setRandomValues / 1000 << " us" << std::endl;
        AMP::pout << "    L1Norm: " << L1Norm / 1000 << " us" << std::endl;
        AMP::pout << "    L2Norm: " << L2Norm / 1000 << " us" << std::endl;
        AMP::pout << "    maxNorm: " << maxNorm / 1000 << " us" << std::endl;
        AMP::pout << "    axpy: " << axpy / 1000 << " us" << std::endl;
        AMP::pout << "    min: " << min / 1000 << " us" << std::endl;
        AMP::pout << "    max: " << max / 1000 << " us" << std::endl;
        AMP::pout << "    dot: " << dot / 1000 << " us" << std::endl;
        AMP::pout << "    multiply: " << multiply / 1000 << " us" << std::endl;
        AMP::pout << "    divide: " << divide / 1000 << " us" << std::endl;
    }
    void print_speedup( const test_times &time0 )
    {
        AMP::pout << "  Speedup: " << std::endl;
        AMP::pout << "    clone: " << speedup( clone, time0.clone ) << std::endl;
        AMP::pout << "    zero: " << speedup( zero, time0.zero ) << std::endl;
        AMP::pout << "    setToScalar: " << speedup( setToScalar, time0.setToScalar ) << std::endl;
        AMP::pout << "    setRandom: " << speedup( setRandomValues, time0.setRandomValues )
                  << std::endl;
        AMP::pout << "    L1Norm: " << speedup( L1Norm, time0.L1Norm ) << std::endl;
        AMP::pout << "    L2Norm: " << speedup( L2Norm, time0.L2Norm ) << std::endl;
        AMP::pout << "    maxNorm: " << speedup( maxNorm, time0.maxNorm ) << std::endl;
        AMP::pout << "    axpy: " << speedup( axpy, time0.axpy ) << std::endl;
        AMP::pout << "    min: " << speedup( min, time0.min ) << std::endl;
        AMP::pout << "    max: " << speedup( max, time0.max ) << std::endl;
        AMP::pout << "    dot: " << speedup( dot, time0.dot ) << std::endl;
        AMP::pout << "    multiply: " << speedup( multiply, time0.multiply ) << std::endl;
        AMP::pout << "    divide: " << speedup( divide, time0.divide ) << std::endl;
    }
};

#define runTest0( TEST )                                                                    \
    do {                                                                                    \
        auto t1 = std::chrono::steady_clock::now();                                         \
        for ( size_t i = 0; i < 50; ++i )                                                   \
            vec->TEST();                                                                    \
        auto t2 = std::chrono::steady_clock::now();                                         \
        times.TEST =                                                                        \
            std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count() / 50.0; \
    } while ( 0 )
#define runTest1( TEST, X )                                                                 \
    do {                                                                                    \
        auto t1 = std::chrono::steady_clock::now();                                         \
        for ( size_t i = 0; i < 50; ++i )                                                   \
            vec->TEST( X );                                                                 \
        auto t2 = std::chrono::steady_clock::now();                                         \
        times.TEST =                                                                        \
            std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count() / 50.0; \
    } while ( 0 )
#define runTest2( TEST, X, Y )                                                              \
    do {                                                                                    \
        auto t1 = std::chrono::steady_clock::now();                                         \
        for ( size_t i = 0; i < 50; ++i )                                                   \
            vec->TEST( X, Y );                                                              \
        auto t2 = std::chrono::steady_clock::now();                                         \
        times.TEST =                                                                        \
            std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count() / 50.0; \
    } while ( 0 )
#define runTest3( TEST, X, Y, Z )                                                           \
    do {                                                                                    \
        auto t1 = std::chrono::steady_clock::now();                                         \
        for ( size_t i = 0; i < 50; ++i )                                                   \
            vec->TEST( X, Y, Z );                                                           \
        auto t2 = std::chrono::steady_clock::now();                                         \
        times.TEST =                                                                        \
            std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count() / 50.0; \
    } while ( 0 )

test_times testPerformance( AMP::LinearAlgebra::Vector::shared_ptr vec )
{
    test_times times;
    // Test the performance of clone
    auto t1     = std::chrono::steady_clock::now();
    auto vec2   = vec->clone();
    auto t2     = std::chrono::steady_clock::now();
    times.clone = std::chrono::duration_cast<std::chrono::nanoseconds>( t2 - t1 ).count();
    auto vec3   = vec->clone();
    vec2->setRandomValues();
    vec3->setRandomValues();
    // Run the tests
    runTest0( zero );
    runTest1( setToScalar, 3.14 );
    runTest0( setRandomValues );
    runTest0( L1Norm );
    runTest0( L2Norm );
    runTest0( maxNorm );
    runTest0( min );
    runTest0( max );
    runTest1( dot, *vec2 );
    runTest3( axpy, 2.5, *vec2, *vec3 );
    runTest2( multiply, *vec2, *vec3 );
    runTest2( multiply, *vec2, *vec3 );

    return times;
}

int main( int argc, char **argv )
{

    AMP::AMPManager::startup( argc, argv );
    PROFILE_DISABLE();

    {
        AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
        int rank = globalComm.getRank();

        size_t N = 4e6;
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
