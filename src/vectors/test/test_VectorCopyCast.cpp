#include "AMP/AMP_TPLs.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/memory.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>

template<typename ALLOC, typename OPS_d, typename OPS_f>
void test_copyCast( size_t N,
                    std::shared_ptr<AMP::LinearAlgebra::Variable> &var,
                    AMP::AMP_MPI &globalComm,
                    AMP::UnitTest &ut,
                    const std::string &test_msg )
{
    using DATA_d = AMP::LinearAlgebra::VectorDataDefault<double, ALLOC>;
    using DATA_f = AMP::LinearAlgebra::VectorDataDefault<float, ALLOC>;

    auto tol  = std::numeric_limits<float>::epsilon();
    auto dvec = AMP::LinearAlgebra::createSimpleVector<double, OPS_d, DATA_d>( N, var, globalComm );
    dvec->setRandomValues();
    auto svec = AMP::LinearAlgebra::createSimpleVector<float, OPS_f, DATA_f>( N, var, globalComm );
    svec->copyCast( dvec );
    auto dvec2 =
        AMP::LinearAlgebra::createSimpleVector<double, OPS_d, DATA_d>( N, var, globalComm );
    dvec2->copyCast( svec );
    auto diff = AMP::LinearAlgebra::createSimpleVector<double, OPS_d, DATA_d>( N, var, globalComm );
    diff->subtract( *dvec, *dvec2 );

    auto err = static_cast<double>( diff->maxNorm() );
    if ( err > tol ) {
        ut.failure( AMP::Utilities::stringf(
            "Cast precision loss %e larger than %e, for %s test", err, tol, test_msg.data() ) );
    } else {
        ut.passes( AMP::Utilities::stringf( "%s tests passed", test_msg.data() ) );
    }
}

int main( int argc, char **argv )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    size_t N = 10;
    auto var = std::make_shared<AMP::LinearAlgebra::Variable>( "vec" );
    test_copyCast<AMP::HostAllocator<void>,
                  AMP::LinearAlgebra::VectorOperationsDefault<double>,
                  AMP::LinearAlgebra::VectorOperationsDefault<float>>(
        N, var, globalComm, ut, "CPU" );
#ifdef AMP_USE_DEVICE
    test_copyCast<AMP::ManagedAllocator<void>,
                  AMP::LinearAlgebra::VectorOperationsDevice<double>,
                  AMP::LinearAlgebra::VectorOperationsDevice<float>>(
        N, var, globalComm, ut, "Managed memory" );
#endif

    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
