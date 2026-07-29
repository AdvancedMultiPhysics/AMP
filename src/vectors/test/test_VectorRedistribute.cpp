#include "AMP/utils/AMPManager.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorHelpers.h"

#include <algorithm>
#include <numeric>
#include <vector>

using namespace AMP::LinearAlgebra;

namespace {

std::shared_ptr<Vector>
createRankWeightedVector( const AMP::AMP_MPI &comm, const std::string &name, double scale )
{
    const size_t local_size = static_cast<size_t>( comm.getRank() + 2 );
    auto var                = std::make_shared<Variable>( name );
    auto vec                = createSimpleVector<double>( local_size, var, comm );

    std::vector<double> values( local_size, 0.0 );
    const size_t global_start = vec->getDOFManager()->beginDOF();
    for ( size_t i = 0; i < local_size; ++i ) {
        values[i] = scale * static_cast<double>( global_start + i + 1 );
    }

    vec->putRawData( values.data() );
    vec->makeConsistent( ScatterType::CONSISTENT_SET );
    return vec;
}

} // namespace

int main( int argc, char **argv )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    {
        AMP::AMP_MPI comm( AMP_COMM_WORLD );
        auto x = createRankWeightedVector( comm, "x", 1.0 );
        auto y = createRankWeightedVector( comm, "y", 2.0 );

        const int reduced_nprocs = std::max( 1, comm.getSize() / 2 );
        auto context = AMP::Utilities::createGroupedRedistributionPlan( comm, reduced_nprocs );

        auto x_red      = VectorHelpers::redistribute( x, context );
        auto y_red      = VectorHelpers::redistribute( y, context );
        auto x_red_into = x_red ? x_red->clone() : nullptr;
        if ( x_red_into ) {
            x_red_into->zero();
        }

        VectorHelpers::redistribute( context, x, x_red_into );

        auto x_scattered = x->clone();
        x_scattered->zero();
        VectorHelpers::scatterRedistributed( context, x_red_into, x_scattered );

        auto gathered_sizes =
            context.groupComm().allGather( static_cast<int>( x->getLocalSize() ) );
        size_t expected_local_size = 0;
        for ( auto size : gathered_sizes ) {
            expected_local_size += static_cast<size_t>( size );
        }

        const size_t global_size = x->getGlobalSize();
        const double expected_x_sum =
            static_cast<double>( global_size ) * static_cast<double>( global_size + 1 ) / 2.0;

        if ( context.isActive() ) {
            if ( !x_red || !y_red ) {
                ut.failure( "redistributed active vectors must be non-null" );
            } else {
                if ( x_red->getComm().getSize() != reduced_nprocs ) {
                    ut.failure( "redistributed vector communicator size is incorrect" );
                }
                if ( x_red->getLocalSize() != expected_local_size ) {
                    ut.failure( "redistributed vector local size is incorrect" );
                }
                if ( x_red->getGlobalSize() != global_size ) {
                    ut.failure( "redistributed vector global size changed" );
                }
                if ( std::abs( static_cast<double>( x_red->sum() ) - expected_x_sum ) > 1e-10 ) {
                    ut.failure( "redistributed vector sum is incorrect" );
                }
                if ( std::abs( static_cast<double>( y_red->sum() ) - 2.0 * expected_x_sum ) >
                     1e-10 ) {
                    ut.failure( "second redistributed vector sum is incorrect" );
                }
                if ( !x_red_into || !x_red_into->equals( *x_red, 1e-10 ) ) {
                    ut.failure( "in-place redistributed vector values are incorrect" );
                }
            }
        } else if ( x_red || y_red ) {
            ut.failure( "inactive ranks must not receive redistributed vectors" );
        }

        if ( !x_scattered->equals( *x, 1e-10 ) ) {
            ut.failure( "scattered redistributed vector values are incorrect" );
        }
    }

    ut.report();
    AMP::AMPManager::shutdown();
    return ut.NumFailGlobal();
}
