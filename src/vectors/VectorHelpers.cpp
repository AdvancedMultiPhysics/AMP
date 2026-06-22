#include "AMP/vectors/VectorHelpers.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra::VectorHelpers {

namespace {

AMP::Utilities::Backend getBackend( const Vector &vec )
{
    auto name = vec.getVectorOperations()->VectorOpName();
    if ( name == "VectorOperationsKokkos" ) {
        return AMP::Utilities::Backend::Kokkos;
    } else if ( name == "VectorOperationsDevice" ) {
        return AMP::Utilities::Backend::Hip_Cuda;
    } else if ( name == "VectorOperationsOpenMP" ) {
        return AMP::Utilities::Backend::OpenMP;
    }
    return AMP::Utilities::Backend::Serial;
}

template<typename TYPE>
struct GatheredLeafValues {
    std::vector<int> sizes;
    std::vector<TYPE> values;
};

void checkParentVector( const Vector &vec,
                        const GroupedRedistributionContext &context,
                        const std::string &name )
{
    AMP_INSIST( !context.groupComm().isNull(), name + ": grouped context is not initialized" );
    AMP_INSIST( !context.parentComm().isNull(), name + ": parent communicator is not initialized" );
    AMP_INSIST( vec.getComm().getSize() == context.parentComm().getSize() &&
                    vec.getComm().getRank() == context.parentComm().getRank(),
                name + ": vector communicator is incompatible with redistribution plan" );
}

size_t sumSizes( const std::vector<int> &sizes )
{
    size_t total = 0;
    for ( auto size : sizes ) {
        AMP_ASSERT( size >= 0 );
        total += static_cast<size_t>( size );
    }
    return total;
}

template<typename TYPE>
GatheredLeafValues<TYPE>
gatherRedistributedLeafValues( std::shared_ptr<const Vector> vec,
                               const GroupedRedistributionContext &context )
{
    std::vector<TYPE> local_values( vec->getLocalSize() );
    if ( !local_values.empty() ) {
        vec->getRawData( local_values.data() );
    }

    constexpr int group_root = 0;
    auto gathered_sizes =
        context.groupComm().gather( static_cast<int>( local_values.size() ), group_root );
    auto gathered_data = context.groupComm().gather( local_values, group_root );

    return { std::move( gathered_sizes ), std::move( gathered_data ) };
}

template<typename TYPE>
void fillRedistributedLeaf( const Vector &src,
                            const GroupedRedistributionContext &context,
                            const GatheredLeafValues<TYPE> &gathered,
                            std::shared_ptr<Vector> out )
{
    AMP_INSIST( out, "VectorHelpers::redistribute: active destination vector is null" );

    auto reduced_local_size = sumSizes( gathered.sizes );
    AMP_INSIST( out->getLocalSize() == reduced_local_size,
                "VectorHelpers::redistribute: destination local size does not match plan" );
    AMP_INSIST( out->getGlobalSize() == src.getGlobalSize(),
                "VectorHelpers::redistribute: destination global size does not match source" );
    AMP_INSIST( out->getComm().getSize() == context.numRanks(),
                "VectorHelpers::redistribute: destination communicator size does not match plan" );

    if ( !gathered.values.empty() ) {
        out->putRawData( gathered.values.data() );
    }
    out->makeConsistent( ScatterType::CONSISTENT_SET );
}

template<typename TYPE>
std::shared_ptr<Vector> createRedistributedLeaf( std::shared_ptr<const Vector> vec,
                                                 const GroupedRedistributionContext &context )
{
    auto gathered = gatherRedistributedLeafValues<TYPE>( vec, context );

    if ( !context.isActive() ) {
        return nullptr;
    }

    auto reduced_local_size = sumSizes( gathered.sizes );

    auto variable = vec->getVariable() ? vec->getVariable()->clone() :
                                         std::make_shared<Variable>( vec->getName() );
    auto dofs     = std::make_shared<AMP::Discretization::DOFManager>( reduced_local_size,
                                                                   context.reducedComm() );

    auto memType = vec->getVectorData()->getMemoryLocation();
    auto backend = getBackend( *vec );
    auto out     = createVector<TYPE>( dofs, variable, false, memType, backend );
    out->setName( vec->getName() );
    out->setUnits( vec->getUnits() );

    fillRedistributedLeaf<TYPE>( *vec, context, gathered, out );
    return out;
}

template<typename TYPE>
void redistributeLeaf( const GroupedRedistributionContext &context,
                       std::shared_ptr<const Vector> src,
                       std::shared_ptr<Vector> dst )
{
    auto gathered = gatherRedistributedLeafValues<TYPE>( src, context );
    if ( context.isActive() ) {
        fillRedistributedLeaf<TYPE>( *src, context, gathered, dst );
    }
}

template<typename TYPE>
void scatterRedistributedLeaf( const GroupedRedistributionContext &context,
                               std::shared_ptr<const Vector> src,
                               std::shared_ptr<Vector> dst )
{
    AMP_INSIST( dst, "VectorHelpers::scatterRedistributed: destination vector is null" );
    checkParentVector( *dst, context, "VectorHelpers::scatterRedistributed" );

    auto local_size = static_cast<int>( dst->getLocalSize() );
    auto sizes      = context.groupComm().allGather( local_size );
    auto total_size = sumSizes( sizes );

    std::vector<TYPE> gathered_values( total_size );
    if ( context.isActive() ) {
        AMP_INSIST( src, "VectorHelpers::scatterRedistributed: active source vector is null" );
        AMP_INSIST( src->getLocalSize() == total_size,
                    "VectorHelpers::scatterRedistributed: source local size does not match plan" );
        AMP_INSIST( src->getGlobalSize() == dst->getGlobalSize(),
                    "VectorHelpers::scatterRedistributed: source global size does not match "
                    "destination" );
        if ( !gathered_values.empty() ) {
            src->getRawData( gathered_values.data() );
        }
    }

    constexpr int group_root = 0;
    if ( !gathered_values.empty() ) {
        context.groupComm().bcast(
            gathered_values.data(), static_cast<int>( gathered_values.size() ), group_root );
    }

    size_t offset = 0;
    for ( int rank = 0; rank < context.groupComm().getRank(); ++rank ) {
        AMP_ASSERT( sizes[rank] >= 0 );
        offset += static_cast<size_t>( sizes[rank] );
    }

    if ( dst->getLocalSize() > 0 ) {
        dst->putRawData( gathered_values.data() + offset );
    }
    dst->makeConsistent( ScatterType::CONSISTENT_SET );
}

} // namespace

std::shared_ptr<Vector> redistribute( std::shared_ptr<const Vector> vec, int new_nprocs )
{
    PROFILE( "VectorHelpers::redistribute" );
    if ( !vec ) {
        return nullptr;
    }
    auto context = AMP::Utilities::createGroupedRedistributionPlan( vec->getComm(), new_nprocs );
    return redistribute( vec, context );
}

std::shared_ptr<Vector> redistribute( std::shared_ptr<const Vector> vec,
                                      const GroupedRedistributionContext &context )
{
    PROFILE( "VectorHelpers::redistributeWithContext" );

    if ( !vec ) {
        return nullptr;
    }

    AMP_INSIST( !context.groupComm().isNull(),
                "Grouped redistribution context is not initialized" );
    checkParentVector( *vec, context, "VectorHelpers::redistribute" );

    if ( auto multivec = std::dynamic_pointer_cast<const MultiVector>( vec ) ) {
        auto components = multivec->getVecs();
        std::vector<Vector::shared_ptr> redistributed;
        redistributed.reserve( components.size() );
        for ( auto component : components ) {
            auto part = redistribute( component, context );
            if ( context.isActive() ) {
                AMP_INSIST( part, "Active redistributed multivector component is null" );
                redistributed.push_back( part );
            }
        }
        if ( !context.isActive() ) {
            return nullptr;
        }
        return MultiVector::create(
            vec->getVariable()->clone(), context.reducedComm(), redistributed );
    }

    auto type = vec->getVectorData()->getType( 0 );
    if ( type == getTypeID<double>() ) {
        return createRedistributedLeaf<double>( vec, context );
    } else if ( type == getTypeID<float>() ) {
        return createRedistributedLeaf<float>( vec, context );
    }

    AMP_ERROR( "Grouped vector redistribution currently supports only float and double vectors" );
}

void redistribute( const GroupedRedistributionContext &context,
                   std::shared_ptr<const Vector> src,
                   std::shared_ptr<Vector> dst )
{
    PROFILE( "VectorHelpers::redistributeInto" );

    if ( !src ) {
        return;
    }

    AMP_INSIST( !context.groupComm().isNull(),
                "Grouped redistribution context is not initialized" );
    checkParentVector( *src, context, "VectorHelpers::redistribute" );

    if ( auto src_multivec = std::dynamic_pointer_cast<const MultiVector>( src ) ) {
        auto src_components = src_multivec->getVecs();
        std::vector<Vector::shared_ptr> dst_components;
        if ( context.isActive() ) {
            auto dst_multivec = std::dynamic_pointer_cast<MultiVector>( dst );
            AMP_INSIST( dst_multivec,
                        "VectorHelpers::redistribute: active multivector destination is null" );
            dst_components = dst_multivec->getVecs();
            AMP_INSIST( dst_components.size() == src_components.size(),
                        "VectorHelpers::redistribute: multivector component count mismatch" );
        }

        for ( size_t i = 0; i < src_components.size(); ++i ) {
            redistribute(
                context, src_components[i], context.isActive() ? dst_components[i] : nullptr );
        }
        if ( context.isActive() ) {
            dst->makeConsistent( ScatterType::CONSISTENT_SET );
        }
        return;
    }

    auto type = src->getVectorData()->getType( 0 );
    if ( type == getTypeID<double>() ) {
        redistributeLeaf<double>( context, src, dst );
    } else if ( type == getTypeID<float>() ) {
        redistributeLeaf<float>( context, src, dst );
    } else {
        AMP_ERROR(
            "Grouped vector redistribution currently supports only float and double vectors" );
    }
}

void scatterRedistributed( const GroupedRedistributionContext &context,
                           std::shared_ptr<const Vector> src,
                           std::shared_ptr<Vector> dst )
{
    PROFILE( "VectorHelpers::scatterRedistributed" );

    if ( !dst ) {
        return;
    }

    AMP_INSIST( !context.groupComm().isNull(),
                "Grouped redistribution context is not initialized" );

    if ( auto dst_multivec = std::dynamic_pointer_cast<MultiVector>( dst ) ) {
        auto dst_components = dst_multivec->getVecs();
        std::vector<Vector::const_shared_ptr> src_components;
        if ( context.isActive() ) {
            auto src_multivec = std::dynamic_pointer_cast<const MultiVector>( src );
            AMP_INSIST( src_multivec,
                        "VectorHelpers::scatterRedistributed: active multivector source is null" );
            src_components = src_multivec->getVecs();
            AMP_INSIST( src_components.size() == dst_components.size(),
                        "VectorHelpers::scatterRedistributed: multivector component count "
                        "mismatch" );
        }

        for ( size_t i = 0; i < dst_components.size(); ++i ) {
            scatterRedistributed(
                context, context.isActive() ? src_components[i] : nullptr, dst_components[i] );
        }
        dst->makeConsistent( ScatterType::CONSISTENT_SET );
        return;
    }

    auto type = dst->getVectorData()->getType( 0 );
    if ( type == getTypeID<double>() ) {
        scatterRedistributedLeaf<double>( context, src, dst );
    } else if ( type == getTypeID<float>() ) {
        scatterRedistributedLeaf<float>( context, src, dst );
    } else {
        AMP_ERROR(
            "Grouped vector redistribution currently supports only float and double vectors" );
    }
}


/****************************************************************
 * Get the desired vectors                                       *
 ****************************************************************/
static std::vector<std::vector<std::shared_ptr<const Vector>>>
getVecs( const std::shared_ptr<const Vector> &vec, const std::vector<std::string> &names )
{
    PROFILE( "VectorHelpers::getVecs" );
    std::vector<std::vector<std::shared_ptr<const Vector>>> vecs( names.size() );
    auto multivec = dynamic_cast<const MultiVector *>( vec.get() );
    if ( multivec ) {
        auto name0 = vec->getName();
        auto vecs0 = multivec->getVecs();
        for ( size_t i = 0; i < names.size(); i++ ) {
            if ( names[i] == name0 ) {
                vecs[i] = vecs0;
            } else {
                for ( auto v : vecs0 ) {
                    if ( v->getName() == names[i] )
                        vecs[i].push_back( v );
                }
            }
        }
    } else {
        auto name = vec->getName();
        for ( size_t i = 0; i < names.size(); i++ ) {
            if ( names[i] == name )
                vecs[i].push_back( vec );
        }
    }
    return vecs;
}


/****************************************************************
 * Perform local computations (double)                           *
 ****************************************************************/
static std::vector<double> computeL1Norm( const std::shared_ptr<const Vector> &vec,
                                          const std::vector<std::string> &names )
{
    auto vecs = getVecs( vec, names );
    std::vector<double> x( names.size(), 0 );
    for ( size_t i = 0; i < x.size(); i++ ) {
        for ( auto &v : vecs[i] ) {
            auto data = v->getVectorData();
            x[i] += static_cast<double>( v->getVectorOperations()->localL1Norm( *data ) );
        }
    }
    return x;
}
std::vector<double> computeL2Norm2( const std::shared_ptr<const Vector> &vec,
                                    const std::vector<std::string> &names )
{
    auto vecs = getVecs( vec, names );
    std::vector<double> x( names.size(), 0 );
    for ( size_t i = 0; i < x.size(); i++ ) {
        for ( auto &v : vecs[i] ) {
            auto data = v->getVectorData();
            x[i] += static_cast<double>( v->getVectorOperations()->localL2Norm2( *data ) );
        }
    }
    return x;
}
std::vector<double> computeMaxNorm( const std::shared_ptr<const Vector> &vec,
                                    const std::vector<std::string> &names )
{
    auto vecs = getVecs( vec, names );
    std::vector<double> x( names.size(), 0 );
    for ( size_t i = 0; i < x.size(); i++ ) {
        for ( auto &v : vecs[i] ) {
            auto data = v->getVectorData();
            auto y    = static_cast<double>( v->getVectorOperations()->localMaxNorm( *data ) );
            x[i]      = std::max( x[i], y );
        }
    }
    return x;
}


/****************************************************************
 * Perform multiple norms                                        *
 ****************************************************************/
std::vector<Scalar> L1Norm( std::shared_ptr<const Vector> vec,
                            const std::vector<std::string> &names )
{
    PROFILE( "VectorHelpers::L1Norm" );
    auto x = computeL1Norm( vec, names );
    vec->getComm().sumReduce( x.data(), x.size() );
    std::vector<Scalar> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = x[i];
    return y;
}
std::vector<Scalar> L2Norm( std::shared_ptr<const Vector> vec,
                            const std::vector<std::string> &names )
{
    PROFILE( "VectorHelpers::L2Norm" );
    auto x = computeL2Norm2( vec, names );
    vec->getComm().sumReduce( x.data(), x.size() );
    std::vector<Scalar> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = sqrt( x[i] );
    return y;
}
std::vector<Scalar> maxNorm( std::shared_ptr<const Vector> vec,
                             const std::vector<std::string> &names )
{
    PROFILE( "VectorHelpers::maxNorm" );
    auto x = computeMaxNorm( vec, names );
    vec->getComm().maxReduce( x.data(), x.size() );
    std::vector<Scalar> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = x[i];
    return y;
}
std::vector<Scalar> localL1Norm( std::shared_ptr<const Vector> vec,
                                 const std::vector<std::string> &names )
{
    PROFILE( "VectorHelpers::localL1Norm" );
    auto x = computeL1Norm( vec, names );
    std::vector<Scalar> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = x[i];
    return y;
}
std::vector<Scalar> localL2Norm( std::shared_ptr<const Vector> vec,
                                 const std::vector<std::string> &names )
{
    PROFILE( "VectorHelpers::localL2Norm" );
    auto x = computeL2Norm2( vec, names );
    std::vector<Scalar> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = sqrt( x[i] );
    return y;
}
std::vector<Scalar> localMaxNorm( std::shared_ptr<const Vector> vec,
                                  const std::vector<std::string> &names )
{
    PROFILE( "VectorHelpers::localMaxNorm" );
    auto x = computeMaxNorm( vec, names );
    std::vector<Scalar> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = x[i];
    return y;
}


} // namespace AMP::LinearAlgebra::VectorHelpers
