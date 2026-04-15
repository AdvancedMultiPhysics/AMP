#include "AMP/vectors/VectorHelpers.h"
#include "AMP/vectors/MultiVector.h"

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra::VectorHelpers {


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
    std::vector<double> x( names.size() );
    for ( size_t i = 0; i < x.size(); i++ ) {
        for ( size_t j = 0; j < vecs[i].size(); j++ ) {
            auto v = vecs[i][j].get();
            x[i] += static_cast<double>(
                v->getVectorOperations()->localL1Norm( *( v->getVectorData() ) ) );
        }
    }
    return x;
}
std::vector<double> computeL2Norm2( const std::shared_ptr<const Vector> &vec,
                                    const std::vector<std::string> &names )
{
    auto vecs = getVecs( vec, names );
    std::vector<double> x( names.size() );
    for ( size_t i = 0; i < x.size(); i++ ) {
        for ( size_t j = 0; j < vecs[i].size(); j++ ) {
            auto v = vecs[i][j].get();
            auto y = static_cast<double>(
                v->getVectorOperations()->localL2Norm( *( v->getVectorData() ) ) );
            x[i] += y * y;
        }
    }
    return x;
}
std::vector<double> computeMaxNorm( const std::shared_ptr<const Vector> &vec,
                                    const std::vector<std::string> &names )
{
    auto vecs = getVecs( vec, names );
    std::vector<double> x( names.size() );
    for ( size_t i = 0; i < x.size(); i++ ) {
        for ( size_t j = 0; j < vecs[i].size(); j++ ) {
            auto v = vecs[i][j].get();
            auto y = static_cast<double>(
                v->getVectorOperations()->localMaxNorm( *( v->getVectorData() ) ) );
            x[i] = std::max( x[i], y );
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
