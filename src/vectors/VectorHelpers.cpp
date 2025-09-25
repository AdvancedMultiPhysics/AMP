#include "AMP/vectors/VectorHelpers.h"
#include "AMP/vectors/MultiVector.h"


namespace AMP::LinearAlgebra::VectorHelpers {


/****************************************************************
 * Perform multiple norms                                        *
 ****************************************************************/
static std::vector<std::vector<std::shared_ptr<const Vector>>>
getVecs( std::shared_ptr<const Vector> vec, const std::vector<std::string> &names )
{
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
std::vector<Scalar> L1Norm( std::shared_ptr<const Vector> vec,
                            const std::vector<std::string> &names )
{
    if ( names.empty() )
        return {};
    auto x = localL1Norm( vec, names );
    std::vector<double> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = static_cast<double>( x[i] );
    vec->getComm().sumReduce( y.data(), y.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        x[i] = y[i];
    return x;
}
std::vector<Scalar> L2Norm( std::shared_ptr<const Vector> vec,
                            const std::vector<std::string> &names )
{
    if ( names.empty() )
        return {};
    auto x = localL2Norm( vec, names );
    std::vector<double> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ ) {
        y[i] = static_cast<double>( x[i] );
        y[i] *= y[i];
    }
    vec->getComm().sumReduce( y.data(), y.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        x[i] = sqrt( y[i] );
    return x;
}
std::vector<Scalar> maxNorm( std::shared_ptr<const Vector> vec,
                             const std::vector<std::string> &names )
{
    if ( names.empty() )
        return {};
    auto x = localMaxNorm( vec, names );
    std::vector<double> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = static_cast<double>( x[i] );
    vec->getComm().maxReduce( y.data(), y.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        x[i] = y[i];
    return x;
}
std::vector<Scalar> localL1Norm( std::shared_ptr<const Vector> vec,
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
    std::vector<Scalar> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = x[i];
    return y;
}
std::vector<Scalar> localL2Norm( std::shared_ptr<const Vector> vec,
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
    std::vector<Scalar> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = sqrt( x[i] );
    return y;
}
std::vector<Scalar> localMaxNorm( std::shared_ptr<const Vector> vec,
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
    std::vector<Scalar> y( x.size() );
    for ( size_t i = 0; i < x.size(); i++ )
        y[i] = x[i];
    return y;
}


} // namespace AMP::LinearAlgebra::VectorHelpers
