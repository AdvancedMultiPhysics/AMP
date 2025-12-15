#include "AMP/vectors/trilinos/thyra/NativeThyraVectorOperations.h"
#include "AMP/vectors/trilinos/thyra/NativeThyraVectorData.h"
#include "AMP/vectors/trilinos/thyra/ThyraVectorWrapper.h"


// Trilinos includes
DISABLE_WARNINGS
// #include "Thyra_SpmdVectorBase_def.hpp"
#include "Thyra_DefaultSpmdVector_def.hpp"
#include "Thyra_VectorStdOps_def.hpp"
#include "Trilinos_version.h"
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {

/************************************************************************
 * Destructor                                                            *
 ************************************************************************/
NativeThyraVectorOperations::~NativeThyraVectorOperations() = default;

Teuchos::RCP<const Thyra::VectorBase<double>>
NativeThyraVectorOperations::getThyraVec( const VectorData &v )
{
    auto data = dynamic_cast<const NativeThyraVectorData *>( &v );
    AMP_ASSERT( data );
    return data->getVec();
}

Teuchos::RCP<Thyra::VectorBase<double>> NativeThyraVectorOperations::getThyraVec( VectorData &v )
{
    auto data = dynamic_cast<NativeThyraVectorData *>( &v );
    AMP_ASSERT( data );
    return data->getVec();
}

void NativeThyraVectorOperations::setToScalar( const Scalar &alpha, VectorData &x )
{
    Thyra::put_scalar<double>( alpha.get<double>(), getThyraVec( x ).ptr() );
}

void NativeThyraVectorOperations::setRandomValues( VectorData &x )
{
    Thyra::randomize<double>( 0.0, 1.0, getThyraVec( x ).ptr() );
}

void NativeThyraVectorOperations::copy( const VectorData &x, VectorData &y )
{
    Thyra::copy<double>( *( getThyraVec( x ) ), getThyraVec( y ).ptr() );
}

void NativeThyraVectorOperations::scale( const Scalar &alpha, VectorData &x )
{
    Thyra::scale<double>( alpha.get<double>(), getThyraVec( x ).ptr() );
}

void NativeThyraVectorOperations::scale( const Scalar &alpha, const VectorData &x, VectorData &y )
{
    Thyra::copy<double>( *( getThyraVec( x ) ), getThyraVec( y ).ptr() );
    Thyra::scale<double>( alpha.get<double>(), getThyraVec( y ).ptr() );
}

void NativeThyraVectorOperations::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    linearSum( 1.0, x, 1.0, y, z );
}

void NativeThyraVectorOperations::subtract( const VectorData &x,
                                            const VectorData &y,
                                            VectorData &z )
{
    linearSum( 1.0, x, -1.0, y, z );
}

void NativeThyraVectorOperations::multiply( const VectorData &x,
                                            const VectorData &y,
                                            VectorData &z )
{
    auto xv = getThyraVec( x );
    auto yv = getThyraVec( y );
    auto zv = getThyraVec( z );
    Thyra::put_scalar<double>( 0.0, zv.ptr() );
    Thyra::ele_wise_prod<double>( 1.0, *xv, *yv, zv.ptr() );
}

void NativeThyraVectorOperations::divide( const VectorData &x, const VectorData &y, VectorData &z )
{
    auto xv = getThyraVec( x );
    auto yv = getThyraVec( y );
    auto zv = getThyraVec( z );
    Thyra::put_scalar<double>( 0.0, zv.ptr() );
    Thyra::ele_wise_divide<double>( 1.0, *xv, *yv, zv.ptr() );
}

void NativeThyraVectorOperations::reciprocal( const VectorData &x, VectorData &y )
{
#if TRILINOS_MAJOR_MINOR_VERSION <= 100800
    Thyra::reciprocal<double>( getThyraVec( y ).ptr(), *( getThyraVec( x ) ) );
#else
    Thyra::reciprocal<double>( *( getThyraVec( x ) ), getThyraVec( y ).ptr() );
#endif
}

void NativeThyraVectorOperations::linearSum( const Scalar &alpha_in,
                                             const VectorData &x,
                                             const Scalar &beta_in,
                                             const VectorData &y,
                                             VectorData &z )
{
    std::vector<double> alpha_vec( 2, 1.0 );
    alpha_vec[0] = alpha_in.get<double>();
    alpha_vec[1] = beta_in.get<double>();
    std::vector<Teuchos::Ptr<const Thyra::VectorBase<double>>> vecs( 2 );
    vecs[0] = getThyraVec( x ).ptr();
    vecs[1] = getThyraVec( y ).ptr();
    Teuchos::ArrayView<double> alpha_view( alpha_vec );
    Teuchos::ArrayView<Teuchos::Ptr<const Thyra::VectorBase<double>>> vecs_view( vecs );
    Thyra::linear_combination<double>( alpha_view, vecs_view, 0.0, getThyraVec( z ).ptr() );
}

void NativeThyraVectorOperations::axpy( const Scalar &alpha,
                                        const VectorData &x,
                                        const VectorData &y,
                                        VectorData &z )
{
    linearSum( alpha.get<double>(), x, 1.0, y, z );
}

void NativeThyraVectorOperations::axpby( const Scalar &alpha,
                                         const Scalar &beta,
                                         const VectorData &x,
                                         VectorData &z )
{
    linearSum( alpha.get<double>(), x, beta, z, z );
}

void NativeThyraVectorOperations::abs( const VectorData &x, VectorData &y )
{
#if TRILINOS_MAJOR_MINOR_VERSION <= 100800
    Thyra::abs<double>( getThyraVec( y ).ptr(), *getThyraVec( x ) );
#else
    Thyra::abs<double>( *getThyraVec( x ), getThyraVec( y ).ptr() );
#endif
}

Scalar NativeThyraVectorOperations::min( const VectorData &x ) const
{
    return Thyra::min<double>( *getThyraVec( x ) );
}

Scalar NativeThyraVectorOperations::max( const VectorData &x ) const
{
    return Thyra::max<double>( *getThyraVec( x ) );
}

Scalar NativeThyraVectorOperations::L1Norm( const VectorData &x ) const
{
    return Thyra::norm_1<double>( *getThyraVec( x ) );
}

Scalar NativeThyraVectorOperations::L2Norm( const VectorData &x ) const
{
    return Thyra::norm_2<double>( *getThyraVec( x ) );
}

Scalar NativeThyraVectorOperations::maxNorm( const VectorData &x ) const
{
    return Thyra::norm_inf<double>( *getThyraVec( x ) );
}

Scalar NativeThyraVectorOperations::dot( const VectorData &x, const VectorData &y ) const
{
    return Thyra::dot<double>( *getThyraVec( x ), *getThyraVec( y ) );
}


} // namespace AMP::LinearAlgebra
