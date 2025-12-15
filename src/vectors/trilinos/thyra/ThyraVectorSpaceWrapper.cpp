#include "AMP/vectors/trilinos/thyra/ThyraVectorSpaceWrapper.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/vectors/trilinos/thyra/ThyraVectorWrapper.h"

DISABLE_WARNINGS
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_VectorSpaceBase.hpp"
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
ThyraVectorSpaceWrapper::ThyraVectorSpaceWrapper(
    std::shared_ptr<const ThyraVectorWrapper> thyra_vec, bool is_range )
    : d_is_range( is_range ), d_thyra_vec( thyra_vec )
{
    AMP_INSIST( thyra_vec != nullptr, "thyra_vec may not be NULL" );
}


/****************************************************************
 * Destructor                                                    *
 ****************************************************************/
ThyraVectorSpaceWrapper::~ThyraVectorSpaceWrapper() = default;


/****************************************************************
 * Virtual functions inherited from VectorSpaceBase              *
 ****************************************************************/
Teuchos::Ordinal ThyraVectorSpaceWrapper::dim() const
{
    if ( !d_is_range )
        return static_cast<Teuchos::Ordinal>( d_thyra_vec->numColumns() );
    return d_thyra_vec->numRows();
}
bool ThyraVectorSpaceWrapper::isCompatible( const Thyra::VectorSpaceBase<double> &vecSpc ) const
{
    const auto *vecSpaceWrapper = dynamic_cast<const ThyraVectorSpaceWrapper *>( &vecSpc );
    if ( vecSpaceWrapper == nullptr )
        return false;
    if ( this == vecSpaceWrapper )
        return true;
    auto dofs1 = d_thyra_vec->getDOFManager();
    auto dofs2 = vecSpaceWrapper->d_thyra_vec->getDOFManager();
    if ( dofs1 == dofs2 )
        return true;
    if ( *dofs1 == *dofs2 )
        return true;
    return false;
}
Teuchos::RCP<const Thyra::VectorSpaceFactoryBase<double>>
ThyraVectorSpaceWrapper::smallVecSpcFcty() const
{
    AMP_ASSERT( d_is_range );
    AMP_ERROR( "Not finished" );
    return Teuchos::RCP<const Thyra::VectorSpaceFactoryBase<double>>();
}
double ThyraVectorSpaceWrapper::scalarProd( const Thyra::VectorBase<double> &x,
                                            const Thyra::VectorBase<double> &y ) const
{
    return x.dot( y );
}
Teuchos::RCP<Thyra::VectorBase<double>> ThyraVectorSpaceWrapper::createMember() const
{
    AMP_ASSERT( d_is_range );
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vecs( 1 );
    vecs[0] = d_thyra_vec->getVec( 0 )->clone();
    return Teuchos::RCP<Thyra::VectorBase<double>>( new ThyraVectorWrapper( vecs ) );
}
Teuchos::RCP<Thyra::MultiVectorBase<double>>
ThyraVectorSpaceWrapper::createMembers( int numMembers ) const
{
    AMP_ASSERT( d_is_range );
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vecs( numMembers );
    for ( int i = 0; i < numMembers; i++ )
        vecs[i] = d_thyra_vec->getVec( 0 )->clone();
    return Teuchos::RCP<Thyra::VectorBase<double>>( new ThyraVectorWrapper( vecs ) );
}
Teuchos::RCP<Thyra::VectorBase<double>>
ThyraVectorSpaceWrapper::createMemberView( const RTOpPack::SubVectorView<double> & ) const
{
    AMP_ERROR( "Not finished" );
    return Teuchos::RCP<Thyra::VectorBase<double>>();
}
Teuchos::RCP<const Thyra::VectorBase<double>>
ThyraVectorSpaceWrapper::createMemberView( const RTOpPack::ConstSubVectorView<double> & ) const
{
    AMP_ERROR( "Not finished" );
    return Teuchos::RCP<const Thyra::VectorBase<double>>();
}
Teuchos::RCP<Thyra::MultiVectorBase<double>> ThyraVectorSpaceWrapper::createMembersView(
    const RTOpPack::SubMultiVectorView<double> &raw_mv ) const
{
    AMP_ASSERT( !d_is_range );
    size_t N_rows = d_thyra_vec->numColumns();
    auto space    = Thyra::defaultSpmdVectorSpace<double>( N_rows );
    auto view     = Thyra::createMembersView<double>( space, raw_mv, "" );
    return view;
}
Teuchos::RCP<const Thyra::MultiVectorBase<double>> ThyraVectorSpaceWrapper::createMembersView(
    const RTOpPack::ConstSubMultiVectorView<double> &raw_mv ) const
{
    AMP_ASSERT( !d_is_range );
    size_t N_rows = d_thyra_vec->numColumns();
    auto space    = Thyra::defaultSpmdVectorSpace<double>( N_rows );
    auto view     = Thyra::createMembersView<double>( space, raw_mv, "" );
    return view;
}
void ThyraVectorSpaceWrapper::scalarProdsImpl( const Thyra::MultiVectorBase<double> &,
                                               const Thyra::MultiVectorBase<double> &,
                                               const Teuchos::ArrayView<double> & ) const
{
    AMP_ERROR( "Not finished" );
}
} // namespace AMP::LinearAlgebra
