#include "AMP/vectors/trilinos/thyra/ThyraVectorWrapper.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/vectors/trilinos/thyra/ThyraVectorSpaceWrapper.h"

DISABLE_WARNINGS
#include "RTOpPack_RTOpT_decl.hpp"
#include "RTOpPack_SPMD_apply_op_def.hpp"
#include "RTOpPack_Types.hpp"
#ifdef AMP_USE_MPI
    #include "Teuchos_DefaultMpiComm.hpp"
#else
    #include "Teuchos_DefaultSerialComm.hpp"
#endif
#include "Thyra_DefaultMultiVectorProductVectorSpace.hpp"
#include "Thyra_DefaultSpmdVectorSpace.hpp"
#include "Thyra_VectorStdOps_def.hpp"
ENABLE_WARNINGS


namespace AMP::LinearAlgebra {


// Some function definitions
template<class TYPE>
static std::vector<ThyraVectorWrapper *>
getPtr( const std::vector<size_t> &block_size, Teuchos::ArrayView<const Teuchos::Ptr<TYPE>> vecs );
template<class TYPE>
static std::vector<const ThyraVectorWrapper *>
getConstPtr( const std::vector<size_t> &block_size,
             Teuchos::ArrayView<const Teuchos::Ptr<const TYPE>> vecs );


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
ThyraVectorWrapper::ThyraVectorWrapper(
    const std::vector<AMP::LinearAlgebra::Vector::shared_ptr> &vecs )
{
    std::vector<size_t> cols( vecs.size() );
    for ( size_t i = 0; i < vecs.size(); i++ )
        cols[i] = i;
    this->initialize( vecs, cols, vecs.size() );
}
ThyraVectorWrapper::ThyraVectorWrapper(
    const std::vector<AMP::LinearAlgebra::Vector::shared_ptr> &vecs,
    const std::vector<size_t> &cols,
    size_t N_cols )
{
    this->initialize( vecs, cols, N_cols );
}
void ThyraVectorWrapper::initialize(
    const std::vector<AMP::LinearAlgebra::Vector::shared_ptr> &vecs,
    const std::vector<size_t> &cols,
    size_t N_cols )
{
    AMP_ASSERT( !vecs.empty() );
    AMP_ASSERT( vecs.size() == cols.size() );
    for ( size_t i = 0; i < vecs.size(); i++ ) {
        AMP_ASSERT( vecs[i] );
        AMP_ASSERT( cols[i] < N_cols );
        /*for (size_t j=0; j<i; j++) {
            AMP_ASSERT(vecs[i]!=vecs[j]);
            AMP_ASSERT(cols[i]>cols[j]);
        }*/
    }
    // Check that the DOFs are compatible for all copies of the vector
    auto dofs1 = vecs[0]->getDOFManager();
    for ( size_t i = 1; i < vecs.size(); i++ ) {
        auto dofs2 = vecs[i]->getDOFManager();
        if ( dofs1 == dofs2 )
            continue;
        if ( *dofs1 != *dofs2 )
            AMP_ERROR( "The DOFManagers for all copies must match" );
    }
    d_vecs   = vecs;
    d_cols   = cols;
    d_N_cols = N_cols;
#ifdef AMP_USE_MPI
    Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm>> mpi_wrapper(
        new Teuchos::OpaqueWrapper<MPI_Comm>( d_vecs[0]->getComm().getCommunicator() ) );
    d_comm = new Teuchos::MpiComm<RTOpPack::index_type>( mpi_wrapper );
#else
    d_comm = new Teuchos::SerialComm<RTOpPack::index_type>();
#endif
}


/****************************************************************
 * Destructor                                                    *
 ****************************************************************/
ThyraVectorWrapper::~ThyraVectorWrapper() { delete d_comm; }


/****************************************************************
 * Get a shared object to this                                   *
 ****************************************************************/
template<class T>
static void nullDeleter( T * )
{
}
std::shared_ptr<const ThyraVectorWrapper> ThyraVectorWrapper::shared_from_this() const
{
    return std::shared_ptr<const ThyraVectorWrapper>( this, nullDeleter<const ThyraVectorWrapper> );
}


/****************************************************************
 * Functions to return the number of rows and columns            *
 ****************************************************************/
size_t ThyraVectorWrapper::numRows() const
{
    // The number of rows is the number of unknowns in the vectors
    return d_vecs[0]->getDOFManager()->numGlobalDOF();
}
size_t ThyraVectorWrapper::numColumns() const
{
    // The number of columns is the number of internal vectors
    return d_cols.size();
}


/****************************************************************
 * Functions to return the vector space                          *
 ****************************************************************/
Teuchos::RCP<const Thyra::VectorSpaceBase<double>> ThyraVectorWrapper::range() const
{
    return Teuchos::RCP<const Thyra::VectorSpaceBase<double>>(
        new ThyraVectorSpaceWrapper( shared_from_this(), true ) );
}
Teuchos::RCP<const Thyra::VectorSpaceBase<double>> ThyraVectorWrapper::domain() const
{
    return Teuchos::RCP<const Thyra::VectorSpaceBase<double>>(
        new ThyraVectorSpaceWrapper( shared_from_this(), false ) );
}
Teuchos::RCP<const Thyra::VectorSpaceBase<double>> ThyraVectorWrapper::space() const
{
    return range();
}


/****************************************************************
 * Public functions derived from Thyra::MultiVectorBase          *
 ****************************************************************/
Teuchos::RCP<Thyra::MultiVectorBase<double>> ThyraVectorWrapper::clone_mv() const
{
    return clone_v();
}


/****************************************************************
 * Public functions derived from Thyra::VectorBase               *
 ****************************************************************/
Teuchos::RCP<Thyra::VectorBase<double>> ThyraVectorWrapper::clone_v() const
{
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vecs( d_vecs.size() );
    for ( size_t i = 0; i < d_vecs.size(); i++ )
        vecs[i] = d_vecs[i]->clone();
    Teuchos::RCP<Thyra::VectorBase<double>> new_vec(
        new ThyraVectorWrapper( vecs, d_cols, d_N_cols ) );
    return new_vec;
}


/****************************************************************
 * Protected functions derived from Thyra::LinearOpBase          *
 ****************************************************************/
bool ThyraVectorWrapper::opSupportedImpl( Thyra::EOpTransp ) const
{
    AMP_ERROR( "Not finished" );
    return false;
}
void ThyraVectorWrapper::applyImpl( const Thyra::EOpTransp M_trans,
                                    const Thyra::MultiVectorBase<double> &X,
                                    const Teuchos::Ptr<Thyra::MultiVectorBase<double>> &Y,
                                    const double alpha,
                                    const double beta ) const
{
    size_t M_dim[2], X_dim[2], Y_dim[2];
    M_dim[0] = this->numRows();    // Number of rows of M
    M_dim[1] = this->numColumns(); // Number of columns of M
    X_dim[0] = X.range()->dim();   // Number of rows of X
    X_dim[1] = X.domain()->dim();  // Number of columns of X
    Y_dim[0] = Y->range()->dim();  // Number of rows of Y
    Y_dim[1] = Y->domain()->dim(); // Number of columns of Y
    if ( M_trans == Thyra::NOTRANS || M_trans == Thyra::CONJ ) {
        // We are computing: Y = alpha*M*X + beta*Y
        const auto *y = dynamic_cast<const ThyraVectorWrapper *>( Y.get() );
        AMP_ASSERT( y );
        AMP_ASSERT( M_dim[1] == X_dim[0] && M_dim[0] == Y_dim[0] && X_dim[1] == Y_dim[1] );
        AMP_ASSERT( M_dim[1] == d_vecs.size() && Y_dim[1] == y->d_vecs.size() );
        auto tmp = d_vecs[0]->clone();
        for ( size_t i = 0; i < X_dim[1]; i++ ) { // Loop over the columns of y
            // We are performing a series of axpby operations:
            //    y(:,i) = alpha*sum(M(:,j)*X(j,i)) + beta*Y(:,i)
            // First compute tmp = sum(M(:,j)*X(j,i))
            tmp->zero();
            for ( size_t j = 0; j < d_vecs.size(); j++ ) {
                AMP_ASSERT( j < d_cols.size() );
                double x = Thyra::get_ele( *( X.col( i ) ), d_cols[j] );
                tmp->axpby( x, 1.0, *d_vecs[j] );
            }
            // Perform the final axpby
            y->d_vecs[i]->axpby( alpha, beta, *tmp );
        }
    } else if ( M_trans == Thyra::TRANS || M_trans == Thyra::CONJTRANS ) {
        // We are computing: Y = alpha*transpose(M)*X + beta*Y
        AMP_ASSERT( M_dim[0] == X_dim[0] && M_dim[1] == Y_dim[0] && X_dim[1] == Y_dim[1] );
        if ( dynamic_cast<const ThyraVectorWrapper *>( &X ) ) {
            const auto *x = dynamic_cast<const ThyraVectorWrapper *>( &X );
            AMP_ASSERT( x );
            size_t N = d_vecs.size();    // Number of columns of M
            size_t M = x->d_vecs.size(); // Number of columns of X
            AMP_ASSERT( Y->domain()->dim() == (Teuchos::Ordinal) M );
            for ( size_t i = 0; i < M; i++ ) {
                Teuchos::RCP<Thyra::VectorBase<double>> Y2 = Y->col( 0 );
                size_t N2 = static_cast<size_t>( Y2->space()->dim() );
                AMP_ASSERT( d_cols.size() == N2 );
                for ( size_t j = 0; j < N; j++ ) {
                    double dot( d_vecs[j]->dot( *x->d_vecs[i] ) );
                    double y = 0.0;
                    if ( beta != 0.0 )
                        y = beta * Thyra::get_ele( *Y2, d_cols[j] );
                    Thyra::set_ele( d_cols[j], alpha * dot + y, Y2.ptr() );
                }
            }
        } else {
            AMP_ERROR( "Not finished" );
        }
    } else {
        AMP_ERROR( "Unknown case" );
    }
}

void ThyraVectorWrapper::assignImpl( double alpha )
{
    for ( auto &vec : d_vecs ) {
        vec->setToScalar( alpha );
    }
}

/****************************************************************
 * Protected functions derived from Thyra::MultiVectorBase       *
 ****************************************************************/
Teuchos::RCP<Thyra::VectorBase<double>> ThyraVectorWrapper::nonconstColImpl( Teuchos::Ordinal j )
{
    std::vector<size_t> cols( 1, j );
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vecs( 1, d_vecs[j] );
    return Teuchos::RCP<ThyraVectorWrapper>( new ThyraVectorWrapper( vecs, cols, d_N_cols ) );
}
Teuchos::RCP<const Thyra::MultiVectorBase<double>>
ThyraVectorWrapper::contigSubViewImpl( const Teuchos::Range1D &colRng ) const
{
    std::vector<size_t> cols;
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vecs;
    for ( Teuchos::Ordinal i = colRng.lbound(); i <= colRng.ubound(); i++ ) {
        cols.push_back( i );
        vecs.push_back( d_vecs[i] );
    }
    return Teuchos::RCP<const ThyraVectorWrapper>( new ThyraVectorWrapper( vecs, cols, d_N_cols ) );
}
Teuchos::RCP<Thyra::MultiVectorBase<double>>
ThyraVectorWrapper::nonconstContigSubViewImpl( const Teuchos::Range1D &colRng )
{
    std::vector<size_t> cols;
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vecs;
    for ( Teuchos::Ordinal i = colRng.lbound(); i <= colRng.ubound(); i++ ) {
        cols.push_back( i );
        vecs.push_back( d_vecs[i] );
    }
    return Teuchos::RCP<ThyraVectorWrapper>( new ThyraVectorWrapper( vecs, cols, d_N_cols ) );
}
Teuchos::RCP<const Thyra::MultiVectorBase<double>>
ThyraVectorWrapper::nonContigSubViewImpl( const Teuchos::ArrayView<const int> &columns ) const
{
    std::vector<size_t> cols;
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vecs;
    for ( int column : columns ) {
        cols.push_back( column );
        vecs.push_back( d_vecs[column] );
    }
    return Teuchos::RCP<const ThyraVectorWrapper>( new ThyraVectorWrapper( vecs, cols, d_N_cols ) );
}
Teuchos::RCP<Thyra::MultiVectorBase<double>>
ThyraVectorWrapper::nonconstNonContigSubViewImpl( const Teuchos::ArrayView<const int> &columns )
{
    std::vector<size_t> cols;
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> vecs;
    for ( int column : columns ) {
        cols.push_back( column );
        vecs.push_back( d_vecs[column] );
    }
    return Teuchos::RCP<ThyraVectorWrapper>( new ThyraVectorWrapper( vecs, cols, d_N_cols ) );
}
void ThyraVectorWrapper::acquireDetachedMultiVectorViewImpl(
    const Teuchos::Range1D &,
    const Teuchos::Range1D &,
    RTOpPack::ConstSubMultiVectorView<double> * ) const
{
    AMP_ERROR( "Not finished" );
}
void ThyraVectorWrapper::releaseDetachedMultiVectorViewImpl(
    RTOpPack::ConstSubMultiVectorView<double> * ) const
{
    AMP_ERROR( "Not finished" );
}
void ThyraVectorWrapper::acquireNonconstDetachedMultiVectorViewImpl(
    const Teuchos::Range1D &, const Teuchos::Range1D &, RTOpPack::SubMultiVectorView<double> * )
{
    AMP_ERROR( "Not finished" );
}
void ThyraVectorWrapper::commitNonconstDetachedMultiVectorViewImpl(
    RTOpPack::SubMultiVectorView<double> * )
{
    AMP_ERROR( "Not finished" );
}


/****************************************************************
 * Protected functions derived from Thyra::VectorBase            *
 ****************************************************************/
void ThyraVectorWrapper::acquireDetachedVectorViewImpl(
    const Teuchos::Range1D &rng, RTOpPack::ConstSubVectorView<double> *sub_vec ) const
{
    AMP_ASSERT( d_vecs.size() == 1 );
    auto vec = getVec( 0 );
    AMP_ASSERT( vec->getLocalSize() > 0 );
    size_t lower = rng.lbound();
    size_t upper = std::min<size_t>( rng.ubound(), vec->getLocalSize() );
    AMP_ASSERT( upper > lower );
    size_t size = upper - lower;
    auto *ptr   = new double[size];
    Teuchos::ArrayRCP<double> array( ptr, rng.lbound(), rng.lbound() + size, true );
    auto *indices = new size_t[size];
    for ( size_t i = 0; i < size; i++ )
        indices[i] = lower + i;
    vec->getValuesByLocalID( size, indices, ptr );
    delete[] indices;
    DISABLE_WARNINGS
    *sub_vec = RTOpPack::ConstSubVectorView<double>( array );
    ENABLE_WARNINGS
}
void ThyraVectorWrapper::releaseDetachedVectorViewImpl(
    RTOpPack::ConstSubVectorView<double> * ) const
{
}
void ThyraVectorWrapper::acquireNonconstDetachedVectorViewImpl(
    const Teuchos::Range1D &rng, RTOpPack::SubVectorView<double> *sub_vec )
{
    AMP_ASSERT( d_vecs.size() == 1 );
    Vector::const_shared_ptr vec = getVec( 0 );
    AMP_ASSERT( vec->getLocalSize() > 0 );
    size_t lower = rng.lbound();
    size_t upper = std::min<size_t>( rng.ubound(), vec->getLocalSize() );
    AMP_ASSERT( upper > lower );
    size_t size = upper - lower;
    auto *ptr   = new double[size];
    Teuchos::ArrayRCP<double> array( ptr, rng.lbound(), rng.lbound() + size, true );
    auto *indices = new size_t[size];
    for ( size_t i = 0; i < size; i++ )
        indices[i] = lower + i;
    vec->getValuesByLocalID( size, indices, ptr );
    delete[] indices;
    DISABLE_WARNINGS
    *sub_vec = RTOpPack::SubVectorView<double>( array );
    ENABLE_WARNINGS
}
void ThyraVectorWrapper::commitNonconstDetachedVectorViewImpl(
    RTOpPack::SubVectorView<double> *sub_vec )
{
    AMP_ASSERT( d_vecs.size() == 1 );
    Vector::shared_ptr vec                 = getVec( 0 );
    const Teuchos::ArrayRCP<double> &array = sub_vec->values();
    size_t lower                           = array.lowerOffset();
    size_t upper                           = array.upperOffset();
    AMP_ASSERT( upper > lower && upper <= vec->getLocalSize() );
    size_t size   = upper - lower;
    auto *indices = new size_t[size];
    for ( size_t i = 0; i < size; i++ )
        indices[i] = lower + i;
    vec->setValuesByLocalID( size, indices, array.get() );
    delete[] indices;
}
void ThyraVectorWrapper::setSubVectorImpl( const RTOpPack::SparseSubVectorT<double> & )
{
    AMP_ERROR( "Not finished" );
}


/****************************************************************
 * applyOpImpl                                                   *
 ****************************************************************/
DISABLE_WARNINGS
void ThyraVectorWrapper::applyOpImpl(
    const RTOpPack::RTOpT<double> &op,
    const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::VectorBase<double>>> &vecs,
    const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::VectorBase<double>>> &targ_vecs,
    const Teuchos::Ptr<RTOpPack::ReductTarget> &reduct_obj,
    const Teuchos::Ordinal ) const
{
    size_t n_blocks = d_vecs[0]->numberOfDataBlocks();
    std::vector<size_t> block_size( n_blocks, 0 );
    for ( size_t i = 0; i < n_blocks; i++ )
        block_size[i] = d_vecs[0]->sizeOfDataBlock( i );
    // Check that all vectors are compatible and get the pointers
    auto vecs_ptr      = getConstPtr( block_size, vecs );
    auto targ_vecs_ptr = getPtr( block_size, targ_vecs );
    // Apply the operation
    auto reduct_obj2 = op.reduct_obj_create();
    for ( size_t k = 0; k < d_vecs.size(); k++ ) {
        for ( size_t j = 0; j < n_blocks; j++ ) {
            std::vector<RTOpPack::ConstSubVectorView<double>> sub_vecs( vecs_ptr.size() );
            std::vector<RTOpPack::SubVectorView<double>> targ_sub_vecs( targ_vecs_ptr.size() );
            for ( size_t i = 0; i < vecs_ptr.size(); i++ ) {
                sub_vecs[i] = RTOpPack::ConstSubVectorView<double>(
                    Teuchos::ArrayRCP<double>( vecs_ptr[i]->d_vecs[k]->getRawDataBlock<double>( j ),
                                               0,
                                               block_size[j],
                                               false ) );
            }
            for ( size_t i = 0; i < targ_vecs_ptr.size(); i++ ) {
                targ_sub_vecs[i] = RTOpPack::SubVectorView<double>( Teuchos::ArrayRCP<double>(
                    targ_vecs_ptr[i]->d_vecs[k]->getRawDataBlock<double>( j ),
                    0,
                    block_size[j],
                    false ) );
            }
            auto sub_vecs2 =
                Teuchos::ArrayView<const RTOpPack::ConstSubVectorView<double>>( sub_vecs );
            auto targ_sub_vecs2 =
                Teuchos::ArrayView<const RTOpPack::SubVectorView<double>>( targ_sub_vecs );
            op.apply_op( sub_vecs2, targ_sub_vecs2, reduct_obj2.ptr() );
        }
    }
    // Reduce the result
    if ( reduct_obj.get() ) {
        RTOpPack::SPMD_all_reduce<double>(
            d_comm,
            op,
            1,
            Teuchos::tuple<const RTOpPack::ReductTarget *>( &*reduct_obj2 ).getRawPtr(),
            Teuchos::tuple<RTOpPack::ReductTarget *>( &*reduct_obj ).getRawPtr() );
    }
    // Change the vector state targ_vecs
    for ( auto &i : targ_vecs_ptr ) {
        for ( size_t j = 0; j < d_vecs.size(); j++ ) {
            if ( i->d_vecs[j]->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED )
                i->d_vecs[j]->setUpdateStatus( AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED );
        }
    }
}

void ThyraVectorWrapper::mvMultiReductApplyOpImpl(
    const RTOpPack::RTOpT<double> &primary_op,
    const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::MultiVectorBase<double>>> &multi_vecs,
    const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::MultiVectorBase<double>>> &targ_multi_vecs,
    const Teuchos::ArrayView<const Teuchos::Ptr<RTOpPack::ReductTarget>> &reduct_objs,
    const Teuchos::Ordinal primary_global_offset ) const
{
    AMP_INSIST( primary_global_offset == 0, "Not programmed for a global offset yet" );
    size_t n_blocks = d_vecs[0]->numberOfDataBlocks();
    std::vector<size_t> block_size( n_blocks, 0 );
    for ( size_t i = 0; i < n_blocks; i++ )
        block_size[i] = d_vecs[0]->sizeOfDataBlock( i );
    // Check that all vectors are compatible and get the pointers
    auto vecs_ptr      = getConstPtr( block_size, multi_vecs );
    auto targ_vecs_ptr = getPtr( block_size, targ_multi_vecs );
    // Apply the operation
    std::vector<Teuchos::RCP<RTOpPack::ReductTarget>> reduct_obj2( d_vecs.size() );
    for ( auto &i : reduct_obj2 )
        i = primary_op.reduct_obj_create();
    for ( size_t k = 0; k < d_vecs.size(); k++ ) {
        for ( size_t j = 0; j < n_blocks; j++ ) {
            std::vector<RTOpPack::ConstSubVectorView<double>> sub_vecs( vecs_ptr.size() );
            std::vector<RTOpPack::SubVectorView<double>> targ_sub_vecs( targ_vecs_ptr.size() );
            for ( size_t i = 0; i < vecs_ptr.size(); i++ ) {
                sub_vecs[i] = RTOpPack::ConstSubVectorView<double>(
                    Teuchos::ArrayRCP<double>( vecs_ptr[i]->d_vecs[k]->getRawDataBlock<double>( j ),
                                               0,
                                               block_size[j],
                                               false ) );
            }
            for ( size_t i = 0; i < targ_vecs_ptr.size(); i++ ) {
                targ_sub_vecs[i] = RTOpPack::SubVectorView<double>( Teuchos::ArrayRCP<double>(
                    targ_vecs_ptr[i]->d_vecs[k]->getRawDataBlock<double>( j ),
                    0,
                    block_size[j],
                    false ) );
            }
            auto sub_vecs2 =
                Teuchos::ArrayView<const RTOpPack::ConstSubVectorView<double>>( sub_vecs );
            auto targ_sub_vecs2 =
                Teuchos::ArrayView<const RTOpPack::SubVectorView<double>>( targ_sub_vecs );
            primary_op.apply_op( sub_vecs2, targ_sub_vecs2, reduct_obj2[k].ptr() );
        }
    }
    // Reduce the result
    if ( reduct_objs.size() == 0 ) {
        // The reduce object does not exist
    } else if ( reduct_objs.size() == (int) d_cols.size() ) {
        // We have one reduce object per column
        for ( size_t i = 0; i < reduct_obj2.size(); i++ ) {
            RTOpPack::SPMD_all_reduce<double>(
                d_comm,
                primary_op,
                1,
                Teuchos::tuple<const RTOpPack::ReductTarget *>( &*reduct_obj2[i] ).getRawPtr(),
                Teuchos::tuple<RTOpPack::ReductTarget *>( &*reduct_objs[i] ).getRawPtr() );
        }
    } else if ( reduct_objs.size() == (int) d_N_cols ) {
        // We have one reduce object per column
        for ( size_t i = 0; i < reduct_obj2.size(); i++ ) {
            RTOpPack::SPMD_all_reduce<double>(
                d_comm,
                primary_op,
                1,
                Teuchos::tuple<const RTOpPack::ReductTarget *>( &*reduct_obj2[i] ).getRawPtr(),
                Teuchos::tuple<RTOpPack::ReductTarget *>( &*reduct_objs[d_cols[i]] ).getRawPtr() );
        }
    } else {
        AMP_ERROR( "Not programmed yet" );
    }
    // Change the vector state targ_vecs
    for ( auto &i : targ_vecs_ptr ) {
        for ( size_t j = 0; j < d_vecs.size(); j++ ) {
            if ( i->d_vecs[j]->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED )
                i->d_vecs[j]->setUpdateStatus( AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED );
        }
    }
}

void ThyraVectorWrapper::mvSingleReductApplyOpImpl(
    const RTOpPack::RTOpT<double> &primary_op,
    const RTOpPack::RTOpT<double> &,
    const Teuchos::ArrayView<const Teuchos::Ptr<const Thyra::MultiVectorBase<double>>> &multi_vecs,
    const Teuchos::ArrayView<const Teuchos::Ptr<Thyra::MultiVectorBase<double>>> &targ_multi_vecs,
    const Teuchos::Ptr<RTOpPack::ReductTarget> &reduct_obj,
    const Teuchos::Ordinal ) const
{
    size_t n_blocks = d_vecs[0]->numberOfDataBlocks();
    std::vector<size_t> block_size( n_blocks, 0 );
    for ( size_t i = 0; i < n_blocks; i++ )
        block_size[i] = d_vecs[0]->sizeOfDataBlock( i );
    // Check that all vectors are compatible and get the pointers
    auto vecs_ptr      = getConstPtr( block_size, multi_vecs );
    auto targ_vecs_ptr = getPtr( block_size, targ_multi_vecs );
    // Apply the operation
    Teuchos::RCP<RTOpPack::ReductTarget> reduct_obj2 = primary_op.reduct_obj_create();
    for ( size_t k = 0; k < d_vecs.size(); k++ ) {
        for ( size_t j = 0; j < n_blocks; j++ ) {
            std::vector<RTOpPack::ConstSubVectorView<double>> sub_vecs( vecs_ptr.size() );
            std::vector<RTOpPack::SubVectorView<double>> targ_sub_vecs( targ_vecs_ptr.size() );
            for ( size_t i = 0; i < vecs_ptr.size(); i++ ) {
                sub_vecs[i] = RTOpPack::ConstSubVectorView<double>(
                    Teuchos::ArrayRCP<double>( vecs_ptr[i]->d_vecs[k]->getRawDataBlock<double>( j ),
                                               0,
                                               block_size[j],
                                               false ) );
            }
            for ( size_t i = 0; i < targ_vecs_ptr.size(); i++ ) {
                targ_sub_vecs[i] = RTOpPack::SubVectorView<double>( Teuchos::ArrayRCP<double>(
                    targ_vecs_ptr[i]->d_vecs[k]->getRawDataBlock<double>( j ),
                    0,
                    block_size[j],
                    false ) );
            }
            auto sub_vecs2 =
                Teuchos::ArrayView<const RTOpPack::ConstSubVectorView<double>>( sub_vecs );
            auto targ_sub_vecs2 =
                Teuchos::ArrayView<const RTOpPack::SubVectorView<double>>( targ_sub_vecs );
            primary_op.apply_op( sub_vecs2, targ_sub_vecs2, reduct_obj2.ptr() );
        }
    }
    // Reduce the result
    if ( reduct_obj.get() ) {
        RTOpPack::SPMD_all_reduce<double>(
            d_comm,
            primary_op,
            1,
            Teuchos::tuple<const RTOpPack::ReductTarget *>( &*reduct_obj2 ).getRawPtr(),
            Teuchos::tuple<RTOpPack::ReductTarget *>( &*reduct_obj ).getRawPtr() );
    }
    // Change the vector state targ_vecs
    for ( auto &i : targ_vecs_ptr ) {
        for ( size_t j = 0; j < d_vecs.size(); j++ ) {
            if ( i->d_vecs[j]->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED )
                i->d_vecs[j]->setUpdateStatus( AMP::LinearAlgebra::UpdateState::LOCAL_CHANGED );
        }
    }
}
ENABLE_WARNINGS


/****************************************************************
 * Helper functions                                              *
 ****************************************************************/
template<class TYPE>
static std::vector<ThyraVectorWrapper *> getPtr( const std::vector<size_t> &block_size,
                                                 Teuchos::ArrayView<const Teuchos::Ptr<TYPE>> vecs )
{
    int n_vecs = vecs.size();
    std::vector<ThyraVectorWrapper *> ptr( n_vecs, nullptr );
    for ( int i = 0; i < n_vecs; i++ ) {
        ptr[i] = dynamic_cast<ThyraVectorWrapper *>( vecs[i].getRawPtr() );
        AMP_INSIST( ptr[i],
                    "All vectors used in applyOpImpl must be of the type ThyraVectorWrapper" );
        for ( size_t j = 0; j < ptr[i]->numVecs(); j++ ) {
            auto tmp        = ptr[i]->getVec( j );
            size_t N_blocks = tmp->numberOfDataBlocks();
            AMP_ASSERT( N_blocks == block_size.size() );
            for ( size_t k = 0; k < N_blocks; k++ )
                AMP_ASSERT( block_size[k] == tmp->sizeOfDataBlock( k ) );
        }
    }
    return ptr;
}
template<class TYPE>
static std::vector<const ThyraVectorWrapper *>
getConstPtr( const std::vector<size_t> &block_size,
             Teuchos::ArrayView<const Teuchos::Ptr<const TYPE>> vecs )
{
    int n_vecs = vecs.size();
    std::vector<const ThyraVectorWrapper *> ptr( n_vecs, nullptr );
    for ( int i = 0; i < n_vecs; i++ ) {
        ptr[i] = dynamic_cast<const ThyraVectorWrapper *>( vecs[i].getRawPtr() );
        AMP_INSIST( ptr[i],
                    "All vectors used in applyOpImpl must be of the type ThyraVectorWrapper" );
        for ( size_t j = 0; j < ptr[i]->numVecs(); j++ ) {
            auto tmp = ptr[i]->getVec( j );
            AMP_ASSERT( tmp->numberOfDataBlocks() == block_size.size() );
            for ( size_t k = 0; k < block_size.size(); k++ )
                AMP_ASSERT( block_size[k] == tmp->sizeOfDataBlock( k ) );
        }
    }
    return ptr;
}
} // namespace AMP::LinearAlgebra
