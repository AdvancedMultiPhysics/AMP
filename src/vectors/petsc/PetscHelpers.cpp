#include "AMP/vectors/petsc/PetscHelpers.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/data/ManagedVectorData.h"

#include "petsc.h"
#include "petsc/private/vecimpl.h"
#include "petscsys.h"
#include "petscvec.h"

#if PETSC_VERSION_LT( 3, 19, 0 )
    #error AMP only supports PETSc 3.19.0 or greater
#endif

namespace PETSC {


static inline void increaseState( Vec x )
{
    PetscObjectStateIncrease( reinterpret_cast<::PetscObject>( x ) );
}


/********************************************************
 * Wrapper class for an AMP vector for PETSc Vec         *
 ********************************************************/
static uint32_t globalHash = AMP::Utilities::hash_char( "PetscVectorWrapper" );
class PetscVectorWrapper :
    public AMP::LinearAlgebra::DataChangeListener,
    public AMP::enable_shared_from_this<PetscVectorWrapper>
{
public:
    PetscVectorWrapper()                             = delete;
    PetscVectorWrapper( const PetscVectorWrapper & ) = delete;
    explicit PetscVectorWrapper( std::shared_ptr<AMP::LinearAlgebra::Vector> vec );
    ~PetscVectorWrapper() override;
    inline Vec &getVec() { return d_petscVec; }
    inline auto getAMP() { return d_vec; }
    inline bool check() const { return hash == globalHash; }
    Vec duplicate();
    void receiveDataChanged() override { increaseState( getVec() ); }

protected:
    Vec d_petscVec;
    uint32_t hash;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_vec;
};
PetscVectorWrapper::PetscVectorWrapper( std::shared_ptr<AMP::LinearAlgebra::Vector> vec )
    : hash( globalHash ), d_vec( vec )
{
    AMP_ASSERT( vec );
    auto comm = vec->getComm();
    VecCreate( comm.getCommunicator(), &d_petscVec );

    d_petscVec->data        = this;
    d_petscVec->petscnative = PETSC_FALSE;

    PETSC::reset_vec_ops( d_petscVec );

    PetscLayoutSetBlockSize( d_petscVec->map, 1 );
    PetscLayoutSetSize( d_petscVec->map, vec->getGlobalSize() );
    PetscLayoutSetLocalSize( d_petscVec->map, vec->getLocalSize() );
    PetscLayoutSetUp( d_petscVec->map );

    if ( ( (PetscObject) d_petscVec )->type_name ) {
        int ierr = PetscFree( ( (PetscObject) d_petscVec )->type_name );
        AMP_INSIST( ierr == 0, "PetscFree returned non-zero error code" );
    }

    int ierr = PetscObjectChangeTypeName( (PetscObject) d_petscVec, "AMPManagedPetscVector" );
    AMP_INSIST( ierr == 0, "PetscObjectChangeTypeName returned non-zero error code" );

    VecSetFromOptions( d_petscVec );

    auto listener = std::dynamic_pointer_cast<DataChangeListener>( shared_from_this() );
    vec->getVectorData()->registerListener( listener );
}
PetscVectorWrapper::~PetscVectorWrapper()
{
    int refct = ( (PetscObject) d_petscVec )->refct;
    AMP_INSIST( refct <= 1, "Deleting a vector still held by PETSc" );
    hash = 0;
}


/********************************************************
 * Get the AMP vector from the PETSc Vec or Mat          *
 ********************************************************/
static inline PetscVectorWrapper *getWrapper( Vec v )
{
    auto p = reinterpret_cast<PetscVectorWrapper *>( v->data );
    AMP_ASSERT( p->check() );
    return p;
}
std::shared_ptr<AMP::LinearAlgebra::Vector> getAMP( Vec v )
{
    auto p = getWrapper( v );
    return p->getAMP();
}
Vec getVec( std::shared_ptr<AMP::LinearAlgebra::Vector> v )
{
    auto ptr = new PetscVectorWrapper( v );
    return ptr->getVec();
}


/********************************************************
 * Destructors                                           *
 ********************************************************/
PetscErrorCode vecDestroy( Vec *v ) { return VecDestroy( v ); }
PetscErrorCode randomDestroy( PetscRandom *random ) { return PetscRandomDestroy( random ); }
PetscErrorCode matDestroy( Mat *mat ) { return MatDestroy( mat ); }


/********************************************************
 * Create random number generator                        *
 ********************************************************/
std::shared_ptr<PetscRandom> genPetscRandom( const AMP::AMP_MPI &comm )
{
    auto destructor = []( auto p ) {
        PETSC::randomDestroy( p );
        delete p;
    };
    std::shared_ptr<PetscRandom> rand( new PetscRandom, destructor );
    PetscRandomCreate( comm.getCommunicator(), rand.get() );
    PetscRandomSetType( *rand, PETSCRAND48 ); // This is a horrible RNG for
                                              // stochastic simulation.  Do not
                                              // use.
    return rand;
}


/********************************************************
 * Override Petsc functions                              *
 ********************************************************/
PetscErrorCode
_AMP_setvaluesblocked( Vec, PetscInt, const PetscInt[], const PetscScalar[], InsertMode )
{
    AMP_ERROR( "31 Not implemented" );
    return 0;
}
PetscErrorCode _AMP_view( Vec, PetscViewer )
{
    AMP_ERROR( "32 Not implemented" );
    return 0;
}
PetscErrorCode _AMP_placearray( Vec, const PetscScalar * )
{
    AMP_ERROR( "33 Not implemented" );
    return 0;
}
PetscErrorCode _AMP_replacearray( Vec, const PetscScalar * )
{
    AMP_ERROR( "34 Not implemented" );
    return 0;
}
PetscErrorCode _AMP_viewnative( Vec, PetscViewer )
{
    AMP_ERROR( "42 Not implemented" );
    return 0;
}
PetscErrorCode _AMP_setlocaltoglobalmapping( Vec, ISLocalToGlobalMapping )
{
    AMP_ERROR( "44 Not implemented" );
    return 0;
}
PetscErrorCode
_AMP_setvalueslocal( Vec, PetscInt, const PetscInt *, const PetscScalar *, InsertMode )
{
    AMP_ERROR( "45 Not implemented" );
    return 0;
}

PetscErrorCode _AMP_getvalues( Vec, PetscInt, const PetscInt[], PetscScalar[] )
{
    AMP_ERROR( "52 Not implemented" );
    return 0;
}
PetscErrorCode _AMP_assemblybegin( Vec ) { return 0; }
PetscErrorCode _AMP_assemblyend( Vec ) { return 0; }
PetscErrorCode _AMP_setoption( Vec, VecOption, PetscBool ) { return 0; }
PetscErrorCode _AMP_load( Vec, PetscViewer )
{
    AMP_ERROR( "48 Not implemented" );
    return 0;
}
PetscErrorCode
_AMP_setvalues( Vec px, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora )
{
    // Inserts or adds values into certain locations of a vector.
    auto x       = getAMP( px );
    auto indices = new size_t[ni];
    auto vals    = new PetscScalar[ni];
    for ( PetscInt i = 0; i < ni; i++ ) {
        indices[i] = static_cast<size_t>( ix[i] );
        vals[i]    = static_cast<PetscScalar>( y[i] );
    }
    if ( iora == INSERT_VALUES ) {
        x->setValuesByGlobalID( ni, indices, vals );
    } else if ( iora == ADD_VALUES ) {
        x->addValuesByGlobalID( ni, indices, vals );
    } else {
        AMP_ERROR( "Invalid option for InsertMode" );
    }
    delete[] indices;
    delete[] vals;
    return 0;
}
PetscErrorCode _AMP_shift( Vec px, PetscScalar s )
{
    auto x   = getAMP( px );
    auto cur = x->begin();
    auto end = x->end();
    while ( cur != end ) {
        *cur = s + ( *cur );
        cur++;
    }
    return 0;
}
PetscErrorCode
_AMP_axpbypcz( Vec c, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec a, Vec b )
{
    auto x      = getAMP( a );
    auto y      = getAMP( b );
    auto z      = getAMP( c );
    auto z_data = z->getVectorData();
    if ( z_data->isAnAliasOf( *x->getVectorData() ) ) {
        z->linearSum( alpha + gamma, *x, beta, *y );
    } else if ( z_data->isAnAliasOf( *y->getVectorData() ) ) {
        z->linearSum( alpha, *x, beta + gamma, *y );
    } else {
        z->linearSum( alpha, *x, gamma, *z );
        z->linearSum( beta, *y, 1., *z );
    }
    increaseState( c );
    return 0;
}
PetscErrorCode _AMP_max( Vec a, PetscInt *p, PetscReal *ans )
{
    if ( p != nullptr ) {
        AMP_ERROR( "Cannot find position for max" );
    }
    auto x = getAMP( a );
    *ans   = static_cast<PetscReal>( x->max() );
    return 0;
}
PetscErrorCode _AMP_min( Vec a, PetscInt *p, PetscReal *ans )
{
    if ( p != nullptr ) {
        AMP_ERROR( "Cannot find position for max" );
    }
    auto x = getAMP( a );
    *ans   = static_cast<PetscReal>( x->min() );
    return 0;
}
PetscErrorCode _AMP_aypx( Vec b, PetscScalar alpha, Vec a )
{
    // y = x + alpha * y
    auto x = getAMP( a );
    auto y = getAMP( b );
    y->linearSum( alpha, *y, 1., *x );
    return 0;
}
PetscErrorCode _AMP_dot_local( Vec a, Vec b, PetscScalar *ans )
{
    auto x = getAMP( a );
    auto y = getAMP( b );
    *ans   = x->getVectorOperations()
               ->localDot( *y->getVectorData(), *x->getVectorData() )
               .get<PetscScalar>();
    return 0;
}
PetscErrorCode _AMP_tdot_local( Vec a, Vec b, PetscScalar *ans )
{
    return _AMP_dot_local( a, b, ans );
}
PetscErrorCode _AMP_mdot_local( Vec a, PetscInt num, const Vec array[], PetscScalar *ans )
{
    for ( PetscInt i = 0; i != num; i++ )
        _AMP_dot_local( a, array[i], ans + i );
    return 0;
}
PetscErrorCode _AMP_mtdot_local( Vec a, PetscInt num, const Vec array[], PetscScalar *ans )
{
    for ( PetscInt i = 0; i != num; i++ )
        _AMP_dot_local( a, array[i], ans + i );
    return 0;
}
PetscErrorCode _AMP_exp( Vec a )
{
    auto x   = getAMP( a );
    auto cur = x->begin();
    auto end = x->end();
    while ( cur != end ) {
        *cur = exp( *cur );
        cur++;
    }
    return 0;
}
PetscErrorCode _AMP_log( Vec a )
{
    auto x   = getAMP( a );
    auto cur = x->begin();
    auto end = x->end();
    while ( cur != end ) {
        *cur = log( *cur );
        cur++;
    }
    return 0;
}
PetscErrorCode _AMP_pointwisemin( Vec a, Vec b, Vec c )
{
    auto x = getAMP( a );
    auto y = getAMP( b );
    auto z = getAMP( c );

    AMP_INSIST( x->getLocalSize() == y->getLocalSize(), "Incompatible vectors" );
    AMP_INSIST( x->getLocalSize() == z->getLocalSize(), "Incompatible vectors" );

    auto xi = x->begin();
    auto xe = x->end();
    auto yi = y->constBegin();
    auto zi = z->constBegin();
    while ( xi != xe ) {
        *xi = std::min( *yi, *zi );
        xi++;
        yi++;
        zi++;
    }
    return 0;
}
PetscErrorCode _AMP_pointwisemax( Vec a, Vec b, Vec c )
{
    auto x = getAMP( a );
    auto y = getAMP( b );
    auto z = getAMP( c );

    AMP_INSIST( x->getLocalSize() == y->getLocalSize(), "Incompatible vectors" );
    AMP_INSIST( x->getLocalSize() == z->getLocalSize(), "Incompatible vectors" );

    auto xi = x->begin();
    auto xe = x->end();
    auto yi = y->constBegin();
    auto zi = z->constBegin();
    while ( xi != xe ) {
        *xi = std::max( *yi, *zi );
        xi++;
        yi++;
        zi++;
    }
    return 0;
}
PetscErrorCode _AMP_pointwisemaxabs( Vec a, Vec b, Vec c )
{
    auto x = getAMP( a );
    auto y = getAMP( b );
    auto z = getAMP( c );

    AMP_INSIST( x->getLocalSize() == y->getLocalSize(), "Incompatible vectors" );
    AMP_INSIST( x->getLocalSize() == z->getLocalSize(), "Incompatible vectors" );

    auto xi = x->begin();
    auto yi = y->begin();
    auto zi = z->begin();
    auto xe = x->end();
    while ( xi != xe ) {
        *xi = std::max( fabs( *yi ), fabs( *zi ) );
        xi++;
        yi++;
        zi++;
    }
    return 0;
}
PetscErrorCode _AMP_pointwisemult( Vec a, Vec b, Vec c )
{
    auto x = getAMP( a );
    auto y = getAMP( b );
    auto z = getAMP( c );
    x->multiply( *y, *z );
    return 0;
}
PetscErrorCode _AMP_pointwisedivide( Vec a, Vec b, Vec c )
{
    auto x = getAMP( a );
    auto y = getAMP( b );
    auto z = getAMP( c );
    x->divide( *y, *z );
    return 0;
}
PetscErrorCode _AMP_sqrt( Vec a )
{
    auto x   = getAMP( a );
    auto cur = x->begin();
    auto end = x->end();
    while ( cur != end ) {
        *cur = std::sqrt( fabs( *cur ) );
        cur++;
    }
    return 0;
}
PetscErrorCode _AMP_setrandom( Vec a, PetscRandom )
{
    // set y[j] = random numbers
    auto x = getAMP( a );
    x->setRandomValues();
    return 0;
}
PetscErrorCode _AMP_conjugate( Vec )
{
    return 0; // Not dealing with complex right now
}
PetscErrorCode _AMP_axpby( Vec b, PetscScalar alpha, PetscScalar beta, Vec a )
{
    auto x = getAMP( a );
    auto y = getAMP( b );
    y->axpby( alpha, beta, *x );
    return 0;
}
PetscErrorCode _AMP_swap( Vec a, Vec b )
{
    auto x = getAMP( a );
    auto y = getAMP( b );
    x->swapVectors( *y );
    increaseState( a );
    increaseState( b );
    return 0;
}
PetscErrorCode _AMP_getsize( Vec a, PetscInt *ans )
{
    auto x      = getAMP( a );
    size_t size = x->getGlobalSize();
    if ( sizeof( PetscInt ) < 8 ) {
        AMP_ASSERT( size < 0x80000000 );
    }
    *ans = (PetscInt) size;
    return 0;
}
PetscErrorCode _AMP_maxpointwisedivide( Vec a, Vec b, PetscReal *res )
{
    auto x              = getAMP( a );
    auto y              = getAMP( b );
    auto cur_x          = x->constBegin();
    auto cur_y          = y->constBegin();
    auto end_x          = x->constEnd();
    PetscReal local_res = 0.0;
    while ( cur_x != end_x ) {
        if ( *cur_y == 0.0 ) {
            local_res = std::max( local_res, fabs( static_cast<PetscReal>( *cur_x ) ) );
        } else {
            local_res =
                std::max( local_res, fabs( static_cast<PetscReal>( ( *cur_x ) / ( *cur_y ) ) ) );
        }
        cur_x++;
        cur_y++;
    }

    *res = x->getComm().maxReduce( local_res );

    return 0;
}
PetscErrorCode _AMP_scale( Vec a, PetscScalar alpha )
{
    auto x = getAMP( a );
    x->scale( alpha );
    return 0;
}
PetscErrorCode _AMP_copy( Vec in, Vec out )
{
    auto x = getAMP( in );
    auto y = getAMP( out );
    y->copyVector( x->shared_from_this() );
    return 0;
}
PetscErrorCode _AMP_maxpy( Vec v, PetscInt num, const PetscScalar *alpha, Vec *vecs )
{
    for ( int i = 0; i != num; i++ )
        VecAXPY( v, alpha[i], vecs[i] );
    return 0;
}
PetscErrorCode _AMP_dot( Vec a, Vec b, PetscScalar *ans )
{
    auto x = getAMP( a );
    auto y = getAMP( b );
    *ans   = static_cast<PetscScalar>( x->dot( *y ) );
    return 0;
}
PetscErrorCode _AMP_mdot( Vec v, PetscInt num, const Vec vec[], PetscScalar *ans )
{
    for ( PetscInt i = 0; i != num; i++ )
        VecDot( v, vec[i], ans + i );
    return 0;
}
PetscErrorCode _AMP_tdot( Vec a, Vec b, PetscScalar *ans )
{
    auto x = getAMP( a );
    auto y = getAMP( b );
    *ans   = static_cast<PetscScalar>( x->dot( *y ) );
    return 0;
}
PetscErrorCode _AMP_mtdot( Vec v, PetscInt num, const Vec vec[], PetscScalar *ans )
{
    for ( PetscInt i = 0; i != num; i++ )
        VecTDot( v, vec[i], ans + i );
    return 0;
}
PetscErrorCode _AMP_l2normanddot( Vec a, Vec b, PetscScalar *dp, PetscReal *nm )
{
    auto x     = getAMP( a );
    auto y     = getAMP( b );
    auto rvals = y->L2NormAndDot( *x );
    *dp        = static_cast<PetscScalar>( rvals.second );
    *nm        = static_cast<PetscReal>( rvals.first );
    return 0;
}
PetscErrorCode _AMP_destroyvecs( PetscInt num, Vec vecArray[] )
{
    for ( PetscInt i = 0; i != num; i++ )
        PETSC::vecDestroy( &vecArray[i] );
    delete[] vecArray;
    return 0;
}
PetscErrorCode _AMP_axpy( Vec out, PetscScalar alpha, Vec in )
{
    auto x = getAMP( in );
    auto y = getAMP( out );
    y->axpy( alpha, *x, *y );
    return 0;
}
PetscErrorCode _AMP_waxpy( Vec w, PetscScalar alpha, Vec x, Vec y )
{
    auto xIn  = getAMP( x );
    auto yIn  = getAMP( y );
    auto wOut = getAMP( w );

    AMP_INSIST( ( wOut != xIn ) && ( wOut != yIn ),
                "ERROR: _AMP_waxpy: w cannot be the same as x or y" );

    wOut->axpy( alpha, *xIn, *yIn );
    return 0;
}
PetscErrorCode _AMP_norm_local( Vec in, NormType type, PetscReal *ans )
{
    auto x   = getAMP( in );
    auto ops = x->getVectorOperations();
    if ( type == NORM_1 )
        *ans = ops->localL1Norm( *x->getVectorData() ).get<PetscReal>();
    else if ( type == NORM_2 )
        *ans = ops->localL2Norm( *x->getVectorData() ).get<PetscReal>();
    else if ( type == NORM_INFINITY )
        *ans = ops->localMaxNorm( *x->getVectorData() ).get<PetscReal>();
    else if ( type == NORM_1_AND_2 ) {
        *ans         = ops->localL1Norm( *x->getVectorData() ).get<PetscReal>();
        *( ans + 1 ) = ops->localL2Norm( *x->getVectorData() ).get<PetscReal>();
    } else
        AMP_ERROR( "Unknown norm type" );
    increaseState( in );
    return 0;
}
PetscErrorCode _AMP_norm( Vec in, NormType type, PetscReal *ans )
{
    auto x = getAMP( in );
    if ( type == NORM_1 )
        *ans = static_cast<PetscReal>( x->L1Norm() );
    else if ( type == NORM_2 )
        *ans = static_cast<PetscReal>( x->L2Norm() );
    else if ( type == NORM_INFINITY )
        *ans = static_cast<PetscReal>( x->maxNorm() );
    else if ( type == NORM_1_AND_2 ) {
        *ans         = static_cast<PetscReal>( x->L1Norm() );
        *( ans + 1 ) = static_cast<PetscReal>( x->L2Norm() );
    } else
        AMP_ERROR( "Unknown norm type" );
    increaseState( in );
    return 0;
}
static inline bool verifyMemory( Vec in, Vec out )
{
    auto p1 = getAMP( in );
    auto p2 = getAMP( out );
    for ( size_t i = 0; i != p1->numberOfDataBlocks(); i++ ) {
        if ( p1->getRawDataBlock<PetscScalar>() == p2->getRawDataBlock<PetscScalar>() )
            return false;
    }
    return true;
}
PetscErrorCode _AMP_duplicate( Vec in, Vec *out )
{
    auto p = getAMP( in );
    *out   = getVec( p->clone() );
    AMP_ASSERT( verifyMemory( in, *out ) );
    return 0;
}
PetscErrorCode _AMP_duplicatevecs( Vec v, PetscInt num, Vec **vecArray )
{
    auto tvecArray = new Vec[num];
    for ( PetscInt i = 0; i != num; i++ )
        VecDuplicate( v, tvecArray + i );
    *vecArray = tvecArray;
    return 0;
}
PetscErrorCode _AMP_restorearray( Vec y, PetscScalar ** )
{
    increaseState( y );
    return 0;
}
PetscErrorCode _AMP_getarray( Vec in, PetscScalar **out )
{
    auto p = getAMP( in );
    *out   = p->getRawDataBlock<PetscScalar>();
    return 0;
}
PetscErrorCode _AMP_getlocalsize( Vec in, PetscInt *out )
{
    auto p = getAMP( in );
    *out   = (PetscInt) p->getLocalSize();
    return 0;
}
PetscErrorCode _AMP_setfromoptions( Vec )
{
    AMP_ERROR( "This should not be thrown" );
    return 0;
}
PetscErrorCode _AMP_reciprocal( Vec v )
{
    auto p = getAMP( v );
    p->reciprocal( *p );
    return 0;
}
PetscErrorCode _AMP_abs( Vec v )
{
    auto p = getAMP( v );
    p->abs( *p );
    increaseState( v );
    return 0;
}
PetscErrorCode _AMP_resetarray( Vec ) { return 0; }
PetscErrorCode _AMP_destroy( Vec v )
{
    auto p = getWrapper( v );
    delete p;
    return 0;
}
PetscErrorCode _AMP_create( Vec ) { return 0; }
PetscErrorCode _AMP_set( Vec x, PetscScalar alpha )
{
    auto p = getAMP( x );
    p->setToScalar( alpha );
    // petsc calls object state increase for this function
    return 0;
}
void reset_vec_ops( Vec t )
{
    // Then, replace the functions
    t->ops->duplicate               = _AMP_duplicate;
    t->ops->duplicatevecs           = _AMP_duplicatevecs;
    t->ops->destroyvecs             = _AMP_destroyvecs;
    t->ops->dot                     = _AMP_dot;
    t->ops->mdot                    = _AMP_mdot;
    t->ops->norm                    = _AMP_norm;
    t->ops->tdot                    = _AMP_tdot;
    t->ops->mtdot                   = _AMP_mtdot;
    t->ops->scale                   = _AMP_scale;
    t->ops->copy                    = _AMP_copy;
    t->ops->set                     = _AMP_set;
    t->ops->swap                    = _AMP_swap;
    t->ops->axpy                    = _AMP_axpy;
    t->ops->axpby                   = _AMP_axpby;
    t->ops->maxpy                   = _AMP_maxpy;
    t->ops->aypx                    = _AMP_aypx;
    t->ops->waxpy                   = _AMP_waxpy;
    t->ops->axpbypcz                = _AMP_axpbypcz;
    t->ops->pointwisemult           = _AMP_pointwisemult;
    t->ops->pointwisedivide         = _AMP_pointwisedivide;
    t->ops->setvalues               = _AMP_setvalues;
    t->ops->assemblybegin           = _AMP_assemblybegin;
    t->ops->assemblyend             = _AMP_assemblyend;
    t->ops->getsize                 = _AMP_getsize;
    t->ops->getlocalsize            = _AMP_getlocalsize;
    t->ops->max                     = _AMP_max;
    t->ops->min                     = _AMP_min;
    t->ops->setrandom               = _AMP_setrandom;
    t->ops->setoption               = _AMP_setoption;
    t->ops->destroy                 = _AMP_destroy;
    t->ops->view                    = _AMP_view;
    t->ops->dot_local               = _AMP_dot_local;
    t->ops->tdot_local              = _AMP_tdot_local;
    t->ops->norm_local              = _AMP_norm_local;
    t->ops->mdot_local              = _AMP_mdot_local;
    t->ops->mtdot_local             = _AMP_mtdot_local;
    t->ops->reciprocal              = _AMP_reciprocal;
    t->ops->conjugate               = _AMP_conjugate;
    t->ops->setlocaltoglobalmapping = _AMP_setlocaltoglobalmapping;
    t->ops->setvalueslocal          = _AMP_setvalueslocal;
    t->ops->maxpointwisedivide      = _AMP_maxpointwisedivide;
    t->ops->load                    = _AMP_load;
    t->ops->pointwisemax            = _AMP_pointwisemax;
    t->ops->pointwisemaxabs         = _AMP_pointwisemaxabs;
    t->ops->pointwisemin            = _AMP_pointwisemin;
    t->ops->getvalues               = _AMP_getvalues;
    t->ops->sqrt                    = _AMP_sqrt;
    t->ops->abs                     = _AMP_abs;
    t->ops->exp                     = _AMP_exp;
    t->ops->log                     = _AMP_log;
    t->ops->shift                   = _AMP_shift;
    t->ops->create                  = _AMP_create;
    t->ops->dotnorm2                = _AMP_l2normanddot;

    auto p = getAMP( t );
    if ( p->getVectorData()->hasContiguousData() ) {
        t->ops->getarray         = _AMP_getarray;
        t->ops->restorearray     = _AMP_restorearray;
        t->ops->placearray       = _AMP_placearray;
        t->ops->replacearray     = _AMP_replacearray;
        t->ops->resetarray       = _AMP_resetarray;
        t->ops->setvaluesblocked = _AMP_setvaluesblocked;
    } else {
        t->ops->getarray         = nullptr;
        t->ops->restorearray     = nullptr;
        t->ops->placearray       = nullptr;
        t->ops->replacearray     = nullptr;
        t->ops->resetarray       = nullptr;
        t->ops->setvaluesblocked = nullptr;
    }
    /*** The following functions do not need to be overridden
      t->ops->setfromoptions = _AMP_setfromoptions;
     ***/
}


} // namespace PETSC
