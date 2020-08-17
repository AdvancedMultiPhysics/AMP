#include "AMP/vectors/petsc/ManagedPetscVector.h"
#include "AMP/vectors/data/VectorDataCPU.h"
#include "AMP/vectors/data/VectorDataIterator.h"

#include "AMP/utils/Utilities.h"
#include "AMP/vectors/petsc/PetscVector.h"
#ifdef USE_EXT_TRILINOS
#include "AMP/vectors/trilinos/epetra/EpetraVectorEngine.h"
#endif


#include "petsc/private/vecimpl.h"
#include "petscsys.h"


// Overridden Petsc functions!
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

#if ( PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR == 0 )
PetscErrorCode _AMP_setoption( Vec, VecOption, PetscTruth ) { return 0; }
PetscErrorCode _AMP_load( PetscViewer, const VecType, Vec * )
{
    AMP_ERROR( "48 Not implemented" );
    return 0;
}
PetscErrorCode _AMP_loadintovector( PetscViewer, Vec )
{
    AMP_ERROR( "40 Not implemented" );
    return 0;
}
PetscErrorCode _AMP_loadintovectornative( PetscViewer, Vec )
{
    AMP_ERROR( "41 Not implemented" );
    return 0;
}
#elif PETSC_VERSION_GE( 3, 2, 0 )
PetscErrorCode _AMP_setoption( Vec, VecOption, PetscBool ) { return 0; }
PetscErrorCode _AMP_load( Vec, PetscViewer )
{
    AMP_ERROR( "48 Not implemented" );
    return 0;
}
#else
#error Not programmed for this version of petsc
#endif


// Inserts or adds values into certain locations of a vector.
PetscErrorCode
_AMP_setvalues( Vec px, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora )
{
    auto x       = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( px->data );
    auto indices = new size_t[ni];
    auto vals    = new double[ni];
    for ( PetscInt i = 0; i < ni; i++ ) {
        indices[i] = static_cast<size_t>( ix[i] );
        vals[i]    = static_cast<double>( y[i] );
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

#if PETSC_VERSION_GE( 3, 7, 5 )
PetscErrorCode _AMP_shift( Vec px, PetscScalar s )
{
    auto x   = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( px->data );
    auto cur = x->VectorData::begin();
    auto end = x->VectorData::end();
    while ( cur != end ) {
        *cur = s + ( *cur );
        cur++;
    }
    return 0;
}
#elif PETSC_VERSION_LT( 3, 7, 5 )
// This function makes no sense wrt the PETSc interface VecShift( Vec, PetscScalar );
PetscErrorCode _AMP_shift( Vec )
{
    AMP_ERROR( "This function cannot be implemented as designed" );
    return 0;
}
#else
#error Not programmed for this version of petsc
#endif

PetscErrorCode
_AMP_axpbypcz( Vec c, PetscScalar alpha, PetscScalar beta, PetscScalar gamma, Vec a, Vec b )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    auto z = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( c->data );
    if ( z->isAnAliasOf( *x ) ) {
        z->linearSum( alpha + gamma, *x, beta, *y );
    } else if ( z->isAnAliasOf( *y ) ) {
        z->linearSum( alpha, *x, beta + gamma, *y );
    } else {
        z->linearSum( alpha, *x, gamma, *z );
        z->linearSum( beta, *y, 1., *z );
    }
    PetscObjectStateIncrease( reinterpret_cast<::PetscObject>( c ) );
    return 0;
}

PetscErrorCode _AMP_max( Vec a, PetscInt *p, PetscReal *ans )
{
    if ( ( p != nullptr ) && ( p != PETSC_NULL ) ) {
        AMP_ERROR( "Cannot find position for max" );
    }
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    *ans   = x->max();
    return 0;
}

PetscErrorCode _AMP_min( Vec a, PetscInt *p, PetscReal *ans )
{
    if ( ( p != nullptr ) && ( p != PETSC_NULL ) ) {
        AMP_ERROR( "Cannot find position for max" );
    }
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    *ans   = x->min();
    return 0;
}

PetscErrorCode _AMP_aypx( Vec b, PetscScalar alpha, Vec a )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    y->linearSum( alpha, *y, 1., *x );
    return 0;
} /* y = x + alpha * y */

PetscErrorCode _AMP_dot_local( Vec a, Vec b, PetscScalar *ans )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    *ans   = x->localDot( *y );
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
    auto x   = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto cur = x->VectorData::begin();
    auto end = x->VectorData::end();
    while ( cur != end ) {
        *cur = exp( *cur );
        cur++;
    }
    return 0;
}

PetscErrorCode _AMP_log( Vec a )
{
    auto x   = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto cur = x->VectorData::begin();
    auto end = x->VectorData::end();
    while ( cur != end ) {
        *cur = log( *cur );
        cur++;
    }
    return 0;
}

PetscErrorCode _AMP_pointwisemin( Vec a, Vec b, Vec c )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    auto z = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( c->data );

    AMP_INSIST( x->getLocalSize() == y->getLocalSize(), "Incompatible vectors" );
    AMP_INSIST( x->getLocalSize() == z->getLocalSize(), "Incompatible vectors" );

    auto xi = x->VectorData::begin();
    auto xe = x->VectorData::end();
    auto yi = y->VectorData::constBegin();
    auto zi = z->VectorData::constBegin();
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
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    auto z = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( c->data );

    AMP_INSIST( x->getLocalSize() == y->getLocalSize(), "Incompatible vectors" );
    AMP_INSIST( x->getLocalSize() == z->getLocalSize(), "Incompatible vectors" );

    auto xi = x->VectorData::begin();
    auto xe = x->VectorData::end();
    auto yi = y->VectorData::constBegin();
    auto zi = z->VectorData::constBegin();
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
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    auto z = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( c->data );

    AMP_INSIST( x->getLocalSize() == y->getLocalSize(), "Incompatible vectors" );
    AMP_INSIST( x->getLocalSize() == z->getLocalSize(), "Incompatible vectors" );

    auto xi = x->VectorData::begin();
    auto yi = y->VectorData::begin();
    auto zi = z->VectorData::begin();
    auto xe = x->VectorData::end();
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
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    auto z = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( c->data );
    x->multiply( *y, *z );
    return 0;
}

PetscErrorCode _AMP_pointwisedivide( Vec a, Vec b, Vec c )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    auto z = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( c->data );
    x->divide( *y, *z );
    return 0;
}

PetscErrorCode _AMP_sqrt( Vec a )
{
    auto x   = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto cur = x->VectorData::begin();
    auto end = x->VectorData::end();
    while ( cur != end ) {
        *cur = sqrt( fabs( *cur ) );
        cur++;
    }
    return 0;
}

PetscErrorCode _AMP_setrandom( Vec a, PetscRandom )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    x->setRandomValues();
    return 0;
} /* set y[j] = random numbers */

PetscErrorCode _AMP_conjugate( Vec )
{
    return 0; // Not dealing with complex right now
}

PetscErrorCode _AMP_axpby( Vec b, PetscScalar alpha, PetscScalar beta, Vec a )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    y->axpby( alpha, beta, *x );
    return 0;
}

PetscErrorCode _AMP_swap( Vec a, Vec b )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    x->swapVectors( *y );
    return 0;
}

PetscErrorCode _AMP_getsize( Vec a, PetscInt *ans )
{
    auto x      = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    size_t size = x->getGlobalSize();
    if ( sizeof( PetscInt ) < 8 ) {
        AMP_ASSERT( size < 0x80000000 );
    }
    *ans = (PetscInt) size;
    return 0;
}

PetscErrorCode _AMP_maxpointwisedivide( Vec a, Vec b, PetscReal *res )
{
    auto x           = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y           = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    auto cur_x       = x->VectorData::constBegin();
    auto cur_y       = y->VectorData::constBegin();
    auto end_x       = x->VectorData::constEnd();
    double local_res = 0.0;
    while ( cur_x != end_x ) {
        if ( *cur_y == 0.0 ) {
            local_res = std::max( local_res, fabs( *cur_x ) );
        } else {
            local_res = std::max( local_res, fabs( ( *cur_x ) / ( *cur_y ) ) );
        }
        cur_x++;
        cur_y++;
    }

    *res = x->getComm().maxReduce( local_res );

    return 0;
}

PetscErrorCode _AMP_scale( Vec a, PetscScalar alpha )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    x->scale( alpha );
    return 0;
}

PetscErrorCode _AMP_copy( Vec in, Vec out )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( in->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( out->data );
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
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    *ans   = x->dot( *y );
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
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( a->data );
    auto y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( b->data );
    *ans   = x->dot( *y );
    return 0;
}

PetscErrorCode _AMP_mtdot( Vec v, PetscInt num, const Vec vec[], PetscScalar *ans )
{
    for ( PetscInt i = 0; i != num; i++ )
        VecTDot( v, vec[i], ans + i );
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
    auto *x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( in->data );
    auto *y = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( out->data );
    y->axpy( alpha, *x, *y );
    return 0;
}


PetscErrorCode _AMP_waxpy( Vec w, PetscScalar alpha, Vec x, Vec y )
{
    auto *xIn  = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( x->data );
    auto *yIn  = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( y->data );
    auto *wOut = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( w->data );

    AMP_INSIST( ( wOut != xIn ) && ( wOut != yIn ),
                "ERROR: _AMP_waxpy: w cannot be the same as x or y" );

    wOut->axpy( alpha, *xIn, *yIn );
    return 0;
}


PetscErrorCode _AMP_norm_local( Vec in, NormType type, PetscReal *ans )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( in->data );
    if ( type == NORM_1 )
        *ans = x->localL1Norm();
    else if ( type == NORM_2 )
        *ans = x->localL2Norm();
    else if ( type == NORM_INFINITY )
        *ans = x->localMaxNorm();
    else if ( type == NORM_1_AND_2 ) {
        *ans         = x->localL1Norm();
        *( ans + 1 ) = x->localL2Norm();
    } else
        AMP_ERROR( "Unknown norm type" );
    if ( type != NORM_1_AND_2 ) {
        PetscObjectComposedDataSetReal(
            reinterpret_cast<::PetscObject>( in ), NormIds[type], ans[0] );
    }
    PetscObjectStateIncrease( reinterpret_cast<::PetscObject>( in ) );
    return 0;
}

PetscErrorCode _AMP_norm( Vec in, NormType type, PetscReal *ans )
{
    auto x = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( in->data );
    if ( type == NORM_1 )
        *ans = x->L1Norm();
    else if ( type == NORM_2 )
        *ans = x->L2Norm();
    else if ( type == NORM_INFINITY )
        *ans = x->maxNorm();
    else if ( type == NORM_1_AND_2 ) {
        *ans         = x->L1Norm();
        *( ans + 1 ) = x->L2Norm();
    } else
        AMP_ERROR( "Unknown norm type" );
    if ( type != NORM_1_AND_2 ) {
        PetscObjectComposedDataSetReal(
            reinterpret_cast<::PetscObject>( in ), NormIds[type], ans[0] );
    }
    PetscObjectStateIncrease( reinterpret_cast<::PetscObject>( in ) );
    return 0;
}

bool _Verify_Memory( AMP::LinearAlgebra::Vector *p1, AMP::LinearAlgebra::Vector *p2 )
{
    for ( size_t i = 0; i != p1->numberOfDataBlocks(); i++ ) {
        if ( p1->getRawDataBlock<double>() == p2->getRawDataBlock<double>() )
            return false;
    }
    return true;
}

PetscErrorCode _AMP_duplicate( Vec in, Vec *out )
{

    auto *p = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( in->data );
    AMP::LinearAlgebra::ManagedPetscVector *dup = p->petscDuplicate();
    AMP_ASSERT( _Verify_Memory( p, dup ) );
    *out = dup->getVec();
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
    PetscObjectStateIncrease( reinterpret_cast<::PetscObject>( y ) );
    return 0;
}

PetscErrorCode _AMP_getarray( Vec in, PetscScalar **out )
{
    auto *p = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( in->data );
    *out    = p->getRawDataBlock<PetscScalar>();
    return 0;
}

PetscErrorCode _AMP_getlocalsize( Vec in, PetscInt *out )
{
    auto *p = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( in->data );
    *out    = (PetscInt) p->getLocalSize();
    return 0;
}

PetscErrorCode _AMP_setfromoptions( Vec )
{
    AMP_ERROR( "This should not be thrown" );
    return 0;
}

PetscErrorCode _AMP_reciprocal( Vec v )
{
    auto *p = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( v->data );
    p->reciprocal( *p );
    return 0;
}

PetscErrorCode _AMP_abs( Vec v )
{
    auto *p = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( v->data );
    p->abs( *p );
    PetscObjectStateIncrease( reinterpret_cast<::PetscObject>( v ) );
    return 0;
}

PetscErrorCode _AMP_resetarray( Vec ) { return 0; }
PetscErrorCode _AMP_destroy( Vec v )
{
    auto *p = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( v->data );
    if ( p->constructedWithPetscDuplicate() ) {
        delete p;
    }
    return 0;
}

PetscErrorCode _AMP_create( Vec ) { return 0; }


PetscErrorCode _AMP_set( Vec x, PetscScalar alpha )
{
    auto *p = reinterpret_cast<AMP::LinearAlgebra::ManagedPetscVector *>( x->data );
    p->setToScalar( alpha );
    // petsc calls object state increase for this function
    return 0;
}

void reset_vec_ops( Vec t )
{

    // Then, replace the functions
    t->ops->duplicate       = _AMP_duplicate;
    t->ops->duplicatevecs   = _AMP_duplicatevecs;
    t->ops->destroyvecs     = _AMP_destroyvecs;
    t->ops->dot             = _AMP_dot;
    t->ops->mdot            = _AMP_mdot;
    t->ops->norm            = _AMP_norm;
    t->ops->tdot            = _AMP_tdot;
    t->ops->mtdot           = _AMP_mtdot;
    t->ops->scale           = _AMP_scale;
    t->ops->copy            = _AMP_copy;
    t->ops->set             = _AMP_set;
    t->ops->swap            = _AMP_swap;
    t->ops->axpy            = _AMP_axpy;
    t->ops->axpby           = _AMP_axpby;
    t->ops->maxpy           = _AMP_maxpy;
    t->ops->aypx            = _AMP_aypx;
    t->ops->waxpy           = _AMP_waxpy;
    t->ops->axpbypcz        = _AMP_axpbypcz;
    t->ops->pointwisemult   = _AMP_pointwisemult;
    t->ops->pointwisedivide = _AMP_pointwisedivide;
    t->ops->setvalues       = _AMP_setvalues;
    t->ops->assemblybegin   = _AMP_assemblybegin;
    t->ops->assemblyend     = _AMP_assemblyend;
    t->ops->getarray        = _AMP_getarray;
    t->ops->getsize         = _AMP_getsize;
    t->ops->getlocalsize    = _AMP_getlocalsize;
    t->ops->restorearray    = _AMP_restorearray;
    t->ops->max             = _AMP_max;
    t->ops->min             = _AMP_min;
    t->ops->setrandom       = _AMP_setrandom;


    t->ops->setoption = _AMP_setoption;


    t->ops->setvaluesblocked        = _AMP_setvaluesblocked;
    t->ops->destroy                 = _AMP_destroy;
    t->ops->view                    = _AMP_view;
    t->ops->placearray              = _AMP_placearray;
    t->ops->replacearray            = _AMP_replacearray;
    t->ops->dot_local               = _AMP_dot_local;
    t->ops->tdot_local              = _AMP_tdot_local;
    t->ops->norm_local              = _AMP_norm_local;
    t->ops->mdot_local              = _AMP_mdot_local;
    t->ops->mtdot_local             = _AMP_mtdot_local;
    t->ops->reciprocal              = _AMP_reciprocal;
    t->ops->conjugate               = _AMP_conjugate;
    t->ops->setlocaltoglobalmapping = _AMP_setlocaltoglobalmapping;
    t->ops->setvalueslocal          = _AMP_setvalueslocal;
    t->ops->resetarray              = _AMP_resetarray;
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
#if ( PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR == 0 )
    t->ops->loadintovector       = _AMP_loadintovector;
    t->ops->loadintovectornative = _AMP_loadintovectornative;
    t->ops->viewnative           = _AMP_viewnative;
#endif

    /*** The following functions do not need to be overridden
      t->ops->setfromoptions = _AMP_setfromoptions;
     ***/
}


namespace AMP {
namespace LinearAlgebra {


void ManagedPetscVector::initPetsc()
{
    auto params  = std::dynamic_pointer_cast<ManagedVectorParameters>( getParameters() );
    AMP_MPI comm = std::dynamic_pointer_cast<VectorData>( params->d_Engine )->getComm();
    VecCreate( comm.getCommunicator(), &d_petscVec );

    d_petscVec->data        = this;
    d_petscVec->petscnative = PETSC_FALSE;

    reset_vec_ops( d_petscVec );

#if ( PETSC_VERSION_MAJOR == 3 && PETSC_VERSION_MINOR == 0 )
    PetscMapInitialize( comm.getCommunicator(), d_petscVec->map );
    PetscMapSetBlockSize( d_petscVec->map, 1 );
    PetscMapSetSize( d_petscVec->map, this->getGlobalSize() );
    PetscMapSetLocalSize( d_petscVec->map, this->getLocalSize() );
    d_petscVec->map->rstart = static_cast<PetscInt>( this->getDOFManager()->beginDOF() );
    d_petscVec->map->rend   = static_cast<PetscInt>( this->getDOFManager()->endDOF() );
#elif PETSC_VERSION_GE( 3, 2, 0 )
    PetscLayoutSetBlockSize( d_petscVec->map, 1 );
    PetscLayoutSetSize( d_petscVec->map, this->getGlobalSize() );
    PetscLayoutSetLocalSize( d_petscVec->map, this->getLocalSize() );
    PetscLayoutSetUp( d_petscVec->map );
#else
#error Not programmed for this version yet
#endif

    d_bMadeWithPetscDuplicate = false;

    const std::string my_name = "AMPManagedPetscVectorReal";
    int ierr                  = 0;

    if ( ( (PetscObject) d_petscVec )->type_name ) {
        ierr = PetscFree( ( (PetscObject) d_petscVec )->type_name );
    }

    ierr =
        PetscObjectChangeTypeName( reinterpret_cast<PetscObject>( d_petscVec ), my_name.c_str() );
    AMP_INSIST( ierr == 0, "PetscObjectChangeTypeName returned non-zero error code" );

    VecSetFromOptions( d_petscVec );
}


ManagedPetscVector::ManagedPetscVector( VectorParameters::shared_ptr params )
    : ManagedVector( params ), PetscVector()
{
    initPetsc();
    auto listener = std::dynamic_pointer_cast<DataChangeListener>( shared_from_this() );
    registerListener( listener );
}


ManagedPetscVector::ManagedPetscVector( Vector::shared_ptr alias )
    : ManagedVector( alias ), PetscVector()
{
    initPetsc();
    auto listener = std::dynamic_pointer_cast<DataChangeListener>( shared_from_this() );
    alias->registerListener( listener );
}


ManagedPetscVector::~ManagedPetscVector()
{
    int refct = ( ( (PetscObject) d_petscVec )->refct );
    if ( !d_bMadeWithPetscDuplicate ) {
        if ( refct > 1 )
            AMP_ERROR( "Deleting a vector still held by PETSc" );
        PETSC::vecDestroy( &d_petscVec );
    }
}


bool ManagedPetscVector::petscHoldsView() const
{
    int refct = ( ( (PetscObject) d_petscVec )->refct );
    if ( !d_bMadeWithPetscDuplicate && refct > 1 )
        return true;
    return false;
}


ManagedPetscVector *ManagedPetscVector::petscDuplicate()
{
    ManagedPetscVector *pAns = rawClone();
    pAns->setVariable( getVariable() );
    pAns->d_bMadeWithPetscDuplicate = true;
    return pAns;
}


void ManagedPetscVector::copyFromPetscVec( Vector &dest, Vec source )
{
    auto params = std::dynamic_pointer_cast<ManagedVectorParameters>(
        dynamic_cast<ManagedVector *>( &dest )->getParameters() );
    if ( !params )
        throw( "Incompatible vector types" );

    if ( sizeof( PetscInt ) < 8 )
        AMP_INSIST( dest.getGlobalSize() < 0x80000000,
                    "PETsc is compiled with 32-bit integers and "
                    "we are trying to use a vector with more "
                    "than 2^31 elements" );

    auto ids       = new PetscInt[dest.getLocalSize()];
    PetscInt begin = dest.getLocalStartID();
    PetscInt end   = begin + dest.getLocalSize() - 1;

    for ( PetscInt i = begin; i < end; i++ )
        ids[i - begin] = i;
    VecGetValues( source, dest.getLocalSize(), ids, dest.getRawDataBlock<double>() );
    delete[] ids;
}


std::shared_ptr<AMP::LinearAlgebra::Vector> ManagedPetscVector::createFromPetscVec( Vec source,
                                                                                    AMP_MPI &comm )
{
#ifdef USE_EXT_TRILINOS
    PetscInt local_size, global_size, local_start, local_end;
    VecGetLocalSize( source, &local_size );
    VecGetSize( source, &global_size );
    VecGetOwnershipRange( source, &local_start, &local_end );
    auto buffer = std::make_shared<VectorDataCPU<double>>( local_start, local_size, global_size );
    auto t      = std::make_shared<ManagedPetscVectorParameters>();
    auto ve_params =
        std::make_shared<EpetraVectorEngineParameters>( local_size, global_size, comm );
    t->d_Engine = std::make_shared<EpetraVectorEngine>( ve_params, buffer );
    auto pRetVal =
        std::make_shared<ManagedPetscVector>( std::dynamic_pointer_cast<VectorParameters>( t ) );
    return pRetVal;
#else
    AMP_ERROR( "General case not programmed yet" );
    NULL_USE( source );
    NULL_USE( comm );
    return std::shared_ptr<AMP::LinearAlgebra::Vector>();
#endif
}


void ManagedPetscVector::swapVectors( Vector &other )
{
    auto tmp = dynamic_cast<ManagedPetscVector *>( &other );
    AMP_ASSERT( tmp != nullptr );
    ParentVector::swapVectors( *tmp );
}

ManagedVector *ManagedPetscVector::getNewRawPtr() const
{
    return new ManagedPetscVector( std::dynamic_pointer_cast<VectorParameters>( d_pParameters ) );
}

bool ManagedPetscVector::constructedWithPetscDuplicate() { return d_bMadeWithPetscDuplicate; }

ManagedPetscVector *ManagedPetscVector::rawClone() const
{
    auto p   = std::make_shared<ManagedPetscVectorParameters>();
    auto vec = std::dynamic_pointer_cast<Vector>( d_Engine );
    if ( vec ) {
        auto vec2   = vec->cloneVector( "ManagedPetscVectorClone" );
        p->d_Buffer = std::dynamic_pointer_cast<VectorData>( vec2 );
        p->d_Engine = std::dynamic_pointer_cast<Vector>( vec2 );
    } else {
        AMP_ERROR( "ManagedPetscVector::rawClone() should not have reached here!" );
    }
    p->d_CommList   = getCommunicationList();
    p->d_DOFManager = getDOFManager();
    auto retVal     = new ManagedPetscVector( p );
    return retVal;
}

Vector::shared_ptr ManagedPetscVector::cloneVector( const Variable::shared_ptr p ) const
{
    Vector::shared_ptr retVal( rawClone() );
    retVal->setVariable( p );
    return retVal;
}


void ManagedPetscVector::assemble() {}


} // namespace LinearAlgebra
} // namespace AMP