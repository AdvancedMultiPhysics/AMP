#include "AMP/vectors/petsc/NativePetscVectorOperations.h"
#include "AMP/vectors/petsc/NativePetscVectorData.h"
#include "AMP/vectors/petsc/PetscHelpers.h"


namespace AMP::LinearAlgebra {

//**********************************************************************
// Static functions that operate on VectorData objects
// Helper function
Vec NativePetscVectorOperations::getPetscVec( VectorData &vx )
{
    auto nx = dynamic_cast<NativePetscVectorData *>( &vx );
    nx->resetArray();
    return nx->getVec();
}

Vec NativePetscVectorOperations::getPetscVec( const VectorData &vx )
{
    auto nx = dynamic_cast<const NativePetscVectorData *>( &vx );
    nx->resetArray();
    return nx->getVec();
}

Vec NativePetscVectorOperations::getConstPetscVec( const VectorData &vx )
{
    auto nx = dynamic_cast<const NativePetscVectorData *>( &vx );
    AMP_ASSERT( nx );
    return nx->getVec();
}

NativePetscVectorData *NativePetscVectorOperations::getNativeVec( VectorData &vx )
{
    return dynamic_cast<NativePetscVectorData *>( &vx );
}

const NativePetscVectorData *NativePetscVectorOperations::getNativeVec( const VectorData &vx )
{
    return dynamic_cast<const NativePetscVectorData *>( &vx );
}

// Function to perform  this = alpha x + beta y + gamma z
void NativePetscVectorOperations::axpbypcz( const Scalar &alpha_in,
                                            const VectorData &vx,
                                            const Scalar &beta_in,
                                            const VectorData &vy,
                                            const Scalar &gamma_in,
                                            VectorData &vz )
{
    Vec x = getConstPetscVec( vx );
    Vec y = getConstPetscVec( vy );
    Vec z = getPetscVec( vz );

    auto alpha = alpha_in.get<double>();
    auto beta  = beta_in.get<double>();
    auto gamma = gamma_in.get<double>();

    if ( x != y && x != z && y != z ) {
        // We can safely perform  z = alpha x + beta y + gamma z
        VecAXPBYPCZ( z, alpha, beta, gamma, x, y );
    } else if ( x != y && x == z ) {
        // x==z:  z = (alpha+gamma)*z + beta*y
        double scale = alpha + gamma;
        VecAXPBY( z, beta, scale, y );
    } else if ( x != y && y == z ) {
        // y==z:  z = (beta+gamma)*z + alpha*x
        double scale = beta + gamma;
        VecAXPBY( z, alpha, scale, x );
    } else if ( x == y && x == z ) {
        // x==y==z:  z = (alpha+beta+gamma)*z
        double scale = alpha + beta + gamma;
        VecScale( z, scale );
    } else {
        AMP_ERROR( "Internal error\n" );
    }
}
void NativePetscVectorOperations::copy( const VectorData &x, VectorData &y )
{
    auto nx = getNativeVec( x );
    auto ny = getNativeVec( y );
    if ( nx && ny ) {
        ny->resetArray();
        VecCopy( nx->getVec(), ny->getVec() );
        y.copyGhostValues( x );
    } else {
        VectorOperationsDefault<double>().copy( x, y );
    }
}

void NativePetscVectorOperations::zero( VectorData &x ) { VecZeroEntries( getPetscVec( x ) ); }

void NativePetscVectorOperations::setToScalar( const Scalar &alpha, VectorData &x )
{
    auto vec = getPetscVec( x );
    VecSet( vec, static_cast<PetscReal>( alpha ) );
}

void NativePetscVectorOperations::setRandomValues( VectorData &x )
{
    auto nx = getNativeVec( x );
    // Get PETSc random context
    if ( !d_PetscRandom )
        d_PetscRandom = PETSC::genPetscRandom( nx->getComm() );
    // Get the native vector and set to random values
    nx->resetArray();
    VecSetRandom( nx->getVec(), *d_PetscRandom );
}

void NativePetscVectorOperations::scale( const Scalar &alpha, const VectorData &x, VectorData &y )
{
    VecCopy( getConstPetscVec( x ), getPetscVec( y ) );
    VecScale( getPetscVec( y ), static_cast<PetscScalar>( alpha ) );
}


void NativePetscVectorOperations::scale( const Scalar &alpha, VectorData &x )
{
    VecScale( getPetscVec( x ), static_cast<PetscScalar>( alpha ) );
}

void NativePetscVectorOperations::add( const VectorData &x, const VectorData &y, VectorData &z )
{
    axpbypcz( 1.0, x, 1.0, y, 0.0, z );
}


void NativePetscVectorOperations::subtract( const VectorData &x,
                                            const VectorData &y,
                                            VectorData &z )
{
    axpbypcz( 1.0, x, -1.0, y, 0.0, z );
}


void NativePetscVectorOperations::multiply( const VectorData &x,
                                            const VectorData &y,
                                            VectorData &z )
{
    VecPointwiseMult( getPetscVec( z ), getConstPetscVec( x ), getConstPetscVec( y ) );
}


void NativePetscVectorOperations::divide( const VectorData &x, const VectorData &y, VectorData &z )
{
    VecPointwiseDivide( getPetscVec( z ), getConstPetscVec( x ), getConstPetscVec( y ) );
}


void NativePetscVectorOperations::reciprocal( const VectorData &x, VectorData &y )
{
    VecCopy( getConstPetscVec( x ), getPetscVec( y ) );
    VecReciprocal( getPetscVec( y ) );
}


void NativePetscVectorOperations::linearSum( const Scalar &alpha,
                                             const VectorData &x,
                                             const Scalar &beta,
                                             const VectorData &y,
                                             VectorData &z )
{
    axpbypcz( alpha, x, beta, y, 0.0, z );
}


void NativePetscVectorOperations::axpy( const Scalar &alpha,
                                        const VectorData &x,
                                        const VectorData &y,
                                        VectorData &z )
{
    axpbypcz( alpha, x, 1.0, y, 0.0, z );
}


void NativePetscVectorOperations::axpby( const Scalar &alpha,
                                         const Scalar &beta,
                                         const VectorData &x,
                                         VectorData &vz )
{
    auto &z = *getNativeVec( vz );
    axpbypcz( alpha, x, beta, z, 0.0, z );
}


void NativePetscVectorOperations::abs( const VectorData &x, VectorData &y )
{
    VecCopy( getConstPetscVec( x ), getPetscVec( y ) );
    VecAbs( getPetscVec( y ) );
}

void NativePetscVectorOperations::addScalar( const VectorData &x,
                                             const Scalar &alpha,
                                             VectorData &y )
{
    auto py = getPetscVec( y );
    VecCopy( getConstPetscVec( x ), py );
    VecShift( py, static_cast<PetscReal>( alpha ) );
}

Scalar NativePetscVectorOperations::min( const VectorData &x ) const
{
    PetscReal val;
    VecMin( getConstPetscVec( x ), nullptr, &val );
    return val;
}

Scalar NativePetscVectorOperations::max( const VectorData &x ) const
{
    PetscReal val;
    VecMax( getConstPetscVec( x ), nullptr, &val );
    return val;
}

Scalar NativePetscVectorOperations::L1Norm( const VectorData &x ) const
{
    PetscReal ans;
    PetscErrorCode ierr = VecNorm( getConstPetscVec( x ), NORM_1, &ans );
    AMP_INSIST( ierr == 0, "Error in NativePetscVectorOperations::L1Norm" );
    return ans;
}

Scalar NativePetscVectorOperations::L2Norm( const VectorData &x ) const
{
    PetscReal ans;
    VecNorm( getConstPetscVec( x ), NORM_2, &ans );
    return ans;
}

Scalar NativePetscVectorOperations::maxNorm( const VectorData &x ) const
{
    PetscReal ans;
    VecNorm( getConstPetscVec( x ), NORM_INFINITY, &ans );
    return ans;
}

Scalar NativePetscVectorOperations::dot( const VectorData &x, const VectorData &y ) const
{
    PetscReal ans;
    VecDot( getConstPetscVec( x ), getConstPetscVec( y ), &ans );
    return ans;
}

Scalar NativePetscVectorOperations::localL1Norm( const VectorData &vx ) const
{
    Vec x = getConstPetscVec( vx );

    PetscReal ans = 0;
    PetscErrorCode ierr;
    ierr = ( *x->ops->norm_local )( x, NORM_1, &ans );
    CHKERRQ( ierr );
    return ans;
}

Scalar NativePetscVectorOperations::localL2Norm( const VectorData &vx ) const
{
    Vec x = getConstPetscVec( vx );

    PetscReal ans = 0;
    PetscErrorCode ierr;

    ierr = ( *x->ops->norm_local )( x, NORM_2, &ans );
    CHKERRQ( ierr );
    return ans;
}

Scalar NativePetscVectorOperations::localMaxNorm( const VectorData &vx ) const
{
    Vec x = getConstPetscVec( vx );

    PetscReal ans = 0;
    PetscErrorCode ierr;

    ierr = ( *x->ops->norm_local )( x, NORM_INFINITY, &ans );
    CHKERRQ( ierr );
    return ans;
}

Scalar NativePetscVectorOperations::localDot( const VectorData &vx, const VectorData &vy ) const
{
    Vec x = getConstPetscVec( vx );

    PetscScalar ans;
    PetscErrorCode ierr;

    ierr = ( *x->ops->dot_local )( getConstPetscVec( vx ), getConstPetscVec( vy ), &ans );
    CHKERRQ( ierr );
    return static_cast<PetscReal>( ans );
}

} // namespace AMP::LinearAlgebra
