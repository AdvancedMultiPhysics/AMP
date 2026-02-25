#include "AMP/vectors/sundials/ManagedSundialsVector.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/vectors/data/ManagedVectorData.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/operations/ManagedVectorOperations.h"


namespace AMP::LinearAlgebra {


[[maybe_unused]] static inline auto getVectorEngine( std::shared_ptr<VectorData> data )
{
    auto managed = std::dynamic_pointer_cast<ManagedVectorData>( data );
    AMP_ASSERT( managed );
    return managed->getVectorEngine();
}
[[maybe_unused]] static inline auto getVectorEngine( std::shared_ptr<const VectorData> data )
{
    auto managed = std::dynamic_pointer_cast<const ManagedVectorData>( data );
    AMP_ASSERT( managed );
    return managed->getVectorEngine();
}


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
ManagedSundialsVector::ManagedSundialsVector( std::shared_ptr<Vector> vec )
    : Vector(), SundialsVector()
{
    AMP_ASSERT( !std::dynamic_pointer_cast<ManagedVectorData>( vec->getVectorData() ) );
    d_VectorOps  = std::make_shared<ManagedVectorOperations>();
    d_VectorData = std::make_shared<ManagedVectorData>( vec );
    d_DOFManager = vec->getDOFManager();
    setVariable( vec->getVariable() );
    // Create N_Vector
    d_n_vector = (N_Vector) malloc( sizeof *d_n_vector );
    AMP_ASSERT( d_n_vector );
    // Attach the content and the ops fields
    d_n_vector->content = this;
    d_n_vector->ops     = ManagedSundialsVector::createNVectorOps();
}


/************************************************************************
 * Destructor for SundialsVector                                         *
 * Frees the memory allocated for the member N_Vector and its ops field  *
 ************************************************************************/
ManagedSundialsVector::~ManagedSundialsVector()
{
    if ( d_n_vector ) {
        if ( d_n_vector->ops ) {
            free( d_n_vector->ops );
            d_n_vector->ops = nullptr;
        }
        free( d_n_vector );
        d_n_vector = nullptr;
    }
}


/************************************************************************
 * Clone the vector                                                      *
 ************************************************************************/
std::unique_ptr<Vector> ManagedSundialsVector::rawClone() const
{
    auto vec    = getVectorEngine( getVectorData() );
    auto vec2   = vec->clone( "ManagedSundialsVectorClone" );
    auto retVal = std::make_unique<ManagedSundialsVector>( vec2 );
    retVal->setVariable( d_Variable );
    return retVal;
}


/**
 * Creates ops, the structure of vector operations which gets attached to the member N_Vector
 * Functions with no_impl in their names are not implemented at the moment
 */
N_Vector_Ops ManagedSundialsVector::createNVectorOps()
{
    N_Vector_Ops ops;
    ops               = (N_Vector_Ops) malloc( sizeof( struct _generic_N_Vector_Ops ) );
    ops->nvclone      = clone_AMP;
    ops->nvcloneempty = cloneempty_no_impl;
    ops->nvdestroy    = freeVectorComponents_AMP;
    // ops->nvspace         = space_no_impl;
    ops->nvspace           = nullptr;
    ops->nvgetarraypointer = getarraypointer_no_impl;
    ops->nvsetarraypointer = setarraypointer_no_impl;
    ops->nvlinearsum       = linearSum_AMP;
    ops->nvconst           = setToScalar_AMP;
    ops->nvprod            = multiply_AMP;
    ops->nvdiv             = divide_AMP;
    ops->nvscale           = scale_AMP;
    ops->nvabs             = abs_AMP;
    ops->nvinv             = reciprocal_AMP;
    ops->nvaddconst        = addScalar_AMP;
    ops->nvdotprod         = dot_AMP;
    ops->nvmaxnorm         = maxNorm_AMP;
    ops->nvwrmsnorm        = WRMSNorm_AMP;
    ops->nvwrmsnormmask    = WRMSNormMask_AMP;
    ops->nvmin             = min_AMP;
    ops->nvwl2norm         = wl2norm_no_impl;
    ops->nvl1norm          = L1Norm_AMP;
    ops->nvcompare         = compare_no_impl;
    ops->nvinvtest         = invtest_no_impl;
    ops->nvconstrmask      = constrmask_no_impl;
    ops->nvminquotient     = minquotient_AMP;
    return ops;
}


N_Vector ManagedSundialsVector::clone_AMP( N_Vector n_vector )
{
    // Extracts the content filed of n_vector
    auto srcAMPVector      = getAMP( n_vector );
    auto ptr               = srcAMPVector->rawClone().release();
    auto newSundialsVector = dynamic_cast<ManagedSundialsVector *>( ptr );
    AMP_ASSERT( newSundialsVector );
    return newSundialsVector->getNVector();
}

N_Vector ManagedSundialsVector::cloneempty_no_impl( N_Vector )
{
    AMP_ERROR( "nvcloneempty not implemented" );
    return N_Vector();
}

void ManagedSundialsVector::freeVectorComponents_AMP( N_Vector v )
{

    auto ptr = getAMP( v );
    delete ptr.get();
}

realtype *ManagedSundialsVector::getarraypointer_no_impl( N_Vector )
{
    AMP_ERROR( "nvgetarraypointer not implemented" );
    return nullptr;
}

void ManagedSundialsVector::setarraypointer_no_impl( realtype *, N_Vector )
{
    AMP_ERROR( "nvsetarraypointer not implemented" );
}

// Set z = a*x + b*y
void ManagedSundialsVector::linearSum_AMP(
    realtype a, N_Vector x, realtype b, N_Vector y, N_Vector z )
{
    auto px = getAMP( x );
    auto py = getAMP( y );
    auto pz = getAMP( z );
    pz->linearSum( a, *px, b, *py );
}

// Set each entry of z to c
void ManagedSundialsVector::setToScalar_AMP( realtype c, N_Vector z )
{
    auto pz = getAMP( z );
    pz->setToScalar( c );
}

// Set z_i = x_i * y_i
void ManagedSundialsVector::multiply_AMP( N_Vector x, N_Vector y, N_Vector z )
{
    auto px = getAMP( x );
    auto py = getAMP( y );
    auto pz = getAMP( z );
    pz->multiply( *px, *py );
}

// Set z_i = x_i / y_i
void ManagedSundialsVector::divide_AMP( N_Vector x, N_Vector y, N_Vector z )
{
    auto px = getAMP( x );
    auto py = getAMP( y );
    auto pz = getAMP( z );
    pz->divide( *px, *py );
}

// Set z_i = c * x_i
void ManagedSundialsVector::scale_AMP( realtype c, N_Vector x, N_Vector z )
{
    auto px = getAMP( x );
    auto pz = getAMP( z );
    pz->scale( c, *px );
}

/**
 * Set z_i = | x_i |
 */

void ManagedSundialsVector::abs_AMP( N_Vector x, N_Vector z )
{
    auto px = getAMP( x );
    auto pz = getAMP( z );
    pz->abs( *px );
}

// Set z_i = 1.0 / x_i
void ManagedSundialsVector::reciprocal_AMP( N_Vector x, N_Vector z )
{
    auto px = getAMP( x );
    auto pz = getAMP( z );
    pz->reciprocal( *px );
}

// Set z_i = x_i + b
void ManagedSundialsVector::addScalar_AMP( N_Vector x, realtype b, N_Vector z )
{
    auto px              = getAMP( x );
    auto pz              = getAMP( z );
    Vector::shared_ptr t = px->clone();
    t->setToScalar( b );
    pz->add( *px, *t );
}

// Returns the dot product of x and y
realtype ManagedSundialsVector::dot_AMP( N_Vector x, N_Vector y )
{
    auto px = getAMP( x );
    auto py = getAMP( y );
    return static_cast<double>( py->dot( *px ) );
}

// Returns the maximum norm of the input vector
realtype ManagedSundialsVector::maxNorm_AMP( N_Vector x )
{
    auto px = getAMP( x );
    return static_cast<double>( px->maxNorm() );
}

// Returns the minimum entry of the input vector
realtype ManagedSundialsVector::min_AMP( N_Vector x )
{
    auto px = getAMP( x );
    return static_cast<double>( px->min() );
}

// Returns the L1 norm of the input vector
realtype ManagedSundialsVector::L1Norm_AMP( N_Vector x )
{
    auto px = getAMP( x );
    return static_cast<double>( px->L1Norm() );
}

/**
 * This function is supposed to return the weighted RMS-norm of the input vector x, with w a
 * weighting vector.
 * Proper implementation needs to be provided.
 */
realtype ManagedSundialsVector::WRMSNorm_AMP( N_Vector x, N_Vector w )
{
    auto px = getAMP( x );
    auto pw = getAMP( w );
    // is this OK?
    return static_cast<double>( px->wrmsNorm( *px, *pw ) );
}


realtype ManagedSundialsVector::WRMSNormMask_AMP( N_Vector x, N_Vector w, N_Vector id )
{
    auto px = getAMP( x );
    auto pw = getAMP( w );
    auto pm = getAMP( id );
    // is this OK?
    return static_cast<double>( px->wrmsNormMask( *px, *pm, *pw ) );
}


realtype ManagedSundialsVector::wl2norm_no_impl( N_Vector, N_Vector )
{
    AMP_ERROR( "nvwl2norm not implemented" );
    return 0;
}

void ManagedSundialsVector::compare_no_impl( realtype, N_Vector, N_Vector )
{
    AMP_ERROR( "nvcompare not implemented" );
}

booleantype ManagedSundialsVector::invtest_no_impl( N_Vector, N_Vector )
{
    AMP_ERROR( "nvinvtest not implemented" );
    return false;
}

booleantype ManagedSundialsVector::constrmask_no_impl( N_Vector, N_Vector, N_Vector )
{
    AMP_ERROR( "nvconstrmask not implemented" );
    return false;
}

realtype ManagedSundialsVector::minquotient_AMP( N_Vector x, N_Vector w )
{
    auto px = getAMP( x );
    auto pw = getAMP( w );
    // is this OK?
    return static_cast<double>( px->minQuotient( *pw ) );
}

std::string ManagedSundialsVector::type() const
{
    return "Managed SUNDIALS Vector" + d_VectorData->VectorDataName();
}

void ManagedSundialsVector::swapVectors( Vector &other )
{
    d_VectorData->swapData( *other.getVectorData() );
}

} // namespace AMP::LinearAlgebra
