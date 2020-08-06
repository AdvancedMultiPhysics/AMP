#include "AMP/vectors/operations/VectorOperations.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/VectorData.h"


namespace AMP {
namespace LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
VectorOperations::VectorOperations() : d_VectorData( nullptr ) {}


/****************************************************************
 * min, max, norms, etc.                                         *
 * Note: these routines require communication                    *
 ****************************************************************/
double VectorOperations::min() const
{
    double ans = localMin();
    if ( hasComm() )
        ans = getComm().minReduce( ans );
    return ans;
}
double VectorOperations::max() const
{
    double ans = localMax();
    if ( hasComm() )
        ans = getComm().maxReduce( ans );
    return ans;
}
double VectorOperations::dot( const VectorOperations &x ) const
{
    double ans = localDot( x );
    if ( hasComm() )
        ans = getComm().sumReduce( ans );
    return ans;
}
double VectorOperations::L1Norm() const
{
    double ans = localL1Norm();
    if ( hasComm() )
        ans = getComm().sumReduce( ans );
    return ans;
}
double VectorOperations::maxNorm() const
{
    double ans = localMaxNorm();
    if ( hasComm() )
        ans = getComm().maxReduce( ans );
    return ans;
}
double VectorOperations::L2Norm() const
{
    double ans = localL2Norm();
    if ( hasComm() )
        ans = sqrt( getComm().sumReduce( ans * ans ) );
    return ans;
}
double VectorOperations::minQuotient( const VectorOperations &x, const VectorOperations &y )
{
    double ans = y.localMinQuotient( x );
    if ( y.hasComm() )
        ans = y.getComm().minReduce( ans );
    AMP_INSIST( ans < std::numeric_limits<double>::max(),
                "denominator is the zero vector on an entire process" );
    return ans;
}
double VectorOperations::wrmsNorm( const VectorOperations &x, const VectorOperations &y )
{
    double ans = y.localWrmsNorm( x );
    if ( y.hasComm() ) {
        size_t N1 = y.getVectorData()->getCommunicationList()->numLocalRows();
        size_t N2 = y.getVectorData()->getCommunicationList()->getTotalSize();
        ans       = sqrt( y.getComm().sumReduce( ans * ans * N1 ) / N2 );
    }
    return ans;
}
double VectorOperations::wrmsNormMask( const VectorOperations &x,
                                       const VectorOperations &y,
                                       const VectorOperations &mask )
{
    double ans = y.localWrmsNormMask( x, mask );
    if ( y.hasComm() ) {
        size_t N1 = y.getVectorData()->getCommunicationList()->numLocalRows();
        size_t N2 = y.getVectorData()->getCommunicationList()->getTotalSize();
        ans       = sqrt( y.getComm().sumReduce( ans * ans * N1 ) / N2 );
    }
    return ans;
}


/****************************************************************
 * equals                                                        *
 * Note: these routines require communication                    *
 ****************************************************************/
bool VectorOperations::equals( const VectorOperations &rhs, double tol ) const
{
    bool equal = localEquals( rhs, tol );
    if ( hasComm() )
        equal = getComm().allReduce( equal );
    return equal;
}

double VectorOperations::min( const VectorData &x ) const
{
    double ans = localMin(x);
    if ( hasComm() )
        ans = getComm().minReduce( ans );
    return ans;
}
double VectorOperations::max( const VectorData &x) const
{
    double ans = localMax(x);
    if ( hasComm() )
        ans = getComm().maxReduce( ans );
    return ans;
}
double VectorOperations::dot( const VectorData &x, const VectorData &y ) const
{
    double ans = localDot( x, y );
    if ( hasComm() )
        ans = getComm().sumReduce( ans );
    return ans;
}
double VectorOperations::L1Norm(const VectorData &x) const
{
    double ans = localL1Norm(x);
    if ( hasComm() )
        ans = getComm().sumReduce( ans );
    return ans;
}
double VectorOperations::maxNorm(const VectorData &x) const
{
    double ans = localMaxNorm(x);
    if ( hasComm() )
        ans = getComm().maxReduce( ans );
    return ans;
}
double VectorOperations::L2Norm(const VectorData &x) const
{
    double ans = localL2Norm(x);
    if ( hasComm() )
        ans = sqrt( getComm().sumReduce( ans * ans ) );
    return ans;
}


} // namespace LinearAlgebra
} // namespace AMP
