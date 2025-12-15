#include "AMP/vectors/operations/VectorOperations.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/vectors/CommunicationList.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/operations/MultiVectorOperations.h"
#include "AMP/vectors/operations/VectorOperationsFactory.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"

#include "ProfilerApp.h"

namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
VectorOperations::VectorOperations() : d_hash( reinterpret_cast<uint64_t>( this ) ) {}

/****************************************************************
 * equals                                                        *
 * Note: these routines require communication                    *
 ****************************************************************/
bool VectorOperations::equals( const VectorData &a, const VectorData &b, const Scalar &tol ) const
{
    PROFILE( "VectorOperations::equals" );

    bool equal = localEquals( a, b, tol );
    auto &comm = b.getComm();
    if ( !comm.isNull() )
        equal = comm.allReduce( equal );
    return equal;
}

Scalar VectorOperations::min( const VectorData &x ) const
{
    PROFILE( "VectorOperations::min" );

    auto ans   = localMin( x );
    auto &comm = x.getComm();
    if ( comm.getSize() > 1 )
        ans = comm.minReduce( ans );
    return ans;
}

Scalar VectorOperations::max( const VectorData &x ) const
{
    PROFILE( "VectorOperations::max" );

    auto ans   = localMax( x );
    auto &comm = x.getComm();
    if ( comm.getSize() > 1 )
        ans = comm.maxReduce( ans );
    return ans;
}

Scalar VectorOperations::sum( const VectorData &x ) const
{
    PROFILE( "VectorOperations::sum" );

    auto ans   = localSum( x );
    auto &comm = x.getComm();
    if ( comm.getSize() > 1 )
        ans = comm.sumReduce( ans );
    return ans;
}

Scalar VectorOperations::mean( const VectorData &x ) const
{
    PROFILE( "VectorOperations::mean" );

    return sum( x ) / Scalar( x.getGlobalSize() );
}

Scalar VectorOperations::dot( const VectorData &x, const VectorData &y ) const
{
    PROFILE( "VectorOperations::dot" );

    auto ans   = localDot( x, y );
    auto &comm = x.getComm();
    if ( comm.getSize() > 1 )
        ans = comm.sumReduce( ans );
    return ans;
}

Scalar VectorOperations::L1Norm( const VectorData &x ) const
{
    PROFILE( "VectorOperations::L1Norm" );

    Scalar ans = localL1Norm( x );
    auto &comm = x.getComm();
    if ( comm.getSize() > 1 )
        ans = comm.sumReduce( ans );
    return ans;
}

Scalar VectorOperations::maxNorm( const VectorData &x ) const
{
    PROFILE( "VectorOperations::maxNorm" );

    Scalar ans = localMaxNorm( x );
    auto &comm = x.getComm();
    if ( comm.getSize() > 1 )
        ans = comm.maxReduce( ans );
    return ans;
}

Scalar VectorOperations::L2Norm( const VectorData &x ) const
{
    PROFILE( "VectorOperations::L2Norm" );

    auto ans   = localL2Norm( x );
    auto &comm = x.getComm();
    if ( comm.getSize() > 1 )
        ans = comm.sumReduce( ans * ans ).sqrt();
    return ans;
}

Scalar VectorOperations::minQuotient( const VectorData &x, const VectorData &y ) const
{
    PROFILE( "VectorOperations::minQuotient" );

    auto ans = localMinQuotient( x, y );
    if ( y.getCommunicationList() )
        ans = y.getComm().minReduce( ans );
    AMP_INSIST( ans < std::numeric_limits<double>::max(),
                "denominator is the zero vector on an entire process" );
    return ans;
}

Scalar VectorOperations::wrmsNorm( const VectorData &x, const VectorData &y ) const
{
    PROFILE( "VectorOperations::wrmsNorm" );

    auto ans = localWrmsNorm( x, y );
    if ( y.getCommunicationList() ) {
        double N1 = y.getCommunicationList()->numLocalRows();
        double N2 = y.getCommunicationList()->getTotalSize();
        auto tmp  = ans * ans * ( N1 / N2 );
        ans       = x.getComm().sumReduce( tmp );
        ans       = ans.sqrt();
    }
    return ans;
}

Scalar VectorOperations::wrmsNormMask( const VectorData &x,
                                       const VectorData &mask,
                                       const VectorData &y ) const
{
    PROFILE( "VectorOperations::wrmsNormMask" );

    auto ans = localWrmsNormMask( x, mask, y );
    if ( y.getCommunicationList() ) {
        double N1 = y.getCommunicationList()->numLocalRows();
        double N2 = y.getCommunicationList()->getTotalSize();
        auto tmp  = ans * ans * ( N1 / N2 );
        ans       = x.getComm().sumReduce( tmp );
        ans       = ans.sqrt();
    }
    return ans;
}


/****************************************************************
 * Get an id                                                     *
 ****************************************************************/
uint64_t VectorOperations::getID() const { return d_hash; }


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void VectorOperations::registerChildObjects( AMP::IO::RestartManager * ) const
{
    AMP_ERROR( "Need to implement registerChildObjects for " + VectorOpName() );
}
void VectorOperations::writeRestart( int64_t ) const
{
    AMP_ERROR( "Need to implement writeRestart for " + VectorOpName() );
}


} // namespace AMP::LinearAlgebra


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::VectorOperations>::DataStoreType(
    std::shared_ptr<const AMP::LinearAlgebra::VectorOperations> data, RestartManager *manager )
    : d_data( data )
{
    d_hash = data->getID();
    d_data->registerChildObjects( manager );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::VectorOperations>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid = createGroup( fid, name );
    writeHDF5( gid, "ClassType", d_data->VectorOpName() );
    d_data->writeRestart( gid );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::LinearAlgebra::VectorOperations>
AMP::IO::RestartManager::DataStoreType<AMP::LinearAlgebra::VectorOperations>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    hid_t gid = openGroup( fid, name );
    auto ops  = AMP::LinearAlgebra::VectorOperationsFactory::create( gid, manager );
    closeGroup( gid );
    return ops;
}
