#include "AMP/vectors/MultiVector.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/VectorSelector.h"
#include "AMP/vectors/data/ManagedVectorData.h"
#include "AMP/vectors/data/MultiVectorData.h"
#include "AMP/vectors/operations/MultiVectorOperations.h"

#include "ProfilerApp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>


namespace AMP::LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
MultiVector::MultiVector( const std::string &name, const AMP_MPI &comm ) : Vector()
{
    d_VectorOps  = std::make_shared<MultiVectorOperations>();
    d_VectorData = std::make_shared<MultiVectorData>( comm );
    d_Variable.reset( new MultiVariable( name ) );
}
MultiVector::MultiVector( const std::string &name,
                          const AMP_MPI &comm,
                          const std::vector<Vector::shared_ptr> &vecs )
    : Vector()
{
    d_VectorOps  = std::make_shared<MultiVectorOperations>();
    d_VectorData = std::make_shared<MultiVectorData>( comm );
    d_Variable.reset( new MultiVariable( name ) );
    addVector( vecs );
}
MultiVector::MultiVector( Vector::shared_ptr vec ) : Vector()
{
    AMP_ASSERT( vec );
    AMP_ASSERT( vec->getDataID() != 0 );
    // Add the vector
    d_vVectors.push_back( vec );
    // Create the MultiVariable
    d_Variable = std::make_shared<MultiVariable>( vec->getVariable() );
    // Create the MultiVectorOperations
    d_VectorOps = std::make_shared<MultiVectorOperations>( vec->getVectorOperations() );
    // Create the multiDOFManager
    d_DOFManager = std::make_shared<AMP::Discretization::multiDOFManager>( vec->getDOFManager() );
    // Create the MultiVectorData
    d_VectorData =
        std::make_shared<MultiVectorData>( vec->getVectorData().get(), d_DOFManager.get() );
    auto listener = std::dynamic_pointer_cast<DataChangeListener>( d_VectorData );
    vec->getVectorData()->registerListener( listener );
}


/****************************************************************
 * create/view                                                   *
 ****************************************************************/
std::shared_ptr<MultiVector> MultiVector::create( std::shared_ptr<Variable> variable,
                                                  const AMP_MPI &comm )
{
    return std::make_shared<MultiVector>( variable->getName(), comm );
}
std::shared_ptr<MultiVector> MultiVector::create( const std::string &name, const AMP_MPI &comm )
{
    return std::make_shared<MultiVector>( name, comm );
}
std::shared_ptr<MultiVector> MultiVector::create( std::shared_ptr<Variable> variable,
                                                  const AMP_MPI &comm,
                                                  const std::vector<Vector::shared_ptr> &vecs )
{
    return std::make_shared<MultiVector>( variable->getName(), comm, vecs );
}
std::shared_ptr<MultiVector> MultiVector::create( const std::string &name,
                                                  const AMP_MPI &comm,
                                                  const std::vector<Vector::shared_ptr> &vecs )
{
    return std::make_shared<MultiVector>( name, comm, vecs );
}
std::shared_ptr<const MultiVector>
MultiVector::const_create( std::shared_ptr<Variable> variable,
                           const AMP_MPI &comm,
                           const std::vector<Vector::const_shared_ptr> &vecs )
{
    std::vector<Vector::shared_ptr> vecs2( vecs.size() );
    for ( size_t i = 0; i < vecs.size(); i++ )
        vecs2[i] = std::const_pointer_cast<Vector>( vecs[i] );
    return std::make_shared<MultiVector>( variable->getName(), comm, vecs2 );
}
std::shared_ptr<const MultiVector>
MultiVector::const_create( const std::string &name,
                           const AMP_MPI &comm,
                           const std::vector<Vector::const_shared_ptr> &vecs )
{
    std::vector<Vector::shared_ptr> vecs2( vecs.size() );
    for ( size_t i = 0; i < vecs.size(); i++ )
        vecs2[i] = std::const_pointer_cast<Vector>( vecs[i] );
    return std::make_shared<MultiVector>( name, comm, vecs2 );
}
std::shared_ptr<MultiVector> MultiVector::view( Vector::shared_ptr vec, const AMP_MPI &comm )
{
    // Check to see if this is a multivector
    auto multivec = std::dynamic_pointer_cast<MultiVector>( vec );
    if ( multivec ) {
        AMP_ASSERT( comm.isNull() || comm.compare( vec->getComm() ) != 0 );
        return multivec;
    }
    // Check to see if the managed vector engine is a multivector
    auto managed = std::dynamic_pointer_cast<ManagedVectorData>( vec->getVectorData() );
    if ( managed )
        return std::dynamic_pointer_cast<MultiVector>( managed->getVectorEngine() );
    // If still don't have a multivector, make one
    if ( comm.isNull() || vec->getComm().compare( comm ) != 0 ) {
        // No change in the communicator, we should be able to create a view quickly
        return std::make_shared<MultiVector>( vec );
    } else {
        return create( vec->getName(), comm, { vec } );
    }
}
std::shared_ptr<const MultiVector> MultiVector::constView( Vector::const_shared_ptr vec,
                                                           const AMP_MPI &comm )
{
    return view( std::const_pointer_cast<Vector>( vec ), comm );
}


/****************************************************************
 * Functions to add/remove vectors                               *
 ****************************************************************/
void MultiVector::addVector( Vector::shared_ptr v )
{
    addVectorHelper( v );
    resetVectorData();
    resetVectorOperations();
}
void MultiVector::addVector( std::vector<Vector::shared_ptr> vecs )
{
    for ( auto &v : vecs )
        addVectorHelper( v );
    resetVectorData();
    resetVectorOperations();
}
void MultiVector::resetVectorOperations()
{
    std::vector<std::shared_ptr<VectorOperations>> operations( d_vVectors.size() );
    for ( size_t i = 0; i < d_vVectors.size(); i++ )
        operations[i] = d_vVectors[i]->getVectorOperations();
    AMP_ASSERT( d_VectorOps );
    auto mvOps = std::dynamic_pointer_cast<MultiVectorOperations>( d_VectorOps );
    AMP_ASSERT( mvOps );
    mvOps->resetVectorOperations( operations );
}
void MultiVector::resetVectorData()
{
    // Create a new multiDOFManager for the multivector
    AMP_ASSERT( !getComm().isNull() );
    std::vector<std::shared_ptr<AMP::Discretization::DOFManager>> managers( d_vVectors.size() );
    for ( size_t i = 0; i < d_vVectors.size(); i++ ) {
        AMP_ASSERT( d_vVectors[i] );
        managers[i] = d_vVectors[i]->getDOFManager();
        AMP_INSIST( managers[i],
                    "All vectors must have a DOFManager for MultiVector to work properly" );
        AMP_ASSERT( managers[i]->numGlobalDOF() == d_vVectors[i]->getGlobalSize() );
        AMP_ASSERT( managers[i]->numLocalDOF() == d_vVectors[i]->getLocalSize() );
    }

    auto dofManager =
        std::dynamic_pointer_cast<AMP::Discretization::multiDOFManager>( d_DOFManager );
    if ( dofManager ) {
        dofManager->reset( managers );
    } else {
        // this only occurs when add vector gets called
        d_DOFManager =
            std::make_shared<AMP::Discretization::multiDOFManager>( getComm(), managers );
    }

    auto mvData = std::dynamic_pointer_cast<MultiVectorData>( d_VectorData );
    AMP_ASSERT( mvData );
    std::vector<VectorData *> dataComponents( d_vVectors.size() );
    for ( size_t i = 0; i < d_vVectors.size(); i++ )
        dataComponents[i] = d_vVectors[i]->Vector::getVectorData().get();
    mvData->resetMultiVectorData( d_DOFManager.get(), dataComponents );
}

void MultiVector::addVectorHelper( Vector::shared_ptr vec )
{
    if ( !vec )
        return;
    auto id = vec->getDataID();
    if ( id == 0 ) {
        // We are dealing with a multivector, empty vector, or something special
        if ( vec->getGlobalSize() == 0 )
            return;
        auto multivec = std::dynamic_pointer_cast<MultiVector>( vec );
        if ( !multivec ) {
            auto managed = std::dynamic_pointer_cast<ManagedVectorData>( vec->getVectorData() );
            if ( managed )
                multivec = std::dynamic_pointer_cast<MultiVector>( managed->getVectorEngine() );
        }
        if ( multivec ) {
            for ( size_t i = 0; i != multivec->getNumberOfSubvectors(); i++ )
                addVectorHelper( multivec->getVector( i ) );
        } else {
            AMP_ERROR( "Not finished" );
            d_vVectors.push_back( vec );
            auto data = Vector::getVectorData();
            AMP_ASSERT( data );
            auto listener = std::dynamic_pointer_cast<DataChangeListener>( data );
            vec->getVectorData()->registerListener( listener );
        }
    } else {
        // We are dealing with a single vector, check if it is already added in some form
        int index = -1;
        for ( size_t j = 0; j < d_vVectors.size(); j++ ) {
            if ( id == d_vVectors[j]->getDataID() )
                index = static_cast<int>( j );
        }
        if ( index == -1 ) {
            // Add the vector
            d_vVectors.push_back( vec );
            auto data = Vector::getVectorData();
            AMP_ASSERT( data );
            auto listener = std::dynamic_pointer_cast<DataChangeListener>( data );
            vec->getVectorData()->registerListener( listener );
        } else {
            // the vector exists, which vector (or both) do we keep?
            auto dof1 = vec->getDOFManager();
            auto dof2 = d_vVectors[index]->getDOFManager();
            if ( dof1 == dof2 ) {
                // We are dealing with the same vector, no need to do anything
            } else {
                AMP_ERROR( "Not finished" );
            }
        }
    }
    // Append the variable if we have a multivariable
    auto multiVar = std::dynamic_pointer_cast<MultiVariable>( d_Variable );
    AMP_ASSERT( multiVar );
    multiVar->add( vec->getVariable() );
}
void MultiVector::eraseVector( Vector::shared_ptr ) { AMP_ERROR( "Needs to be fixed" ); }
void MultiVector::replaceSubVector( Vector::shared_ptr oldVec, Vector::shared_ptr newVec )
{
    AMP_ASSERT( oldVec );
    AMP_ASSERT( newVec );
    AMP_INSIST( oldVec->getDOFManager() == newVec->getDOFManager(),
                "oldVec and newVec must chare the same DOFManager" );
    int pos = -1;
    for ( size_t i = 0; i < d_vVectors.size(); ++i ) {
        if ( d_vVectors[i] == oldVec ) {
            pos = i;
            break;
        }
    } // end for i
    AMP_INSIST( pos != -1, "oldVec was not found" );
    if ( pos >= 0 )
        d_vVectors[pos] = newVec;
    reset();
}


/****************************************************************
 * Select into the vector                                        *
 ****************************************************************/
Vector::shared_ptr MultiVector::selectInto( const VectorSelector &s )
{
    // Check if we are dealing with a component selector
    if ( dynamic_cast<const VS_Components *>( &s ) )
        return s.subset( shared_from_this() );
    if ( dynamic_cast<const MultiSelector *>( &s ) )
        return s.subset( shared_from_this() );
    // Check if this vector matches (only need to deal with select by name for now)
    auto s_name = dynamic_cast<const VS_ByVariableName *>( &s );
    if ( s_name || dynamic_cast<const VS_MultiVariable *>( &s ) ) {
        auto retVec = Vector::selectInto( s );
        if ( retVec )
            return retVec;
    }
    // Subset each vector
    std::vector<Vector::shared_ptr> subvectors;
    for ( size_t i = 0; i != d_vVectors.size(); i++ ) {
        // Subset the individual vector
        auto retVec2 = d_vVectors[i]->selectInto( s );
        if ( retVec2 ) {
            if ( retVec2->getGlobalSize() > 0 )
                subvectors.push_back( retVec2 );
        }
    }
    // Check if we have an empty vector or single vector
    AMP_MPI comm = s.communicator( *this );
    bool test    = subvectors.empty();
    if ( subvectors.size() == 1 )
        test = comm.compare( subvectors[0]->getComm() ) != 0;
    if ( comm.allReduce( test ) )
        return subvectors.empty() ? nullptr : subvectors[0];
    // Construct the new multivector
    std::string vecName = getName();
    if ( s_name )
        vecName = s_name->getName();
    // Add the subsets to the multivector
    auto retVec = MultiVector::create( vecName, comm, subvectors );
    if ( retVec->getGlobalSize() == 0 )
        retVec.reset();
    return retVec;
}
Vector::const_shared_ptr MultiVector::selectInto( const VectorSelector &s ) const
{
    return const_cast<MultiVector *>( this )->selectInto( s );
}


/****************************************************************
 * Other functions                                               *
 ****************************************************************/
bool MultiVector::containsPointer( const Vector::shared_ptr p ) const
{
    for ( size_t i = 0; i != d_vVectors.size(); i++ ) {
        if ( d_vVectors[i].get() == p.get() ) {
            return true;
        }
    }
    return false;
}


/****************************************************************
 * Get vectors                                                   *
 ****************************************************************/
Vector::shared_ptr MultiVector::getVector( size_t i ) { return d_vVectors[i]; }
Vector::const_shared_ptr MultiVector::getVector( size_t i ) const { return d_vVectors[i]; }
std::vector<std::shared_ptr<const Vector>> MultiVector::getVecs() const
{
    std::vector<std::shared_ptr<const Vector>> vecs( d_vVectors.size() );
    for ( size_t i = 0; i < d_vVectors.size(); i++ )
        vecs[i] = d_vVectors[i];
    return vecs;
}
size_t MultiVector::getNumberOfSubvectors() const { return d_vVectors.size(); }


/****************************************************************
 * Misc functions                                                *
 ****************************************************************/
void MultiVector::reset()
{
    for ( auto &vec : d_vVectors )
        vec->reset();
    // reset multivector dof manager
    resetVectorData();
    resetVectorOperations();
}

void MultiVector::swapVectors( Vector &other )
{
    for ( size_t i = 0; i != d_vVectors.size(); i++ )
        d_vVectors[i]->swapVectors( getVector( other, i ) );
}
std::unique_ptr<Vector> MultiVector::rawClone() const
{
    std::unique_ptr<MultiVector> retVec( new MultiVector() );
    retVec->d_vVectors.resize( d_vVectors.size() );
    for ( size_t i = 0; i != d_vVectors.size(); i++ )
        retVec->d_vVectors[i] = d_vVectors[i]->clone();
    retVec->d_units      = d_units;
    retVec->d_Variable   = d_Variable->clone();
    retVec->d_DOFManager = d_DOFManager;
    retVec->d_VectorData = std::make_shared<MultiVectorData>( getComm() );
    retVec->resetVectorData();
    retVec->d_VectorOps = d_VectorOps->cloneOperations();
    return retVec;
}

std::string MultiVector::type() const { return "MultiVector"; }

MultiVector::~MultiVector() = default;

const Vector::shared_ptr &MultiVector::getVector( const Vector &rhs, size_t which ) const
{
    auto x = dynamic_cast<const MultiVector *>( &rhs );
    AMP_ASSERT( x );
    AMP_ASSERT( which < x->d_vVectors.size() );
    return x->d_vVectors[which];
}

Vector::shared_ptr &MultiVector::getVector( Vector &rhs, size_t which ) const
{
    auto x = dynamic_cast<MultiVector *>( &rhs );
    AMP_ASSERT( x );
    AMP_ASSERT( which < x->d_vVectors.size() );
    return x->d_vVectors[which];
}


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
void MultiVector::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    Vector::registerChildObjects( manager );
    for ( auto vec : d_vVectors )
        manager->registerObject( vec );
}
void MultiVector::writeRestart( int64_t fid ) const
{
    Vector::writeRestart( fid );
    std::vector<uint64_t> vecs;
    for ( auto vec2 : d_vVectors )
        vecs.push_back( vec2->getID() );
    IO::writeHDF5( fid, "vecs", vecs );
}
MultiVector::MultiVector( int64_t fid, AMP::IO::RestartManager *manager ) : Vector( fid, manager )
{
    std::vector<uint64_t> vecs;
    IO::readHDF5( fid, "vecs", vecs );
    d_vVectors.resize( vecs.size() );
    for ( size_t i = 0; i < vecs.size(); i++ )
        d_vVectors[i] = manager->getData<AMP::LinearAlgebra::Vector>( vecs[i] );
}


} // namespace AMP::LinearAlgebra
