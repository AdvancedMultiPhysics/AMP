#include "math.h"

#include "AMP/discretization/DOF_Manager.h"
#include "AMP/vectors/operations/VectorOperationsDefault.h"
#include "ArrayVector.h"

namespace AMP {
namespace LinearAlgebra {


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
template<typename T, typename FUN, typename Allocator>
ArrayVector<T, FUN, Allocator>::ArrayVector() : Vector(), d_globalSize( 0 )
{
  d_VectorOps = std::make_shared<VectorOperationsDefault<T>>();
}

template<typename T, typename FUN, typename Allocator>
Vector::shared_ptr ArrayVector<T, FUN, Allocator>::create( const std::vector<size_t> &localSize,
                                                           Variable::shared_ptr var )
{
    std::shared_ptr<ArrayVector<T, FUN, Allocator>> retVal( new ArrayVector<T, FUN, Allocator>() );
    retVal->setVariable( var );
    retVal->resize( localSize );
    const auto N = retVal->getArray().length();
    AMP_MPI comm( AMP_COMM_SELF );
    auto DOFs            = std::make_shared<AMP::Discretization::DOFManager>( N, comm );
    retVal->d_DOFManager = DOFs;
    retVal->setCommunicationList(
        AMP::LinearAlgebra::CommunicationList::createEmpty( DOFs->numLocalDOF(), comm ) );
    retVal->d_globalSize = N;
    retVal->d_comm       = comm;
    return retVal;
}

template<typename T, typename FUN, typename Allocator>
Vector::shared_ptr ArrayVector<T, FUN, Allocator>::create( const std::vector<size_t> &localSize,
                                                           Variable::shared_ptr var,
                                                           AMP_MPI comm )
{
    std::shared_ptr<ArrayVector<T, FUN, Allocator>> retVal( new ArrayVector<T, FUN, Allocator>() );
    retVal->setVariable( var );
    retVal->resize( localSize );
    const auto N         = retVal->getArray().length();
    auto DOFs            = std::make_shared<AMP::Discretization::DOFManager>( N, comm );
    retVal->d_DOFManager = DOFs;
    retVal->setCommunicationList(
        AMP::LinearAlgebra::CommunicationList::createEmpty( DOFs->numLocalDOF(), comm ) );
    retVal->d_comm       = comm;
    retVal->d_globalSize = comm.sumReduce( N );
    return retVal;
}

template<typename T, typename FUN, typename Allocator>
Vector::shared_ptr
ArrayVector<T, FUN, Allocator>::create( Variable::shared_ptr var,
                                        AMP::Discretization::DOFManager::shared_ptr DOFs,
                                        AMP::LinearAlgebra::CommunicationList::shared_ptr commlist )
{
    std::shared_ptr<ArrayVector<T, FUN, Allocator>> retVal( new ArrayVector<T, FUN, Allocator>() );
    retVal->setVariable( var );
    retVal->d_DOFManager = DOFs;
    retVal->setCommunicationList( commlist );
    retVal->d_comm = DOFs->getComm();
    AMP_ERROR( "This routine is not complete" );
    return retVal;
}

template<typename T, typename FUN, typename Allocator>
inline Vector::shared_ptr
ArrayVector<T, FUN, Allocator>::cloneVector( const Variable::shared_ptr name ) const
{
    const auto &array = this->getArray();
    std::vector<size_t> size( array.size().begin(), array.size().end() );
    return create( size, name, this->getComm() );
}

template<typename T, typename FUN, typename Allocator>
void ArrayVector<T, FUN, Allocator>::swapVectors( Vector &rhs )
{
    // get internal arrays
    auto &internalArray = this->getArray();
    auto &otherArray    = dynamic_cast<ArrayVector<T, FUN, Allocator> &>( rhs ).getArray();
    // reset views
    internalArray.swap( otherArray );
}

template<typename T, typename FUN, typename Allocator>
void ArrayVector<T, FUN, Allocator>::aliasVector( Vector & )
{
    AMP_ERROR( "Not implemented" );
}

template<typename T, typename FUN, typename Allocator>
void ArrayVector<T, FUN, Allocator>::resize( const std::vector<size_t> &localDims )
{
    d_array.resize( localDims );
}


template<typename T, typename FUN, typename Allocator>
void ArrayVector<T, FUN, Allocator>::putRawData( const double *buf )
{
    auto &array = this->getArray();
    array.copy( buf );
}

template<typename T, typename FUN, typename Allocator>
void ArrayVector<T, FUN, Allocator>::copyOutRawData( double *buf ) const
{
    auto &array = this->getArray();
    array.copyTo( buf );
}
} // namespace LinearAlgebra
} // namespace AMP
