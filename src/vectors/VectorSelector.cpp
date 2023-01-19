#include "AMP/vectors/VectorSelector.h"
#include "AMP/vectors/CommVariable.h"
#include "AMP/vectors/MeshVariable.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/StridedVariable.h"
#include "AMP/vectors/SubsetVariable.h"


namespace AMP::LinearAlgebra {


/********************************************************
 * VectorSelector                                        *
 ********************************************************/
VectorSelector::~VectorSelector() = default;
bool VectorSelector::isSelected( const Vector & ) const { return true; }
AMP_MPI VectorSelector::communicator( const Vector &p ) const { return p.getComm(); }
std::shared_ptr<Vector> VectorSelector::subset( std::shared_ptr<Vector> p ) const { return p; }
std::shared_ptr<const Vector> VectorSelector::subset( std::shared_ptr<const Vector> p ) const
{
    return p;
}


/********************************************************
 * VS_ByVariableName                                     *
 ********************************************************/
VS_ByVariableName::VS_ByVariableName( std::string name ) : d_VecName( std::move( name ) ) {}
bool VS_ByVariableName::isSelected( const Vector &v ) const { return v.getName() == d_VecName; }
std::shared_ptr<Vector> VS_ByVariableName::subset( std::shared_ptr<Vector> vec ) const
{
    auto var = vec->getVariable();
    if ( var ) {
        if ( var->getName() == d_VecName )
            return vec;
    }
    return std::shared_ptr<Vector>();
}
std::shared_ptr<const Vector> VS_ByVariableName::subset( std::shared_ptr<const Vector> vec ) const
{
    auto var = vec->getVariable();
    if ( var ) {
        if ( var->getName() == d_VecName )
            return vec;
    }
    return std::shared_ptr<const Vector>();
}


/********************************************************
 * VS_Stride                                             *
 ********************************************************/
VS_Stride::VS_Stride( size_t a, size_t b ) : d_Offset( a ), d_Stride( b ) {}
bool VS_Stride::isSelected( const Vector & ) const { return true; }
std::shared_ptr<Vector> VS_Stride::subset( std::shared_ptr<Vector> p ) const
{
    auto variable = std::make_shared<StridedVariable>( p->getName(), d_Offset, d_Stride );
    auto vector   = SubsetVariable::view( p, variable );
    return vector;
}
std::shared_ptr<const Vector> VS_Stride::subset( std::shared_ptr<const Vector> p ) const
{
    auto variable = std::make_shared<StridedVariable>( p->getName(), d_Offset, d_Stride );
    auto vector   = SubsetVariable::view( p, variable );
    return vector;
}


/********************************************************
 * VS_Stride                                             *
 ********************************************************/
VS_Components::VS_Components( size_t index ) : d_index{ index } {}
VS_Components::VS_Components( std::vector<size_t> index ) : d_index( std::move( index ) )
{
    std::sort( d_index.begin(), d_index.end() );
}
bool VS_Components::isSelected( const Vector &vec ) const
{
    size_t N = vec.getNumberOfComponents();
    for ( auto index : d_index ) {
        if ( index < N )
            return true;
    }
    return false;
}
std::shared_ptr<Vector> VS_Components::subset( std::shared_ptr<Vector> p ) const
{
    if ( d_index.empty() )
        return nullptr;
    size_t N = p->getNumberOfComponents();
    if ( N == 1 ) {
        if ( d_index[0] == 0 )
            return p;
    } else if ( std::dynamic_pointer_cast<MultiVector>( p ) ) {
        auto multivec = std::dynamic_pointer_cast<MultiVector>( p );
        std::vector<std::shared_ptr<Vector>> vecs;
        auto index = d_index;
#if 1
        std::vector<size_t> nc( N );
        size_t i = 0;
        for ( auto vec : *multivec ) {
            nc[i] = vec->getNumberOfComponents();
            ++i;
        }

        for ( i = 1u; i < N; ++i )
            nc[i] += nc[i - 1];

        for ( auto &ic : index ) {
            for ( i = 0u; i < N; ++i ) {
                if ( nc[i] > ic ) {
                    auto cv = multivec->getVector( i );
                    AMP_ASSERT( cv );
                    auto sc   = ( i > 0 ) ? ic - nc[i - 1] : ic;
                    auto vec2 = cv->selectInto( VS_Components( { sc } ) );
                    if ( vec2 )
                        vecs.push_back( vec2 );
                    break;
                }
            }
        }
#else
        for ( auto vec : *multivec ) {
            auto N2   = vec->getNumberOfComponents();
            auto vec2 = vec->selectInto( VS_Components( index ) );
            if ( vec2 )
                vecs.push_back( vec );
            std::vector<size_t> index2;
            for ( auto &i : index ) {
                if ( i >= N2 )
                    index2.push_back( i - N2 );
            }
            std::swap( index, index2 );
        }
#endif
        return MultiVector::create( p->getName(), p->getComm(), vecs );
    } else {
        AMP_ERROR( "Not finished: " + p->type() );
    }
    return nullptr;
}
std::shared_ptr<const Vector> VS_Components::subset( std::shared_ptr<const Vector> p ) const
{
    return subset( std::const_pointer_cast<Vector>( p ) );
}


/********************************************************
 * VS_Comm                                               *
 ********************************************************/
VS_Comm::VS_Comm( const AMP_MPI &comm )
{
    AMP_ASSERT( !comm.isNull() );
    d_comm = comm;
}
bool VS_Comm::isSelected( const Vector & ) const { return true; }
AMP_MPI VS_Comm::communicator( const Vector &p ) const
{
    return AMP_MPI::intersect( d_comm, p.getComm() );
}
std::shared_ptr<Vector> VS_Comm::subset( std::shared_ptr<Vector> p ) const
{
    auto variable = std::make_shared<CommVariable>( p->getName(), communicator( *p ) );
    auto vector   = SubsetVariable::view( p, variable );
    return vector;
}
std::shared_ptr<const Vector> VS_Comm::subset( std::shared_ptr<const Vector> p ) const
{
    auto variable = std::make_shared<CommVariable>( p->getName(), communicator( *p ) );
    auto vector   = SubsetVariable::view( p, variable );
    return vector;
}


/********************************************************
 * VS_Mesh                                               *
 ********************************************************/
VS_Mesh::VS_Mesh( std::shared_ptr<AMP::Mesh::Mesh> mesh, bool useMeshComm )
    : d_useMeshComm( useMeshComm ), d_mesh( mesh )
{
}
bool VS_Mesh::isSelected( const Vector & ) const { return true; }
AMP_MPI VS_Mesh::communicator( const Vector &p ) const
{
    if ( d_useMeshComm ) {
        if ( d_mesh )
            return AMP_MPI::intersect( p.getComm(), d_mesh->getComm() );
        return AMP_MPI( AMP_COMM_WORLD );
    }
    return p.getComm();
}
std::shared_ptr<Vector> VS_Mesh::subset( std::shared_ptr<Vector> p ) const
{
    if ( !d_mesh )
        return std::shared_ptr<Vector>();
    auto variable = std::make_shared<MeshVariable>( p->getName(), d_mesh, d_useMeshComm );
    auto vector   = SubsetVariable::view( p, variable );
    return vector;
}
std::shared_ptr<const Vector> VS_Mesh::subset( std::shared_ptr<const Vector> p ) const
{
    if ( !d_mesh )
        return std::shared_ptr<Vector>();
    auto variable = std::make_shared<MeshVariable>( p->getName(), d_mesh, d_useMeshComm );
    auto vector   = SubsetVariable::view( p, variable );
    return vector;
}


/********************************************************
 * VS_MeshIterator                                       *
 ********************************************************/
VS_MeshIterator::VS_MeshIterator( const AMP::Mesh::MeshIterator &iterator,
                                  const AMP::AMP_MPI &comm )
    : d_comm( comm ), d_iterator( iterator )
{
}
bool VS_MeshIterator::isSelected( const Vector & ) const { return true; }
std::shared_ptr<Vector> VS_MeshIterator::subset( std::shared_ptr<Vector> p ) const
{
    auto variable = std::make_shared<MeshIteratorVariable>( p->getName(), d_iterator, d_comm );
    auto vector   = SubsetVariable::view( p, variable );
    return vector;
}
std::shared_ptr<const Vector> VS_MeshIterator::subset( std::shared_ptr<const Vector> p ) const
{
    auto variable = std::make_shared<MeshIteratorVariable>( p->getName(), d_iterator, d_comm );
    auto vector   = SubsetVariable::view( p, variable );
    return vector;
}


} // namespace AMP::LinearAlgebra
