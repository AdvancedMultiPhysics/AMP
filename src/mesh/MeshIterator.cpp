#include "AMP/mesh/MeshIterator.h"
#include "AMP/IO/HDF.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/mesh/structured/structuredMeshIterator.h"


namespace AMP::Mesh {


/********************************************************
 * Empty Mesh iterator                                   *
 ********************************************************/
class EmptyMeshIterator final : public MeshIteratorBase
{
public:
    std::string className() const override { return "EmptyMeshIterator"; }
    void setPos( size_t pos ) { AMP_INSIST( pos == 0, "Attempting to iterate empty iterator" ); }
    MeshIteratorBase &operator++() override { AMP_ERROR( "Attempting to iterate empty iterator" ); }
    MeshIteratorBase &operator--() override { AMP_ERROR( "Attempting to iterate empty iterator" ); }
    MeshIteratorBase &operator+=( int ) override
    {
        AMP_ERROR( "Attempting to iterate empty iterator" );
    }
    bool operator==( const MeshIteratorBase &rhs ) const override { return rhs.size() == 0; }
    bool operator!=( const MeshIteratorBase &rhs ) const override { return rhs.size() == 0; }
    MeshIterator begin() const override { return MeshIterator::create<EmptyMeshIterator>(); }
    std::unique_ptr<MeshIteratorBase> clone() const override
    {
        return std::make_unique<EmptyMeshIterator>();
    }
    void registerChildObjects( AMP::IO::RestartManager * ) const override
    {
        AMP_ERROR( "Not finished" );
    }
    void writeRestart( int64_t ) const override { AMP_ERROR( "Not finished" ); }
};


/********************************************************
 *  Write/read restart data (MeshIteratorBase)           *
 ********************************************************/
void MeshIteratorBase::registerChildObjects( AMP::IO::RestartManager * ) const {}
void MeshIteratorBase::writeRestart( int64_t fid ) const
{
    IO::writeHDF5( fid, "typeHash", d_typeHash );
    IO::writeHDF5( fid, "iteratorType", d_iteratorType );
    IO::writeHDF5( fid, "size", d_size );
    IO::writeHDF5( fid, "pos", d_pos );
}
MeshIteratorBase::MeshIteratorBase( int64_t fid, AMP::IO::RestartManager * )
{
    IO::readHDF5( fid, "typeHash", d_typeHash );
    IO::readHDF5( fid, "iteratorType", d_iteratorType );
    IO::readHDF5( fid, "size", d_size );
    IO::readHDF5( fid, "pos", d_pos );
}


/********************************************************
 * MeshIterator constructors                             *
 ********************************************************/
static constexpr auto MeshIteratorType = AMP::getTypeID<MeshIterator>().hash;
static_assert( MeshIteratorType != 0 );
MeshIterator::MeshIterator() : it( new EmptyMeshIterator() ) {}
MeshIterator::MeshIterator( MeshIterator &&rhs ) : it( nullptr ) { std::swap( it, rhs.it ); }
MeshIterator::MeshIterator( const MeshIterator &rhs ) : it( rhs.it->clone().release() ) {}
MeshIterator &MeshIterator::operator=( MeshIterator &&rhs )
{
    std::swap( it, rhs.it );
    return *this;
}
MeshIterator &MeshIterator::operator=( const MeshIterator &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    delete it;
    it = rhs.it->clone().release();
    return *this;
}
MeshIteratorBase *MeshIterator::release()
{
    auto ptr = it;
    it       = nullptr;
    return ptr;
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void MeshIterator::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    it->registerChildObjects( manager );
}
void MeshIterator::writeRestart( int64_t fid ) const { it->writeRestart( fid ); }
MeshIterator::MeshIterator( int64_t ) : MeshIterator() { AMP_ERROR( "Not finished" ); }


/********************************************************
 * Increment/Decrement the iterator                      *
 ********************************************************/
MeshIterator MeshIterator::operator++( int )
{
    // Postfix increment (increment and return temporary object)
    auto tmp = *this; // Create a temporary iterator
    it->operator++(); // apply operator
    return tmp;       // return temporary result
}
MeshIterator MeshIterator::operator--( int )
{
    // Postfix decrement (increment and return temporary object)
    auto tmp = *this; // Create a temporary iterator
    it->operator--(); // apply operator
    return tmp;       // return temporary result
}
MeshIterator MeshIterator::operator+( int n ) const
{
    auto tmp = *this;    // Create a temporary iterator
    tmp.operator+=( n ); // Increment temporary d_iterator
    return tmp;          // return temporary d_iterator
}
MeshIterator MeshIterator::operator-( int n ) const
{
    auto tmp = *this;     // Create a temporary iterator
    tmp.operator+=( -n ); // Increment temporary d_iterator
    return tmp;           // return temporary d_iterator
}
MeshIterator MeshIterator::operator+( const MeshIterator &it ) const
{
    return operator+( (int) it.pos() );
}
MeshIterator MeshIterator::operator-( const MeshIterator &it ) const
{
    return operator+( -static_cast<int>( it.pos() ) );
}
MeshIterator &MeshIterator::operator+=( const MeshIterator &it )
{
    return operator+=( (int) it.pos() );
}
MeshIterator &MeshIterator::operator-=( int n ) { return operator+=( -n ); }
MeshIterator &MeshIterator::operator-=( const MeshIterator &it )
{
    return operator+=( -static_cast<int>( it.pos() ) );
}


} // namespace AMP::Mesh


/********************************************************
 *  Restart operations                                   *
 ********************************************************/
template<>
AMP::IO::RestartManager::DataStoreType<AMP::Mesh::MeshIterator>::DataStoreType(
    std::shared_ptr<const AMP::Mesh::MeshIterator> data, RestartManager *manager )
{
    d_hash = data->getID();
    if ( d_hash == AMP::Mesh::MeshIteratorType )
        d_data = std::make_shared<AMP::Mesh::MeshIterator>();
    else
        d_data = data;
    d_data->registerChildObjects( manager );
}
template<>
void AMP::IO::RestartManager::DataStoreType<AMP::Mesh::MeshIterator>::write(
    hid_t fid, const std::string &name ) const
{
    hid_t gid = createGroup( fid, name );
    writeHDF5( gid, "ClassType", d_data->className() );
    d_data->writeRestart( gid );
    closeGroup( gid );
}
template<>
std::shared_ptr<AMP::Mesh::MeshIterator>
AMP::IO::RestartManager::DataStoreType<AMP::Mesh::MeshIterator>::read(
    hid_t fid, const std::string &name, RestartManager *manager ) const
{
    if ( d_hash == AMP::Mesh::MeshIteratorType )
        return std::make_shared<AMP::Mesh::MeshIterator>();
    hid_t gid = openGroup( fid, name );
    std::string type;
    readHDF5( gid, "ClassType", type );
    // Load the object (we will need to replace the if/else with a factory)
    std::unique_ptr<AMP::Mesh::MeshIteratorBase> it;
    if ( type == "structuredMeshIterator" ) {
        it = std::make_unique<AMP::Mesh::structuredMeshIterator>( gid, manager );
    } else {
        AMP_ERROR( "Unknown MeshIterator: " + type );
    }
    closeGroup( gid );
    return std::make_shared<AMP::Mesh::MeshIterator>( std::move( it ) );
}
