#include "AMP/mesh/MeshIterator.h"
#include "AMP/IO/HDF.h"
#include "AMP/IO/RestartManager.h"
#include "AMP/mesh/structured/structuredMeshIterator.h"


namespace AMP::Mesh {


// unused global variable to prevent compiler warning
static MeshElement nullElement;


/********************************************************
 * Constructors                                          *
 ********************************************************/
static constexpr auto MeshIteratorType = AMP::getTypeID<MeshIterator>().hash;
static_assert( MeshIteratorType != 0 );
MeshIterator::MeshIterator()
    : d_iterator( nullptr ),
      d_typeHash( MeshIteratorType ),
      d_iteratorType( Type::RandomAccess ),
      d_size( 0 ),
      d_pos( 0 ),
      d_element( nullptr )
{
}
MeshIterator::MeshIterator( MeshIterator &&rhs )
    : d_iterator( nullptr ),
      d_typeHash( MeshIteratorType ),
      d_iteratorType( rhs.d_iteratorType ),
      d_size( 0 ),
      d_pos( 0 ),
      d_element( nullptr )
{
    if ( rhs.d_iterator == nullptr && rhs.d_typeHash == MeshIteratorType ) {
        d_iterator = nullptr;
    } else if ( rhs.d_typeHash != MeshIteratorType ) {
        d_iterator = rhs.clone();
    } else {
        d_iterator     = rhs.d_iterator;
        rhs.d_iterator = nullptr;
    }
}
MeshIterator::MeshIterator( const MeshIterator &rhs )
    : d_iterator( nullptr ),
      d_typeHash( MeshIteratorType ),
      d_iteratorType( rhs.d_iteratorType ),
      d_size( 0 ),
      d_pos( 0 ),
      d_element( nullptr )
{
    if ( rhs.d_iterator == nullptr && rhs.d_typeHash == MeshIteratorType ) {
        d_iterator = nullptr;
    } else if ( rhs.d_typeHash != MeshIteratorType ) {
        d_iterator = rhs.clone();
    } else {
        d_iterator = rhs.d_iterator->clone();
    }
}
MeshIterator &MeshIterator::operator=( MeshIterator &&rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    if ( d_iterator != nullptr ) {
        // Delete the existing element
        delete d_iterator;
        d_iterator = nullptr;
    }
    d_typeHash     = MeshIteratorType;
    d_iteratorType = rhs.d_iteratorType;
    d_size         = 0;
    d_pos          = 0;
    d_element      = nullptr;
    if ( rhs.d_iterator == nullptr && rhs.d_typeHash == MeshIteratorType ) {
        d_iterator = nullptr;
    } else if ( rhs.d_typeHash != MeshIteratorType ) {
        d_iterator = rhs.clone();
    } else {
        d_iterator     = rhs.d_iterator;
        rhs.d_iterator = nullptr;
    }
    return *this;
}
MeshIterator &MeshIterator::operator=( const MeshIterator &rhs )
{
    if ( this == &rhs ) // protect against invalid self-assignment
        return *this;
    if ( d_iterator != nullptr ) {
        // Delete the existing element
        delete d_iterator;
        d_iterator = nullptr;
    }
    d_typeHash     = MeshIteratorType;
    d_iteratorType = rhs.d_iteratorType;
    d_size         = 0;
    d_pos          = 0;
    d_element      = nullptr;
    if ( rhs.d_iterator == nullptr && rhs.d_typeHash == MeshIteratorType ) {
        d_iterator = nullptr;
    } else if ( rhs.d_typeHash != MeshIteratorType ) {
        d_iterator = rhs.clone();
    } else {
        d_iterator = rhs.d_iterator->clone();
    }
    return *this;
}
MeshIterator::MeshIterator( MeshIterator *rhs )
    : d_iterator( nullptr ),
      d_typeHash( MeshIteratorType ),
      d_iteratorType( Type::RandomAccess ),
      d_size( 0 ),
      d_pos( 0 ),
      d_element( nullptr )
{
    if ( rhs->d_iterator ) {
        std::swap( d_iterator, rhs->d_iterator );
        delete rhs;
    } else {
        d_iterator = rhs;
    }
    d_iteratorType = d_iterator->d_iteratorType;
}


/********************************************************
 * Destructor                                            *
 ********************************************************/
MeshIterator::~MeshIterator()
{
    if ( d_iterator != nullptr )
        delete d_iterator;
    d_iterator = nullptr;
}


/********************************************************
 * Get the underlying class name                         *
 ********************************************************/
std::string MeshIterator::className() const
{
    if ( d_iterator == nullptr )
        return "MeshIterator";
    auto name = d_iterator->className();
    AMP_DEBUG_ASSERT( name != "MeshIterator" );
    return name;
}


/********************************************************
 * Clone the iterator                                    *
 ********************************************************/
MeshIterator *MeshIterator::clone() const
{
    if ( d_iterator )
        return d_iterator->clone();
    return new MeshIterator();
}


/********************************************************
 * Return the iterator type                              *
 ********************************************************/
MeshIterator::Type MeshIterator::type() const
{
    if ( d_iterator == nullptr )
        return d_iteratorType;
    return d_iterator->d_iteratorType;
}


/********************************************************
 * Return the begin or end d_iterator                    *
 ********************************************************/
MeshIterator MeshIterator::begin() const
{
    if ( d_iterator == nullptr )
        return MeshIterator();
    return d_iterator->begin();
}
MeshIterator MeshIterator::end() const
{
    if ( d_iterator == nullptr )
        return MeshIterator();
    return d_iterator->end();
}


/********************************************************
 * Iterator comparisons                                  *
 ********************************************************/
bool MeshIterator::operator==( const MeshIterator &rhs ) const
{
    if ( this->size() == 0 && rhs.size() == 0 )
        return true;
    if ( this->size() != rhs.size() || this->position() != rhs.position() )
        return false;
    if ( d_iterator == nullptr )
        return rhs.d_iterator == nullptr;
    return d_iterator->operator==( rhs );
}
bool MeshIterator::operator!=( const MeshIterator &rhs ) const
{
    if ( this->size() == 0 && rhs.size() == 0 )
        return false;
    if ( this->size() != rhs.size() || this->position() != rhs.position() )
        return true;
    if ( d_iterator == nullptr )
        return rhs.d_iterator != nullptr;
    return d_iterator->operator!=( rhs );
}

/****************************************************************
 *Get a unique hash id for the iterator                          *
 ****************************************************************/
uint64_t MeshIterator::getID() const
{
    if ( empty() )
        return AMP::Mesh::MeshIteratorType;
    return reinterpret_cast<uint64_t>( rawIterator() );
}


/****************************************************************
 * Write/Read restart data                                       *
 ****************************************************************/
void MeshIterator::registerChildObjects( AMP::IO::RestartManager *manager ) const
{
    if ( d_iterator )
        d_iterator->registerChildObjects( manager );
}
void MeshIterator::writeRestart( int64_t fid ) const
{
    if ( d_iterator ) {
        d_iterator->writeRestart( fid );
    } else {
        IO::writeHDF5( fid, "typeHash", d_typeHash );
        IO::writeHDF5( fid, "iteratorType", d_iteratorType );
        IO::writeHDF5( fid, "size", d_size );
        IO::writeHDF5( fid, "pos", d_pos );
    }
}
MeshIterator::MeshIterator( int64_t fid ) : MeshIterator()
{
    IO::readHDF5( fid, "typeHash", d_typeHash );
    IO::readHDF5( fid, "iteratorType", d_iteratorType );
    IO::readHDF5( fid, "size", d_size );
    IO::readHDF5( fid, "pos", d_pos );
}

/********************************************************
 * Increment/Decrement the iterator                      *
 ********************************************************/
MeshIterator &MeshIterator::operator++()
{
    AMP_DEBUG_ASSERT( d_iterator );
    return d_iterator->operator++();
}
MeshIterator &MeshIterator::operator--()
{
    AMP_DEBUG_ASSERT( d_iterator );
    return d_iterator->operator--();
}
MeshIterator MeshIterator::operator++( int )
{
    // Postfix increment (increment and return temporary object)
    auto tmp = clone(); // Create a temporary variable
    this->operator++(); // apply operator
    return tmp;         // return temporary result
}
MeshIterator MeshIterator::operator--( int )
{
    // Postfix decrement (increment and return temporary object)
    auto tmp = clone(); // Create a temporary variable
    --( *this );        // apply operator
    return tmp;         // return temporary result
}


/********************************************************
 * Random access iterators                               *
 ********************************************************/
MeshIterator &MeshIterator::operator+=( int n )
{
    if ( d_iterator != nullptr )
        return d_iterator->operator+=( n );
    if ( n >= 0 ) {
        for ( int i = 0; i < n; i++ ) {
            this->operator++();
        } // increment d_iterator
    } else {
        for ( int i = 0; i < -n; i++ ) {
            this->operator--();
        } // decrement d_iterator
    }
    return *this;
}
MeshIterator MeshIterator::operator+( int n ) const
{
    auto tmp = clone();   // Create a temporary iterator
    tmp->operator+=( n ); // Increment temporary d_iterator
    return tmp;           // return temporary d_iterator
}
MeshIterator MeshIterator::operator-( int n ) const
{
    auto tmp = clone();    // Create a temporary iterator
    tmp->operator+=( -n ); // Increment temporary d_iterator
    return tmp;            // return temporary d_iterator
}
MeshIterator MeshIterator::operator+( const MeshIterator &it ) const
{
    return operator+( (int) it.position() );
}
MeshIterator MeshIterator::operator-( const MeshIterator &it ) const
{
    return this->operator+( -static_cast<int>( it.position() ) );
}
MeshIterator &MeshIterator::operator+=( const MeshIterator &it )
{
    if ( d_iterator != nullptr )
        return d_iterator->operator+=( (int) it.position() );
    return this->operator+=( (int) it.position() );
}
MeshIterator &MeshIterator::operator-=( int n )
{
    if ( d_iterator != nullptr )
        return d_iterator->operator-=( n );
    return this->operator+=( -n );
}
MeshIterator &MeshIterator::operator-=( const MeshIterator &it )
{
    if ( d_iterator != nullptr )
        return d_iterator->operator-=( (int) it.position() );
    return this->operator+=( -static_cast<int>( it.position() ) );
}


/********************************************************
 * Functions for de-referencing the d_iterator           *
 ********************************************************/
MeshElement &MeshIterator::operator[]( int i )
{
    if ( d_iterator != nullptr )
        return d_iterator->operator[]( i );
    AMP_ERROR( "Dereferencing d_iterator with offset is not supported by default" );
    return this->operator*(); // This line never executes and would return the wrong object
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
    std::shared_ptr<AMP::Mesh::MeshIterator> it;
    if ( type == "structuredMeshIterator" ) {
        it = std::make_shared<AMP::Mesh::structuredMeshIterator>( gid, manager );
    } else {
        AMP_ERROR( "Unknown MeshIterator: " + type );
    }
    closeGroup( gid );
    return it;
}
