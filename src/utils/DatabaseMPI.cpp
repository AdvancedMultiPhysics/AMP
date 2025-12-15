#include "AMP/utils/AMP_MPI.I"
#include "AMP/utils/Database.hpp"
#include "AMP/utils/FactoryStrategy.hpp"

#include <cstddef>


using KeyDataFactory = AMP::FactoryStrategy<AMP::KeyData>;


/****************************************************************************
 * pack/unpack functions                                                     *
 ****************************************************************************/
template<>
size_t AMP::packSize<AMP::Database>( const AMP::Database &db )
{
    return db.packSize();
}
template<>
size_t AMP::pack<AMP::Database>( const AMP::Database &db, std::byte *buffer )
{
    return db.pack( buffer );
}
template<>
size_t AMP::unpack<AMP::Database>( AMP::Database &db, const std::byte *buffer )
{
    return db.unpack( buffer );
}

size_t AMP::Database::packSize() const
{
    size_t bytes = 0;
    bytes += AMP::packSize( d_check ); // Pack d_check
    bytes += AMP::packSize( d_name );  // Pack d_name
    bytes += sizeof( d_keys.size() );  // Pack number of keys/data
    for ( size_t i = 0; i < d_keys.size(); i++ ) {
        bytes += AMP::packSize( d_keys[i] );                 // Pack key value
        bytes += AMP::packSize( d_data[i]->getClassType() ); // Pack type of KeyData
        bytes += d_data[i]->packSize();                      // Pack KeyData
    }
    return bytes;
}
size_t AMP::Database::pack( std::byte *p ) const
{
    size_t N = 0;
    N += AMP::pack( d_check, &p[N] );
    N += AMP::pack( d_name, &p[N] );
    N += AMP::pack( d_keys.size(), &p[N] );
    for ( size_t i = 0; i < d_keys.size(); i++ ) {
        N += AMP::pack( d_keys[i], &p[N] );                 // Pack key value
        N += AMP::pack( d_data[i]->getClassType(), &p[N] ); // Pack type of KeyData
        N += d_data[i]->pack( &p[N] );                      // Pack KeyData
    }
    return N;
}
size_t AMP::Database::unpack( const std::byte *p )
{
    size_t N = 0;
    d_hash.clear();
    d_keys.clear();
    d_data.clear();
    size_t N_key = 0;
    N += AMP::unpack( d_check, &p[N] ); // Unpack d_check
    N += AMP::unpack( d_name, &p[N] );  // Unpack d_name
    N += AMP::unpack( N_key, &p[N] );   // Unpack number of keys/data
    d_hash.resize( N_key, 0 );
    d_keys.resize( N_key );
    d_data.resize( N_key );
    d_used.resize( N_key, true );
    for ( size_t i = 0; i < d_keys.size(); i++ ) {
        AMP::typeID type;
        N += AMP::unpack( d_keys[i], &p[N] );            // Unpack key value
        N += AMP::unpack( type, &p[N] );                 // Unpack type of KeyData
        d_hash[i] = hashString( d_keys[i] );             // Hash the key
        d_data[i] = KeyDataFactory::create( type.name ); // Create the KeyData
        N += d_data[i]->unpack( &p[N] );                 // Unpack KeyData
    }
    return N;
}


/****************************************************************************
 * Explicit instantiation                                                    *
 ****************************************************************************/
INSTANTIATE_MPI_BCAST( std::shared_ptr<AMP::Database> );
INSTANTIATE_MPI_SENDRECV( std::shared_ptr<AMP::Database> );
