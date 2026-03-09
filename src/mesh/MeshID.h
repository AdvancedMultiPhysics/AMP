#ifndef included_AMP_MeshID
#define included_AMP_MeshID

#include <cstdint>
#include <memory>
#include <string>

namespace AMP::Mesh {


//! Enumeration for basic mesh-based quantities
enum class GeomType : uint8_t {
    Vertex      = 0,
    Edge        = 1,
    Face        = 2,
    Cell        = 3,
    Polytope_4D = 4,
    Polytope_5D = 5,
    Nullity     = 0xFF
};


struct MeshElementID;


/**
 * \struct MeshID
 * \brief  A structure used to identify the mesh
 * \details  This structure provides a unique id that can be used to identify a mesh.
 */
struct MeshID {

public:
    // Constructors used to initialize key values
    constexpr MeshID() : data( 0xFFFFFFFFFFFFFFFF ) {}
    constexpr MeshID( unsigned int root, unsigned int local_id )
        : data( ( static_cast<uint64_t>( root ) << 32 ) + local_id )
    {
    }
    constexpr MeshID( uint64_t id ) : data( id ) {}
    constexpr uint64_t getData() const { return data; }
    constexpr int getRoot() const { return data >> 32; }
    constexpr int getLocalID() const { return data & 0xFFFFFFFF; }
    // Hash the id
    constexpr uint64_t getHash() const
    {
        uint64_t d1  = data & 0xFFFF;
        uint64_t d2  = ( data >> 16 ) & 0xFFFF;
        uint64_t d3  = ( data >> 32 ) & 0xFFFF;
        uint64_t d4  = data >> 48;
        uint64_t tmp = ( d1 << 48 ) | ( d3 << 32 ) | ( d2 << 16 ) | d4;
        return 0xF958F86D61D1B36B ^ tmp;
    }
    // Overload key operators
    constexpr bool operator==( const MeshID &rhs ) const { return data == rhs.data; }
    constexpr bool operator!=( const MeshID &rhs ) const { return data != rhs.data; }
    constexpr bool operator>=( const MeshID &rhs ) const { return data >= rhs.data; }
    constexpr bool operator<=( const MeshID &rhs ) const { return data <= rhs.data; }
    constexpr bool operator>( const MeshID &rhs ) const { return data > rhs.data; }
    constexpr bool operator<( const MeshID &rhs ) const { return data < rhs.data; }
    // Check if id is null
    constexpr bool isNull() const { return data == 0xFFFFFFFFFFFFFFFF; }

private:
    // We will store the data as a 64-bit data type
    // We do not want the user to get direct access to the data
    uint64_t data;
};


/**
 * \struct ElementID
 * \brief  A structure used to identify an element within a mesh
 * \details  This structure provides a unique id that can be used to identify
 *     a mesh element within a single unique mesh.  To identify a global unique
 *     element MeshElementID should be used.
 */
struct ElementID {

public:
    // Constructors used to initialize key values
    constexpr ElementID() : data( 0x000000FFFFFFFFFF ) {}
    constexpr explicit ElementID( bool isLocal,
                                  GeomType type_id,
                                  uint32_t local_ID,
                                  uint32_t owner_rank_id )
        : data( 0 )
    {
        // Set the bit for is_local
        uint32_t tmp = 0x00000000;
        if ( isLocal )
            tmp = 0x80000000;
        // Add the owner_rank
        tmp += ( 0x007FFFFF & owner_rank_id ) << 8;
        // Add the type_id
        tmp += static_cast<uint8_t>( type_id );
        // Combine the above data with the local_ID
        data = ( ( (uint64_t) tmp ) << 32 ) + ( (uint64_t) local_ID );
    }
    constexpr explicit ElementID( uint64_t id ) : data( id ) {}
    // Overload key operators
    constexpr bool operator==( const ElementID &rhs ) const
    {
        return ( ( data ^ rhs.data ) << 1 ) == 0;
    }
    constexpr bool operator!=( const ElementID &rhs ) const
    {
        return ( ( data ^ rhs.data ) << 1 ) != 0;
    }
    constexpr bool operator>=( const ElementID &rhs ) const
    {
        return ( data << 1 ) >= ( rhs.data << 1 );
    }
    constexpr bool operator<=( const ElementID &rhs ) const
    {
        return ( data << 1 ) <= ( rhs.data << 1 );
    }
    constexpr bool operator>( const ElementID &rhs ) const
    {
        return ( data << 1 ) > ( rhs.data << 1 );
    }
    constexpr bool operator<( const ElementID &rhs ) const
    {
        return ( data << 1 ) < ( rhs.data << 1 );
    }
    // Access local data
    constexpr uint64_t getData() const { return data; }
    constexpr bool is_local() const { return ( data >> 63 ) != 0; }
    constexpr GeomType type() const { return static_cast<GeomType>( ( data >> 32 ) & 0x00FF ); }
    constexpr unsigned int local_id() const { return data & 0x00000000FFFFFFFF; }
    constexpr unsigned int owner_rank() const { return ( data >> 40 ) & 0x007FFFFF; }
    constexpr bool isNull() const { return data == 0x000000FFFFFFFFFF; }
    constexpr void set_is_local( bool isLocal )
    {
        if ( isLocal )
            data |= 0x8000000000000000;
        else
            data &= 0x7FFFFFFFFFFFFFFF;
    }

private:
    // We will store the data as a 64-bit unsigned integer
    // The first bit refer to if the element is local
    // The next 23 bits refer to the processor id
    // The next 8 bits refer to the element type
    // The next 32 bits refer to the local id
    uint64_t data;
};


/**
 * \struct MeshElementID
 * \brief  A structure used to identify the mesh element
 * \details  This structure provides a unique id that can be used to identify a mesh element.
 */
struct MeshElementID {

public:
    // Constructors used to initialize key values
    constexpr MeshElementID() {}
    constexpr explicit MeshElementID(
        bool isLocal, GeomType type, unsigned int local_ID, unsigned int rank, MeshID mesh_ID )
        : meshId( mesh_ID ), elemId( isLocal, type, local_ID, rank )
    {
    }
    constexpr explicit MeshElementID( MeshID mesh_ID, ElementID elem_id )
        : meshId( mesh_ID ), elemId( elem_id )
    {
    }
    constexpr void resetElemID( ElementID elem_id ) { elemId = elem_id; }
    // Overload key operators
    constexpr bool operator==( const MeshElementID &rhs ) const
    {
        return meshId == rhs.meshId && elemId == rhs.elemId;
    }
    constexpr bool operator!=( const MeshElementID &rhs ) const
    {
        return meshId != rhs.meshId || elemId != rhs.elemId;
    }
    constexpr bool operator>=( const MeshElementID &rhs ) const
    {
        if ( meshId != rhs.meshId )
            return meshId > rhs.meshId;
        return elemId >= rhs.elemId;
    }
    constexpr bool operator>( const MeshElementID &rhs ) const
    {
        if ( meshId != rhs.meshId )
            return meshId > rhs.meshId;
        return elemId > rhs.elemId;
    }
    constexpr bool operator<( const MeshElementID &rhs ) const
    {
        if ( meshId != rhs.meshId )
            return meshId < rhs.meshId;
        return elemId < rhs.elemId;
    }
    constexpr bool operator<=( const MeshElementID &rhs ) const
    {
        if ( meshId != rhs.meshId )
            return meshId < rhs.meshId;
        return elemId <= rhs.elemId;
    }
    // Access local data
    constexpr bool is_local() const { return elemId.is_local(); }
    constexpr GeomType type() const { return elemId.type(); }
    constexpr unsigned int local_id() const { return elemId.local_id(); }
    constexpr unsigned int owner_rank() const { return elemId.owner_rank(); }
    constexpr MeshID meshID() const { return meshId; }
    constexpr ElementID elemID() const { return elemId; }
    constexpr void set_is_local( bool isLocal ) { elemId.set_is_local( isLocal ); }
    constexpr bool isNull() const { return meshId.isNull() || elemId.isNull(); }

private:
    MeshID meshId;
    ElementID elemId;
};


// Stream operators
std::ostream &operator<<( std::ostream &out, GeomType x );
std::ostream &operator<<( std::ostream &out, MeshID x );
std::ostream &operator<<( std::ostream &out, ElementID x );
std::ostream &operator<<( std::ostream &out, MeshElementID x );
std::string to_string( GeomType x );


// Arithmetic operators
GeomType operator+( GeomType, int ) noexcept;
GeomType operator-( GeomType, int ) noexcept;
GeomType operator+( int, GeomType ) noexcept;
GeomType operator-( int, GeomType ) noexcept;


} // namespace AMP::Mesh


#endif
