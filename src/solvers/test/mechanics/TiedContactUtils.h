
#ifndef included_AMP_TiedContactUtils
#define included_AMP_TiedContactUtils

#include <vector>

class MeshAndNodeId
{
public:
    unsigned int d_meshId;
    unsigned int d_nodeId;

    bool operator==( MeshAndNodeId const &other ) const
    {
        return ( ( d_meshId == other.d_meshId ) && ( d_nodeId == other.d_nodeId ) );
    }

    bool operator<( MeshAndNodeId const &other ) const
    {
        if ( d_meshId == other.d_meshId ) {
            return ( d_nodeId < other.d_nodeId );
        } else {
            return ( d_meshId < other.d_meshId );
        }
        return false;
    }

    bool operator!=( MeshAndNodeId const &other ) const { return ( !( ( *this ) == other ) ); }

    bool operator<=( MeshAndNodeId const &other ) const
    {
        return ( ( ( *this ) == other ) || ( ( *this ) < other ) );
    }

    bool operator>( MeshAndNodeId const &other ) const { return ( !( ( *this ) <= other ) ); }

    bool operator>=( MeshAndNodeId const &other ) const { return ( !( ( *this ) < other ) ); }

    MeshAndNodeId &operator=( MeshAndNodeId const &other )
    {
        if ( this != ( &other ) ) {
            d_meshId = other.d_meshId;
            d_nodeId = other.d_nodeId;
        }
        return ( *this );
    }

    MeshAndNodeId()
    {
        d_meshId = static_cast<unsigned int>( -1 );
        d_nodeId = static_cast<unsigned int>( -1 );
    }

    MeshAndNodeId( MeshAndNodeId const &other )
    {
        d_meshId = other.d_meshId;
        d_nodeId = other.d_nodeId;
    }
};

class PointAndId
{
public:
    double d_pt[3];
    unsigned int d_id;
    constexpr static double const PRECISION = 1.0e-12;

    bool idEquals( PointAndId const &other ) const
    {
        if ( d_id != other.d_id ) {
            return false;
        }
        return true;
    }

    bool pointEquals( PointAndId const &other ) const
    {
        for ( int i = 0; i < 3; i++ ) {
            if ( fabs( d_pt[i] - other.d_pt[i] ) > PRECISION ) {
                return false;
            }
        }
        return true;
    }

    bool operator==( PointAndId const &other ) const
    {
        return ( ( pointEquals( other ) ) && ( idEquals( other ) ) );
    }

    bool operator<( PointAndId const &other ) const
    {
        for ( int i = 0; i < 3; i++ ) {
            if ( ( d_pt[i] - other.d_pt[i] ) > PRECISION ) {
                return false;
            }
            if ( ( other.d_pt[i] - d_pt[i] ) > PRECISION ) {
                return true;
            }
        }
        return ( d_id < other.d_id );
    }

    bool operator!=( PointAndId const &other ) const { return ( !( ( *this ) == other ) ); }

    bool operator<=( PointAndId const &other ) const
    {
        return ( ( ( *this ) == other ) || ( ( *this ) < other ) );
    }

    bool operator>( PointAndId const &other ) const { return ( !( ( *this ) <= other ) ); }

    bool operator>=( PointAndId const &other ) const { return ( !( ( *this ) < other ) ); }

    PointAndId &operator=( PointAndId const &other )
    {
        if ( this != ( &other ) ) {
            d_id = other.d_id;
            for ( int i = 0; i < 3; i++ ) {
                d_pt[i] = other.d_pt[i];
            }
        }
        return ( *this );
    }

    PointAndId()
    {
        d_id = static_cast<unsigned int>( -1 );
        for ( int i = 0; i < 3; i++ ) {
            d_pt[i] = 0.0;
        }
    }

    PointAndId( PointAndId const &other )
    {
        d_id = other.d_id;
        for ( int i = 0; i < 3; i++ ) {
            d_pt[i] = other.d_pt[i];
        }
    }
};

static void createNodeSet( AMP::Mesh::MeshManager::Adapter::shared_ptr mesh,
                           const unsigned int bndId,
                           std::vector<PointAndId> &nodeSet );

static void createMasterSlaveMap( AMP::Mesh::MeshManager::Adapter::shared_ptr masterMeshAdapter,
                                  AMP::Mesh::MeshManager::Adapter::shared_ptr slaveMeshAdapter,
                                  const unsigned int masterId,
                                  const unsigned int slaveId,
                                  std::vector<unsigned int> &masterContactNodes,
                                  std::vector<unsigned int> &slaveContactNodes,
                                  std::vector<unsigned int> &masterVolumeNodes,
                                  std::vector<unsigned int> &slaveGeomType::CellNodes );

#include "TiedContactUtils.hpp"

#endif
