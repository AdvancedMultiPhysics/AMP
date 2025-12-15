#include <algorithm>
#include <iostream>

static void createNodeSet( AMP::Mesh::MeshManager::Adapter::shared_ptr mesh,
                           const unsigned int bndId,
                           std::vector<PointAndId> &nodeSet )
{
    auto bnd     = mesh->beginOwnedBoundary( bndId );
    auto end_bnd = mesh->endOwnedBoundary( bndId );

    nodeSet.clear();

    for ( ; bnd != end_bnd; ++bnd ) {
        PointAndId tmp;
        tmp.d_id    = bnd->globalID();
        tmp.d_pt[0] = bnd->x();
        tmp.d_pt[1] = bnd->y();
        tmp.d_pt[2] = bnd->z();
        nodeSet.push_back( tmp );
    } // end for bnd

    std::sort( nodeSet.begin(), nodeSet.end() );

    std::cout << "Created contact surface nodeset for " << ( mesh->getMeshName() ) << std::endl;
}

static void createMasterSlaveMap( AMP::Mesh::MeshManager::Adapter::shared_ptr masterMeshAdapter,
                                  AMP::Mesh::MeshManager::Adapter::shared_ptr slaveMeshAdapter,
                                  const unsigned int masterId,
                                  const unsigned int slaveId,
                                  std::vector<unsigned int> &masterContactNodes,
                                  std::vector<unsigned int> &slaveContactNodes,
                                  std::vector<unsigned int> &masterVolumeNodes,
                                  std::vector<unsigned int> &slaveGeomType::CellNodes )
{
    std::vector<PointAndId> masterContactSet;
    std::vector<PointAndId> slaveContactSet;

    createNodeSet( masterMeshAdapter, masterId, masterContactSet );
    createNodeSet( slaveMeshAdapter, slaveId, slaveContactSet );

    std::cout << "Master contact surface has " << ( masterContactSet.size() ) << " nodes."
              << std::endl;
    std::cout << "Slave contact surface has " << ( slaveContactSet.size() ) << " nodes."
              << std::endl;

    const size_t numContactNodes = masterContactSet.size();

    AMP_INSIST( ( masterContactSet.size() == slaveContactSet.size() ),
                "Masterset size differs from Slaveset size" );

    for ( size_t i = 0; i < numContactNodes; i++ ) {
        double mX        = masterContactSet[i].d_pt[0];
        double mY        = masterContactSet[i].d_pt[1];
        double mZ        = masterContactSet[i].d_pt[2];
        double sX        = slaveContactSet[i].d_pt[0];
        double sY        = slaveContactSet[i].d_pt[1];
        double sZ        = slaveContactSet[i].d_pt[2];
        unsigned int mId = masterContactSet[i].d_id;
        unsigned int sId = slaveContactSet[i].d_id;
        auto msg         = AMP::Utilities::stringf(
            "Master point (%g, %g, %g) and Slave point (%g, %g, %g) differ.\n Master ID = %u "
                    "and Slave ID = %u.\n",
            mX,
            mY,
            mZ,
            sX,
            sY,
            sZ,
            mId,
            sId );
        AMP_INSIST( masterContactSet[i].pointEquals( slaveContactSet[i] ), msg );
    }

    masterContactNodes.resize( numContactNodes );
    slaveContactNodes.resize( numContactNodes );

    for ( size_t i = 0; i < numContactNodes; i++ ) {
        masterContactNodes[i] = masterContactSet[i].d_id;
    }

    for ( size_t i = 0; i < numContactNodes; i++ ) {
        slaveContactNodes[i] = slaveContactSet[i].d_id;
    }

    std::cout << "Master GeomType::Cell Node Size = ";
    std::cout << ( masterMeshAdapter->numLocalNodes() ) << std::endl;

    std::cout << "Slave GeomType::Cell Node Size = ";
    std::cout << ( slaveMeshAdapter->numLocalNodes() ) << std::endl;

    masterVolumeNodes.resize( masterMeshAdapter->numLocalNodes() );
    slaveGeomType::CellNodes.resize( slaveMeshAdapter->numLocalNodes() );

    for ( size_t i = 0; i < masterVolumeNodes.size(); i++ ) {
        masterVolumeNodes[i] = i;
    }

    for ( size_t i = 0; i < slaveGeomType::CellNodes.size(); i++ ) {
        slaveGeomType::CellNodes[i] = i;
    }

    for ( size_t i = 0; i < numContactNodes; i++ ) {
        AMP_ASSERT( masterContactNodes[i] < masterVolumeNodes.size() );
        AMP_ASSERT( slaveContactNodes[i] < slaveGeomType::CellNodes.size() );
        masterVolumeNodes[masterContactNodes[i]]       = static_cast<unsigned int>( -1 );
        slaveGeomType::CellNodes[slaveContactNodes[i]] = static_cast<unsigned int>( -1 );
    }

    std::vector<unsigned int> tmpMaster;
    for ( size_t i = 0; i < masterVolumeNodes.size(); i++ ) {
        if ( masterVolumeNodes[i] != ( static_cast<unsigned int>( -1 ) ) ) {
            tmpMaster.push_back( masterVolumeNodes[i] );
        }
    }

    std::vector<unsigned int> tmpSlave;
    for ( size_t i = 0; i < slaveGeomType::CellNodes.size(); i++ ) {
        if ( slaveGeomType::CellNodes[i] != ( static_cast<unsigned int>( -1 ) ) ) {
            tmpSlave.push_back( slaveGeomType::CellNodes[i] );
        }
    }

    masterVolumeNodes        = tmpMaster;
    slaveGeomType::CellNodes = tmpSlave;

    std::cout << "Created Master-Slave map for Tied-Contact." << std::endl;
}
