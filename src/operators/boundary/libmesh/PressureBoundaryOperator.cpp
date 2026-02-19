#include "AMP/operators/boundary/libmesh/PressureBoundaryOperator.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/VectorBuilder.h"

DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/cell_hex8.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/node.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS


namespace AMP::Operator {

PressureBoundaryOperator::PressureBoundaryOperator(
    std::shared_ptr<const OperatorParameters> params )
    : BoundaryOperator( params )
{
    AMP_ASSERT( params->d_db->keyExists( "BoundaryID" ) );
    short int bndId = params->d_db->getScalar<int>( "BoundaryID" );

    // ASSUMPTION: Each boundary face element is associated with an unique volume element. This will
    // be true
    // if the face element is truly on the boundary of the mesh.
    // NOTE: The processor that owns a face element may not own the
    // corresponding volume element.

    AMP::AMP_MPI meshComm = d_Mesh->getComm();
    const int npes        = meshComm.getSize();

    std::vector<std::vector<double>> volElemMap( npes );
    std::vector<std::vector<unsigned int>> sideMap( npes );
    std::vector<std::vector<AMP::Mesh::MeshElementID>> idMap( npes );

    for ( auto &elem : d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 ) ) {
        auto sides = elem.getElements( AMP::Mesh::GeomType::Face );
        for ( size_t s = 0; s < sides.size(); ++s ) {
            if ( sides[s]->isOnBoundary( bndId ) ) {
                auto sideId        = sides[s]->globalID();
                unsigned int owner = sideId.owner_rank();
                auto vertices      = elem.getElements( AMP::Mesh::GeomType::Vertex );
                for ( auto &vertice : vertices ) {
                    auto pt = vertice->coord();
                    for ( auto &elem : pt )
                        volElemMap[owner].push_back( elem );
                    auto nodeId = vertice->globalID();
                    idMap[owner].push_back( nodeId );
                }
                sideMap[owner].push_back( s );
            }
        }
    }

    std::vector<int> sendCnts( npes );
    for ( int i = 0; i < npes; ++i ) {
        sendCnts[i] = sideMap[i].size();
    } // end i

    std::vector<int> recvCnts( npes );
    meshComm.allToAll<int>( 1, &( sendCnts[0] ), &( recvCnts[0] ) );

    std::vector<int> sendDisps( npes );
    std::vector<int> recvDisps( npes );
    sendDisps[0] = 0;
    recvDisps[0] = 0;
    for ( int i = 1; i < npes; ++i ) {
        sendDisps[i] = sendDisps[i - 1] + sendCnts[i - 1];
        recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
    } // end i

    std::vector<double> sendVolElemList;
    std::vector<unsigned int> sendSideList;
    std::vector<AMP::Mesh::MeshElementID> sendIDlist;
    for ( int i = 0; i < npes; ++i ) {
        sendSideList.insert( sendSideList.end(), sideMap[i].begin(), sideMap[i].end() );
        sendIDlist.insert( sendIDlist.end(), idMap[i].begin(), idMap[i].end() );
        sendVolElemList.insert( sendVolElemList.end(), volElemMap[i].begin(), volElemMap[i].end() );
        volElemMap[i].clear();
        sideMap[i].clear();
        idMap[i].clear();
    } // end i
    volElemMap.clear();
    sideMap.clear();
    idMap.clear();

    std::vector<unsigned int> recvSideList( recvDisps[npes - 1] + recvCnts[npes - 1] );

    unsigned int *sendSideListPtr = nullptr;
    unsigned int *recvSideListPtr = nullptr;
    if ( !( sendSideList.empty() ) ) {
        sendSideListPtr = &( sendSideList[0] );
    }
    if ( !( recvSideList.empty() ) ) {
        recvSideListPtr = &( recvSideList[0] );
    }
    meshComm.allToAll<unsigned int>( sendSideListPtr,
                                     &( sendCnts[0] ),
                                     &( sendDisps[0] ),
                                     recvSideListPtr,
                                     &( recvCnts[0] ),
                                     &( recvDisps[0] ),
                                     true );
    sendSideList.clear();

    for ( int i = 0; i < npes; ++i ) {
        sendCnts[i] *= 8;
        sendDisps[i] *= 8;
        recvCnts[i] *= 8;
        recvDisps[i] *= 8;
    } // end i

    std::vector<AMP::Mesh::MeshElementID> recvIDlist( 8 * ( recvSideList.size() ) );

    AMP::Mesh::MeshElementID *sendIDlistPtr = nullptr;
    AMP::Mesh::MeshElementID *recvIDlistPtr = nullptr;
    if ( !( sendIDlist.empty() ) ) {
        sendIDlistPtr = &( sendIDlist[0] );
    }
    if ( !( recvIDlist.empty() ) ) {
        recvIDlistPtr = &( recvIDlist[0] );
    }
    meshComm.allToAll<AMP::Mesh::MeshElementID>( sendIDlistPtr,
                                                 &( sendCnts[0] ),
                                                 &( sendDisps[0] ),
                                                 recvIDlistPtr,
                                                 &( recvCnts[0] ),
                                                 &( recvDisps[0] ),
                                                 true );
    sendIDlist.clear();

    for ( int i = 0; i < npes; ++i ) {
        sendCnts[i] *= 3;
        sendDisps[i] *= 3;
        recvCnts[i] *= 3;
        recvDisps[i] *= 3;
    } // end i

    std::vector<double> recvVolElemList( 24 * ( recvSideList.size() ) );

    double *sendVolElemListPtr = nullptr;
    double *recvVolElemListPtr = nullptr;
    if ( !( sendVolElemList.empty() ) ) {
        sendVolElemListPtr = &( sendVolElemList[0] );
    }
    if ( !( recvVolElemList.empty() ) ) {
        recvVolElemListPtr = &( recvVolElemList[0] );
    }
    meshComm.allToAll<double>( sendVolElemListPtr,
                               &( sendCnts[0] ),
                               &( sendDisps[0] ),
                               recvVolElemListPtr,
                               &( recvCnts[0] ),
                               &( recvDisps[0] ),
                               true );
    sendVolElemList.clear();
    sendCnts.clear();
    recvCnts.clear();
    sendDisps.clear();
    recvDisps.clear();

    auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
    auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );
    auto qruleType   = libMesh::Utility::string_to_enum<libMeshEnums::QuadratureType>( "QGAUSS" );
    auto feType      = std::make_shared<libMesh::FEType>( feTypeOrder, feFamily );
    libMeshEnums::Order qruleOrder = feType->default_quadrature_order();
    std::shared_ptr<libMesh::QBase> qrule(
        ( libMesh::QBase::build( qruleType, 2, qruleOrder ) ).release() );

    AMP_ASSERT( params->d_db->keyExists( "Value" ) );
    const auto val = params->d_db->getScalar<double>( "Value" );

    std::vector<double> pressure( 12 * ( recvSideList.size() ) );

    for ( size_t i = 0; i < recvSideList.size(); ++i ) {
        libMesh::Elem *elem = new libMesh::Hex8;
        for ( int j = 0; j < 8; ++j ) {
            elem->set_node( j ) = new libMesh::Node( recvVolElemList[( 24 * i ) + ( 3 * j ) + 0],
                                                     recvVolElemList[( 24 * i ) + ( 3 * j ) + 1],
                                                     recvVolElemList[( 24 * i ) + ( 3 * j ) + 2],
                                                     j );
        } // end j

        std::shared_ptr<libMesh::FEBase> fe(
            ( libMesh::FEBase::build( 3, ( *feType ) ) ).release() );
        fe->attach_quadrature_rule( qrule.get() );
        fe->reinit( elem, recvSideList[i] );

        AMP_ASSERT( qrule->n_points() == 4 );

        const auto &normals = fe->get_normals();
        AMP_ASSERT( normals.size() == 4 );

        for ( size_t qp = 0; qp < qrule->n_points(); ++qp ) {
            for ( int d = 0; d < 3; ++d ) {
                double tractionVal                    = val * ( normals[qp]( d ) );
                pressure[( 12 * i ) + ( 3 * qp ) + d] = tractionVal;
            } // end d
        }     // end qp

        for ( size_t j = 0; j < elem->n_nodes(); ++j ) {
            delete ( elem->node_ptr( j ) );
            elem->set_node( j ) = nullptr;
        } // end for j
        delete elem;
        elem = nullptr;
    } // end i

    auto tmp_db = std::make_shared<AMP::Database>( "Dummy" );
    AMP_ASSERT( params->d_db->keyExists( "Variable" ) );
    std::string varName = params->d_db->getString( "Variable" );
    tmp_db->putScalar( "Variable", varName );
    AMP_ASSERT( params->d_db->keyExists( "ResidualMode" ) );
    tmp_db->putScalar( "ResidualMode", ( params->d_db->getScalar<bool>( "ResidualMode" ) ) );

    auto tracOpParams              = std::make_shared<TractionBoundaryOperatorParameters>( tmp_db );
    tracOpParams->d_Mesh           = d_Mesh;
    tracOpParams->d_traction       = pressure;
    tracOpParams->d_volumeElements = recvVolElemList;
    tracOpParams->d_sideNumbers    = recvSideList;
    tracOpParams->d_nodeID         = recvIDlist;

    d_tractionOp.reset( new TractionBoundaryOperator( tracOpParams ) );
}
} // namespace AMP::Operator
