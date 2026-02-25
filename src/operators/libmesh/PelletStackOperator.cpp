#include "AMP/operators/libmesh/PelletStackOperator.h"
#include "AMP/operators/map/NodeToNodeMap.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorSelector.h"


namespace AMP::Operator {

PelletStackOperator::PelletStackOperator(
    std::shared_ptr<const PelletStackOperatorParameters> params )
    : Operator( params )
{
    d_totalNumberOfPellets = params->d_db->getScalar<int>( "TOTAL_NUMBER_OF_PELLETS" );
    d_useSerial            = params->d_db->getWithDefault<bool>( "USE_SERIAL", false );
    d_onlyZcorrection      = params->d_db->getWithDefault<bool>( "ONLY_Z_CORRECTION", false );
    AMP_ASSERT( !( d_useSerial && d_onlyZcorrection ) );
    d_masterId = params->d_db->getScalar<int>( "MASTER" );
    d_slaveId  = params->d_db->getScalar<int>( "SLAVE" );
    if ( params->d_db->keyExists( "SCALING_FACTOR" ) ) {
        d_useScaling    = true;
        d_scalingFactor = params->d_db->getScalar<double>( "SCALING_FACTOR" );
    } else {
        d_useScaling = false;
    }
    d_frozenVectorSet   = false;
    std::string varName = params->d_db->getString( "Variable" );
    d_var.reset( new AMP::LinearAlgebra::Variable( varName ) );
    std::string meshNamePrefix = params->d_db->getString( "MeshNamePrefix" );
    d_currentPellet            = params->d_currentPellet;
    d_pelletStackComm          = params->d_pelletStackComm;
    d_n2nMaps                  = params->d_n2nMaps;
    for ( unsigned int pellId = 0; pellId < d_totalNumberOfPellets; pellId++ ) {
        auto pellId2Str = AMP::Utilities::stringf( "%u", pellId + 1 );
        auto meshName   = meshNamePrefix + "_" + pellId2Str;
        auto currMesh   = d_Mesh->Subset( meshName );
        if ( currMesh == nullptr )
            continue;
        currMesh->setName(
            meshName ); // This is needed since subset may change the name and we rely on it later
        d_pelletIds.push_back( pellId );
        d_meshes.push_back( currMesh );
    } // end for pellId
}

void PelletStackOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );
    auto myParams = std::dynamic_pointer_cast<const PelletStackOperatorParameters>( params );
    if ( myParams )
        d_currentPellet = myParams->d_currentPellet;
}

int PelletStackOperator::getLocalIndexForPellet( unsigned int pellId )
{
    for ( size_t i = 0; i < d_pelletIds.size(); i++ ) {
        if ( d_pelletIds[i] == pellId ) {
            return i;
        }
    } // end for i
    return -1;
}

void PelletStackOperator::applyUnscaling( AMP::LinearAlgebra::Vector::shared_ptr f )
{
    auto subF    = f->subsetVectorForVariable( d_var );
    auto dof_map = subF->getDOFManager();
    auto bnd     = d_Mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_slaveId, 0 );
    auto end_bnd = bnd.end();
    for ( ; bnd != end_bnd; ++bnd ) {
        std::vector<size_t> bndGlobalIds;
        dof_map->getDOFs( bnd->globalID(), bndGlobalIds );
        for ( auto &bndGlobalId : bndGlobalIds ) {
            double val = subF->getLocalValueByGlobalID( bndGlobalId );
            val /= d_scalingFactor;
            subF->setValuesByGlobalID( 1, &bndGlobalId, &val );
        } // end for j
    }     // end for bnd
}

void PelletStackOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                 AMP::LinearAlgebra::Vector::shared_ptr f )
{
    if ( d_onlyZcorrection ) {
        applyOnlyZcorrection( f );
    } else {
        if ( !d_frozenVectorSet ) {
            d_frozenVectorForMaps = d_n2nMaps->getFrozenVector();
            d_frozenVectorSet     = true;
        }
        if ( d_useSerial ) {
            applySerial( u, f );
        } else {
            applyXYZcorrection( u, f );
        }
    }
}

void PelletStackOperator::applyOnlyZcorrection( AMP::LinearAlgebra::Vector::shared_ptr &u )
{
    std::vector<double> finalMaxZdispsList;
    computeZscan( u, finalMaxZdispsList );
    auto subU    = u->subsetVectorForVariable( d_var );
    auto dof_map = subU->getDOFManager();
    for ( size_t i = 0; i < d_pelletIds.size(); ++i ) {
        if ( d_pelletIds[i] > 0 ) {
            auto nd     = d_meshes[i]->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
            auto end_nd = nd.end();
            for ( ; nd != end_nd; ++nd ) {
                std::vector<size_t> dofIds;
                dof_map->getDOFs( nd->globalID(), dofIds );
                subU->addValuesByGlobalID( 1, &dofIds[2], &finalMaxZdispsList[d_pelletIds[i] - 1] );
            } // end for nd
        }
    } // end for i
}

void PelletStackOperator::applyXYZcorrection( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                              AMP::LinearAlgebra::Vector::shared_ptr &r )
{
    AMP_ASSERT( d_frozenVectorSet );
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    // commenting out next line of code instead. Possibly introducing
    // bug involving uninitialized values
    // Currently not filling r with zeros as r may contain valid values
    // BP, Nov 3, 2015
    //      r->copyVector(f);
    d_frozenVectorForMaps->zero();
    d_n2nMaps->apply( u, nullVec );
    auto subU    = d_frozenVectorForMaps->subsetVectorForVariable( d_var );
    auto subR    = r->subsetVectorForVariable( d_var );
    auto dof_map = subR->getDOFManager();
    auto bnd     = d_Mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_slaveId, 0 );
    auto end_bnd = bnd.end();
    for ( ; bnd != end_bnd; ++bnd ) {
        std::vector<size_t> bndGlobalIds;
        dof_map->getDOFs( bnd->globalID(), bndGlobalIds );
        for ( auto &bndGlobalId : bndGlobalIds ) {
            double val = subU->getLocalValueByGlobalID( bndGlobalId );
            subR->addValuesByGlobalID( 1, &bndGlobalId, &val );
        } // end for j
    }     // end for bnd
    std::vector<double> finalMaxZdispsList;
    computeZscan( u, finalMaxZdispsList );
    for ( size_t i = 0; i < d_pelletIds.size(); ++i ) {
        if ( d_pelletIds[i] > 1 ) {
            bnd = d_meshes[i]->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_slaveId, 0 );
            end_bnd = bnd.end();
            for ( ; bnd != end_bnd; ++bnd ) {
                std::vector<size_t> bndGlobalIds;
                dof_map->getDOFs( bnd->globalID(), bndGlobalIds );
                subR->addValuesByGlobalID(
                    1, &bndGlobalIds[2], &finalMaxZdispsList[d_pelletIds[i] - 2] );
            } // end for bnd
        }
    } // end for i
}

void PelletStackOperator::applySerial( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                       AMP::LinearAlgebra::Vector::shared_ptr &r )
{
    AMP_ASSERT( d_frozenVectorSet );
    AMP_ASSERT( d_currentPellet > 0 );
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    int currPellIdx = getLocalIndexForPellet( d_currentPellet );
    int prevPellIdx = getLocalIndexForPellet( d_currentPellet - 1 );
    size_t numMaps  = d_n2nMaps->getNumberOfOperators();
    if ( currPellIdx != -1 ) {
        for ( size_t m = 0; m < numMaps; m++ ) {
            auto currMap = std::dynamic_pointer_cast<AMP::Operator::NodeToNodeMap>(
                d_n2nMaps->getOperator( m ) );
            if ( currMap->getMesh( 2 ) == d_meshes[currPellIdx] ) {
                currMap->applyStart( u, nullVec );
                break;
            }
        } // end for m
    }
    if ( prevPellIdx != -1 ) {
        for ( size_t m = 0; m < numMaps; m++ ) {
            auto currMap = std::dynamic_pointer_cast<AMP::Operator::NodeToNodeMap>(
                d_n2nMaps->getOperator( m ) );
            if ( currMap->getMesh( 1 ) == d_meshes[prevPellIdx] ) {
                currMap->applyStart( u, nullVec );
                break;
            }
        } // end for m
    }
    if ( currPellIdx != -1 ) {
        for ( size_t m = 0; m < numMaps; m++ ) {
            auto currMap = std::dynamic_pointer_cast<AMP::Operator::NodeToNodeMap>(
                d_n2nMaps->getOperator( m ) );
            if ( ( currMap->getMesh( 2 ) ) == d_meshes[currPellIdx] ) {
                currMap->applyFinish( u, nullVec );
                break;
            }
        } // end for m
    }
    if ( prevPellIdx != -1 ) {
        for ( size_t m = 0; m < numMaps; m++ ) {
            auto currMap = std::dynamic_pointer_cast<AMP::Operator::NodeToNodeMap>(
                d_n2nMaps->getOperator( m ) );
            if ( ( currMap->getMesh( 1 ) ) == d_meshes[prevPellIdx] ) {
                currMap->applyFinish( u, nullVec );
                break;
            }
        } // end for m
    }

    //      AMP::LinearAlgebra::Vector::const_shared_ptr subF =
    //      f->subsetVectorForVariable(d_var);
    auto subR    = r->subsetVectorForVariable( d_var );
    auto subU    = d_frozenVectorForMaps->subsetVectorForVariable( d_var );
    auto dof_map = subR->getDOFManager();
    if ( currPellIdx != -1 ) {
        // commenting out next line of code instead. Possibly introducing
        // bug involving uninitialized values
        // Currently not filling r with zeros as r may contain valid values
        // BP, Nov 3, 2015
        //        subR->copyVector(subF);
        auto bnd = d_meshes[currPellIdx]->getBoundaryIDIterator(
            AMP::Mesh::GeomType::Vertex, d_slaveId, 0 );
        auto end_bnd = bnd.end();
        for ( ; bnd != end_bnd; ++bnd ) {
            std::vector<size_t> bndGlobalIds;
            dof_map->getDOFs( bnd->globalID(), bndGlobalIds );
            for ( auto &bndGlobalId : bndGlobalIds ) {
                double val = subU->getLocalValueByGlobalID( bndGlobalId );
                subR->addValuesByGlobalID( 1, &bndGlobalId, &val );
            } // end for j
        }     // end for bnd
    }
}

void PelletStackOperator::computeZscan( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                        std::vector<double> &finalMaxZdispsList )
{
    auto subU    = u->subsetVectorForVariable( d_var );
    auto dof_map = subU->getDOFManager();
    std::vector<double> myMaxZdisps( d_pelletIds.size(), 0.0 );
    for ( size_t i = 0; i < d_pelletIds.size(); i++ ) {
        auto bnd = d_meshes[i]->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_masterId, 0 );
        auto end_bnd = bnd.end();
        for ( ; bnd != end_bnd; ++bnd ) {
            std::vector<size_t> bndGlobalIds;
            dof_map->getDOFs( bnd->globalID(), bndGlobalIds );
            double val = subU->getLocalValueByGlobalID( bndGlobalIds[2] );
            if ( fabs( myMaxZdisps[i] ) < fabs( val ) ) {
                myMaxZdisps[i] = val;
            }
        } // end for bnd
    }     // end for i

    std::vector<int> recvCnts( d_pelletStackComm.getSize() );
    d_pelletStackComm.allGather<int>( d_pelletIds.size(), &( recvCnts[0] ) );

    std::vector<int> recvDisps( recvCnts.size() );
    recvDisps[0] = 0;
    for ( size_t i = 1; i < recvDisps.size(); i++ ) {
        recvDisps[i] = recvDisps[i - 1] + recvCnts[i - 1];
    } // end for i

    std::vector<unsigned int> allPelletIds( ( *( recvDisps.end() - 1 ) ) +
                                            ( *( recvCnts.end() - 1 ) ) );
    d_pelletStackComm.allGather<unsigned int>( &( d_pelletIds[0] ),
                                               d_pelletIds.size(),
                                               &( allPelletIds[0] ),
                                               &( recvCnts[0] ),
                                               &( recvDisps[0] ),
                                               true );

    std::vector<double> allPelletMaxZdisps( allPelletIds.size() );
    d_pelletStackComm.allGather<double>( &( myMaxZdisps[0] ),
                                         d_pelletIds.size(),
                                         &( allPelletMaxZdisps[0] ),
                                         &( recvCnts[0] ),
                                         &( recvDisps[0] ),
                                         true );

    finalMaxZdispsList.resize( d_totalNumberOfPellets, 0.0 );
    for ( size_t i = 0; i < allPelletIds.size(); i++ ) {
        if ( fabs( allPelletMaxZdisps[i] ) > fabs( finalMaxZdispsList[allPelletIds[i]] ) ) {
            finalMaxZdispsList[allPelletIds[i]] = allPelletMaxZdisps[i];
        }
    } // end for i

    for ( size_t i = 1; i < d_totalNumberOfPellets; i++ ) {
        finalMaxZdispsList[i] += finalMaxZdispsList[i - 1];
    } // end for i
}
} // namespace AMP::Operator
