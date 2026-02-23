#include "AMP/operators/map/libmesh/Map3Dto1D.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"

// Libmesh files
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/face_quad4.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/node.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS


namespace AMP::Operator {


Map3Dto1D::Map3Dto1D( std::shared_ptr<const OperatorParameters> params ) : MapOperator( params )
{
    auto myparams = std::dynamic_pointer_cast<const MapOperatorParameters>( params );
    reset( myparams );
}


void Map3Dto1D::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );

    auto myparams = std::dynamic_pointer_cast<const MapOperatorParameters>( params );

    AMP_INSIST( myparams, "NULL parameter" );
    AMP_INSIST( myparams->d_db, "NULL database" );
    d_Mesh    = myparams->d_Mesh;
    d_MapMesh = myparams->d_MapMesh;
    d_MapComm = myparams->d_MapComm;
    if ( d_MapComm.isNull() && d_MapMesh )
        d_MapComm = d_MapMesh->getComm();
    if ( d_MapComm.isNull() && d_Mesh )
        d_MapComm = d_Mesh->getComm();
    AMP_INSIST( !d_MapComm.isNull(), "NULL communicator" );
    AMP_INSIST( d_MapComm.sumReduce<int>( d_MapMesh ? 1 : 0 ) > 0, "Somebody must own the mesh" );

    d_useGaussVec = myparams->d_db->getWithDefault<bool>( "UseGaussVec", false );

    if ( d_useGaussVec ) {
        AMP::Mesh::MeshIterator iterator =
            d_MapMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, d_boundaryId, 0 );
        libmeshElements.reinit( iterator );
    }

    AMP_INSIST( myparams->d_db->keyExists( "InputVariable" ), "key not found" );
    std::string inpVar = myparams->d_db->getString( "InputVariable" );
    d_inpVariable.reset( new AMP::LinearAlgebra::Variable( inpVar ) );

    AMP_INSIST( myparams->d_db->keyExists( "OutputVariable" ), "key not found" );
    std::string outVar = myparams->d_db->getString( "OutputVariable" );
    d_outVariable.reset( new AMP::LinearAlgebra::Variable( outVar ) );
}


void Map3Dto1D::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                       AMP::LinearAlgebra::Vector::shared_ptr f )
{
    if ( !outputVec )
        setVector( f );
    if ( d_useGaussVec ) {
        apply_Gauss( u, f );
    } else {
        apply_Nodal( u, f );
    }
}

void Map3Dto1D::apply_Gauss( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr )
{
    const unsigned int numPoints = outputVec->getLocalSize();
    std::vector<double> mapValues( numPoints, 0 );
    std::vector<int> numFaceGauss( numPoints, 0 );

    // Get the local contributions to the map
    if ( d_MapMesh ) {
        AMP_ASSERT( u );
        AMP::LinearAlgebra::Vector::const_shared_ptr inputVec = subsetInputVector( u );
        AMP_ASSERT( inputVec );
        AMP_ASSERT( inputVec->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

        std::shared_ptr<AMP::Discretization::DOFManager> dof_map = inputVec->getDOFManager();

        if ( d_iDebugPrintInfoLevel > 5 ) {
            AMP::pout << "The input to Map3Dto1D " << std::endl;
            AMP::pout << inputVec << std::endl;
        }

        AMP_ASSERT( outputVec );

        // Get an iterator over the side elements
        AMP::Mesh::MeshIterator bnd =
            d_MapMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, d_boundaryId, 0 );
        AMP::Mesh::MeshIterator end_bnd = bnd.end();

        // Iterator for the solid-clad boundary
        for ( ; bnd != end_bnd; ++bnd ) {
            auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
            auto feFamily = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );

            auto d_feType = std::make_shared<libMesh::FEType>( feTypeOrder, feFamily );
            std::shared_ptr<libMesh::FEBase> d_fe(
                ( libMesh::FEBase::build( 2, ( *d_feType ) ) ).release() );
            d_fe->get_xyz();

            auto qruleOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "SECOND" );
            std::shared_ptr<libMesh::QBase> d_qrule(
                ( libMesh::QBase::build( "QGAUSS", 2, qruleOrder ) ).release() );

            d_fe->attach_quadrature_rule( d_qrule.get() );

            d_fe->reinit( libmeshElements.getElement( bnd->globalID() ) );

            // Get the current position and DOF
            auto coordinates = d_fe->get_xyz();

            std::vector<size_t> ids;
            dof_map->getDOFs( bnd->globalID(), ids );

            std::vector<double> zcoords;
            for ( size_t i = 0; i < ids.size(); i++ )
                zcoords.push_back( coordinates[i]( 2 ) );

            std::sort( zcoords.begin(), zcoords.end() );

            std::vector<double> tmpZcoords = zcoords;
            std::vector<int> tmpIds( zcoords.size() );

            for ( size_t i = 0; i < ids.size(); i++ ) {
                tmpIds[i] = i;
            }

            std::vector<int> originalGaussOrder( zcoords.size() );

            for ( size_t i = 0; i < ids.size(); i++ ) {
                double myZ = coordinates[i]( 2 );
                for ( unsigned int j = 0; j < tmpZcoords.size(); j++ ) {
                    if ( fabs( tmpZcoords[j] - myZ ) <= 1.e-12 ) {
                        originalGaussOrder[tmpIds[j]] = i;
                        tmpZcoords.erase( tmpZcoords.begin() + j );
                        tmpIds.erase( tmpIds.begin() + j );
                        break;
                    }
                }
            }

            std::vector<double> z( 4, 0 );
            std::vector<double> y( 4, 0 );
            std::vector<double> x( 4, 0 );
            for ( int i = 0; i < 4; i++ ) {
                x[i] = coordinates[originalGaussOrder[i]]( 0 );
                y[i] = coordinates[originalGaussOrder[i]]( 1 );
                z[i] = coordinates[originalGaussOrder[i]]( 2 );
            }

            int pickId;
            if ( std::pow( ( y[0] - y[3] ), 2 ) + std::pow( ( x[0] - x[3] ), 2 ) <
                 std::pow( ( y[0] - y[2] ), 2 ) + std::pow( ( x[0] - x[2] ), 2 ) ) {
                pickId = 3;
            } else {
                pickId = 2;
            }

            // Iterator for the fluid boundary
            for ( unsigned int i = 0; i < numPoints; i++ ) {

                double cur_gauss = d_zLocations[i];

                // Section of the Clad boundary map corresponding to the fluid Element
                if ( cur_gauss >= z[0] && cur_gauss <= z[pickId] ) {
                    mapValues[i] +=
                        ( ( inputVec )->getValueByGlobalID( ids[originalGaussOrder[0]] ) *
                              ( z[pickId] - cur_gauss ) +
                          ( inputVec )->getValueByGlobalID( ids[originalGaussOrder[pickId]] ) *
                              ( cur_gauss - z[0] ) ) /
                        ( z[pickId] - z[0] );
                    numFaceGauss[i] += 1;
                }
            } // end for i
        } // end for bnd
    } // end if

    // Gather the results from all processors
    std::vector<double> aggMapValues( numPoints );
    std::vector<int> aggNumFaceGauss( numPoints );
    d_MapComm.sumReduce( (double *) &( mapValues[0] ), (double *) &( aggMapValues[0] ), numPoints );
    d_MapComm.sumReduce( (int *) &( numFaceGauss[0] ), (int *) &( aggNumFaceGauss[0] ), numPoints );

    // Store the results
    for ( size_t i = 0; i < numPoints; i++ ) {
        const double val = aggMapValues[i] / static_cast<double>( aggNumFaceGauss[i] );
        outputVec->setValuesByLocalID( 1, &i, &val );
    }

    if ( d_iDebugPrintInfoLevel > 4 ) {
        AMP::pout << "The output to Map3Dto1D " << std::endl;
        AMP::pout << outputVec << std::endl;
    }
}


void Map3Dto1D::apply_Nodal( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                             AMP::LinearAlgebra::Vector::shared_ptr )
{

    const unsigned int numPoints = outputVec->getLocalSize();
    std::vector<double> mapValues( numPoints, 0 );
    std::vector<int> numFaceNodes( numPoints, 0 );

    // Get the local contributions to the map
    if ( d_MapMesh ) {
        AMP_ASSERT( u );

        // Subset u for the local vector of interest
        AMP::LinearAlgebra::Vector::const_shared_ptr inputVec = subsetInputVector( u );
        AMP_ASSERT( inputVec );
        AMP_ASSERT( inputVec->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

        std::shared_ptr<AMP::Discretization::DOFManager> dof_map = inputVec->getDOFManager();

        if ( d_iDebugPrintInfoLevel > 5 ) {
            AMP::pout << "The input to Map3Dto1D " << std::endl;
            AMP::pout << inputVec << std::endl;
        }

        AMP_ASSERT( outputVec );

        // Get an iterator over the side elements
        AMP::Mesh::MeshIterator bnd =
            d_MapMesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, d_boundaryId, 0 );
        AMP::Mesh::MeshIterator end_bnd = bnd.end();

        // Iterator for the solid-clad boundary
        for ( ; bnd != end_bnd; ++bnd ) {

            auto &cur_side = *bnd;
            auto nodes     = cur_side.getElements( AMP::Mesh::GeomType::Vertex );
            AMP_ASSERT( nodes.size() == 4 );

            std::vector<double> zcoords;
            for ( auto &node : nodes ) {
                auto coord = node.coord();
                zcoords.push_back( coord[2] );
            }

            std::sort( zcoords.begin(), zcoords.end() );

            std::vector<double> tmpZcoords = zcoords;
            std::vector<int> tmpIds( zcoords.size() );

            for ( size_t i = 0; i < nodes.size(); i++ ) {
                tmpIds[i] = i;
            }

            std::vector<int> originalNodeOrder( zcoords.size() );

            for ( size_t i = 0; i < nodes.size(); i++ ) {
                auto coord = nodes[i].coord();
                double myZ = coord[2];
                for ( unsigned int j = 0; j < tmpZcoords.size(); j++ ) {
                    if ( fabs( tmpZcoords[j] - myZ ) <= 1.e-12 ) {
                        originalNodeOrder[tmpIds[j]] = i;
                        tmpZcoords.erase( tmpZcoords.begin() + j );
                        tmpIds.erase( tmpIds.begin() + j );
                        break;
                    }
                }
            }

            std::vector<double> z( 4, 0 );
            std::vector<double> y( 4, 0 );
            std::vector<double> x( 4, 0 );
            for ( int i = 0; i < 4; i++ ) {
                auto coord = nodes[originalNodeOrder[i]].coord();
                x[i]       = coord[0];
                y[i]       = coord[1];
                z[i]       = coord[2];
            }

            int pickId;
            if ( std::pow( ( y[0] - y[3] ), 2 ) + std::pow( ( x[0] - x[3] ), 2 ) <
                 std::pow( ( y[0] - y[2] ), 2 ) + std::pow( ( x[0] - x[2] ), 2 ) ) {
                pickId = 3;
            } else {
                pickId = 2;
            }

            // Iterator for the fluid boundary
            for ( unsigned int i = 0; i < numPoints; i++ ) {

                double cur_node = d_zLocations[i];

                // Section of the Clad boundary map corresponding to the fluid Element
                if ( cur_node >= z[0] && cur_node <= z[pickId] ) {
                    std::vector<size_t> dof1;
                    std::vector<size_t> dof2;
                    dof_map->getDOFs( nodes[originalNodeOrder[0]].globalID(), dof1 );
                    dof_map->getDOFs( nodes[originalNodeOrder[pickId]].globalID(), dof2 );
                    AMP_ASSERT( dof1.size() == 1 && dof2.size() == 1 );

                    mapValues[i] +=
                        ( inputVec->getValueByGlobalID( dof1[0] ) * ( z[pickId] - cur_node ) +
                          inputVec->getValueByGlobalID( dof2[0] ) * ( cur_node - z[0] ) ) /
                        ( z[pickId] - z[0] );

                    numFaceNodes[i] += 1;
                }
            } // end for i
        } // end for bnd
    }

    // Gather the results from all processors
    std::vector<double> aggMapValues( numPoints );
    std::vector<int> aggNumFaceNodes( numPoints );
    d_MapComm.sumReduce( (double *) &( mapValues[0] ), (double *) &( aggMapValues[0] ), numPoints );
    d_MapComm.sumReduce( (int *) &( numFaceNodes[0] ), (int *) &( aggNumFaceNodes[0] ), numPoints );

    // Store the results
    for ( size_t i = 0; i < numPoints; i++ ) {
        const double val = aggMapValues[i] / static_cast<double>( aggNumFaceNodes[i] );
        outputVec->setValuesByLocalID( 1, &i, &val );
    }

    if ( d_iDebugPrintInfoLevel > 4 ) {
        AMP::pout << "The output to Map3Dto1D " << std::endl;
        AMP::pout << outputVec << std::endl;
    }
}
} // namespace AMP::Operator
