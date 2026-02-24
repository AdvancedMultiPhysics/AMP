#include "AMP/operators/boundary/MassMatrixCorrection.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/utils/Database.h"


namespace AMP::Operator {


/****************************************************************
 * Create the appropriate parameters                             *
 ****************************************************************/
static std::shared_ptr<const MassMatrixCorrectionParameters>
convert( std::shared_ptr<const OperatorParameters> inParams )
{
    AMP_ASSERT( inParams );
    if ( std::dynamic_pointer_cast<const MassMatrixCorrectionParameters>( inParams ) )
        return std::dynamic_pointer_cast<const MassMatrixCorrectionParameters>( inParams );
    auto bndParams = std::dynamic_pointer_cast<const BoundaryOperatorParameters>( inParams );
    AMP_ASSERT( bndParams );
    auto linearOperator = std::dynamic_pointer_cast<LinearOperator>( bndParams->d_volumeOperator );
    auto params         = std::make_shared<MassMatrixCorrectionParameters>( inParams->d_db );
    params->d_Mesh      = inParams->d_Mesh;
    params->d_variable  = linearOperator->getOutputVariable();
    params->d_inputMatrix = linearOperator->getMatrix();
    return params;
}


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
MassMatrixCorrection::MassMatrixCorrection( std::shared_ptr<const OperatorParameters> inParams )
    : BoundaryOperator( inParams ), d_bSetIdentityOnDiagonal( false )
{
    auto params = convert( inParams );
    AMP_ASSERT( params );
    d_variable = params->d_variable;
    reset( params );
}


/****************************************************************
 * reset                                                         *
 ****************************************************************/
void MassMatrixCorrection::resetBoundaryIds(
    std::shared_ptr<const MassMatrixCorrectionParameters> params )
{
    AMP_INSIST( ( ( ( params->d_db ).get() ) != nullptr ), "NULL database" );
    bool skipParams          = params->d_db->getWithDefault<bool>( "skip_params", true );
    d_bSetIdentityOnDiagonal = params->d_db->getWithDefault<bool>( "setIdentityOnDiagonal", false );

    if ( !skipParams ) {
        AMP_INSIST( params->d_db->keyExists( "number_of_ids" ),
                    "Key ''number_of_ids'' is missing!" );
        int numIds = params->d_db->getScalar<int>( "number_of_ids" );

        d_boundaryIds.resize( numIds );
        d_dofIds.resize( numIds );

        char key[100];
        for ( int j = 0; j < numIds; j++ ) {
            snprintf( key, sizeof key, "id_%d", j );
            AMP_INSIST( params->d_db->keyExists( key ), "Key is missing!" );
            d_boundaryIds[j] = params->d_db->getScalar<int>( key );

            snprintf( key, sizeof key, "number_of_dofs_%d", j );
            AMP_INSIST( params->d_db->keyExists( key ), "Key is missing!" );
            int numDofIds = params->d_db->getScalar<int>( key );

            d_dofIds[j].resize( numDofIds );
            for ( int i = 0; i < numDofIds; i++ ) {
                snprintf( key, sizeof key, "dof_%d_%d", j, i );
                AMP_INSIST( params->d_db->keyExists( key ), "Key is missing!" );
                d_dofIds[j][i] = params->d_db->getScalar<int>( key );
            } // end for i
        }     // end for j
    }
}
void MassMatrixCorrection::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );

    auto myParams = std::dynamic_pointer_cast<const MassMatrixCorrectionParameters>( params );

    AMP_INSIST( myParams, "NULL parameters" );

    resetBoundaryIds( myParams );

    double diagVal = d_bSetIdentityOnDiagonal ? 1.0 : 0.0;

    auto inputMatrix = myParams->d_inputMatrix;
    AMP_INSIST( inputMatrix, "NULL matrix" );

    auto inVec   = inputMatrix->createInputVector();
    auto dof_map = inVec->getDOFManager();

    unsigned int numIds = d_boundaryIds.size();

    for ( unsigned int k = 0; k < numIds; k++ ) {

        auto bnd =
            d_Mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_boundaryIds[k], 0 );
        auto end_bnd = bnd.end();

        for ( ; bnd != end_bnd; ++bnd ) {
            std::vector<size_t> bndGlobalIds;
            dof_map->getDOFs( bnd->globalID(), bndGlobalIds );

            auto neighbors = bnd->getNeighbors();

            for ( auto &neighbor : neighbors ) {
                if ( !neighbor.isNull() )
                    AMP_ASSERT( neighbor != *bnd );
            } // end for el

            for ( unsigned int j = 0; j < d_dofIds[k].size(); ++j ) {
                for ( unsigned int i = 0; i < bndGlobalIds.size(); ++i ) {

                    if ( d_dofIds[k][j] == i ) {
                        inputMatrix->setValueByGlobalID(
                            bndGlobalIds[j], bndGlobalIds[j], diagVal );
                    } else {
                        inputMatrix->setValueByGlobalID(
                            bndGlobalIds[d_dofIds[k][j]], bndGlobalIds[i], 0.0 );
                        inputMatrix->setValueByGlobalID(
                            bndGlobalIds[i], bndGlobalIds[d_dofIds[k][j]], 0.0 );
                    }
                } // end for i
                for ( auto &neighbor : neighbors ) {
                    if ( neighbor.isNull() )
                        continue;
                    std::vector<size_t> nhDofIds;
                    dof_map->getDOFs( neighbor.globalID(), nhDofIds );
                    for ( auto &nhDofId : nhDofIds ) {
                        inputMatrix->setValueByGlobalID(
                            bndGlobalIds[d_dofIds[k][j]], nhDofId, 0.0 );
                        inputMatrix->setValueByGlobalID(
                            nhDofId, bndGlobalIds[d_dofIds[k][j]], 0.0 );
                    } // end for i
                }     // end for n
            }         // end for j
        }             // end for bnd
    }                 // end for k

    // This does consistent for both "Sum-into" and "set".
    inputMatrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}
} // namespace AMP::Operator
