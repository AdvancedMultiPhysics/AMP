#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/boundary/BoundaryOperatorParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"


using AMP::Utilities::stringf;


namespace AMP::Operator {


/****************************************************************
 * Create the appropriate parameters                             *
 ****************************************************************/
static std::shared_ptr<const DirichletMatrixCorrectionParameters>
convert( std::shared_ptr<const OperatorParameters> inParams )
{
    AMP_ASSERT( inParams );
    if ( std::dynamic_pointer_cast<const DirichletMatrixCorrectionParameters>( inParams ) )
        return std::dynamic_pointer_cast<const DirichletMatrixCorrectionParameters>( inParams );
    auto bndParams = std::dynamic_pointer_cast<const BoundaryOperatorParameters>( inParams );
    AMP_ASSERT( bndParams );
    auto linearOperator = std::dynamic_pointer_cast<LinearOperator>( bndParams->d_volumeOperator );
    auto matrixCorrectionParameters =
        std::make_shared<DirichletMatrixCorrectionParameters>( inParams->d_db );
    matrixCorrectionParameters->d_Mesh        = inParams->d_Mesh;
    matrixCorrectionParameters->d_variable    = linearOperator->getOutputVariable();
    matrixCorrectionParameters->d_inputMatrix = linearOperator->getMatrix();
    return matrixCorrectionParameters;
}


/****************************************************************
 * Constructors                                                  *
 ****************************************************************/
DirichletMatrixCorrection::DirichletMatrixCorrection(
    std::shared_ptr<const OperatorParameters> inParams )
    : BoundaryOperator( inParams )
{
    auto params = convert( inParams );
    AMP_ASSERT( params );
    d_variable                 = params->d_variable;
    d_computedAddRHScorrection = false;
    d_symmetricCorrection      = params->d_db->getWithDefault<bool>( "symmetric_correction", true );
    d_zeroDirichletBlock   = params->d_db->getWithDefault<bool>( "zero_dirichlet_block", false );
    d_skipRHSsetCorrection = params->d_db->getWithDefault<bool>( "skip_rhs_correction", true );
    d_skipRHSaddCorrection =
        params->d_db->getWithDefault<bool>( "skip_rhs_add_correction", d_skipRHSsetCorrection );
    d_applyMatrixCorrectionWasCalled = false;
    reset( params );
}


/****************************************************************
 * Reset                                                         *
 ****************************************************************/
void DirichletMatrixCorrection::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );
    auto myParams = std::dynamic_pointer_cast<const DirichletMatrixCorrectionParameters>( params );
    parseParams( myParams );

    d_inputMatrix = myParams->d_inputMatrix;
    AMP_INSIST( d_inputMatrix, "NULL matrix" );
    d_inputMatrix->makeConsistent(
        AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD ); // Check that we can call makeConsistent

    if ( d_skipRHSsetCorrection )
        AMP_ASSERT( d_skipRHSaddCorrection );
    if ( !d_skipRHSaddCorrection )
        AMP_ASSERT( !d_skipRHSsetCorrection );

    d_applyMatrixCorrectionWasCalled = false;

    initRhsCorrectionSet();

    if ( d_skipRHSaddCorrection )
        applyMatrixCorrection();
}
void DirichletMatrixCorrection::parseParams(
    std::shared_ptr<const DirichletMatrixCorrectionParameters> params )
{
    AMP_INSIST( params->d_db, "NULL database" );
    bool skipParams = params->d_db->getWithDefault<bool>( "skip_params", false );

    if ( !skipParams || !d_initialized ) {
        d_symmetricCorrection = params->d_db->getWithDefault<bool>( "symmetric_correction", true );
        d_zeroDirichletBlock  = params->d_db->getWithDefault<bool>( "zero_dirichlet_block", false );
        d_skipRHSsetCorrection = params->d_db->getWithDefault<bool>( "skip_rhs_correction", true );
        d_skipRHSaddCorrection =
            params->d_db->getWithDefault<bool>( "skip_rhs_add_correction", d_skipRHSsetCorrection );

        if ( d_symmetricCorrection == false )
            d_skipRHSaddCorrection = true;

        int numIds = params->d_db->getScalar<int>( "number_of_ids" );

        d_boundaryIds.resize( numIds );
        d_dofIds.resize( numIds );

        for ( int j = 0; j < numIds; ++j ) {
            d_boundaryIds[j] = params->d_db->getScalar<int>( stringf( "id_%d", j ) );
            int numDofIds    = params->d_db->getScalar<int>( stringf( "number_of_dofs_%d", j ) );
            d_dofIds[j].resize( numDofIds );
            for ( int i = 0; i < numDofIds; ++i ) {
                d_dofIds[j][i] = params->d_db->getScalar<int>( stringf( "dof_%d_%d", j, i ) );
            }
        }

        if ( !d_skipRHSsetCorrection ) {
            d_dirichletValues.resize( numIds );
            for ( int j = 0; j < numIds; ++j ) {
                int numDofIds = d_dofIds[j].size();
                d_dirichletValues[j].resize( numDofIds );
                for ( int i = 0; i < numDofIds; ++i ) {
                    auto key                = stringf( "value_%d_%d", j, i );
                    d_dirichletValues[j][i] = params->d_db->getWithDefault<double>( key, 0.0 );
                }
            }
        }

        d_initialized = true;
    }
}


/****************************************************************
 * applyMatrixCorrection                                         *
 ****************************************************************/
void DirichletMatrixCorrection::applyMatrixCorrection()
{
    AMP_ASSERT( !d_applyMatrixCorrectionWasCalled );
    d_applyMatrixCorrectionWasCalled = true;

    auto dof_map = d_inputMatrix->getRightDOFManager();
    AMP_ASSERT( ( *dof_map ) == ( *d_inputMatrix->getLeftDOFManager() ) );
    AMP_ASSERT( ( *dof_map ) == ( *d_inputMatrix->getRightDOFManager() ) );

    std::vector<size_t> bndDofIds, nhDofIds;
    for ( size_t k = 0; k < d_boundaryIds.size(); ++k ) {
        for ( auto &node :
              d_Mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, d_boundaryIds[k], 0 ) ) {
            dof_map->getDOFs( node.globalID(), bndDofIds );

            // Get neighbors does not include the calling node itself.
            // Get neighbors also returns remote neighbors
            // The calling node must be owned locally.
            auto neighbors = node.getNeighbors();
            for ( auto &neighbor : neighbors ) {
                if ( neighbor )
                    AMP_INSIST( *neighbor != node, "boundary node neighbor includes self" );
            }

            for ( auto &elem : d_dofIds[k] ) {
                for ( unsigned int i = 0; i < bndDofIds.size(); ++i ) {
                    if ( elem == i ) {
                        if ( d_zeroDirichletBlock ) {
                            d_inputMatrix->setValueByGlobalID( bndDofIds[i], bndDofIds[i], 0.0 );
                        } else {
                            d_inputMatrix->setValueByGlobalID( bndDofIds[i], bndDofIds[i], 1.0 );
                        }
                    } else {
                        d_inputMatrix->setValueByGlobalID( bndDofIds[elem], bndDofIds[i], 0.0 );
                        if ( d_symmetricCorrection ) {
                            d_inputMatrix->setValueByGlobalID( bndDofIds[i], bndDofIds[elem], 0.0 );
                        }
                    }
                }
                for ( auto &neighbor : neighbors ) {
                    if ( !neighbor )
                        continue;
                    dof_map->getDOFs( neighbor->globalID(), nhDofIds );
                    for ( auto &nhDofId : nhDofIds ) {
                        d_inputMatrix->setValueByGlobalID( bndDofIds[elem], nhDofId, 0.0 );
                        if ( d_symmetricCorrection ) {
                            d_inputMatrix->setValueByGlobalID( nhDofId, bndDofIds[elem], 0.0 );
                        }
                    }
                }
            }
        }
    }

    // This does consistent for both "Sum-into" and "set".
    d_inputMatrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


void DirichletMatrixCorrection::initRhsCorrectionSet()
{
    if ( !d_skipRHSsetCorrection ) {
        int numIds = d_dofIds.size();
        char key[100];
        auto tmp_db = std::make_shared<AMP::Database>( "Dummy" );
        tmp_db->putScalar( "skip_params", false );
        tmp_db->putScalar( "isAttachedToVolumeOperator", false );
        tmp_db->putScalar( "number_of_ids", numIds );
        tmp_db->putScalar( "print_info_level", d_iDebugPrintInfoLevel );
        for ( int j = 0; j < numIds; j++ ) {
            int numDofIds = d_dofIds[j].size();

            snprintf( key, sizeof key, "id_%d", j );
            tmp_db->putScalar( key, d_boundaryIds[j] );

            snprintf( key, sizeof key, "number_of_dofs_%d", j );
            tmp_db->putScalar( key, numDofIds );

            for ( int i = 0; i < numDofIds; i++ ) {
                snprintf( key, sizeof key, "dof_%d_%d", j, i );
                tmp_db->putScalar( key, d_dofIds[j][i] );

                snprintf( key, sizeof key, "value_%d_%d", j, i );
                tmp_db->putScalar( key, d_dirichletValues[j][i] );
            } // end for i
        } // end for j

        auto setDispOpParams = std::make_shared<DirichletVectorCorrectionParameters>( tmp_db );
        setDispOpParams->d_variable = d_variable;
        setDispOpParams->d_Mesh     = d_Mesh;

        if ( !d_rhsCorrectionSet ) {
            d_rhsCorrectionSet.reset( new DirichletVectorCorrection( setDispOpParams ) );
        } else {
            d_rhsCorrectionSet->reset( setDispOpParams );
        }
    }
}


void DirichletMatrixCorrection::initRhsCorrectionAdd( AMP::LinearAlgebra::Vector::shared_ptr rhs )
{
    AMP_ASSERT( !d_applyMatrixCorrectionWasCalled );

    if ( !d_skipRHSsetCorrection ) {
        if ( !d_skipRHSaddCorrection ) {
            if ( !d_dispVals ) {
                d_dispVals = ( subsetOutputVector( rhs ) )->clone();
                AMP_ASSERT( ( *( d_dispVals->getVariable() ) ) == ( *d_variable ) );
            }

            d_dispVals->zero();

            AMP::LinearAlgebra::Vector::shared_ptr emptyVec;
            d_rhsCorrectionSet->apply( emptyVec, d_dispVals );

            if ( !d_rhsCorrectionAdd ) {
                d_rhsCorrectionAdd = d_dispVals->clone();
            }

            d_inputMatrix->mult( d_dispVals, d_rhsCorrectionAdd );

            d_rhsCorrectionAdd->scale( -1.0 );
        }
    }
}


void DirichletMatrixCorrection::addRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs )
{
    if ( !d_skipRHSaddCorrection ) {
        if ( !d_applyMatrixCorrectionWasCalled ) {
            initRhsCorrectionAdd( rhs );
            applyMatrixCorrection();
        } // end if
        auto myRhs = subsetOutputVector( rhs );
        myRhs->add( *myRhs, *d_rhsCorrectionAdd );
    }
}


void DirichletMatrixCorrection::setRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs )
{
    if ( !d_skipRHSsetCorrection ) {
        d_rhsCorrectionSet->apply( nullptr, rhs );
    }
}
} // namespace AMP::Operator
