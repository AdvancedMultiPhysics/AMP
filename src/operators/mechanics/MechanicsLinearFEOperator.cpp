#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"


namespace AMP::Operator {


static std::shared_ptr<const MechanicsLinearFEOperatorParameters>
convert( std::shared_ptr<const OperatorParameters> params )
{
    if ( std::dynamic_pointer_cast<const MechanicsLinearFEOperatorParameters>( params ) )
        return std::dynamic_pointer_cast<const MechanicsLinearFEOperatorParameters>( params );
    auto db = params->d_db;
    // first create a MechanicsMaterialModel
    auto model_db = db->getDatabase( "LocalModel" );
    auto model    = std::dynamic_pointer_cast<MechanicsMaterialModel>(
        ElementPhysicsModelFactory::createElementPhysicsModel( model_db ) );
    // next create a ElementOperation object
    auto elem_db       = db->getDatabase( "MechanicsElement" );
    auto mechanicsElem = ElementOperationFactory::createElementOperation( elem_db );
    // now create the linear mechanics operator parameters
    AMP_ASSERT( db->getString( "name" ) == "MechanicsLinearFEOperator" );
    auto dofManager = AMP::Discretization::simpleDOFManager::create(
        params->d_Mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );
    auto myparams             = std::make_shared<MechanicsLinearFEOperatorParameters>( db );
    myparams->d_Mesh          = params->d_Mesh;
    myparams->d_materialModel = model;
    myparams->d_elemOp        = mechanicsElem;
    myparams->d_inDofMap      = dofManager;
    myparams->d_outDofMap     = dofManager;
    return myparams;
}


MechanicsLinearFEOperator::MechanicsLinearFEOperator(
    std::shared_ptr<const OperatorParameters> params )
    : MechanicsLinearFEOperator( convert( params ), true )
{
}
MechanicsLinearFEOperator::MechanicsLinearFEOperator(
    std::shared_ptr<const MechanicsLinearFEOperatorParameters> params, bool )
    : LinearFEOperator( params )
{
    AMP_INSIST( params, "NULL parameter" );
    d_useUpdatedLagrangian = params->d_db->getWithDefault<bool>( "USE_UPDATED_LAGRANGIAN", false );
    if ( d_useUpdatedLagrangian ) {
        d_mechLinULElem =
            std::dynamic_pointer_cast<MechanicsLinearUpdatedLagrangianElement>( d_elemOp );
        AMP_INSIST( d_mechLinULElem,
                    "d_elemOp is not of type MechanicsLinearUpdatedLagrangianElement" );
    } else {
        d_mechLinElem = std::dynamic_pointer_cast<MechanicsLinearElement>( d_elemOp );
        AMP_INSIST( d_mechLinElem, "d_elemOp is not of type MechanicsLinearElement" );
    }
    d_materialModel = params->d_materialModel;

    AMP_INSIST( params->d_db->keyExists( "InputVariable" ), "key not found" );
    std::string inpVarName = params->d_db->getString( "InputVariable" );
    d_inputVariable.reset( new AMP::LinearAlgebra::Variable( inpVarName ) );

    AMP_INSIST( params->d_db->keyExists( "OutputVariable" ), "key not found" );
    std::string outVarName = params->d_db->getString( "OutputVariable" );
    d_outputVariable.reset( new AMP::LinearAlgebra::Variable( outVarName ) );

    if ( d_useUpdatedLagrangian ) {
        d_refXYZ = AMP::LinearAlgebra::createVector(
            d_inDofMap, d_inputVariable, true, d_memory_location );
        d_refXYZ->zero();

        auto el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
        auto end_el = el.end();

        for ( ; el != end_el; ++el ) {
            d_currNodes               = el->getElements( AMP::Mesh::GeomType::Vertex );
            size_t numNodesInCurrElem = d_currNodes.size();

            getDofIndicesForCurrentElement();

            createCurrentLibMeshElement();

            std::vector<double> elementRefXYZ;
            elementRefXYZ.resize( 3 * numNodesInCurrElem );

            d_mechLinULElem->initializeForCurrentElement( d_currElemPtr, d_materialModel );
            d_mechLinULElem->initializeReferenceXYZ( elementRefXYZ );

            for ( size_t j = 0; j < numNodesInCurrElem; j++ ) {
                for ( size_t i = 0; i < 3; i++ ) {
                    d_refXYZ->setValuesByGlobalID(
                        1, &d_dofIndices[j][i], &elementRefXYZ[( 3 * j ) + i] );
                } // end of i
            } // end of j

            destroyCurrentLibMeshElement();
        } // end of el

        d_refXYZ->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    } // end of UpdatedLagrangian condition.

    bool isAttachedToNonlinearOperator =
        params->d_db->getWithDefault<bool>( "isAttachedToNonlinearOperator", false );

    if ( isAttachedToNonlinearOperator ) {
        bool isNonlinearOperatorInitialized =
            params->d_db->getWithDefault<bool>( "isNonlinearOperatorInitialized", false );
        if ( isNonlinearOperatorInitialized ) {
            reset( params );
        } else {
            auto tmpInVec = AMP::LinearAlgebra::createVector(
                d_inDofMap, d_inputVariable, true, d_memory_location );
            auto tmpOutVec = AMP::LinearAlgebra::createVector(
                d_outDofMap, d_outputVariable, true, d_memory_location );
            d_matrix = AMP::LinearAlgebra::createMatrix( tmpInVec, tmpOutVec );
        }
    } else {
        reset( params );
    }
}

void MechanicsLinearFEOperator::preAssembly( std::shared_ptr<const OperatorParameters> oparams )
{
    if ( d_useUpdatedLagrangian ) {
        auto params =
            std::dynamic_pointer_cast<const MechanicsLinearFEOperatorParameters>( oparams );
        AMP_INSIST( params, "NULL params" );

        if ( !d_dispVec && params->d_dispVec ) {
            d_dispVec = params->d_dispVec->clone();
        }
        if ( d_dispVec ) {
            if ( params->d_dispVec ) {
                d_dispVec->copyVector( params->d_dispVec );
                d_dispVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
            } else {
                d_dispVec.reset();
            }
        }
    }
    d_matrix->zero();
    d_materialModel->preLinearAssembly();
}

void MechanicsLinearFEOperator::preElementOperation( const AMP::Mesh::MeshElement &elem )
{
    d_currNodes               = elem.getElements( AMP::Mesh::GeomType::Vertex );
    size_t numNodesInCurrElem = d_currNodes.size();

    getDofIndicesForCurrentElement();

    createCurrentLibMeshElement();

    size_t num_local_dofs = ( 3 * numNodesInCurrElem );
    d_elementStiffnessMatrix.resize( num_local_dofs );
    for ( size_t r = 0; r < num_local_dofs; r++ ) {
        d_elementStiffnessMatrix[r].resize( num_local_dofs );
        for ( size_t c = 0; c < num_local_dofs; c++ ) {
            d_elementStiffnessMatrix[r][c] = 0.0;
        } // end for c
    } // end for r

    if ( d_useUpdatedLagrangian ) {
        std::vector<std::vector<double>> elementInputVectors(
            Mechanics::TOTAL_NUMBER_OF_VARIABLES );
        std::vector<double> elementRefXYZ;
        elementInputVectors[Mechanics::DISPLACEMENT].resize( num_local_dofs );
        elementRefXYZ.resize( num_local_dofs );
        for ( size_t r = 0; r < numNodesInCurrElem; r++ ) {
            for ( size_t d = 0; d < 3; d++ ) {
                if ( d_dispVec ) {
                    elementInputVectors[Mechanics::DISPLACEMENT][( 3 * r ) + d] =
                        d_dispVec->getValueByGlobalID( d_dofIndices[r][d] );
                } else {
                    elementInputVectors[Mechanics::DISPLACEMENT][( 3 * r ) + d] = 0.0;
                }
                elementRefXYZ[( 3 * r ) + d] = d_refXYZ->getValueByGlobalID( d_dofIndices[r][d] );
            } // end for d
        } // end for r
        d_mechLinULElem->initializeForCurrentElement( d_currElemPtr, d_materialModel );
        d_mechLinULElem->setElementVectors( elementInputVectors );
        d_mechLinULElem->setElementStiffnessMatrix( d_elementStiffnessMatrix );
        d_mechLinULElem->assignReferenceXYZ( elementRefXYZ );
    } else {
        d_mechLinElem->initializeForCurrentElement( d_currElemPtr, d_materialModel );
        d_mechLinElem->setElementStiffnessMatrix( d_elementStiffnessMatrix );
    }
}

void MechanicsLinearFEOperator::postElementOperation()
{
    PROFILE( "postElementOperation", 5 );
    size_t numNodesInCurrElem = d_dofIndices.size();
    for ( size_t r = 0; r < numNodesInCurrElem; r++ ) {
        for ( size_t dr = 0; dr < 3; dr++ ) {
            for ( size_t c = 0; c < numNodesInCurrElem; c++ ) {
                for ( size_t dc = 0; dc < 3; dc++ ) {
                    d_matrix->addValueByGlobalID(
                        d_dofIndices[r][dr],
                        d_dofIndices[c][dc],
                        d_elementStiffnessMatrix[( 3 * r ) + dr][( 3 * c ) + dc] );
                } // end for dc
            } // end for c
        } // end for dr
    } // end for r
    destroyCurrentLibMeshElement();
}

void MechanicsLinearFEOperator::postAssembly()
{
    d_materialModel->postLinearAssembly();
    d_matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}

void MechanicsLinearFEOperator::printStressAndStrain( AMP::LinearAlgebra::Vector::shared_ptr disp,
                                                      const std::string &fname )
{
    FILE *fp = fopen( fname.c_str(), "w" );
    fprintf( fp,
             "x, y, z, Stresses(11, 22, 33, 23, 13, 12), Strains(11, 22, 33, 23, 13, 12) \n\n" );

    disp->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    d_materialModel->preLinearAssembly();

    auto el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    auto end_el = el.end();

    for ( ; el != end_el; ++el ) {
        d_currNodes               = el->getElements( AMP::Mesh::GeomType::Vertex );
        size_t numNodesInCurrElem = d_currNodes.size();

        getDofIndicesForCurrentElement();

        createCurrentLibMeshElement();

        std::vector<double> elementInputVector( 3 * numNodesInCurrElem );
        for ( size_t r = 0; r < numNodesInCurrElem; r++ ) {
            for ( size_t d = 0; d < 3; d++ ) {
                elementInputVector[( 3 * r ) + d] = disp->getValueByGlobalID( d_dofIndices[r][d] );
            } // end for d
        } // end for r

        d_mechLinElem->initializeForCurrentElement( d_currElemPtr, d_materialModel );
        d_mechLinElem->printStressAndStrain( fp, elementInputVector );

        destroyCurrentLibMeshElement();
    } // end for el

    d_materialModel->postLinearAssembly();
    fprintf( fp, "\n\n" );
    fclose( fp );
}

void MechanicsLinearFEOperator::getDofIndicesForCurrentElement()
{
    d_dofIndices.resize( d_currNodes.size() );
    for ( size_t j = 0; j < d_currNodes.size(); j++ ) {
        d_inDofMap->getDOFs( d_currNodes[j]->globalID(), d_dofIndices[j] );
    } // end of j
}
} // namespace AMP::Operator
