#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include "ProfilerApp.h"

#include <cstring>

namespace AMP::Operator {


static std::shared_ptr<const VolumeIntegralOperatorParameters>
convert( std::shared_ptr<const OperatorParameters> params )
{
    if ( std::dynamic_pointer_cast<const VolumeIntegralOperatorParameters>( params ) )
        return std::dynamic_pointer_cast<const VolumeIntegralOperatorParameters>( params );
    auto db = params->d_db;
    // first create a source physics model
    std::shared_ptr<AMP::Database> model_db;
    if ( db->keyExists( "LocalModel" ) )
        model_db = db->getDatabase( "LocalModel" );
    else if ( db->keyExists( "SourcePhysicsModel" ) )
        model_db = db->getDatabase( "SourcePhysicsModel" );
    std::shared_ptr<ElementPhysicsModel> elementPhysicsModel;
    if ( model_db )
        elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel( model_db );
    auto sourcePhysicsModel = std::dynamic_pointer_cast<SourcePhysicsModel>( elementPhysicsModel );
    // next create a ElementOperation object
    AMP_INSIST( db->keyExists( "SourceElement" ), "Key ''SourceElement'' is missing!" );
    auto sourceElem =
        ElementOperationFactory::createElementOperation( db->getDatabase( "SourceElement" ) );
    // now create the source operator parameters
    auto volumeIntegralParameters    = std::make_shared<VolumeIntegralOperatorParameters>( db );
    volumeIntegralParameters->d_Mesh = params->d_Mesh;
    volumeIntegralParameters->d_sourcePhysicsModel = sourcePhysicsModel;
    volumeIntegralParameters->d_elemOp             = sourceElem;
    return volumeIntegralParameters;
}


VolumeIntegralOperator::VolumeIntegralOperator( std::shared_ptr<const OperatorParameters> params )
    : VolumeIntegralOperator( convert( params ), true )
{
}
VolumeIntegralOperator::VolumeIntegralOperator(
    std::shared_ptr<const VolumeIntegralOperatorParameters> params, bool )
    : NonlinearFEOperator( params )
{
    AMP_INSIST( params, "NULL parameter!" );
    AMP_INSIST( params->d_db, "NULL database!" );

    d_srcNonlinElem = std::dynamic_pointer_cast<SourceNonlinearElement>( d_elemOp );
    AMP_INSIST( d_srcNonlinElem, "d_elemOp is not of type SourceNonlinearElement" );

    if ( params->d_sourcePhysicsModel ) {
        d_sourcePhysicsModel = params->d_sourcePhysicsModel;
    } else if ( params->d_db->keyExists( "SourcePhysicsModel" ) ) {
        d_sourcePhysicsModel = params->d_sourcePhysicsModel;
    }

    auto primaryDb = params->d_db->getDatabase( "ActiveInputVariables" );

    int numPrimaryVariables   = params->d_db->getScalar<int>( "Number_Active_Variables" );
    int numAuxillaryVariables = params->d_db->getScalar<int>( "Number_Auxillary_Variables" );

    d_inpVariables.reset( new AMP::LinearAlgebra::MultiVariable( "myInpVar" ) );
    d_auxVariables.reset( new AMP::LinearAlgebra::MultiVariable( "myAuxVar" ) );

    for ( int i = 0; i < numPrimaryVariables; i++ ) {
        std::shared_ptr<AMP::LinearAlgebra::Variable> dummyVar;
        d_inpVariables->add( dummyVar );
    }

    d_inVec.resize( numPrimaryVariables );

    AMP_INSIST( ( numAuxillaryVariables == 0 ),
                "Verify this works before using it; the Interface "
                "to SourcePhysicsModel.h does not appear to be "
                "complete." );
    if ( numAuxillaryVariables > 0 ) {
        AMP_INSIST( params->d_auxVec, "NULL Auxillary Vector!" );
    }
    d_multiAuxPtr = params->d_auxVec;
    d_auxVec.resize( numAuxillaryVariables );

    for ( int var = 0; var < numPrimaryVariables; var++ ) {
        char key[100];
        snprintf( key, sizeof key, "ActiveVariable_%d", (int) var );
        std::string varName = primaryDb->getString( key );
        auto inpVar         = std::make_shared<AMP::LinearAlgebra::Variable>( varName );
        d_inpVariables->setVariable( var, inpVar );
    }

    std::string outVar = params->d_db->getString( "OutputVariable" );
    d_outVariable.reset( new AMP::LinearAlgebra::Variable( outVar ) );

    d_isInputType =
        params->d_db->getWithDefault<std::string>( "InputVariableType", "IntegrationPointScalar" );

    // d_bMatrixAndVectorsCloned=false;

    init( params );
}


void VolumeIntegralOperator::preAssembly( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                          AMP::LinearAlgebra::Vector::shared_ptr r )
{
    PROFILE( "preAssembly", 1 );
    AMP_INSIST( ( u ), "NULL Input Vector" );

    AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
    AMP::LinearAlgebra::Vector::const_shared_ptr meshSubsetPrimary, meshSubsetAuxillary;

    if ( d_inpVariables->numVariables() > 0 )
        meshSubsetPrimary = u->select( meshSelector );
    for ( size_t var = 0; var < d_inpVariables->numVariables(); var++ ) {
        auto primaryVariable = d_inpVariables->getVariable( var );
        d_inVec[var]         = meshSubsetPrimary->subsetVectorForVariable( primaryVariable );
        AMP_ASSERT( d_inVec[var] );
        AMP_ASSERT( d_inVec[var]->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
    }

    if ( d_auxVariables->numVariables() > 0 )
        meshSubsetAuxillary = d_multiAuxPtr->select( meshSelector );
    for ( size_t var = 0; var < d_auxVariables->numVariables(); var++ ) {
        auto auxillaryVariable = d_auxVariables->getVariable( var );
        d_auxVec[var]          = meshSubsetAuxillary->subsetVectorForVariable( auxillaryVariable );
        AMP_ASSERT( d_auxVec[var] );
        AMP_ASSERT( d_auxVec[var]->getUpdateStatus() ==
                    AMP::LinearAlgebra::UpdateState::UNCHANGED );
    }

    // subsetOutputVector is from Operator.h
    d_outVec = this->subsetOutputVector( r );
    d_outVec->zero();

    if ( d_inpVariables->numVariables() > 0 ) {
        d_elementDofMap = d_inVec[0]->getDOFManager();
    } else if ( d_auxVariables->numVariables() > 0 ) {
        d_elementDofMap = d_auxVec[0]->getDOFManager();
    }

    d_nodeDofMap = d_outVec->getDOFManager();

    if ( d_isInputType == "NodalScalar" ) {
        for ( unsigned int var = 0; var < d_inpVariables->numVariables(); var++ ) {
            auto tmp = d_inVec[var]->getDOFManager()->getIterator()->globalID();
            if ( tmp.type() != AMP::Mesh::GeomType::Vertex )
                AMP_ERROR( "Input vector isn't really a NodalScalar" );
        }
        for ( unsigned int var = 0; var < d_auxVariables->numVariables(); var++ ) {
            auto tmp = d_auxVec[var]->getDOFManager()->getIterator()->globalID();
            if ( tmp.type() != AMP::Mesh::GeomType::Vertex )
                AMP_ERROR( "aux vector isn't really a NodalScalar" );
        }
    }
}


void VolumeIntegralOperator::preElementOperation( const AMP::Mesh::MeshElement &elem )
{
    PROFILE( "preElementOperation", 5 );
    d_currNodes = elem.getElements( AMP::Mesh::GeomType::Vertex );

    std::vector<size_t> elemDofIds;
    d_elementDofMap->getDOFs( elem.globalID(), elemDofIds );

    getNodeDofIndicesForCurrentElement();

    std::vector<std::vector<double>> elementInputVectors( d_inpVariables->numVariables() );
    std::vector<std::vector<double>> elementAuxVectors( d_auxVariables->numVariables() );

    if ( d_isInputType == "IntegrationPointScalar" ) {
        AMP_INSIST(
            !elemDofIds.empty(),
            "d_elementDofMap does not contain element, but type is IntegrationPointScalar" );
        for ( unsigned int var = 0; var < d_inpVariables->numVariables(); var++ ) {
            elementInputVectors[var].resize( elemDofIds.size() );
            d_inVec[var]->getValuesByGlobalID(
                elemDofIds.size(), &elemDofIds[0], &elementInputVectors[var][0] );
        }
        for ( unsigned int var = 0; var < d_auxVariables->numVariables(); var++ ) {
            elementAuxVectors[var].resize( elemDofIds.size() );
            d_auxVec[var]->getValuesByGlobalID(
                elemDofIds.size(), &elemDofIds[0], &elementAuxVectors[var][0] );
        }
    } else if ( d_isInputType == "NodalScalar" ) {
        AMP_INSIST( elemDofIds.empty(),
                    "d_elementDofMap contains elements, but type is NodalScalar" );
        for ( unsigned int var = 0; var < d_inpVariables->numVariables(); var++ ) {
            elementInputVectors[var].resize( d_dofIndices.size() );
            d_inVec[var]->getValuesByGlobalID(
                d_dofIndices.size(), &d_dofIndices[0], &elementInputVectors[var][0] );
        }
        for ( unsigned int var = 0; var < d_auxVariables->numVariables(); var++ ) {
            elementAuxVectors[var].resize( d_dofIndices.size() );
            d_auxVec[var]->getValuesByGlobalID(
                d_dofIndices.size(), &d_dofIndices[0], &elementAuxVectors[var][0] );
        }
    }

    d_elementOutputVector.resize( d_dofIndices.size() );
    // Reinitialize the std::vector to zero (resize does not do this)
    for ( unsigned int i = 0; i < d_dofIndices.size(); i++ )
        d_elementOutputVector[i] = 0.0;

    d_srcNonlinElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                  d_sourcePhysicsModel );

    d_srcNonlinElem->setElementVectors(
        elementInputVectors, elementAuxVectors, d_elementOutputVector );
}


void VolumeIntegralOperator::postElementOperation()
{
    PROFILE( "postElementOperation", 5 );
    d_outVec->addValuesByGlobalID(
        d_dofIndices.size(), &d_dofIndices[0], &d_elementOutputVector[0] );
}


void VolumeIntegralOperator::postAssembly()
{
    PROFILE( "postAssembly", 1 );
    d_outVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}


void VolumeIntegralOperator::init( std::shared_ptr<const VolumeIntegralOperatorParameters> )
{
    auto el = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    d_srcNonlinElem->setElementFlags( d_isInputType );
    for ( d_currElemIdx = 0; d_currElemIdx < el.size(); ++d_currElemIdx, ++el ) {
        d_currNodes = el->getElements( AMP::Mesh::GeomType::Vertex );
        d_srcNonlinElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                      d_sourcePhysicsModel );
    } // end for el
    d_currElemIdx = static_cast<unsigned int>( -1 );
}


void VolumeIntegralOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );
    d_outVec.reset();
}


std::shared_ptr<OperatorParameters>
VolumeIntegralOperator::getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u )
{
    auto tmp_db = std::make_shared<AMP::Database>( "Dummy" );
    tmp_db->putScalar( "name", "VolumeIntegralOperator" );
    auto outParams = std::make_shared<VolumeIntegralOperatorParameters>( tmp_db );

    outParams->d_sourcePhysicsModel = d_sourcePhysicsModel;
    outParams->d_pVector            = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( u );
    return outParams;
}


void VolumeIntegralOperator::getNodeDofIndicesForCurrentElement()
{
    d_dofIndices.resize( d_currNodes.size() );
    std::vector<size_t> dofs;
    for ( unsigned int j = 0; j < d_currNodes.size(); j++ ) {
        d_nodeDofMap->getDOFs( d_currNodes[j].globalID(), dofs );
        AMP_ASSERT( dofs.size() == 1 );
        d_dofIndices[j] = dofs[0];
    } // end of j
}
} // namespace AMP::Operator
