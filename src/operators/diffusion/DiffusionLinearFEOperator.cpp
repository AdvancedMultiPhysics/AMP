#include "DiffusionLinearFEOperator.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Operator {

DiffusionLinearFEOperator::DiffusionLinearFEOperator(
    std::shared_ptr<const OperatorParameters> inParams )
    : LinearFEOperator( inParams )
{
    auto params = std::dynamic_pointer_cast<const DiffusionLinearFEOperatorParameters>( inParams );
    AMP_INSIST( params, "NULL parameter" );

    d_diffLinElem = std::dynamic_pointer_cast<DiffusionLinearElement>( d_elemOp );

    AMP_INSIST( d_diffLinElem, "d_elemOp is not of type DiffusionLinearElement" );

    d_useConstantTemperature   = params->d_db->getWithDefault<bool>( "FixedTemperature", false );
    d_useConstantConcentration = params->d_db->getWithDefault<bool>( "FixedConcentration", false );
    d_useConstantBurnup        = params->d_db->getWithDefault<bool>( "FixedBurnup", false );

    auto inpVar     = params->d_db->getString( "InputVariable" );
    d_inputVariable = std::make_shared<AMP::LinearAlgebra::Variable>( inpVar );

    auto outVar      = params->d_db->getString( "OutputVariable" );
    d_outputVariable = std::make_shared<AMP::LinearAlgebra::Variable>( outVar );

    reset( params );
}


void DiffusionLinearFEOperator::set_temperature(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> temperature )
{
    d_temperature = temperature;
}
void DiffusionLinearFEOperator::set_concentration(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> concentration )
{
    d_concentration = concentration;
}
void DiffusionLinearFEOperator::set_burnup(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> burnup )
{
    d_burnup = burnup;
}


void DiffusionLinearFEOperator::preAssembly( std::shared_ptr<const OperatorParameters> oparams )
{
    auto params = std::dynamic_pointer_cast<const DiffusionLinearFEOperatorParameters>( oparams );

    if ( d_iDebugPrintInfoLevel > 7 ) {
        AMP::pout << "DiffusionLinearFEOperator::preAssembly, entering" << std::endl;
    }

    d_transportModel = params->d_transportModel;

    for ( auto [name, vec] : params->d_inputVecs ) {
        if ( name == "temperature" )
            d_temperature = vec;
        else if ( name == "concentration" )
            d_concentration = vec;
        else if ( name == "burnup" )
            d_burnup = vec;
        else
            AMP_ERROR( "Unknown input vector: " + name );
    }

    d_matrix->zero();

    d_transportModel->preLinearAssembly();

    if ( d_iDebugPrintInfoLevel > 7 ) {
        AMP::pout << "DiffusionLinearFEOperator::preAssembly, leaving" << std::endl;
    }
}


void DiffusionLinearFEOperator::postAssembly()
{

    if ( d_iDebugPrintInfoLevel > 7 ) {
        AMP::pout << "DiffusionLinearFEOperator::postAssembly, entering" << std::endl;
    }

    d_transportModel->postLinearAssembly();

    d_matrix->makeConsistent();

    if ( d_iDebugPrintInfoLevel > 7 ) {
        AMP::pout << "DiffusionLinearFEOperator::postAssembly, leaving" << std::endl;
    }
}


void DiffusionLinearFEOperator::preElementOperation( const AMP::Mesh::MeshElement &elem )
{

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionLinearFEOperator::preElementOperation, entering" << std::endl;

    d_currNodes = elem.getElements( AMP::Mesh::GeomType::Vertex );

    unsigned int num_local_dofs = d_currNodes.size();

    d_elementStiffnessMatrix.resize( num_local_dofs );
    for ( unsigned int r = 0; r < num_local_dofs; r++ ) {
        d_elementStiffnessMatrix[r].resize( num_local_dofs );
        for ( unsigned int c = 0; c < num_local_dofs; c++ ) {
            d_elementStiffnessMatrix[r][c] = 0;
        }
    }

    std::vector<double> localTemperature;
    std::vector<double> localConcentration;
    std::vector<double> localBurnup;

    if ( !d_useConstantTemperature && d_temperature ) {
        localTemperature.resize( num_local_dofs );
        auto DOF = d_temperature->getDOFManager();
        std::vector<size_t> dofs;
        for ( size_t r = 0; r < d_currNodes.size(); r++ ) {
            DOF->getDOFs( d_currNodes[r].globalID(), dofs );
            AMP_ASSERT( dofs.size() == 1 );
            localTemperature[r] = d_temperature->getValueByGlobalID( dofs[0] );
        }
    }


    if ( !d_useConstantConcentration && d_concentration ) {
        localConcentration.resize( num_local_dofs );
        auto DOF = d_concentration->getDOFManager();
        std::vector<size_t> dofs;
        for ( size_t r = 0; r < d_currNodes.size(); r++ ) {
            DOF->getDOFs( d_currNodes[r].globalID(), dofs );
            AMP_ASSERT( dofs.size() == 1 );
            localConcentration[r] = d_concentration->getValueByGlobalID( dofs[0] );
        }
    }

    if ( !d_useConstantBurnup && d_burnup ) {
        localBurnup.resize( num_local_dofs );
        auto DOF = d_burnup->getDOFManager();
        std::vector<size_t> dofs;
        for ( size_t r = 0; r < d_currNodes.size(); r++ ) {
            DOF->getDOFs( d_currNodes[r].globalID(), dofs );
            AMP_ASSERT( dofs.size() == 1 );
            localBurnup[r] = d_burnup->getValueByGlobalID( dofs[0] );
        }
    }

    createCurrentLibMeshElement();

    d_diffLinElem->initializeForCurrentElement( d_currElemPtr, d_transportModel );

    d_diffLinElem->setElementStiffnessMatrix( d_elementStiffnessMatrix );

    d_diffLinElem->setElementVectors(
        num_local_dofs, localTemperature, localConcentration, localBurnup );

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionLinearFEOperator::preElementOperation, leaving" << std::endl;
}


void DiffusionLinearFEOperator::postElementOperation()
{

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionLinearFEOperator::postElementOperation, entering" << std::endl;

    std::vector<size_t> d_dofIndices( d_currNodes.size() ), dofs( 1 );
    for ( size_t i = 0; i < d_currNodes.size(); i++ ) {
        d_inDofMap->getDOFs( d_currNodes[i].globalID(), dofs );
        AMP_ASSERT( dofs.size() == 1 );
        d_dofIndices[i] = dofs[0];
    }

    for ( size_t r = 0; r < d_dofIndices.size(); r++ ) {
        for ( size_t c = 0; c < d_dofIndices.size(); c++ ) {
            d_matrix->addValueByGlobalID(
                d_dofIndices[r], d_dofIndices[c], d_elementStiffnessMatrix[r][c] );
        }
    }

    destroyCurrentLibMeshElement();

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionLinearFEOperator::postElementOperation, leaving" << std::endl;
}


std::shared_ptr<DiffusionTransportModel> DiffusionLinearFEOperator::getTransportModel()
{
    return d_transportModel;
}
} // namespace AMP::Operator
