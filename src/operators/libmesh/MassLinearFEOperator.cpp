#include "AMP/operators/libmesh/MassLinearFEOperator.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Operator {


static std::shared_ptr<const MassLinearFEOperatorParameters>
convert( std::shared_ptr<const OperatorParameters> params )
{
    if ( std::dynamic_pointer_cast<const MassLinearFEOperatorParameters>( params ) )
        return std::dynamic_pointer_cast<const MassLinearFEOperatorParameters>( params );
    auto db = params->d_db;
    // first create a source physics model
    auto model_db            = db->getDatabase( "LocalModel" );
    auto elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel( model_db );
    auto densityModel        = std::dynamic_pointer_cast<MassDensityModel>( elementPhysicsModel );
    AMP_INSIST( densityModel, "NULL density model" );
    // next create a ElementOperation object
    auto densityLinElem =
        ElementOperationFactory::createElementOperation( db->getDatabase( "MassElement" ) );
    // now create the linear density operator
    AMP_ASSERT( db->getString( "name" ) == "MassLinearFEOperator" );
    auto scalarDofs = AMP::Discretization::simpleDOFManager::create(
        params->d_Mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto densityOpParams            = std::make_shared<MassLinearFEOperatorParameters>( db );
    densityOpParams->d_Mesh         = params->d_Mesh;
    densityOpParams->d_densityModel = densityModel;
    densityOpParams->d_elemOp       = densityLinElem;
    densityOpParams->d_inDofMap     = scalarDofs;
    densityOpParams->d_outDofMap    = scalarDofs;
    return densityOpParams;
}


MassLinearFEOperator::MassLinearFEOperator( std::shared_ptr<const OperatorParameters> params )
    : MassLinearFEOperator( convert( params ), true )
{
}
MassLinearFEOperator::MassLinearFEOperator(
    std::shared_ptr<const MassLinearFEOperatorParameters> params, bool )
    : LinearFEOperator( params )
{
    AMP_INSIST( params, "NULL parameter" );

    d_massLinElem = std::dynamic_pointer_cast<MassLinearElement>( d_elemOp );

    AMP_INSIST( ( ( d_massLinElem.get() ) != nullptr ),
                "d_elemOp is not of type MassLinearElement" );

    d_densityModel = params->d_densityModel;

    d_useConstantTemperature = params->d_db->keyExists( "FixedTemperature" );

    d_useConstantConcentration = params->d_db->keyExists( "FixedConcentration" );

    d_useConstantBurnup = params->d_db->keyExists( "FixedBurnup" );

    d_constantTemperatureValue = params->d_db->getWithDefault<double>( "FixedTemperature", 273.0 );

    d_constantConcentrationValue =
        params->d_db->getWithDefault<double>( "FixedConcentration", 0.0 );

    d_constantBurnupValue = params->d_db->getWithDefault<double>( "FixedBurnup", 0.0 );
    // d_inpVariable.reset(new AMP::Mesh::NodalScalarVariable("inpVar"));
    // d_outVariable.reset(new AMP::Mesh::NodalScalarVariable("outVar"));
    std::string inpVar = params->d_db->getString( "InputVariable" );
    d_inpVariable      = std::make_shared<AMP::LinearAlgebra::Variable>( inpVar );

    std::string outVar = params->d_db->getString( "OutputVariable" );
    d_outVariable      = std::make_shared<AMP::LinearAlgebra::Variable>( outVar );

    reset( params );
}


std::shared_ptr<AMP::LinearAlgebra::Variable> MassLinearFEOperator::getInputVariable() const
{
    return d_inpVariable;
}


std::shared_ptr<AMP::LinearAlgebra::Variable> MassLinearFEOperator::getOutputVariable() const
{
    return d_outVariable;
}


void MassLinearFEOperator::preAssembly( std::shared_ptr<const AMP::Operator::OperatorParameters> )
{
    d_matrix->zero();

    d_densityModel->preLinearAssembly();
}


void MassLinearFEOperator::postAssembly()
{
    d_densityModel->postLinearAssembly();

    d_matrix->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}


void MassLinearFEOperator::preElementOperation( const AMP::Mesh::MeshElement &elem )
{
    d_currNodes                 = elem.getElements( AMP::Mesh::GeomType::Vertex );
    unsigned int num_local_dofs = d_currNodes.size();

    d_elementMassMatrix.resize( num_local_dofs );
    for ( unsigned int r = 0; r < num_local_dofs; r++ ) {
        d_elementMassMatrix[r].resize( num_local_dofs );
        for ( unsigned int c = 0; c < num_local_dofs; c++ ) {
            d_elementMassMatrix[r][c] = 0;
        }
    }

    std::vector<double> localTemperature( num_local_dofs );
    std::vector<double> localConcentration( num_local_dofs );
    std::vector<double> localBurnup( num_local_dofs );
    std::vector<size_t> dofs;

    if ( d_useConstantTemperature ) {
        for ( size_t r = 0; r < d_currNodes.size(); r++ ) {
            localTemperature[r] = d_constantTemperatureValue;
        }
    } else {
        std::shared_ptr<AMP::Discretization::DOFManager> DOF = d_temperature->getDOFManager();
        for ( size_t r = 0; r < d_currNodes.size(); r++ ) {
            DOF->getDOFs( d_currNodes[r]->globalID(), dofs );
            AMP_ASSERT( dofs.size() == 1 );
            localTemperature[r] = d_temperature->getValueByGlobalID( dofs[0] );
        }
    }

    if ( d_useConstantConcentration ) {
        for ( size_t r = 0; r < d_currNodes.size(); r++ ) {
            localConcentration[r] = d_constantConcentrationValue;
        }
    } else {
        std::shared_ptr<AMP::Discretization::DOFManager> DOF = d_concentration->getDOFManager();
        for ( size_t r = 0; r < d_currNodes.size(); r++ ) {
            DOF->getDOFs( d_currNodes[r]->globalID(), dofs );
            AMP_ASSERT( dofs.size() == 1 );
            localConcentration[r] = d_concentration->getValueByGlobalID( dofs[0] );
        }
    }

    if ( d_useConstantBurnup ) {
        for ( size_t r = 0; r < d_currNodes.size(); r++ ) {
            localBurnup[r] = d_constantBurnupValue;
        }
    } else {
        std::shared_ptr<AMP::Discretization::DOFManager> DOF = d_burnup->getDOFManager();
        for ( size_t r = 0; r < d_currNodes.size(); r++ ) {
            DOF->getDOFs( d_currNodes[r]->globalID(), dofs );
            AMP_ASSERT( dofs.size() == 1 );
            localBurnup[r] = d_burnup->getValueByGlobalID( dofs[0] );
        }
    }

    createCurrentLibMeshElement();

    d_massLinElem->initializeForCurrentElement( d_currElemPtr, d_densityModel );

    d_massLinElem->setElementMassMatrix( d_elementMassMatrix );

    d_massLinElem->setElementVectors( localTemperature, localConcentration, localBurnup );
}


void MassLinearFEOperator::postElementOperation()
{
    PROFILE( "postElementOperation", 5 );

    std::vector<size_t> d_dofIndices( d_currNodes.size() ), dofs( 1 );
    for ( size_t i = 0; i < d_currNodes.size(); i++ ) {
        d_inDofMap->getDOFs( d_currNodes[i]->globalID(), dofs );
        AMP_ASSERT( dofs.size() == 1 );
        d_dofIndices[i] = dofs[0];
    }

    for ( size_t r = 0; r < d_dofIndices.size(); r++ ) {
        for ( size_t c = 0; c < d_dofIndices.size(); c++ ) {
            d_matrix->addValueByGlobalID(
                d_dofIndices[r], d_dofIndices[c], d_elementMassMatrix[r][c] );
        }
    }

    destroyCurrentLibMeshElement();
}
} // namespace AMP::Operator
