#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementOperationParameters.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperatorParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include "ProfilerApp.h"

#include <cstring>
#include <iostream>


namespace AMP::Operator {


static std::vector<std::string> getActiveVariables( std::shared_ptr<const AMP::Database> db,
                                                    const std::string &key )
{
    std::vector<std::string> vars;
    if ( db->isDatabase( key ) ) {
        auto activeDB = db->getDatabase( key );
        for ( auto key2 : activeDB->getAllKeys() )
            vars.push_back( activeDB->getString( key2 ) );
    } else {
        vars = db->getVector<std::string>( key );
    }
    return vars;
}


std::shared_ptr<AMP::LinearAlgebra::Variable>
DiffusionNonlinearFEOperator::createInputVariable( const std::string &name, int varId )
{
    if ( varId == -1 ) {
        return d_inpVariables->clone( name );
    } else {
        return d_inpVariables->getVariable( varId )->clone( name );
    }
}


std::shared_ptr<AMP::LinearAlgebra::Variable>
DiffusionNonlinearFEOperator::createOutputVariable( const std::string &name, int varId )
{
    (void) varId;
    return d_outVariable->clone( name );
}


std::shared_ptr<AMP::LinearAlgebra::Variable> DiffusionNonlinearFEOperator::getInputVariable() const
{
    return d_inpVariables;
}


std::shared_ptr<AMP::LinearAlgebra::Variable>
DiffusionNonlinearFEOperator::getOutputVariable() const
{
    return d_outVariable;
}


std::shared_ptr<DiffusionTransportModel> DiffusionNonlinearFEOperator::getTransportModel()
{
    return d_transportModel;
}


std::map<std::string, AMP::LinearAlgebra::Vector::shared_ptr>
DiffusionNonlinearFEOperator::getFrozen()
{
    std::map<std::string, AMP::LinearAlgebra::Vector::shared_ptr> frozen;
    for ( auto &[name, data] : d_active ) {
        if ( data.isFrozen )
            frozen[name] = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( data.frozen );
    }
    return frozen;
}


void DiffusionNonlinearFEOperator::setVector( const std::string &name,
                                              AMP::LinearAlgebra::Vector::shared_ptr frozenVec )
{
    auto it = d_active.find( name );
    AMP_INSIST( it != d_active.end(), "Variable " + name + " not found" );
    AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
    auto mVec = frozenVec->select( meshSelector );
    auto vec  = mVec->subsetVectorForVariable( name );
    vec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    it->second.frozen = vec;
}


static std::shared_ptr<const DiffusionNonlinearFEOperatorParameters>
convert( std::shared_ptr<const OperatorParameters> params )
{
    if ( std::dynamic_pointer_cast<const DiffusionNonlinearFEOperatorParameters>( params ) )
        return std::dynamic_pointer_cast<const DiffusionNonlinearFEOperatorParameters>( params );
    auto db = params->d_db;
    // first create a DiffusionTransportModel
    auto model_db            = db->getDatabase( "LocalModel" );
    auto elementPhysicsModel = ElementPhysicsModelFactory::createElementPhysicsModel( model_db );
    auto transportModel = std::dynamic_pointer_cast<DiffusionTransportModel>( elementPhysicsModel );
    AMP_INSIST( transportModel, "NULL transport model" );
    // next create a ElementOperation object
    auto diffusionElem =
        ElementOperationFactory::createElementOperation( db->getDatabase( "DiffusionElement" ) );
    // now create the nonlinear diffusion operator parameters
    AMP_ASSERT( db->getString( "name" ) == "DiffusionNonlinearFEOperator" );
    auto diffusionOpParams    = std::make_shared<DiffusionNonlinearFEOperatorParameters>( db );
    diffusionOpParams->d_Mesh = params->d_Mesh;
    diffusionOpParams->d_transportModel = transportModel;
    diffusionOpParams->d_elemOp         = diffusionElem;
    // populate the parameters with frozen active variable vectors
    diffusionOpParams->d_FrozenVecs.clear();
    auto scalarDOF = AMP::Discretization::simpleDOFManager::create(
        params->d_Mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    for ( auto name : getActiveVariables( db, "ActiveInputVariables" ) ) {
        auto var    = std::make_shared<AMP::LinearAlgebra::Variable>( name );
        auto memLoc = AMP::Utilities::memoryLocationFromString(
            db->getWithDefault<std::string>( "MemoryLocation", "host" ) );
        auto vec = AMP::LinearAlgebra::createVector( scalarDOF, var, true, memLoc );
        if ( db->getWithDefault<bool>( "Freeze" + name, false ) )
            diffusionOpParams->d_FrozenVecs[name] = vec;
    }
    return diffusionOpParams;
}


DiffusionNonlinearFEOperator::DiffusionNonlinearFEOperator(
    std::shared_ptr<const OperatorParameters> params )
    : DiffusionNonlinearFEOperator( convert( params ), true )
{
}
DiffusionNonlinearFEOperator::DiffusionNonlinearFEOperator(
    std::shared_ptr<const DiffusionNonlinearFEOperatorParameters> params, bool )
    : NonlinearFEOperator( params )
{
    AMP_INSIST( params, "NULL parameter!" );

    d_diffNonlinElem = std::dynamic_pointer_cast<DiffusionNonlinearElement>( d_elemOp );

    AMP_INSIST( d_diffNonlinElem, "d_elemOp is not of type DiffusionNonlinearElement" );

    d_transportModel = params->d_transportModel;

    auto activeVariables = getActiveVariables( params->d_db, "ActiveInputVariables" );
    for ( auto &name : activeVariables ) {
        InputVectorStruct data;
        data.isFrozen  = params->d_db->getWithDefault<bool>( "Freeze" + name, false );
        d_active[name] = data;
    }

    d_PrincipalVariable = params->d_db->getString( "PrincipalVariable" );
    if ( d_active.find( d_PrincipalVariable ) == d_active.end() )
        AMP_ERROR( "must have Principal_Variable active" );
    d_diffNonlinElem->setPrincipalVariable( d_PrincipalVariable );

    resetFrozen( params );

    d_inpVariables.reset( new AMP::LinearAlgebra::MultiVariable( "InputVariables" ) );

    for ( auto &[name, data] : d_active ) {
        auto dummyVar = std::make_shared<AMP::LinearAlgebra::Variable>( name );
        d_inpVariables->add( dummyVar );
        if ( data.isFrozen ) {
            data.vec = data.frozen;
        }
        if ( data.vec )
            AMP_ASSERT( data.vec->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
    }

    d_outVariable.reset(
        new AMP::LinearAlgebra::Variable( params->d_db->getString( "OutputVariable" ) ) );

    init( params );
}


void DiffusionNonlinearFEOperator::preAssembly( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                                AMP::LinearAlgebra::Vector::shared_ptr r )
{
    PROFILE( "preAssembly", 1 );

    AMP_INSIST( u, "NULL Input Vector!" );
    AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
    auto u_meshVec = u->select( meshSelector );

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionNonlinearFEOperator::preAssembly, entering" << std::endl;

    for ( auto &[name, data] : d_active ) {
        if ( data.isFrozen ) {
            data.vec = data.frozen;
            std::const_pointer_cast<AMP::LinearAlgebra::Vector>( data.frozen )
                ->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        } else {
            data.vec = u_meshVec->subsetVectorForVariable( name );
        }

        AMP_ASSERT( data.vec );
        AMP_ASSERT( data.vec->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
        if ( d_iDebugPrintInfoLevel > 5 )
            std::cout << "Max Value inside preAssembly: " << data.vec->max() << std::endl;
    }

    d_outVec = subsetOutputVector( r );
    d_outVec->zero();

    d_transportModel->preNonlinearAssembly();

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionNonlinearFEOperator::preAssembly, leaving" << std::endl;
}


void DiffusionNonlinearFEOperator::postAssembly()
{
    PROFILE( "postAssembly", 1 );

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionNonlinearFEOperator::postAssembly, entering" << std::endl;

    d_transportModel->postNonlinearAssembly();
    d_outVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionNonlinearFEOperator::postAssembly, leaving" << std::endl;
}


void DiffusionNonlinearFEOperator::preElementOperation( const AMP::Mesh::MeshElement &elem )
{
    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionNonlinearFEOperator::preElementOperation, entering" << std::endl;

    std::map<std::string, std::vector<double>> elementInputVectors;

    d_currNodes = elem.getElements( AMP::Mesh::GeomType::Vertex );
    std::vector<AMP::Mesh::MeshElementID> ids( d_currNodes.size() );
    for ( size_t i = 0; i < d_currNodes.size(); i++ )
        ids[i] = d_currNodes[i].globalID();

    std::vector<size_t> dofs( d_currNodes.size() );
    for ( auto &[name, data] : d_active ) {
        auto DOF = data.vec->getDOFManager();
        DOF->getDOFs( ids, dofs );
        AMP_ASSERT( dofs.size() == d_currNodes.size() );
        std::vector<double> tmp( dofs.size() );
        data.vec->getValuesByGlobalID( dofs.size(), &dofs[0], &tmp[0] );
        elementInputVectors[name] = std::move( tmp );
    }

    d_elementOutputVector.resize( d_currNodes.size() );
    for ( unsigned int i = 0; i < d_currNodes.size(); i++ )
        d_elementOutputVector[i] = 0.0;

    d_diffNonlinElem->setElementVectors( std::move( elementInputVectors ), d_elementOutputVector );

    d_diffNonlinElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                   d_transportModel );

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionNonlinearFEOperator::preElementOperation, leaving" << std::endl;
}


void DiffusionNonlinearFEOperator::postElementOperation()
{
    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionNonlinearFEOperator::postElementOperation, entering" << std::endl;

    std::vector<AMP::Mesh::MeshElementID> ids( d_currNodes.size() );
    for ( size_t i = 0; i < d_currNodes.size(); i++ )
        ids[i] = d_currNodes[i].globalID();

    auto DOF = d_outVec->getDOFManager();
    std::vector<size_t> dofs( d_currNodes.size() );
    DOF->getDOFs( ids, dofs );
    AMP_ASSERT( dofs.size() == d_currNodes.size() );

    d_outVec->addValuesByGlobalID( dofs.size(), &dofs[0], &d_elementOutputVector[0] );

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionNonlinearFEOperator::postElementOperation, leaving" << std::endl;
}


void DiffusionNonlinearFEOperator::init(
    std::shared_ptr<const DiffusionNonlinearFEOperatorParameters> )
{
    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionNonlinearFEOperator::init, entering" << std::endl;

    auto el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    auto end_el = el.end();
    for ( d_currElemIdx = 0; el != end_el; ++el, ++d_currElemIdx ) {
        d_currNodes = el->getElements( AMP::Mesh::GeomType::Vertex );
        d_diffNonlinElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                       d_transportModel );
        d_diffNonlinElem->initTransportModel();
    }
    d_currElemIdx = static_cast<unsigned int>( -1 );

    if ( d_iDebugPrintInfoLevel > 7 )
        AMP::pout << "DiffusionNonlinearFEOperator::init, leaving" << std::endl;
}


void DiffusionNonlinearFEOperator::reset( std::shared_ptr<const OperatorParameters> inParams )
{
    AMP_ASSERT( inParams );

    auto params =
        std::dynamic_pointer_cast<const DiffusionNonlinearFEOperatorParameters>( inParams );

    for ( auto [name, vec] : params->d_FrozenVecs ) {
        auto it = d_active.find( name );
        if ( it != d_active.end() ) {
            it->second.vec = vec;
            AMP_ASSERT( vec->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
        }
    }

    resetFrozen( params );
    for ( auto [name, vec] : params->d_FrozenVecs ) {
        auto it = d_active.find( name );
        if ( it == d_active.end() )
            continue;
        if ( it->second.isFrozen ) {
            it->second.vec = it->second.frozen;
            AMP_ASSERT( it->second.vec->getUpdateStatus() ==
                        AMP::LinearAlgebra::UpdateState::UNCHANGED );
        }
    }
}


std::shared_ptr<OperatorParameters> DiffusionNonlinearFEOperator::getJacobianParameters(
    AMP::LinearAlgebra::Vector::const_shared_ptr u )
{
    auto db = std::make_shared<AMP::Database>( "Dummy" );
    AMP::LinearAlgebra::Vector::const_shared_ptr u_meshVec;
    if ( u ) {
        AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
        u_meshVec = u->select( meshSelector );
    }

    // set up a database for the linear operator params
    db->putScalar( "name", "DiffusionLinearFEOperator" );
    db->putScalar( "InputVariable", d_PrincipalVariable );
    db->putScalar( "OutputVariable", d_outVariable->getName() );
    for ( auto &[name, data] : d_active )
        db->putScalar( "Fixed" + name, true );

    // create the linear operator params
    auto outParams = std::make_shared<DiffusionLinearFEOperatorParameters>( db );

    // create the linear element object
    auto elem_db = std::make_shared<AMP::Database>( "Dummy" );
    db->putScalar( "TransportAtGaussPoints", d_diffNonlinElem->getTransportAtGauss() );
    auto eparams       = std::make_shared<ElementOperationParameters>( elem_db );
    auto linearElement = std::make_shared<DiffusionLinearElement>( eparams );

    // add miscellaneous to output parameters
    auto createDOFManager = []( std::shared_ptr<AMP::Mesh::Mesh> mesh ) {
        return AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    };
    outParams->d_transportModel = d_transportModel;
    outParams->d_elemOp         = linearElement;
    outParams->d_Mesh           = d_Mesh;
    outParams->d_inDofMap       = createDOFManager( d_Mesh );
    outParams->d_outDofMap      = outParams->d_inDofMap;

    // add variables to parameters
    for ( auto &[name, data] : d_active ) {
        if ( data.isFrozen ) {
            outParams->d_inputVecs[name] =
                std::const_pointer_cast<AMP::LinearAlgebra::Vector>( data.frozen );
        } else if ( u_meshVec ) {
            auto vec = std::const_pointer_cast<AMP::LinearAlgebra::Vector>(
                u_meshVec->subsetVectorForVariable( name ) );
            outParams->d_inputVecs[name] = vec;
        }
    }

    return outParams;
}


void DiffusionNonlinearFEOperator::resetFrozen(
    std::shared_ptr<const DiffusionNonlinearFEOperatorParameters> params )
{
    for ( auto &[name, data] : d_active ) {
        if ( data.isFrozen ) {
            data.frozen.reset();
            auto it = params->d_FrozenVecs.find( name );
            if ( it != params->d_FrozenVecs.end() )
                data.frozen = it->second;
        }
    }
}


bool DiffusionNonlinearFEOperator::isValidVector( AMP::LinearAlgebra::Vector::const_shared_ptr u )
{
    auto property = d_transportModel->getProperty();
    auto names    = property->get_arguments();
    size_t nnames = names.size();
    std::string argname;
    bool found = false;
    for ( size_t i = 0; i < nnames; i++ ) {
        if ( names[i] == d_PrincipalVariable ) {
            argname = d_PrincipalVariable;
            found   = true;
            break;
        }
    }

    bool result = true;
    AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
    auto u_meshVec = u->select( meshSelector );
    if ( found ) {
        auto uinp = u_meshVec->subsetVectorForVariable( d_PrincipalVariable );
        std::vector<double> vals( uinp->getLocalSize() );
        size_t nit = 0;
        for ( auto &elem : *uinp ) {
            vals[nit] = elem;
            nit++;
        }
        result = property->in_range( argname, vals );
    }

    return result;
}


std::vector<std::string> DiffusionNonlinearFEOperator::getNonPrincipalVariableIds()
{
    std::vector<std::string> vars;
    for ( auto &[name, data] : d_active ) {
        if ( name != d_PrincipalVariable )
            vars.push_back( name );
    }
    return vars;
}


} // namespace AMP::Operator
