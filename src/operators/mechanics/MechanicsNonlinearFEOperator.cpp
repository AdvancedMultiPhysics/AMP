#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/operators/ElementOperationFactory.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/mechanics/MechanicsLinearElement.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperatorParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

#include "libmesh/cell_hex8.h"
#include "libmesh/node.h"


namespace AMP::Operator {


static std::shared_ptr<const MechanicsNonlinearFEOperatorParameters>
convert( std::shared_ptr<const OperatorParameters> params )
{
    if ( std::dynamic_pointer_cast<const MechanicsNonlinearFEOperatorParameters>( params ) )
        return std::dynamic_pointer_cast<const MechanicsNonlinearFEOperatorParameters>( params );
    auto db = params->d_db;
    // first create a MechanicsMaterialModel
    auto model_db = db->getDatabase( "LocalModel" );
    auto model    = std::dynamic_pointer_cast<MechanicsMaterialModel>(
        ElementPhysicsModelFactory::createElementPhysicsModel( model_db ) );
    // next create a ElementOperation object
    auto elem_db       = db->getDatabase( "MechanicsElement" );
    auto mechanicsElem = ElementOperationFactory::createElementOperation( elem_db );
    // now create the nonlinear mechanics operator parameters
    AMP_ASSERT( db->getString( "name" ) == "MechanicsNonlinearFEOperator" );
    auto scalarDofs = AMP::Discretization::simpleDOFManager::create(
        params->d_Mesh, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    auto vecDofs = AMP::Discretization::simpleDOFManager::create(
        params->d_Mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );
    auto myparams             = std::make_shared<MechanicsNonlinearFEOperatorParameters>( db );
    myparams->d_Mesh          = params->d_Mesh;
    myparams->d_materialModel = model;
    myparams->d_elemOp        = mechanicsElem;
    myparams->d_dofMap[Mechanics::DISPLACEMENT]         = vecDofs;
    myparams->d_dofMap[Mechanics::TEMPERATURE]          = scalarDofs;
    myparams->d_dofMap[Mechanics::BURNUP]               = scalarDofs;
    myparams->d_dofMap[Mechanics::OXYGEN_CONCENTRATION] = scalarDofs;
    myparams->d_dofMap[Mechanics::LHGR]                 = scalarDofs;
    return myparams;
}


MechanicsNonlinearFEOperator::MechanicsNonlinearFEOperator(
    std::shared_ptr<const OperatorParameters> params )
    : MechanicsNonlinearFEOperator( convert( params ), true )
{
}
MechanicsNonlinearFEOperator::MechanicsNonlinearFEOperator(
    std::shared_ptr<const MechanicsNonlinearFEOperatorParameters> params, bool )
    : NonlinearFEOperator( params )
{
    AMP_INSIST( params, "NULL parameter!" );
    AMP_INSIST( params->d_db, "NULL database!" );

    auto db = params->d_db;
    d_resetReusesRadialReturn =
        params->d_db->getWithDefault<bool>( "RESET_REUSES_RADIAL_RETURN", true );
    d_jacobianReusesRadialReturn =
        params->d_db->getWithDefault<bool>( "JACOBIAN_REUSES_RADIAL_RETURN", true );

    d_useUpdatedLagrangian = params->d_db->getWithDefault<bool>( "USE_UPDATED_LAGRANGIAN", false );

    if ( d_useUpdatedLagrangian ) {
        d_mechNULElem =
            std::dynamic_pointer_cast<MechanicsNonlinearUpdatedLagrangianElement>( d_elemOp );
        // next create a linear ElementOperation object
        AMP_INSIST( db->keyExists( "MechanicsLinearElement" ),
                    "Key ''MechanicsLinearElement'' is missing!" );
        d_mechLULElem = ElementOperationFactory::createElementOperation(
            db->getDatabase( "MechanicsLinearElement" ) );
    } else {
        d_mechNonlinElem = std::dynamic_pointer_cast<MechanicsNonlinearElement>( d_elemOp );
        // next create a linear ElementOperation object
        AMP_INSIST( db->keyExists( "MechanicsLinearElement" ),
                    "Key ''MechanicsLinearElement'' is missing!" );
        d_mechLinElem = ElementOperationFactory::createElementOperation(
            db->getDatabase( "MechanicsLinearElement" ) );
    }

    if ( d_useUpdatedLagrangian ) {
        AMP_INSIST( d_mechNULElem,
                    "d_elemOp is not of type MechanicsNonlinearUpdatedLagrangianElement" );
    } else {
        AMP_INSIST( d_mechNonlinElem, "d_elemOp is not of type MechanicsNonlinearElement" );
    }

    d_materialModel = params->d_materialModel;

    d_isActive.resize( Mechanics::TOTAL_NUMBER_OF_VARIABLES );
    d_isFrozen.resize( Mechanics::TOTAL_NUMBER_OF_VARIABLES );
    d_inVec.resize( Mechanics::TOTAL_NUMBER_OF_VARIABLES );
    if ( d_useUpdatedLagrangian ) {
        d_inVec_pre.resize(
            Mechanics::TOTAL_NUMBER_OF_VARIABLES ); // storage for variables at the previous config
    }

    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        d_dofMap[i] = ( params->d_dofMap )[i];
    } // end for i

    AMP_INSIST( params->d_db->keyExists( "ActiveInputVariables" ), "key not found" );
    std::shared_ptr<AMP::Database> activeInpVar_db =
        params->d_db->getDatabase( "ActiveInputVariables" );

    d_inpVariables.reset( new AMP::LinearAlgebra::MultiVariable( "myInpVar" ) );
    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        std::shared_ptr<AMP::LinearAlgebra::Variable> dummyVar;
        d_inpVariables->add( dummyVar );
    } // end for i

    AMP_INSIST( activeInpVar_db->keyExists( "DISPLACEMENT" ), "DISPLACEMENT must be active" );

    std::vector<std::string> keysForVariables( Mechanics::TOTAL_NUMBER_OF_VARIABLES );
    keysForVariables[Mechanics::DISPLACEMENT]         = "DISPLACEMENT";
    keysForVariables[Mechanics::TEMPERATURE]          = "TEMPERATURE";
    keysForVariables[Mechanics::BURNUP]               = "BURNUP";
    keysForVariables[Mechanics::OXYGEN_CONCENTRATION] = "OXYGEN_CONCENTRATION";
    keysForVariables[Mechanics::LHGR]                 = "LHGR";

    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        if ( i == Mechanics::DISPLACEMENT ) {
            d_isActive[i] = true;
            d_isFrozen[i] = false;
        } else {
            d_isActive[i] = activeInpVar_db->keyExists( keysForVariables[i] );
            d_isFrozen[i] =
                params->d_db->getWithDefault<bool>( "FREEZE_" + keysForVariables[i], true );
        }
        if ( d_isActive[i] ) {
            std::string varName = activeInpVar_db->getString( keysForVariables[i] );
            auto dummyVar       = std::make_shared<AMP::LinearAlgebra::Variable>( varName );
            d_inpVariables->setVariable( i, dummyVar );
            if ( d_isFrozen[i] ) {
                if ( params->d_FrozenVec[i] ) {
                    setVector( i, params->d_FrozenVec[i] );
                }
            }
        }
    } // end for i

    // memory for reference coordinates (used in UL formulation)
    // memory for variables in the previous config
    if ( d_useUpdatedLagrangian ) {
        d_refXYZ = AMP::LinearAlgebra::createVector(
            d_dofMap[Mechanics::DISPLACEMENT],
            d_inpVariables->getVariable( Mechanics::DISPLACEMENT ),
            true,
            d_memory_location );
        d_refXYZ->zero();
        for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
            if ( d_isActive[i] ) {
                d_inVec_pre[i] = AMP::LinearAlgebra::createVector(
                    d_dofMap[i], d_inpVariables->getVariable( i ), true, d_memory_location );
                d_inVec_pre[i]->zero();
            }
        }
    }

    if ( params->d_ReferenceTemperature ) {
        setReferenceTemperature( params->d_ReferenceTemperature );
    }

    AMP_INSIST( params->d_db->keyExists( "OutputVariable" ), "key not found" );
    std::string outVarName = params->d_db->getString( "OutputVariable" );
    d_outVariable.reset( new AMP::LinearAlgebra::Variable( outVarName ) );

    d_isInitialized = false;
}

void MechanicsNonlinearFEOperator::preAssembly( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                                std::shared_ptr<AMP::LinearAlgebra::Vector> r )
{
    AMP_INSIST( u, "NULL Input Vector" );

    if ( !d_isInitialized ) {
        init();
    }

    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        if ( d_isActive[i] ) {
            if ( !( d_isFrozen[i] ) ) {
                std::shared_ptr<AMP::LinearAlgebra::Variable> var =
                    d_inpVariables->getVariable( i );
                AMP::LinearAlgebra::Vector::const_shared_ptr vector = mySubsetVector( u, var );
                setVector( i, vector );
            }
        }
    } // end for i

    d_outVec = mySubsetVector( r, d_outVariable );
    d_outVec->zero();

    d_materialModel->preNonlinearAssembly();

    if ( d_useUpdatedLagrangian == true ) {
        d_mechNULElem->zeroOutGaussPointCount();
    }
}

void MechanicsNonlinearFEOperator::postAssembly()
{
    d_materialModel->postNonlinearAssembly();
    d_outVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
}

void MechanicsNonlinearFEOperator::preElementOperation( const AMP::Mesh::MeshElement &elem )
{
    d_currNodes                     = elem.getElements( AMP::Mesh::GeomType::Vertex );
    unsigned int numNodesInCurrElem = d_currNodes.size();

    AMP_ASSERT( numNodesInCurrElem == 8 );

    getDofIndicesForCurrentElement( Mechanics::DISPLACEMENT, d_dofIndices );

    std::vector<std::vector<size_t>> auxDofIds;
    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        if ( i != Mechanics::DISPLACEMENT ) {
            if ( d_isActive[i] ) {
                getDofIndicesForCurrentElement( i, auxDofIds );
                break;
            }
        }
    } // end for i

    std::vector<std::vector<double>> elementInputVectors( Mechanics::TOTAL_NUMBER_OF_VARIABLES );
    std::vector<std::vector<double>> elementInputVectors_pre(
        Mechanics::TOTAL_NUMBER_OF_VARIABLES );

    elementInputVectors[Mechanics::DISPLACEMENT].resize( 3 * numNodesInCurrElem );
    if ( d_useUpdatedLagrangian ) {
        elementInputVectors_pre[Mechanics::DISPLACEMENT].resize( 3 * numNodesInCurrElem );
    }
    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        if ( i != Mechanics::DISPLACEMENT ) {
            if ( d_isActive[i] ) {
                elementInputVectors[i].resize( numNodesInCurrElem );
                if ( d_useUpdatedLagrangian ) {
                    elementInputVectors_pre[i].resize( numNodesInCurrElem );
                }
            }
        }
    } // end for i

    std::vector<double> elementRefXYZ;
    elementRefXYZ.resize( 3 * numNodesInCurrElem );

    for ( unsigned int r = 0; r < numNodesInCurrElem; r++ ) {
        for ( unsigned int d = 0; d < 3; d++ ) {
            elementInputVectors[Mechanics::DISPLACEMENT][( 3 * r ) + d] =
                ( d_inVec[Mechanics::DISPLACEMENT] )->getValueByGlobalID( d_dofIndices[r][d] );
            if ( d_useUpdatedLagrangian ) {
                elementInputVectors_pre[Mechanics::DISPLACEMENT][( 3 * r ) + d] =
                    ( d_inVec_pre[Mechanics::DISPLACEMENT] )
                        ->getValueByGlobalID( d_dofIndices[r][d] );
                elementRefXYZ[( 3 * r ) + d] = d_refXYZ->getValueByGlobalID( d_dofIndices[r][d] );
            }
        } // end d
        if ( d_isActive[Mechanics::TEMPERATURE] ) {
            elementInputVectors[Mechanics::TEMPERATURE][r] =
                ( d_inVec[Mechanics::TEMPERATURE] )->getValueByGlobalID( auxDofIds[r][0] );
            if ( d_useUpdatedLagrangian ) {
                elementInputVectors_pre[Mechanics::TEMPERATURE][r] =
                    ( d_inVec_pre[Mechanics::TEMPERATURE] )->getValueByGlobalID( auxDofIds[r][0] );
            }
        }
        if ( d_isActive[Mechanics::BURNUP] ) {
            elementInputVectors[Mechanics::BURNUP][r] =
                ( d_inVec[Mechanics::BURNUP] )->getValueByGlobalID( auxDofIds[r][0] );
            if ( d_useUpdatedLagrangian ) {
                elementInputVectors_pre[Mechanics::BURNUP][r] =
                    ( d_inVec_pre[Mechanics::BURNUP] )->getValueByGlobalID( auxDofIds[r][0] );
            }
        }
        if ( d_isActive[Mechanics::OXYGEN_CONCENTRATION] ) {
            elementInputVectors[Mechanics::OXYGEN_CONCENTRATION][r] =
                ( d_inVec[Mechanics::OXYGEN_CONCENTRATION] )->getValueByGlobalID( auxDofIds[r][0] );
            if ( d_useUpdatedLagrangian ) {
                elementInputVectors_pre[Mechanics::OXYGEN_CONCENTRATION][r] =
                    ( d_inVec_pre[Mechanics::OXYGEN_CONCENTRATION] )
                        ->getValueByGlobalID( auxDofIds[r][0] );
            }
        }
        if ( d_isActive[Mechanics::LHGR] ) {
            elementInputVectors[Mechanics::LHGR][r] =
                ( d_inVec[Mechanics::LHGR] )->getValueByGlobalID( auxDofIds[r][0] );
            if ( d_useUpdatedLagrangian ) {
                elementInputVectors_pre[Mechanics::LHGR][r] =
                    ( d_inVec_pre[Mechanics::LHGR] )->getValueByGlobalID( auxDofIds[r][0] );
            }
        }
    } // end r

    d_elementOutputVector.resize( 3 * numNodesInCurrElem );
    for ( auto &ve : d_elementOutputVector ) {
        ve = 0.0;
    } // end i

    if ( d_useUpdatedLagrangian ) {
        d_mechNULElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                    d_materialModel );
        d_mechNULElem->setElementVectors(
            elementInputVectors, elementInputVectors_pre, d_elementOutputVector );
        d_mechNULElem->assignReferenceXYZ( elementRefXYZ );
    } else {
        d_mechNonlinElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                       d_materialModel );
        d_mechNonlinElem->setElementVectors( elementInputVectors, d_elementOutputVector );
    }
}

void MechanicsNonlinearFEOperator::postElementOperation()
{
    PROFILE( "postElementOperation", 5 );
    AMP_ASSERT( d_dofIndices.size() == 8 );
    for ( unsigned int r = 0; r < d_dofIndices.size(); r++ ) {
        AMP_ASSERT( d_dofIndices[r].size() == 3 );
        for ( unsigned int d = 0; d < 3; d++ ) {
            d_outVec->addValuesByGlobalID(
                1, &d_dofIndices[r][d], &d_elementOutputVector[( 3 * r ) + d] );
        } // end for d
    } // end for r
}

void MechanicsNonlinearFEOperator::init()
{
    d_isInitialized = true;

    AMP::Mesh::MeshIterator el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    AMP::Mesh::MeshIterator end_el = el.end();

    d_materialModel->preNonlinearInit( d_resetReusesRadialReturn, d_jacobianReusesRadialReturn );

    if ( d_useUpdatedLagrangian ) {
        d_mechNULElem->preNonlinearElementInit();
    }

    for ( d_currElemIdx = 0; el != end_el; ++el, ++d_currElemIdx ) {
        d_currNodes                     = el->getElements( AMP::Mesh::GeomType::Vertex );
        unsigned int numNodesInCurrElem = d_currNodes.size();

        if ( d_useUpdatedLagrangian ) {
            getDofIndicesForCurrentElement( Mechanics::DISPLACEMENT, d_dofIndices );
        }

        std::vector<std::vector<size_t>> auxDofIds;
        if ( d_isActive[Mechanics::TEMPERATURE] ) {
            getDofIndicesForCurrentElement( Mechanics::TEMPERATURE, auxDofIds );
        }

        std::vector<double> localVector;
        if ( d_isActive[Mechanics::TEMPERATURE] ) {
            localVector.resize( numNodesInCurrElem );
            for ( unsigned int r = 0; r < numNodesInCurrElem; r++ ) {
                localVector[r] = d_referenceTemperature->getValueByGlobalID( auxDofIds[r][0] );
            }
        }

        std::vector<double> elementRefXYZ;
        if ( d_useUpdatedLagrangian ) {
            elementRefXYZ.resize( 3 * numNodesInCurrElem );
        }

        if ( d_useUpdatedLagrangian ) {
            d_mechNULElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                        d_materialModel );
            d_mechNULElem->initMaterialModel( localVector );
            d_mechNULElem->initializeReferenceXYZ( elementRefXYZ );
        } else {
            d_mechNonlinElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                           d_materialModel );
            d_mechNonlinElem->initMaterialModel( localVector );
        }

        if ( d_useUpdatedLagrangian ) {
            for ( unsigned int j = 0; j < numNodesInCurrElem; j++ ) {
                for ( unsigned int i = 0; i < 3; i++ ) {
                    d_refXYZ->setValuesByGlobalID(
                        1, &d_dofIndices[j][i], &elementRefXYZ[( 3 * j ) + i] );
                } // end for i
            } // end for j
        }

    } // end for el
    d_currElemIdx = static_cast<unsigned int>( -1 );

    if ( d_useUpdatedLagrangian ) {
        d_refXYZ->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }

    d_materialModel->postNonlinearInit();
}


void MechanicsNonlinearFEOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );

    if ( !d_isInitialized )
        init();

    auto myParams =
        std::dynamic_pointer_cast<const MechanicsNonlinearFEOperatorParameters>( params );

    AMP_INSIST( myParams, "Null parameter!" );

    if ( d_resetReusesRadialReturn ) {
        d_materialModel->globalReset();
    } else {
        AMP::Mesh::MeshIterator el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
        AMP::Mesh::MeshIterator end_el = el.end();

        setVector( Mechanics::DISPLACEMENT, myParams->d_EquilibriumVec[Mechanics::DISPLACEMENT] );

        if ( d_isActive[Mechanics::TEMPERATURE] ) {
            if ( myParams->d_EquilibriumVec[Mechanics::TEMPERATURE] ) {
                setVector( Mechanics::TEMPERATURE,
                           myParams->d_EquilibriumVec[Mechanics::TEMPERATURE] );
            }
        }

        if ( d_isActive[Mechanics::BURNUP] ) {
            if ( myParams->d_EquilibriumVec[Mechanics::BURNUP] ) {
                setVector( Mechanics::BURNUP, myParams->d_EquilibriumVec[Mechanics::BURNUP] );
            }
        }

        if ( d_isActive[Mechanics::OXYGEN_CONCENTRATION] ) {
            if ( myParams->d_EquilibriumVec[Mechanics::OXYGEN_CONCENTRATION] ) {
                setVector( Mechanics::OXYGEN_CONCENTRATION,
                           myParams->d_EquilibriumVec[Mechanics::OXYGEN_CONCENTRATION] );
            }
        }

        if ( d_isActive[Mechanics::LHGR] ) {
            if ( myParams->d_EquilibriumVec[Mechanics::LHGR] ) {
                setVector( Mechanics::LHGR, myParams->d_EquilibriumVec[Mechanics::LHGR] );
            }
        }

        d_outVec.reset();

        d_materialModel->preNonlinearReset();

        for ( d_currElemIdx = 0; el != end_el; ++el, ++d_currElemIdx ) {
            auto RESET = MechanicsNonlinearElement::MaterialUpdateType::RESET;
            if ( d_useUpdatedLagrangian ) {
                updateMaterialForUpdatedLagrangianElement( RESET, *el );
            } else {
                updateMaterialForElement( RESET, *el );
            }
        } // end for el
        d_currElemIdx = static_cast<unsigned int>( -1 );

        d_materialModel->postNonlinearReset();
    }

    if ( d_useUpdatedLagrangian == true ) {
        d_mechNULElem->resetElementInfo();
    }

    if ( d_useUpdatedLagrangian ) {
        // Copy values in pre
        for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
            if ( d_isActive[i] ) {
                d_inVec_pre[i]->copyVector( d_inVec[i] );
                d_inVec_pre[i]->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
            }
        }
    }

    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        if ( d_isActive[i] ) {
            if ( d_isFrozen[i] ) {
                if ( myParams->d_FrozenVec[i] ) {
                    setVector( i, myParams->d_FrozenVec[i] );
                }
            }
        }
    } // end for i
}

std::shared_ptr<OperatorParameters> MechanicsNonlinearFEOperator::getJacobianParameters(
    AMP::LinearAlgebra::Vector::const_shared_ptr u_in )
{
    if ( !d_isInitialized )
        init();

    auto u = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( u_in );

    // set up a database for the linear operator params
    auto tmp_db = std::make_shared<AMP::Database>( "Dummy" );
    tmp_db->putScalar( "name", "MechanicsLinearFEOperator" );
    tmp_db->putScalar( "USE_UPDATED_LAGRANGIAN", d_useUpdatedLagrangian );
    tmp_db->putScalar( "isAttachedToNonlinearOperator", true );
    tmp_db->putScalar( "isNonlinearOperatorInitialized", true );
    tmp_db->putScalar( "InputVariable",
                       d_inpVariables->getVariable( Mechanics::DISPLACEMENT )->getName() );
    tmp_db->putScalar( "OutputVariable", d_outVariable->getName() );
    tmp_db->putScalar( "reset_reuses_matrix", true );

    // create the linear operator params
    auto outParams             = std::make_shared<MechanicsLinearFEOperatorParameters>( tmp_db );
    outParams->d_materialModel = getMaterialModel();
    outParams->d_elemOp        = d_useUpdatedLagrangian ? d_mechLULElem : d_mechLinElem;
    outParams->d_Mesh          = d_Mesh;
    outParams->d_outDofMap     = outParams->d_inDofMap;

    // add miscellaneous to output parameters
    auto createDOFManager = []( std::shared_ptr<AMP::Mesh::Mesh> mesh ) {
        return AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );
    };

    outParams->d_inDofMap = createDOFManager( d_Mesh );

    // If updated-lagrangian is being used, then displacement and other variable has to be passed to
    // the linear element level.
    if ( d_useUpdatedLagrangian ) {
        auto displacementVector =
            mySubsetVector( u, d_inpVariables->getVariable( Mechanics::DISPLACEMENT ) );
        outParams->d_dispVec = displacementVector;
    }

    if ( d_jacobianReusesRadialReturn == false ) {
        auto dispVector =
            mySubsetVector( u, d_inpVariables->getVariable( Mechanics::DISPLACEMENT ) );
        setVector( Mechanics::DISPLACEMENT, dispVector );

        if ( d_isActive[Mechanics::TEMPERATURE] ) {
            if ( !( d_isFrozen[Mechanics::TEMPERATURE] ) ) {
                auto tempVector =
                    mySubsetVector( u, d_inpVariables->getVariable( Mechanics::TEMPERATURE ) );
                setVector( Mechanics::TEMPERATURE, tempVector );
            }
        }

        if ( d_isActive[Mechanics::BURNUP] ) {
            if ( !( d_isFrozen[Mechanics::BURNUP] ) ) {
                auto burnVector =
                    mySubsetVector( u, ( d_inpVariables->getVariable( Mechanics::BURNUP ) ) );
                setVector( Mechanics::BURNUP, burnVector );
            }
        }

        if ( d_isActive[Mechanics::OXYGEN_CONCENTRATION] ) {
            if ( !( d_isFrozen[Mechanics::OXYGEN_CONCENTRATION] ) ) {
                auto oxyVector = mySubsetVector(
                    u, d_inpVariables->getVariable( Mechanics::OXYGEN_CONCENTRATION ) );
                setVector( Mechanics::OXYGEN_CONCENTRATION, oxyVector );
            }
        }

        if ( d_isActive[Mechanics::LHGR] ) {
            if ( !( d_isFrozen[Mechanics::LHGR] ) ) {
                auto lhgrVar    = d_inpVariables->getVariable( Mechanics::LHGR );
                auto lhgrVector = mySubsetVector( u, lhgrVar );
                setVector( Mechanics::LHGR, lhgrVector );
            }
        }

        d_outVec.reset();

        d_materialModel->preNonlinearJacobian();

        auto el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
        auto end_el = el.end();

        for ( d_currElemIdx = 0; el != end_el; ++el, ++d_currElemIdx ) {
            auto JACOBIAN = MechanicsNonlinearElement::MaterialUpdateType::JACOBIAN;
            if ( d_useUpdatedLagrangian ) {
                updateMaterialForUpdatedLagrangianElement( JACOBIAN, *el );
            } else {
                updateMaterialForElement( JACOBIAN, *el );
            }
        } // end for el
        d_currElemIdx = static_cast<unsigned int>( -1 );

        d_materialModel->postNonlinearJacobian();
    }

    return outParams;
}

void MechanicsNonlinearFEOperator::printStressAndStrain(
    AMP::LinearAlgebra::Vector::const_shared_ptr u, const std::string &fname )
{
    if ( !d_isInitialized ) {
        init();
    }

    auto el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    auto end_el = el.end();

    auto fp = fopen( fname.c_str(), "w" );

    std::shared_ptr<AMP::LinearAlgebra::Variable> dispVar =
        d_inpVariables->getVariable( Mechanics::DISPLACEMENT );
    auto dispVector = mySubsetVector( u, dispVar );
    setVector( Mechanics::DISPLACEMENT, dispVector );

    if ( d_isActive[Mechanics::TEMPERATURE] ) {
        if ( !( d_isFrozen[Mechanics::TEMPERATURE] ) ) {
            auto tempVar    = d_inpVariables->getVariable( Mechanics::TEMPERATURE );
            auto tempVector = mySubsetVector( u, tempVar );
            setVector( Mechanics::TEMPERATURE, tempVector );
        }
    }

    if ( d_isActive[Mechanics::BURNUP] ) {
        if ( !( d_isFrozen[Mechanics::BURNUP] ) ) {
            auto burnVar    = d_inpVariables->getVariable( Mechanics::BURNUP );
            auto burnVector = mySubsetVector( u, burnVar );
            setVector( Mechanics::BURNUP, burnVector );
        }
    }

    if ( d_isActive[Mechanics::OXYGEN_CONCENTRATION] ) {
        if ( !( d_isFrozen[Mechanics::OXYGEN_CONCENTRATION] ) ) {
            auto oxyVar    = d_inpVariables->getVariable( Mechanics::OXYGEN_CONCENTRATION );
            auto oxyVector = mySubsetVector( u, oxyVar );
            setVector( Mechanics::OXYGEN_CONCENTRATION, oxyVector );
        }
    }

    if ( d_isActive[Mechanics::LHGR] ) {
        if ( !( d_isFrozen[Mechanics::LHGR] ) ) {
            auto lhgrVar    = d_inpVariables->getVariable( Mechanics::LHGR );
            auto lhgrVector = mySubsetVector( u, lhgrVar );
            setVector( Mechanics::LHGR, lhgrVector );
        }
    }

    d_materialModel->preNonlinearAssembly();

    for ( d_currElemIdx = 0; el != end_el; ++el, ++d_currElemIdx ) {
        d_currNodes                     = el->getElements( AMP::Mesh::GeomType::Vertex );
        unsigned int numNodesInCurrElem = d_currNodes.size();

        getDofIndicesForCurrentElement( Mechanics::DISPLACEMENT, d_dofIndices );

        std::vector<std::vector<size_t>> auxDofIds;
        for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
            if ( i != Mechanics::DISPLACEMENT ) {
                if ( d_isActive[i] ) {
                    getDofIndicesForCurrentElement( i, auxDofIds );
                    break;
                }
            }
        } // end for i

        std::vector<std::vector<double>> elementInputVectors(
            Mechanics::TOTAL_NUMBER_OF_VARIABLES );
        std::vector<std::vector<double>> elementInputVectors_pre(
            Mechanics::TOTAL_NUMBER_OF_VARIABLES );

        elementInputVectors[Mechanics::DISPLACEMENT].resize( 3 * numNodesInCurrElem );
        if ( d_useUpdatedLagrangian ) {
            elementInputVectors_pre[Mechanics::DISPLACEMENT].resize( 3 * numNodesInCurrElem );
        }
        for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
            if ( i != Mechanics::DISPLACEMENT ) {
                if ( d_isActive[i] ) {
                    elementInputVectors[i].resize( numNodesInCurrElem );
                    if ( d_useUpdatedLagrangian ) {
                        elementInputVectors_pre[i].resize( numNodesInCurrElem );
                    }
                }
            }
        } // end for i

        for ( unsigned int r = 0; r < numNodesInCurrElem; r++ ) {
            for ( unsigned int d = 0; d < 3; d++ ) {
                elementInputVectors[Mechanics::DISPLACEMENT][( 3 * r ) + d] =
                    ( d_inVec[Mechanics::DISPLACEMENT] )->getValueByGlobalID( d_dofIndices[r][d] );
                if ( d_useUpdatedLagrangian ) {
                    elementInputVectors_pre[Mechanics::DISPLACEMENT][( 3 * r ) + d] =
                        ( d_inVec_pre[Mechanics::DISPLACEMENT] )
                            ->getValueByGlobalID( d_dofIndices[r][d] );
                }
            } // end for d
            for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
                if ( i != Mechanics::DISPLACEMENT ) {
                    if ( d_isActive[i] ) {
                        elementInputVectors[i][r] =
                            ( d_inVec[i] )->getValueByGlobalID( auxDofIds[r][0] );
                        if ( d_useUpdatedLagrangian ) {
                            elementInputVectors_pre[i][r] =
                                ( d_inVec_pre[i] )->getValueByGlobalID( auxDofIds[r][0] );
                        }
                    }
                }
            } // end for i
        } // end for r

        d_mechNonlinElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                       d_materialModel );

        d_mechNonlinElem->printStressAndStrain( fp, elementInputVectors );
    } // end for el
    d_currElemIdx = static_cast<unsigned int>( -1 );

    d_materialModel->postNonlinearAssembly();

    fprintf( fp, "\n\n" );

    fclose( fp );
}

void MechanicsNonlinearFEOperator::updateMaterialForElementCommonFunction(
    const AMP::Mesh::MeshElement &elem,
    std::vector<std::vector<double>> &elementInputVectors,
    std::vector<std::vector<double>> &elementInputVectors_pre )
{
    d_currNodes                     = elem.getElements( AMP::Mesh::GeomType::Vertex );
    unsigned int numNodesInCurrElem = d_currNodes.size();

    getDofIndicesForCurrentElement( Mechanics::DISPLACEMENT, d_dofIndices );

    std::vector<std::vector<size_t>> auxDofIds;
    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        if ( i != Mechanics::DISPLACEMENT ) {
            if ( d_isActive[i] ) {
                getDofIndicesForCurrentElement( i, auxDofIds );
                break;
            }
        }
    } // end for i

    elementInputVectors[Mechanics::DISPLACEMENT].resize( 3 * numNodesInCurrElem );
    if ( d_useUpdatedLagrangian ) {
        elementInputVectors_pre[Mechanics::DISPLACEMENT].resize( 3 * numNodesInCurrElem );
    }
    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        if ( i != Mechanics::DISPLACEMENT ) {
            if ( d_isActive[i] ) {
                elementInputVectors[i].resize( numNodesInCurrElem );
                if ( d_useUpdatedLagrangian ) {
                    elementInputVectors_pre[i].resize( numNodesInCurrElem );
                }
            }
        }
    } // end for i

    std::vector<double> elementRefXYZ;
    elementRefXYZ.resize( 3 * numNodesInCurrElem );

    for ( unsigned int r = 0; r < numNodesInCurrElem; r++ ) {
        for ( unsigned int d = 0; d < 3; d++ ) {
            elementInputVectors[Mechanics::DISPLACEMENT][( 3 * r ) + d] =
                ( d_inVec[Mechanics::DISPLACEMENT] )->getValueByGlobalID( d_dofIndices[r][d] );
            if ( d_useUpdatedLagrangian ) {
                elementInputVectors_pre[Mechanics::DISPLACEMENT][( 3 * r ) + d] =
                    ( d_inVec_pre[Mechanics::DISPLACEMENT] )
                        ->getValueByGlobalID( d_dofIndices[r][d] );
                elementRefXYZ[( 3 * r ) + d] = d_refXYZ->getValueByGlobalID( d_dofIndices[r][d] );
            }
        } // end for d
        for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
            if ( i != Mechanics::DISPLACEMENT ) {
                if ( d_isActive[i] ) {
                    elementInputVectors[i][r] =
                        ( d_inVec[i] )->getValueByGlobalID( auxDofIds[r][0] );
                    if ( d_useUpdatedLagrangian ) {
                        elementInputVectors_pre[i][r] =
                            ( d_inVec_pre[i] )->getValueByGlobalID( auxDofIds[r][0] );
                    }
                }
            }
        } // end for i
    } // end for r

    if ( d_useUpdatedLagrangian ) {
        d_mechNULElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                    d_materialModel );
        d_mechNULElem->assignReferenceXYZ( elementRefXYZ );
    } else {
        d_mechNonlinElem->initializeForCurrentElement( d_currElemPtrs[d_currElemIdx],
                                                       d_materialModel );
    }
}

void MechanicsNonlinearFEOperator::getDofIndicesForCurrentElement(
    int varId, std::vector<std::vector<size_t>> &dofIds )
{
    dofIds.resize( d_currNodes.size() );
    for ( unsigned int j = 0; j < d_currNodes.size(); j++ ) {
        d_dofMap[varId]->getDOFs( d_currNodes[j].globalID(), dofIds[j] );
    } // end of j
}

void MechanicsNonlinearFEOperator::setVector(
    unsigned int id, AMP::LinearAlgebra::Vector::const_shared_ptr frozenVec )
{
    std::shared_ptr<AMP::LinearAlgebra::Variable> var = d_inpVariables->getVariable( id );
    d_inVec[id]                                       = mySubsetVector( frozenVec, var );
    AMP_ASSERT( d_inVec[id]->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
}

void MechanicsNonlinearFEOperator::setReferenceTemperature(
    AMP::LinearAlgebra::Vector::const_shared_ptr refTemp )
{
    std::shared_ptr<AMP::LinearAlgebra::Variable> var =
        d_inpVariables->getVariable( Mechanics::TEMPERATURE );
    d_referenceTemperature = mySubsetVector( refTemp, var );
    AMP_ASSERT( d_referenceTemperature->getUpdateStatus() ==
                AMP::LinearAlgebra::UpdateState::UNCHANGED );
    if ( d_useUpdatedLagrangian ) {
        d_inVec_pre[Mechanics::TEMPERATURE]->copyVector( d_referenceTemperature );
        d_inVec_pre[Mechanics::TEMPERATURE]->makeConsistent(
            AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }
}

AMP::LinearAlgebra::Vector::shared_ptr
MechanicsNonlinearFEOperator::mySubsetVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                                              std::shared_ptr<AMP::LinearAlgebra::Variable> var )
{
    if ( !vec )
        return nullptr;
    if ( d_Mesh ) {
        AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
        auto meshSubsetVec = vec->select( meshSelector );
        return meshSubsetVec->subsetVectorForVariable( var );
    } else {
        return vec->subsetVectorForVariable( var );
    }
}

AMP::LinearAlgebra::Vector::const_shared_ptr
MechanicsNonlinearFEOperator::mySubsetVector( AMP::LinearAlgebra::Vector::const_shared_ptr vec,
                                              std::shared_ptr<AMP::LinearAlgebra::Variable> var )
{
    if ( !vec )
        return nullptr;
    if ( d_Mesh ) {
        AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
        auto meshSubsetVec = vec->select( meshSelector );
        return meshSubsetVec->subsetVectorForVariable( var );
    } else {
        return vec->subsetVectorForVariable( var );
    }
}

void MechanicsNonlinearFEOperator::updateMaterialForElement(
    MechanicsNonlinearElement::MaterialUpdateType updateType, const AMP::Mesh::MeshElement &elem )
{
    std::vector<std::vector<double>> elementInputVectors1( Mechanics::TOTAL_NUMBER_OF_VARIABLES );
    std::vector<std::vector<double>> elementInputVectors_pre1(
        Mechanics::TOTAL_NUMBER_OF_VARIABLES );

    updateMaterialForElementCommonFunction( elem, elementInputVectors1, elementInputVectors_pre1 );

    d_mechNonlinElem->updateMaterialModel( updateType, elementInputVectors1 );
}

void MechanicsNonlinearFEOperator::updateMaterialForUpdatedLagrangianElement(
    MechanicsNonlinearElement::MaterialUpdateType updateType, const AMP::Mesh::MeshElement &elem )
{
    std::vector<std::vector<double>> elementInputVectors2( Mechanics::TOTAL_NUMBER_OF_VARIABLES );
    std::vector<std::vector<double>> elementInputVectors_pre2(
        Mechanics::TOTAL_NUMBER_OF_VARIABLES );

    updateMaterialForElementCommonFunction( elem, elementInputVectors2, elementInputVectors_pre2 );

    d_mechNULElem->updateMaterialModel(
        updateType, elementInputVectors2, elementInputVectors_pre2 );
}


} // namespace AMP::Operator
