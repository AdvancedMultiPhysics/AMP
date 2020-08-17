
#include "MechanicsNonlinearFEOperator.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"
#include "MechanicsLinearFEOperatorParameters.h"
#include "libmesh/cell_hex8.h"
#include "libmesh/node.h"

namespace AMP {
namespace Operator {

MechanicsNonlinearFEOperator::MechanicsNonlinearFEOperator(
    const std::shared_ptr<MechanicsNonlinearFEOperatorParameters> &params )
    : NonlinearFEOperator( params )
{
    AMP_INSIST( ( ( params.get() ) != nullptr ), "NULL parameter!" );
    AMP_INSIST( ( ( ( params->d_db ).get() ) != nullptr ), "NULL database!" );

    d_resetReusesRadialReturn =
        ( params->d_db )->getWithDefault( "RESET_REUSES_RADIAL_RETURN", true );
    d_jacobianReusesRadialReturn =
        ( params->d_db )->getWithDefault( "JACOBIAN_REUSES_RADIAL_RETURN", true );

    d_useUpdatedLagrangian = ( params->d_db )->getWithDefault( "USE_UPDATED_LAGRANGIAN", false );

    if ( d_useUpdatedLagrangian ) {
        d_mechNULElem =
            std::dynamic_pointer_cast<MechanicsNonlinearUpdatedLagrangianElement>( d_elemOp );
    } else {
        d_mechNonlinElem = std::dynamic_pointer_cast<MechanicsNonlinearElement>( d_elemOp );
    }

    if ( d_useUpdatedLagrangian ) {
        AMP_INSIST( ( ( d_mechNULElem.get() ) != nullptr ),
                    "d_elemOp is not of type MechanicsNonlinearUpdatedLagrangianElement" );
    } else {
        AMP_INSIST( ( ( d_mechNonlinElem.get() ) != nullptr ),
                    "d_elemOp is not of type MechanicsNonlinearElement" );
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
        AMP::LinearAlgebra::Variable::shared_ptr dummyVar;
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
                ( params->d_db )->getWithDefault( "FREEZE_" + keysForVariables[i], true );
        }
        if ( d_isActive[i] ) {
            std::string varName = activeInpVar_db->getString( keysForVariables[i] );
            AMP::LinearAlgebra::Variable::shared_ptr dummyVar(
                new AMP::LinearAlgebra::Variable( varName ) );
            d_inpVariables->setVariable( i, dummyVar );
            if ( d_isFrozen[i] ) {
                if ( params->d_FrozenVec[i] != nullptr ) {
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
            true );
        d_refXYZ->zero();
        for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
            if ( d_isActive[i] ) {
                d_inVec_pre[i] = AMP::LinearAlgebra::createVector(
                    d_dofMap[i], d_inpVariables->getVariable( i ), true );
                d_inVec_pre[i]->zero();
            }
        }
    }

    if ( params->d_ReferenceTemperature != nullptr ) {
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
    AMP_INSIST( ( u != nullptr ), "NULL Input Vector" );

    if ( !d_isInitialized ) {
        init();
    }

    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        if ( d_isActive[i] ) {
            if ( !( d_isFrozen[i] ) ) {
                AMP::LinearAlgebra::Variable::shared_ptr var = d_inpVariables->getVariable( i );
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
    d_outVec->makeConsistent( AMP::LinearAlgebra::Vector::ScatterType::CONSISTENT_ADD );
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
    AMP_ASSERT( d_dofIndices.size() == 8 );
    for ( unsigned int r = 0; r < d_dofIndices.size(); r++ ) {
        AMP_ASSERT( d_dofIndices[r].size() == 3 );
        for ( unsigned int d = 0; d < 3; d++ ) {
            d_outVec->addValueByGlobalID( d_dofIndices[r][d],
                                          d_elementOutputVector[( 3 * r ) + d] );
        } // end for d
    }     // end for r
}

void MechanicsNonlinearFEOperator::init()
{
    d_isInitialized = true;

    AMP::Mesh::MeshIterator el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Volume, 0 );
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
                    d_refXYZ->setValueByGlobalID( d_dofIndices[j][i],
                                                  elementRefXYZ[( 3 * j ) + i] );
                } // end for i
            }     // end for j
        }

    } // end for el
    d_currElemIdx = static_cast<unsigned int>( -1 );

    if ( d_useUpdatedLagrangian ) {
        d_refXYZ->makeConsistent( AMP::LinearAlgebra::Vector::ScatterType::CONSISTENT_SET );
    }

    d_materialModel->postNonlinearInit();
}


void MechanicsNonlinearFEOperator::reset( const std::shared_ptr<OperatorParameters> &params )
{
    if ( !d_isInitialized ) {
        init();
    }

    std::shared_ptr<MechanicsNonlinearFEOperatorParameters> myParams =
        std::dynamic_pointer_cast<MechanicsNonlinearFEOperatorParameters>( params );

    AMP_INSIST( ( ( myParams.get() ) != nullptr ), "Null parameter!" );

    if ( d_resetReusesRadialReturn ) {
        d_materialModel->globalReset();
    } else {
        AMP::Mesh::MeshIterator el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Volume, 0 );
        AMP::Mesh::MeshIterator end_el = el.end();

        setVector( Mechanics::DISPLACEMENT, myParams->d_EquilibriumVec[Mechanics::DISPLACEMENT] );

        if ( d_isActive[Mechanics::TEMPERATURE] ) {
            if ( myParams->d_EquilibriumVec[Mechanics::TEMPERATURE] != nullptr ) {
                setVector( Mechanics::TEMPERATURE,
                           myParams->d_EquilibriumVec[Mechanics::TEMPERATURE] );
            }
        }

        if ( d_isActive[Mechanics::BURNUP] ) {
            if ( myParams->d_EquilibriumVec[Mechanics::BURNUP] != nullptr ) {
                setVector( Mechanics::BURNUP, myParams->d_EquilibriumVec[Mechanics::BURNUP] );
            }
        }

        if ( d_isActive[Mechanics::OXYGEN_CONCENTRATION] ) {
            if ( myParams->d_EquilibriumVec[Mechanics::OXYGEN_CONCENTRATION] != nullptr ) {
                setVector( Mechanics::OXYGEN_CONCENTRATION,
                           myParams->d_EquilibriumVec[Mechanics::OXYGEN_CONCENTRATION] );
            }
        }

        if ( d_isActive[Mechanics::LHGR] ) {
            if ( myParams->d_EquilibriumVec[Mechanics::LHGR] != nullptr ) {
                setVector( Mechanics::LHGR, myParams->d_EquilibriumVec[Mechanics::LHGR] );
            }
        }

        d_outVec.reset();

        d_materialModel->preNonlinearReset();

        for ( d_currElemIdx = 0; el != end_el; ++el, ++d_currElemIdx ) {
            if ( d_useUpdatedLagrangian ) {
                updateMaterialForUpdatedLagrangianElement<
                    MechanicsNonlinearUpdatedLagrangianElement::RESET>( *el );
            } else {
                updateMaterialForElement<MechanicsNonlinearElement::RESET>( *el );
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
                d_inVec_pre[i]->makeConsistent(
                    AMP::LinearAlgebra::Vector::ScatterType::CONSISTENT_SET );
            }
        }
    }

    for ( unsigned int i = 0; i < Mechanics::TOTAL_NUMBER_OF_VARIABLES; i++ ) {
        if ( d_isActive[i] ) {
            if ( d_isFrozen[i] ) {
                if ( myParams->d_FrozenVec[i] != nullptr ) {
                    setVector( i, myParams->d_FrozenVec[i] );
                }
            }
        }
    } // end for i
}

std::shared_ptr<OperatorParameters> MechanicsNonlinearFEOperator::getJacobianParameters(
    AMP::LinearAlgebra::Vector::const_shared_ptr u_in )
{
    if ( !d_isInitialized ) {
        init();
    }

    AMP::LinearAlgebra::Vector::shared_ptr u =
        std::const_pointer_cast<AMP::LinearAlgebra::Vector>( u_in );

    // set up a database for the linear operator params
    std::shared_ptr<AMP::Database> tmp_db( new AMP::Database( "Dummy" ) );
    tmp_db->putScalar( "reset_reuses_matrix", true );
    tmp_db->putScalar( "isAttachedToNonlinearOperator", true );
    tmp_db->putScalar( "isNonlinearOperatorInitialized", true );

    // create the linear operator params
    std::shared_ptr<MechanicsLinearFEOperatorParameters> outParams(
        new MechanicsLinearFEOperatorParameters( tmp_db ) );

    // If updated-lagrangian is being used, then displacement and other variable has to be passed to
    // the linear element
    // level.
    if ( d_useUpdatedLagrangian ) {
        AMP::LinearAlgebra::Vector::shared_ptr displacementVector =
            mySubsetVector( u, ( d_inpVariables->getVariable( Mechanics::DISPLACEMENT ) ) );
        outParams->d_dispVec = displacementVector;
        ( outParams->d_dispVec )
            ->makeConsistent( AMP::LinearAlgebra::Vector::ScatterType::CONSISTENT_SET );
    }

    if ( d_jacobianReusesRadialReturn == false ) {
        AMP::LinearAlgebra::Vector::shared_ptr dispVector =
            mySubsetVector( u, ( d_inpVariables->getVariable( Mechanics::DISPLACEMENT ) ) );
        setVector( Mechanics::DISPLACEMENT, dispVector );

        if ( d_isActive[Mechanics::TEMPERATURE] ) {
            if ( !( d_isFrozen[Mechanics::TEMPERATURE] ) ) {
                AMP::LinearAlgebra::Vector::shared_ptr tempVector =
                    mySubsetVector( u, ( d_inpVariables->getVariable( Mechanics::TEMPERATURE ) ) );
                setVector( Mechanics::TEMPERATURE, tempVector );
            }
        }

        if ( d_isActive[Mechanics::BURNUP] ) {
            if ( !( d_isFrozen[Mechanics::BURNUP] ) ) {
                AMP::LinearAlgebra::Vector::shared_ptr burnVector =
                    mySubsetVector( u, ( d_inpVariables->getVariable( Mechanics::BURNUP ) ) );
                setVector( Mechanics::BURNUP, burnVector );
            }
        }

        if ( d_isActive[Mechanics::OXYGEN_CONCENTRATION] ) {
            if ( !( d_isFrozen[Mechanics::OXYGEN_CONCENTRATION] ) ) {
                AMP::LinearAlgebra::Vector::shared_ptr oxyVector = mySubsetVector(
                    u, ( d_inpVariables->getVariable( Mechanics::OXYGEN_CONCENTRATION ) ) );
                setVector( Mechanics::OXYGEN_CONCENTRATION, oxyVector );
            }
        }

        if ( d_isActive[Mechanics::LHGR] ) {
            if ( !( d_isFrozen[Mechanics::LHGR] ) ) {
                AMP::LinearAlgebra::Variable::shared_ptr lhgrVar =
                    d_inpVariables->getVariable( Mechanics::LHGR );
                AMP::LinearAlgebra::Vector::shared_ptr lhgrVector = mySubsetVector( u, lhgrVar );
                setVector( Mechanics::LHGR, lhgrVector );
            }
        }

        d_outVec.reset();

        d_materialModel->preNonlinearJacobian();

        AMP::Mesh::MeshIterator el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Volume, 0 );
        AMP::Mesh::MeshIterator end_el = el.end();

        for ( d_currElemIdx = 0; el != end_el; ++el, ++d_currElemIdx ) {
            if ( d_useUpdatedLagrangian ) {
                updateMaterialForUpdatedLagrangianElement<
                    MechanicsNonlinearUpdatedLagrangianElement::JACOBIAN>( *el );
            } else {
                updateMaterialForElement<MechanicsNonlinearElement::JACOBIAN>( *el );
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

    AMP::Mesh::MeshIterator el     = d_Mesh->getIterator( AMP::Mesh::GeomType::Volume, 0 );
    AMP::Mesh::MeshIterator end_el = el.end();

    FILE *fp = fopen( fname.c_str(), "w" );

    AMP::LinearAlgebra::Variable::shared_ptr dispVar =
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
        }     // end for r

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
    }     // end for r

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
    AMP::LinearAlgebra::Variable::shared_ptr var = d_inpVariables->getVariable( id );
    d_inVec[id]                                  = mySubsetVector( frozenVec, var );
    AMP_ASSERT( d_inVec[id]->getUpdateStatus() ==
                AMP::LinearAlgebra::Vector::UpdateState::UNCHANGED );
}

void MechanicsNonlinearFEOperator::setReferenceTemperature(
    AMP::LinearAlgebra::Vector::const_shared_ptr refTemp )
{
    AMP::LinearAlgebra::Variable::shared_ptr var =
        d_inpVariables->getVariable( Mechanics::TEMPERATURE );
    d_referenceTemperature = mySubsetVector( refTemp, var );
    AMP_ASSERT( d_referenceTemperature->getUpdateStatus() ==
                AMP::LinearAlgebra::Vector::UpdateState::UNCHANGED );
    if ( d_useUpdatedLagrangian ) {
        d_inVec_pre[Mechanics::TEMPERATURE]->copyVector( d_referenceTemperature );
        d_inVec_pre[Mechanics::TEMPERATURE]->makeConsistent(
            AMP::LinearAlgebra::Vector::ScatterType::CONSISTENT_SET );
    }
}

AMP::LinearAlgebra::Vector::shared_ptr
MechanicsNonlinearFEOperator::mySubsetVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                                              AMP::LinearAlgebra::Variable::shared_ptr var )
{
    if ( d_Mesh.get() != nullptr ) {
        AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
        AMP::LinearAlgebra::Vector::shared_ptr meshSubsetVec =
            vec->select( meshSelector, vec->getVariable()->getName() );
        return meshSubsetVec->subsetVectorForVariable( var );
    } else {
        return vec->subsetVectorForVariable( var );
    }
}

AMP::LinearAlgebra::Vector::const_shared_ptr
MechanicsNonlinearFEOperator::mySubsetVector( AMP::LinearAlgebra::Vector::const_shared_ptr vec,
                                              AMP::LinearAlgebra::Variable::shared_ptr var )
{
    if ( d_Mesh.get() != nullptr ) {
        AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
        AMP::LinearAlgebra::Vector::const_shared_ptr meshSubsetVec =
            vec->constSelect( meshSelector, vec->getVariable()->getName() );
        return meshSubsetVec->constSubsetVectorForVariable( var );
    } else {
        return vec->constSubsetVectorForVariable( var );
    }
}
} // namespace Operator
} // namespace AMP