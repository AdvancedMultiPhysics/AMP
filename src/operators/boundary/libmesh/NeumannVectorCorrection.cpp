#include "AMP/operators/boundary/libmesh/NeumannVectorCorrection.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrectionParameters.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/VectorSelector.h"

// Libmesh files
DISABLE_WARNINGS
#include "libmesh/libmesh_config.h"
#undef LIBMESH_ENABLE_REFERENCE_COUNTING
#include "libmesh/auto_ptr.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/face_quad4.h"
#include "libmesh/node.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS

#include <string>


using AMP::Utilities::stringf;


namespace AMP::Operator {


/****************************************************************
 * Create the appropriate parameters                             *
 ****************************************************************/
static std::shared_ptr<const NeumannVectorCorrectionParameters>
convert( std::shared_ptr<const OperatorParameters> inParams )
{
    AMP_ASSERT( inParams );
    if ( std::dynamic_pointer_cast<const NeumannVectorCorrectionParameters>( inParams ) )
        return std::dynamic_pointer_cast<const NeumannVectorCorrectionParameters>( inParams );
    auto bndParams = std::dynamic_pointer_cast<const BoundaryOperatorParameters>( inParams );
    AMP_ASSERT( bndParams );
    auto params        = std::make_shared<NeumannVectorCorrectionParameters>( inParams->d_db );
    params->d_Mesh     = inParams->d_Mesh;
    params->d_variable = bndParams->d_volumeOperator->getOutputVariable();
    if ( params->d_db->keyExists( "LocalModel" ) ) {
        auto model_db = params->d_db->getDatabase( "LocalModel" );
        auto model    = ElementPhysicsModelFactory::createElementPhysicsModel( model_db );
        params->d_robinPhysicsModel = std::dynamic_pointer_cast<RobinPhysicsModel>( model );
    }
    return params;
}


/****************************************************************
 * Constructor                                                   *
 ****************************************************************/
NeumannVectorCorrection::NeumannVectorCorrection(
    std::shared_ptr<const OperatorParameters> inParams )
    : BoundaryOperator( inParams )
{
    auto params = convert( inParams );
    AMP_ASSERT( params );
    d_params = params;

    d_isConstantFlux      = false;
    d_isFluxGaussPtVector = false;

    auto feTypeOrderName = params->d_db->getWithDefault<std::string>( "FE_ORDER", "FIRST" );
    auto feFamilyName    = params->d_db->getWithDefault<std::string>( "FE_FAMILY", "LAGRANGE" );
    auto qruleTypeName   = params->d_db->getWithDefault<std::string>( "QRULE_TYPE", "QGAUSS" );
    auto qruleOrderName  = params->d_db->getWithDefault<std::string>( "QRULE_ORDER", "DEFAULT" );

    // Create the libmesh qruleOrder, qruleType, and FEType
    auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( feTypeOrderName );
    auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( feFamilyName );
    d_type.reset( new libMesh::FEType( feTypeOrder, feFamily ) );
    d_qruleType = libMesh::Utility::string_to_enum<libMeshEnums::QuadratureType>( qruleTypeName );
    if ( qruleOrderName == "DEFAULT" ) {
        d_qruleOrder = d_type->default_quadrature_order();
    } else {
        d_qruleOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( qruleOrderName );
    }

    d_variable = params->d_variable;

    reset( params );
}


void NeumannVectorCorrection::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );

    auto myparams = std::dynamic_pointer_cast<const NeumannVectorCorrectionParameters>( params );

    AMP_INSIST( myparams, "NULL parameters" );
    AMP_INSIST( myparams->d_db, "NULL database" );
    AMP_INSIST( myparams->d_db->keyExists( "number_of_ids" ), "Key ''number_of_ids'' is missing!" );

    int numBndIds         = myparams->d_db->getScalar<int>( "number_of_ids" );
    d_isConstantFlux      = myparams->d_db->getWithDefault<bool>( "constant_flux", true );
    d_isFluxGaussPtVector = myparams->d_db->getWithDefault<bool>( "IsFluxGaussPtVector", true );

    d_boundaryIds.resize( numBndIds );
    d_dofIds.resize( numBndIds );
    d_neumannValues.resize( numBndIds );
    d_IsCoupledBoundary.resize( numBndIds );

    for ( int j = 0; j < numBndIds; j++ ) {
        d_boundaryIds[j] = myparams->d_db->getScalar<int>( stringf( "id_%d", j ) );

        int numDofIds = myparams->d_db->getScalar<int>( stringf( "number_of_dofs_%d", j ) );

        d_IsCoupledBoundary[j] =
            params->d_db->getWithDefault<bool>( stringf( "IsCoupledBoundary_%d", j ), false );

        d_dofIds[j].resize( numDofIds );
        d_neumannValues[j].resize( numDofIds );
        for ( int i = 0; i < numDofIds; i++ ) {
            d_dofIds[j][i] = myparams->d_db->getScalar<int>( stringf( "dof_%d_%d", j, i ) );

            if ( d_isConstantFlux ) {
                auto key              = stringf( "value_%d_%d", j, i );
                d_neumannValues[j][i] = myparams->d_db->getScalar<double>( key );
            } else {
                d_variableFlux = myparams->d_variableFlux;
            }
        }
    }

    if ( myparams->d_robinPhysicsModel ) {
        d_robinPhysicsModel = myparams->d_robinPhysicsModel;
    }

    // Create the libmesh elements
    AMP::Mesh::MeshIterator iterator;
    for ( auto &elem : d_boundaryIds ) {
        auto iterator2 = d_Mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, elem, 0 );
        iterator = AMP::Mesh::Mesh::getIterator( AMP::Mesh::SetOP::Union, iterator, iterator2 );
    }
    d_libmeshElements.reinit( iterator, d_qruleType, d_qruleOrder, d_type );
}


void NeumannVectorCorrection::addRHScorrection(
    AMP::LinearAlgebra::Vector::shared_ptr rhsCorrection )
{

    auto myRhs = subsetInputVector( rhsCorrection );

    auto gammaValue = ( d_params->d_db )->getWithDefault<double>( "gamma", 1.0 );

    auto rInternal  = myRhs->clone();
    auto dofManager = rInternal->getDOFManager();
    rInternal->zero();

    unsigned int numBndIds = d_boundaryIds.size();
    std::vector<size_t> dofs;
    std::vector<std::vector<size_t>> dofIndices;
    std::vector<size_t> fluxDofs;

    for ( unsigned int j = 0; j < numBndIds; j++ ) {
        if ( !d_IsCoupledBoundary[j] ) {
            unsigned int numDofIds = d_dofIds[j].size();

            if ( !d_isConstantFlux ) {
                d_variableFlux->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
            }

            for ( unsigned int k = 0; k < numDofIds; k++ ) {

                auto bnd =
                    d_Mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Face, d_boundaryIds[j], 0 );

                for ( auto &elem : bnd ) {

                    d_currNodes = elem.getElements( AMP::Mesh::GeomType::Vertex );
                    unsigned int numNodesInCurrElem = d_currNodes.size();

                    dofIndices.resize( numNodesInCurrElem );
                    for ( unsigned int i = 0; i < numNodesInCurrElem; i++ ) {
                        dofManager->getDOFs( d_currNodes[i]->globalID(), dofIndices[i] );
                    }

                    std::shared_ptr<AMP::Discretization::DOFManager> fluxDOFManager;
                    if ( !d_isConstantFlux && d_isFluxGaussPtVector ) {
                        fluxDOFManager = d_variableFlux->getDOFManager();
                        fluxDOFManager->getDOFs( elem.globalID(), fluxDofs );
                    }

                    // Get the current libmesh element
                    const libMesh::FEBase *fe  = d_libmeshElements.getFEBase( elem.globalID() );
                    const libMesh::QBase *rule = d_libmeshElements.getQBase( elem.globalID() );
                    AMP_ASSERT( fe );
                    AMP_ASSERT( rule );
                    const unsigned int numGaussPts = rule->n_points();

                    const std::vector<std::vector<libMesh::Real>> phi = fe->get_phi();
                    const std::vector<libMesh::Real> djxw             = fe->get_JxW();

                    std::vector<std::vector<double>> temp( 1 );
                    std::vector<double> gamma( numGaussPts, gammaValue );

                    dofs.resize( numNodesInCurrElem );
                    for ( size_t n = 0; n < dofIndices.size(); n++ )
                        dofs[n] = dofIndices[n][d_dofIds[j][k]];

                    for ( size_t qp = 0; qp < numGaussPts; qp++ ) {
                        if ( d_isConstantFlux ) {
                            temp[0].push_back( d_neumannValues[j][k] );
                        } else {
                            if ( d_isFluxGaussPtVector ) {
                                temp[0].push_back(
                                    d_variableFlux->getValueByGlobalID( fluxDofs[qp] ) );
                            } else {
                                libMesh::Real Tqp = 0.0;
                                for ( size_t n = 0; n < dofIndices.size(); n++ ) {
                                    Tqp +=
                                        phi[n][qp] * d_variableFlux->getValueByGlobalID( dofs[n] );
                                }
                                temp[0].push_back( Tqp );
                            }
                        }
                    }

                    if ( d_robinPhysicsModel ) {
                        d_robinPhysicsModel->getConductance( gamma, gamma, temp );
                    }

                    std::vector<double> flux( dofIndices.size(), 0.0 );

                    for ( unsigned int i = 0; i < dofIndices.size(); i++ ) // Loop over nodes
                    {
                        for ( unsigned int qp = 0; qp < numGaussPts; qp++ ) {
                            flux[i] += ( gamma[qp] ) * djxw[qp] * phi[i][qp] * temp[0][qp];
                        }
                    }

                    rInternal->addValuesByGlobalID(
                        (int) dofs.size(), (size_t *) &( dofs[0] ), &( flux[0] ) );
                }
            }
        }
    }

    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_ADD );
    myRhs->add( *myRhs, *rInternal );
}


void NeumannVectorCorrection::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                     AMP::LinearAlgebra::Vector::shared_ptr r )
{
    (void) u;
    (void) r;
    // Do Nothing
}

std::shared_ptr<OperatorParameters>
NeumannVectorCorrection::getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr )
{
    auto db = std::make_shared<AMP::Database>( "Dummy" );
    db->putScalar( "name", "NeumannVectorCorrection" );
    db->putScalar( "FE_ORDER", "FIRST" );
    db->putScalar( "FE_FAMILY", "LAGRANGE" );
    db->putScalar( "QRULE_TYPE", "QGAUSS" );
    db->putScalar( "DIMENSION", 2 );
    db->putScalar( "QRULE_ORDER", "DEFAULT" );
    db->putScalar( "skip_params", true );
    db->putScalar( "number_of_ids", d_boundaryIds.size() );
    db->putScalar( "constant_flux", d_isConstantFlux );
    for ( int i = 0; i < (int) d_boundaryIds.size(); i++ ) {
        db->putScalar( stringf( "id_%i", i ), d_boundaryIds[i] );
        db->putScalar( stringf( "number_of_dofs_%i", i ), d_dofIds[i].size() );
        db->putScalar( stringf( "IsCoupledBoundary_%i", i ), d_IsCoupledBoundary[i] );
        for ( int j = 0; j < (int) d_dofIds[i].size(); j++ ) {
            db->putScalar( stringf( "dof_%i_%i", i, j ), d_dofIds[i][j] );
            if ( d_isConstantFlux ) {
                db->putScalar( stringf( "value_%i_%i", i, j ), d_neumannValues[i][j] );
            } else {
                db->putScalar( stringf( "value_%i_%i", i, j ), 0 );
            }
        }
    }

    auto outParams    = std::make_shared<NeumannVectorCorrectionParameters>( db );
    outParams->d_Mesh = d_Mesh;

    return outParams;
}


void NeumannVectorCorrection::setFrozenVector( AMP::LinearAlgebra::Vector::shared_ptr f )
{
    auto f2 = f;
    if ( d_Mesh )
        f2 = f->select( AMP::LinearAlgebra::VS_Mesh( d_Mesh ) );
    if ( !f2 )
        return;
    if ( !d_Frozen )
        d_Frozen = AMP::LinearAlgebra::MultiVector::create( "frozenMultiVec", d_Mesh->getComm() );
    std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( d_Frozen )->addVector( f2 );
}


void NeumannVectorCorrection::setVariableFlux( const AMP::LinearAlgebra::Vector::shared_ptr &flux )
{
    if ( d_Mesh ) {
        AMP::LinearAlgebra::VS_Mesh meshSelector( d_Mesh );
        auto meshSubsetVec = flux->select( meshSelector );
        d_variableFlux     = meshSubsetVec->subsetVectorForVariable( d_variable );
    } else {
        d_variableFlux = flux->subsetVectorForVariable( d_variable );
    }
}
} // namespace AMP::Operator
