#include "AMP/operators/subchannel/CoupledChannelToCladMapOperator.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/StructuredMeshHelper.h"
#include "AMP/operators/subchannel/CoupledChannelToCladMapOperatorParameters.h"
#include "AMP/operators/subchannel/SubchannelConstants.h"


namespace AMP::Operator {


CoupledChannelToCladMapOperator::CoupledChannelToCladMapOperator(
    std::shared_ptr<const CoupledChannelToCladMapOperatorParameters> params )
    : Operator( params )
{
    AMP_ASSERT( params->d_thermalMapOperator );
    AMP_ASSERT( params->d_densityMapOperator );
    d_thermalMapOperator     = params->d_thermalMapOperator;
    d_densityMapOperator     = params->d_densityMapOperator;
    d_flowVariable           = params->d_variable;
    d_Mesh                   = params->d_subchannelMesh;
    d_subchannelPhysicsModel = params->d_subchannelPhysicsModel;
    d_subchannelTemperature  = params->d_vector;
    if ( d_Mesh ) {
        AMP_ASSERT( d_subchannelPhysicsModel );
    }
    if ( d_subchannelTemperature ) {
        d_subchannelDensity = d_subchannelTemperature->clone();
        std::shared_ptr<AMP::LinearAlgebra::Variable> densityVariable(
            new AMP::LinearAlgebra::Variable( "Density" ) );
        d_subchannelDensity->setVariable( densityVariable );
    }
}


void CoupledChannelToCladMapOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                             AMP::LinearAlgebra::Vector::shared_ptr )
{

    // Compute the subchannel temperature and density
    if ( d_Mesh ) {
        const double h_scale = 1.0 / Subchannel::scaleEnthalpy; // Scale to correct units
        const double P_scale = 1.0 / Subchannel::scalePressure; // Scale to correct units

        AMP::LinearAlgebra::Vector::const_shared_ptr uInternal = subsetInputVector( u );

        std::shared_ptr<AMP::Discretization::DOFManager> faceDOFManager =
            uInternal->getDOFManager();
        std::shared_ptr<AMP::Discretization::DOFManager> scalarFaceDOFManager =
            d_subchannelTemperature->getDOFManager();

        AMP::Mesh::MeshIterator face =
            AMP::Mesh::StructuredMeshHelper::getXYFaceIterator( d_Mesh, 0 );
        // AMP::Mesh::MeshIterator end_face = face.end();

        std::vector<size_t> dofs;
        std::vector<size_t> scalarDofs;
        for ( size_t i = 0; i < face.size(); i++ ) {
            faceDOFManager->getDOFs( face->globalID(), dofs );
            scalarFaceDOFManager->getDOFs( face->globalID(), scalarDofs );
            std::map<std::string, std::shared_ptr<std::vector<double>>> temperatureArgMap;
            temperatureArgMap.insert(
                std::make_pair( std::string( "enthalpy" ),
                                std::make_shared<std::vector<double>>(
                                    1, h_scale * uInternal->getValueByGlobalID( dofs[0] ) ) ) );
            temperatureArgMap.insert(
                std::make_pair( std::string( "pressure" ),
                                std::make_shared<std::vector<double>>(
                                    1, P_scale * uInternal->getValueByGlobalID( dofs[1] ) ) ) );
            std::vector<double> temperatureResult( 1 );
            std::vector<double> specificVolume( 1 );
            d_subchannelPhysicsModel->getProperty(
                "Temperature", temperatureResult, temperatureArgMap );
            d_subchannelPhysicsModel->getProperty(
                "SpecificVolume", specificVolume, temperatureArgMap );
            d_subchannelTemperature->setValuesByGlobalID(
                1, &scalarDofs[0], &temperatureResult[0] );
            const double val = 1.0 / specificVolume[0];
            d_subchannelDensity->setValuesByGlobalID( 1, &scalarDofs[0], &val );
            ++face;
        }

        d_subchannelTemperature->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_subchannelDensity->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }

    // Map the temperature and density back to the clad
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    d_thermalMapOperator->apply( d_subchannelTemperature, nullVec );
    d_densityMapOperator->apply( d_subchannelDensity, nullVec );
}
} // namespace AMP::Operator
