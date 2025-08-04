#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/utils/UtilityMacros.h"

#ifdef AMP_USE_LIBMESH
    #include "AMP/operators/ManufacturedDiffusionTransportModel.h"
    #include "AMP/operators/diffusion/DiffusionCylindricalTransportModel.h"
    #include "AMP/operators/diffusion/DiffusionTransportModel.h"
    #include "AMP/operators/diffusion/DiffusionTransportTensorModel.h"
    #include "AMP/operators/libmesh/MassDensityModel.h"
    #include "AMP/operators/libmesh/PelletContactConductanceModel.h"
    #include "AMP/operators/libmesh/SourcePhysicsModel.h"
    #include "AMP/operators/mechanics/ElasticDamageThermalStrainModel.h"
    #include "AMP/operators/mechanics/GeneralCladThermalCreepPlasticModel.h"
    #include "AMP/operators/mechanics/IsotropicElasticModel.h"
    #include "AMP/operators/mechanics/PericElastoViscoPlasticModel.h"
    #include "AMP/operators/mechanics/ThermalStrainMaterialModel.h"
    #include "AMP/operators/mechanics/ThermalVonMisesMatModel.h"
    #include "AMP/operators/mechanics/VonMisesElastoPlasticModel.h"
    #include "AMP/operators/mechanics/VonMises_IsotropicKinematicHardening.h"
    #include "AMP/operators/subchannel/ConvectiveHeatCoefficient.h"
    #include "AMP/operators/subchannel/SubchannelPhysicsModel.h"
#endif


#define resetElementPhysicsModel( NAME )                        \
    do {                                                        \
        if ( name == #NAME )                                    \
            retElementPhysicsModel.reset( new NAME( params ) ); \
    } while ( 0 )


namespace AMP::Operator {


std::shared_ptr<ElementPhysicsModel> ElementPhysicsModelFactory::createElementPhysicsModel(
    std::shared_ptr<Database> elementPhysicsModelDb )
{
    std::shared_ptr<ElementPhysicsModel> retElementPhysicsModel;
    std::shared_ptr<ElementPhysicsModelParameters> params;

    AMP_INSIST(
        elementPhysicsModelDb,
        "ElementPhysicsModelFactory::createElementPhysicsModel:: NULL Database object input" );

    std::string name = elementPhysicsModelDb->getString( "name" );

    params.reset( new ElementPhysicsModelParameters( elementPhysicsModelDb ) );

#ifdef AMP_USE_LIBMESH
    resetElementPhysicsModel( IsotropicElasticModel );
    resetElementPhysicsModel( ThermalStrainMaterialModel );
    resetElementPhysicsModel( VonMisesElastoPlasticModel );
    resetElementPhysicsModel( ThermalVonMisesMatModel );
    resetElementPhysicsModel( GeneralCladThermalCreepPlasticModel );
    resetElementPhysicsModel( VonMises_IsotropicKinematicHardening );
    resetElementPhysicsModel( ElasticDamageThermalStrainModel );
    resetElementPhysicsModel( PericElastoViscoPlasticModel );
    resetElementPhysicsModel( DiffusionTransportModel );
    resetElementPhysicsModel( DiffusionTransportTensorModel );
    resetElementPhysicsModel( DiffusionCylindricalTransportModel );
    resetElementPhysicsModel( PelletContactConductanceModel );
    resetElementPhysicsModel( ConvectiveHeatCoefficient );
    resetElementPhysicsModel( SourcePhysicsModel );
    resetElementPhysicsModel( MassDensityModel );
    resetElementPhysicsModel( ManufacturedDiffusionTransportModel );
    resetElementPhysicsModel( SubchannelPhysicsModel );
#endif

    AMP_INSIST( retElementPhysicsModel != nullptr, "requested model " + name + " is invalid" );
    return retElementPhysicsModel;
}


} // namespace AMP::Operator
