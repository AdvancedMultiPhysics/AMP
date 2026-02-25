#include "AMP/operators/mechanics/ThermalStrainMaterialModel.h"
#include "AMP/materials/Property.h"
#include "AMP/operators/mechanics/IsotropicElasticModel.h"
#include "AMP/operators/mechanics/MechanicsConstants.h"
#include "AMP/utils/Utilities.h"


using namespace AMP::Operator::UpdatedLagrangianUtils;


namespace AMP::Operator {

ThermalStrainMaterialModel::ThermalStrainMaterialModel(
    std::shared_ptr<MechanicsMaterialModelParameters> params )
    : MechanicsMaterialModel( params )
{
    AMP_INSIST( params, "NULL parameter" );

    AMP_INSIST( params->d_db, "NULL database" );

    d_Is_Source = params->d_db->getWithDefault<bool>( "THERMAL_STRAIN_AS_SOURCE_TERM", false );

    if ( d_useMaterialsLibrary == false ) {

        d_Is_Source = params->d_db->getWithDefault<bool>( "THERMAL_STRAIN_AS_SOURCE_TERM", false );

        AMP_INSIST( params->d_db->keyExists( "THERMAL_EXPANSION_COEFFICIENT" ),
                    "Missing key: THERMAL_EXPANSION_COEFFICIENT" );

        default_alpha = params->d_db->getScalar<double>( "THERMAL_EXPANSION_COEFFICIENT" );

        AMP_INSIST( params->d_db->keyExists( "Youngs_Modulus" ), "Missing key: Youngs_Modulus" );

        AMP_INSIST( params->d_db->keyExists( "Poissons_Ratio" ), "Missing key: Poissons_Ratio" );

        default_E = params->d_db->getScalar<double>( "Youngs_Modulus" );

        default_Nu = params->d_db->getScalar<double>( "Poissons_Ratio" );

        if ( d_useUpdatedLagrangian == true ) {
            constructConstitutiveMatrixUpdatedLagrangian( default_E, default_Nu );
        } else {
            constructConstitutiveMatrix( default_E, default_Nu );
        }
    }

    default_TEMPERATURE = params->d_db->getWithDefault<double>( "Default_Temperature", 310.0 );

    default_BURNUP = params->d_db->getWithDefault<double>( "Default_Burnup", 0.0 );

    default_OXYGEN_CONCENTRATION =
        params->d_db->getWithDefault<double>( "Default_Oxygen_Concentration", 0.0 );

    for ( size_t i = 0; i < 6; i++ ) {
        for ( size_t j = 0; j < 6; j++ ) {
            d_constitutiveMatrix[i][j]    = 0.;
            d_constitutiveMatrix_UL[i][j] = 0.;
        }
    }
    d_gaussPtCnt                 = 0;
    d_resetReusesRadialReturn    = false;
    d_jacobianReusesRadialReturn = false;
}

void ThermalStrainMaterialModel::preNonlinearInit( bool resetReusesRadialReturn,
                                                   bool jacobianReusesRadialReturn )
{
    d_resetReusesRadialReturn    = resetReusesRadialReturn;
    d_jacobianReusesRadialReturn = jacobianReusesRadialReturn;

    d_E.clear();
    d_Nu.clear();
    d_alpha.clear();

    d_detULF.clear();

    d_EquilibriumStress.clear();
    d_EquilibriumStrain.clear();
    d_EquilibriumTemperature.clear();

    d_tmp1Stress.clear();
    d_tmp1Strain.clear();
    d_tmp1Temperature.clear();
}

void ThermalStrainMaterialModel::nonlinearInitGaussPointOperation( double tempAtGaussPoint )
{
    if ( d_useMaterialsLibrary == false ) {
        d_E.push_back( default_E );
        d_Nu.push_back( default_Nu );
        d_alpha.push_back( default_alpha );
    } else {
        d_E.push_back( 0.0 );
        d_Nu.push_back( 0.0 );
        d_alpha.push_back( 0.0 );
    }

    d_detULF.push_back( 1.0 );

    for ( int i = 0; i < 6; i++ ) {
        d_EquilibriumStress.push_back( 0 );
        d_EquilibriumStrain.push_back( 0 );

        d_tmp1Stress.push_back( 0 );
        d_tmp1Strain.push_back( 0 );
    } // end for i

    d_EquilibriumTemperature.push_back( tempAtGaussPoint );

    d_tmp1Temperature.push_back( tempAtGaussPoint );
}

void ThermalStrainMaterialModel::globalReset()
{
    AMP_INSIST( ( d_resetReusesRadialReturn == true ), "Inconsistent options!" );

    d_EquilibriumStress = d_tmp1Stress;
    d_EquilibriumStrain = d_tmp1Strain;

    d_EquilibriumTemperature = d_tmp1Temperature;
}

void ThermalStrainMaterialModel::nonlinearResetGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    double strain_np1[6];

    double Identity[3][3];
    for ( unsigned int i = 0; i < 3; i++ ) {
        for ( unsigned int j = 0; j < 3; j++ ) {
            Identity[i][j] = 0.0;
        }
        Identity[i][i] = 1.0;
    }

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        strain_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    d_tmp1Temperature[d_gaussPtCnt] = strain[Mechanics::TEMPERATURE][0];

    double Temp = strain[Mechanics::TEMPERATURE][0];

    double *stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    Thermal_Strain_Gauss_Point( &strain_np1[0], Temp, stress, strain, Identity, Identity );
}

void ThermalStrainMaterialModel::nonlinearResetGaussPointOperation_UL(
    const std::vector<std::vector<double>> &strain, double R_n[3][3], double R_np1[3][3] )
{
    double strain_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        strain_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    d_tmp1Temperature[d_gaussPtCnt] = strain[Mechanics::TEMPERATURE][0];

    double Temp = strain[Mechanics::TEMPERATURE][0];

    double *stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    Thermal_Strain_Gauss_Point( &strain_np1[0], Temp, stress, strain, R_n, R_np1 );
}

void ThermalStrainMaterialModel::postNonlinearReset()
{
    d_EquilibriumStress = d_tmp1Stress;
    d_EquilibriumStrain = d_tmp1Strain;

    d_EquilibriumTemperature = d_tmp1Temperature;
}

void ThermalStrainMaterialModel::nonlinearJacobianGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    if ( d_useMaterialsLibrary == true ) {
        computeEvalv( strain );
    }
}

void ThermalStrainMaterialModel::nonlinearJacobianGaussPointOperation_UL(
    const std::vector<std::vector<double>> &strain, double[3][3], double[3][3] )
{
    if ( d_useMaterialsLibrary == true ) {
        computeEvalv( strain );
    }
}

void ThermalStrainMaterialModel::getInternalStress( const std::vector<std::vector<double>> &strain,
                                                    double *&stress )
{
    AMP_INSIST( ( d_Is_Source == false ), "Inconsistent options!" );

    double strain_np1[6];

    double Identity[3][3];
    for ( unsigned int i = 0; i < 3; i++ ) {
        for ( unsigned int j = 0; j < 3; j++ ) {
            Identity[i][j] = 0.0;
        }
        Identity[i][i] = 1.0;
    }

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        strain_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    double Temp = strain[Mechanics::TEMPERATURE][0];

    d_tmp1Temperature[d_gaussPtCnt] = strain[Mechanics::TEMPERATURE][0];

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    Thermal_Strain_Gauss_Point( &strain_np1[0], Temp, stress, strain, Identity, Identity );
}

void ThermalStrainMaterialModel::getInternalStress_UL(
    const std::vector<std::vector<double>> &strain,
    double *&stress,
    double R_n[3][3],
    double R_np1[3][3],
    double detF )
{
    AMP_INSIST( ( d_Is_Source == false ), "Inconsistent options!" );

    double strain_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        strain_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    double Temp = strain[Mechanics::TEMPERATURE][0];

    d_tmp1Temperature[d_gaussPtCnt] = strain[Mechanics::TEMPERATURE][0];

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    d_detULF[d_gaussPtCnt] = detF;

    Thermal_Strain_Gauss_Point( &strain_np1[0], Temp, stress, strain, R_n, R_np1 );
}

void ThermalStrainMaterialModel::getExternalStress( double *&stress )
{
    AMP_INSIST( ( d_Is_Source == true ), "Inconsistent options!" );

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    // Thermal_Strain_Gauss_Point( strain_np1, Temp, stress );
}

void ThermalStrainMaterialModel::getConstitutiveMatrix( double *&constitutiveMatrix )
{
    // Consistent Tangent

    if ( d_useMaterialsLibrary == false ) {
        constitutiveMatrix = &( d_constitutiveMatrix[0][0] );
    }

    if ( d_useMaterialsLibrary == true ) {
        double pass_E  = d_E[d_gaussPtCnt];
        double pass_Nu = d_Nu[d_gaussPtCnt];

        constructConstitutiveMatrix( pass_E, pass_Nu );
        constitutiveMatrix = &( d_constitutiveMatrix[0][0] );
    }
}

void ThermalStrainMaterialModel::getConstitutiveMatrixUpdatedLagrangian(
    double constitutiveMatrix[6][6], double[3][3] )
{
    // Consistent Tangent
    double pass_E  = d_E[d_gaussPtCnt];
    double pass_Nu = d_Nu[d_gaussPtCnt];
    constructConstitutiveMatrixUpdatedLagrangian( pass_E, pass_Nu );

    if ( d_useJaumannRate == true ) {
        for ( auto &elem : d_constitutiveMatrix_UL ) {
            for ( double &j : elem ) {
                j /= d_detULF[d_gaussPtCnt];
            }
        }
    }

    for ( int i = 0; i < 6; i++ ) {
        for ( int j = 0; j < 6; j++ ) {
            constitutiveMatrix[i][j] = d_constitutiveMatrix_UL[i][j];
        }
    }
}

void ThermalStrainMaterialModel::constructConstitutiveMatrix( const double passed_E,
                                                              const double passed_Nu )
{
    double E  = passed_E;
    double Nu = passed_Nu;

    double G = E / ( 2.0 * ( 1.0 + Nu ) );
    double K = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) );

    for ( auto &elem : d_constitutiveMatrix ) {
        for ( double &j : elem ) {
            j = 0.0;
        }
    }

    for ( int i = 0; i < 3; i++ ) {
        d_constitutiveMatrix[i][i] += ( 2.0 * G );
        d_constitutiveMatrix[i + 3][i + 3] += ( 1.0 * G );
    }

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            d_constitutiveMatrix[i][j] += ( K - ( ( 2.0 / 3.0 ) * G ) );
        } // end for j
    }     // end for i
}

void ThermalStrainMaterialModel::constructConstitutiveMatrixUpdatedLagrangian(
    const double passed_E, const double passed_Nu )
{
    double E  = passed_E;
    double Nu = passed_Nu;

    double G = E / ( 2.0 * ( 1.0 + Nu ) );
    double K = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) );

    for ( auto &elem : d_constitutiveMatrix_UL ) {
        for ( double &j : elem ) {
            j = 0.0;
        }
    }

    for ( int i = 0; i < 3; i++ ) {
        d_constitutiveMatrix_UL[i][i] += ( 2.0 * G );
        d_constitutiveMatrix_UL[i + 3][i + 3] += ( G );
    }

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            d_constitutiveMatrix_UL[i][j] += ( K - ( ( 2.0 / 3.0 ) * G ) );
        } // end for j
    }     // end for i
}

void ThermalStrainMaterialModel::computeEvalv( const std::vector<std::vector<double>> &strain )
{
    if ( d_useMaterialsLibrary == true ) {

        std::vector<double> tempVec;
        std::vector<double> burnupVec;
        std::vector<double> oxygenVec;

        if ( strain[Mechanics::TEMPERATURE].empty() ) {
            tempVec.push_back( default_TEMPERATURE );
        } else {
            tempVec = strain[Mechanics::TEMPERATURE];
        }

        if ( strain[Mechanics::BURNUP].empty() ) {
            burnupVec.push_back( default_BURNUP );
        } else {
            burnupVec = strain[Mechanics::BURNUP];
        }

        if ( strain[Mechanics::OXYGEN_CONCENTRATION].empty() ) {
            oxygenVec.push_back( default_OXYGEN_CONCENTRATION );
        } else {
            oxygenVec = strain[Mechanics::OXYGEN_CONCENTRATION];
        }

        std::vector<double> YM( 1 );
        std::vector<double> PR( 1 );
        std::vector<double> TEC( 1 );

        std::string ymString  = "YoungsModulus";
        std::string prString  = "PoissonRatio";
        std::string tecString = "ThermalExpansion";

        std::map<std::string, std::vector<double> &> args = { { "temperature", tempVec },
                                                              { "concentration", oxygenVec },
                                                              { "burnup", burnupVec } };

        d_material->property( ymString )->evalv( YM, {}, args );
        d_material->property( prString )->evalv( PR, {}, args );
        d_material->property( tecString )->evalv( TEC, {}, args );

        if ( !( strain[Mechanics::TEMPERATURE].empty() ) ) {
            tempVec[0] = d_EquilibriumTemperature[d_gaussPtCnt];
        }

        std::vector<double> TEC_1( 1 );
        d_material->property( tecString )->evalv( TEC_1, {}, args );

        d_E[d_gaussPtCnt]     = YM[0];
        d_Nu[d_gaussPtCnt]    = PR[0];
        d_alpha[d_gaussPtCnt] = ( TEC[0] + TEC_1[0] ) / 2.0;
    }
}

void ThermalStrainMaterialModel::Thermal_Strain_Gauss_Point(
    const double *stra_np1,
    const double Temp,
    double *stre_np1,
    const std::vector<std::vector<double>> &strain,
    double R_n[3][3],
    double R_np1[3][3] )
{
    if ( d_useMaterialsLibrary == true ) {
        computeEvalv( strain );

        double pass_E  = d_E[d_gaussPtCnt];
        double pass_Nu = d_Nu[d_gaussPtCnt];

        if ( d_useUpdatedLagrangian == true ) {
            constructConstitutiveMatrixUpdatedLagrangian( pass_E, pass_Nu );
        } else {
            constructConstitutiveMatrix( pass_E, pass_Nu );
        }
    }

    double thermal_expansion_coefficient = d_alpha[d_gaussPtCnt];

    const double *stre_n = &( d_EquilibriumStress[6 * d_gaussPtCnt] );
    const double *stra_n = &( d_EquilibriumStrain[6 * d_gaussPtCnt] );

    double stress_n[6], S[3][3], Sr[3][3], stress_np1[6];

    if ( d_useUpdatedLagrangian == true ) {
        S[0][0] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 0];
        S[1][1] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 1];
        S[2][2] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 2];
        S[1][2] = S[2][1] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 3];
        S[0][2] = S[2][0] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 4];
        S[0][1] = S[1][0] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 5];

        if ( d_useJaumannRate == false ) {
            pullbackCorotational( R_n, S, Sr );
            stress_n[0] = Sr[0][0];
            stress_n[1] = Sr[1][1];
            stress_n[2] = Sr[2][2];
            stress_n[3] = ( 0.5 * ( Sr[1][2] + Sr[2][1] ) );
            stress_n[4] = ( 0.5 * ( Sr[0][2] + Sr[2][0] ) );
            stress_n[5] = ( 0.5 * ( Sr[0][1] + Sr[1][0] ) );
        }

        if ( d_useJaumannRate == true ) {
            jaumannToCauchy( R_n, S );
            stress_n[0] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 0] + S[0][0];
            stress_n[1] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 1] + S[1][1];
            stress_n[2] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 2] + S[2][2];
            stress_n[3] =
                d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 3] + ( 0.5 * ( S[1][2] + S[2][1] ) );
            stress_n[4] =
                d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 4] + ( 0.5 * ( S[0][2] + S[2][0] ) );
            stress_n[5] =
                d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 5] + ( 0.5 * ( S[0][1] + S[1][0] ) );
        }
    }

    double delta_thermal_stress[6], delta_thermal_strain[6];

    double Temperature_n   = ( d_EquilibriumTemperature[d_gaussPtCnt] );
    double Temperature_np1 = Temp;

    if ( d_iDebugPrintInfoLevel > 11 ) {
        for ( int i = 0; i < 6; i++ ) {
            std::cout << "stra_np1[" << i << "]=" << stra_np1[i] << std::endl;
            std::cout << "stra_n[" << i << "]=" << stra_n[i] << std::endl;
        }
    }

    for ( int i = 0; i < 6; i++ ) {
        delta_thermal_stress[i] = 0.0;
        delta_thermal_strain[i] = 0.0;
    }

    for ( int i = 0; i < 3; i++ ) {
        delta_thermal_strain[i] +=
            thermal_expansion_coefficient * ( Temperature_np1 - Temperature_n );
    }

    if ( d_useUpdatedLagrangian == false ) {
        if ( d_Is_Source ) {
            for ( int i = 0; i < 6; i++ ) {
                for ( int j = 0; j < 6; j++ ) {
                    delta_thermal_stress[i] += d_constitutiveMatrix[i][j] * delta_thermal_strain[j];
                }
            }
            for ( int i = 0; i < 6; i++ ) {
                stre_np1[i] = delta_thermal_stress[i];
            }
        } else {
            for ( int i = 0; i < 6; i++ ) {
                for ( int j = 0; j < 6; j++ ) {
                    delta_thermal_stress[i] +=
                        d_constitutiveMatrix[i][j] *
                        ( stra_np1[j] - stra_n[j] - delta_thermal_strain[j] );
                }
            }
            for ( int i = 0; i < 6; i++ ) {
                stre_np1[i] = stre_n[i] + delta_thermal_stress[i];
            }
        }
    } else {
        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                delta_thermal_stress[i] +=
                    d_constitutiveMatrix_UL[i][j] * ( stra_np1[j] - delta_thermal_strain[j] );
            }
        }
        for ( int i = 0; i < 6; i++ ) {
            stress_np1[i] = stress_n[i] + delta_thermal_stress[i];
        }
    }

    if ( d_useUpdatedLagrangian == true ) {
        if ( d_useJaumannRate == false ) {
            S[0][0] = stress_np1[0];
            S[1][1] = stress_np1[1];
            S[2][2] = stress_np1[2];
            S[1][2] = S[2][1] = stress_np1[3];
            S[0][2] = S[2][0] = stress_np1[4];
            S[0][1] = S[1][0] = stress_np1[5];
            pushforwardCorotational( R_np1, S, Sr );
            stress_np1[0] = Sr[0][0];
            stress_np1[1] = Sr[1][1];
            stress_np1[2] = Sr[2][2];
            stress_np1[3] = 0.5 * ( Sr[1][2] + Sr[2][1] );
            stress_np1[4] = 0.5 * ( Sr[0][2] + Sr[2][0] );
            stress_np1[5] = 0.5 * ( Sr[0][1] + Sr[1][0] );
        }

        for ( int i = 0; i < 6; i++ ) {
            stre_np1[i] = stress_np1[i];
        }
    }
}
} // namespace AMP::Operator
