#include "AMP/operators/mechanics/ElasticDamageThermalStrainModel.h"
#include "AMP/materials/Property.h"
#include "AMP/operators/mechanics/IsotropicElasticModel.h"
#include "AMP/operators/mechanics/MechanicsConstants.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Operator {

ElasticDamageThermalStrainModel::ElasticDamageThermalStrainModel(
    std::shared_ptr<MechanicsMaterialModelParameters> params )
    : MechanicsMaterialModel( params ), d_constitutiveMatrix{ { 0 } }
{
    AMP_INSIST( ( ( params.get() ) != nullptr ), "NULL parameter" );

    AMP_INSIST( ( ( ( params->d_db ).get() ) != nullptr ), "NULL database" );

    d_Is_Source = params->d_db->getWithDefault<bool>( "THERMAL_STRAIN_AS_SOURCE_TERM", false );

    if ( d_useMaterialsLibrary == false ) {
        // IsotropicElasticModel C_Elastic(params);

        d_Is_Source = params->d_db->getWithDefault<bool>( "THERMAL_STRAIN_AS_SOURCE_TERM", false );

        AMP_INSIST( params->d_db->keyExists( "THERMAL_EXPANSION_COEFFICIENT" ),
                    "Missing key: THERMAL_EXPANSION_COEFFICIENT" );

        default_alpha = params->d_db->getScalar<double>( "THERMAL_EXPANSION_COEFFICIENT" );

        AMP_INSIST( params->d_db->keyExists( "Youngs_Modulus" ), "Missing key: Youngs_Modulus" );

        AMP_INSIST( params->d_db->keyExists( "Poissons_Ratio" ), "Missing key: Poissons_Ratio" );

        default_E = params->d_db->getScalar<double>( "Youngs_Modulus" );

        default_Nu = params->d_db->getScalar<double>( "Poissons_Ratio" );

        constructConstitutiveMatrix( default_E, default_Nu );

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                d_initialConstitutiveMatrix[i][j] = d_constitutiveMatrix[i][j];
            }
        }

        // double * tmpDoublePtr;

        // C_Elastic.getConstitutiveMatrix(tmpDoublePtr);

        /*for(int  i = 0; i < 6; i++) {
          for(int j = 0; j < 6; j++) {
            d_constitutiveMatrix[i][j] = tmpDoublePtr[(6*i) + j];
          }
        }*/
    }

    d_DamageThreshold = params->d_db->getWithDefault<double>( "Initial_Damage_Threshold", 0.5 );

    d_CriticalDamageThreshold =
        params->d_db->getWithDefault<double>( "Critical_Damage_Threshold", 0.4 );

    default_TEMPERATURE = params->d_db->getWithDefault<double>( "Default_Temperature", 310.0 );

    default_BURNUP = params->d_db->getWithDefault<double>( "Default_Burnup", 0.0 );

    default_OXYGEN_CONCENTRATION =
        params->d_db->getWithDefault<double>( "Default_Oxygen_Concentration", 0.0 );

    d_gaussPtCnt                 = 0;
    d_resetReusesRadialReturn    = false;
    d_jacobianReusesRadialReturn = false;
}

void ElasticDamageThermalStrainModel::preNonlinearInit( bool resetReusesRadialReturn,
                                                        bool jacobianReusesRadialReturn )
{
    d_resetReusesRadialReturn    = resetReusesRadialReturn;
    d_jacobianReusesRadialReturn = jacobianReusesRadialReturn;

    AMP_INSIST(
        ( ( d_resetReusesRadialReturn == true ) && ( d_jacobianReusesRadialReturn == true ) ),
        "The current code works only with true values for resetReusesRadialReturn and "
        "jacobianReusesRadialReturn" );

    d_E.clear();
    d_Nu.clear();
    d_alpha.clear();

    d_EquilibriumStress.clear();
    d_EquilibriumStrain.clear();
    d_EquilibriumTemperature.clear();
    d_EquilibriumTau.clear();
    d_EquilibriumDamage.clear();
    d_EquilibriumDamageThreshold.clear();

    d_tmp1Stress.clear();
    d_tmp1Strain.clear();
    d_tmp1Temperature.clear();
    d_tmp1Tau.clear();
    d_tmp1Damage.clear();
    d_tmp1DamageThreshold.clear();

    d_tmp2Stress.clear();
    d_tmp2Tau.clear();
    d_tmp2Damage.clear();
    d_tmp2DamageThreshold.clear();

    d_tmp1ThermalStrain_Axial.clear();
    d_tmp1ThermalStrain_Radial.clear();
    d_EquilibriumThermalStrain_Axial.clear();
    d_EquilibriumThermalStrain_Radial.clear();

    d_InitialDamageVec.clear();
    d_CriticalDamageVec.clear();
}

void ElasticDamageThermalStrainModel::nonlinearInitGaussPointOperation( double tempAtGaussPoint )
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

    for ( int i = 0; i < 6; i++ ) {
        d_EquilibriumStress.push_back( 0 );
        d_EquilibriumStrain.push_back( 0 );

        d_tmp1Stress.push_back( 0 );
        d_tmp1Strain.push_back( 0 );
    } // end for i

    d_EquilibriumTemperature.push_back( tempAtGaussPoint );
    d_EquilibriumTau.push_back( 0.0 );
    d_EquilibriumDamage.push_back( 0.0 );
    d_EquilibriumDamageThreshold.push_back( d_DamageThreshold );

    if ( d_checkCladOrPellet == true ) {
        if ( d_useMaterialsLibrary == true ) {
            d_tmp1ThermalStrain_Axial.push_back( 0.0 );
            d_tmp1ThermalStrain_Radial.push_back( 0.0 );
            d_EquilibriumThermalStrain_Axial.push_back( 0.0 );
            d_EquilibriumThermalStrain_Radial.push_back( 0.0 );
        }
    }

    d_tmp1Temperature.push_back( tempAtGaussPoint );
    d_tmp1Tau.push_back( 0.0 );
    d_tmp1Damage.push_back( 0.0 );
    d_tmp1DamageThreshold.push_back( d_DamageThreshold );

    if ( !d_jacobianReusesRadialReturn ) {
        for ( int i = 0; i < 6; i++ ) {
            d_tmp2Stress.push_back( 0.0 );
        }
        d_tmp2Tau.push_back( 0.0 );
        d_tmp2Damage.push_back( 0.0 );
        d_tmp2DamageThreshold.push_back( d_DamageThreshold );
    }

    d_InitialDamageVec.push_back( d_DamageThreshold );
    d_CriticalDamageVec.push_back( d_CriticalDamageThreshold );
}

void ElasticDamageThermalStrainModel::globalReset()
{
    AMP_INSIST( ( d_resetReusesRadialReturn == true ), "Inconsistent options!" );

    d_EquilibriumStress = d_tmp1Stress;
    d_EquilibriumStrain = d_tmp1Strain;

    d_EquilibriumTemperature     = d_tmp1Temperature;
    d_EquilibriumTau             = d_tmp1Tau;
    d_EquilibriumDamage          = d_tmp1Damage;
    d_EquilibriumDamageThreshold = d_tmp1DamageThreshold;

    if ( d_checkCladOrPellet == true ) {
        if ( d_useMaterialsLibrary == true ) {
            d_EquilibriumThermalStrain_Axial  = d_tmp1ThermalStrain_Axial;
            d_EquilibriumThermalStrain_Radial = d_tmp1ThermalStrain_Radial;
        }
    }

    if ( d_iDebugPrintInfoLevel > 11 ) {
        AMP::pout << "d_tmp1Tau[799]=" << d_tmp1Tau[799]
                  << "  d_tmp1Damage[799]=" << d_tmp1Damage[799] << std::endl;
        AMP::pout << "d_tmp1Tau[599]=" << d_tmp1Tau[599]
                  << "  d_tmp1Damage[599]=" << d_tmp1Damage[599] << std::endl;
        AMP::pout << "d_tmp1Tau[853]=" << d_tmp1Tau[853]
                  << "  d_tmp1Damage[853]=" << d_tmp1Damage[853] << std::endl;
        AMP::pout << "d_tmp1Tau[478]=" << d_tmp1Tau[478]
                  << "  d_tmp1Damage[478]=" << d_tmp1Damage[478] << std::endl;
        AMP::pout << "d_InitialDamageVec[799]=" << d_InitialDamageVec[799]
                  << "  d_CriticalDamageVec[799]=" << d_CriticalDamageVec[799] << std::endl;
        AMP::pout << "d_InitialDamageVec[599]=" << d_InitialDamageVec[599]
                  << "  d_CriticalDamageVec[599]=" << d_CriticalDamageVec[599] << std::endl;
        AMP::pout << "d_InitialDamageVec[853]=" << d_InitialDamageVec[853]
                  << "  d_CriticalDamageVec[853]=" << d_CriticalDamageVec[853] << std::endl;
        AMP::pout << "d_InitialDamageVec[478]=" << d_InitialDamageVec[478]
                  << "  d_CriticalDamageVec[478]=" << d_CriticalDamageVec[478] << std::endl;
    }
}

void ElasticDamageThermalStrainModel::nonlinearResetGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    double strain_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        strain_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    d_tmp1Temperature[d_gaussPtCnt] = strain[Mechanics::TEMPERATURE][0];

    double Temp = strain[Mechanics::TEMPERATURE][0];

    double *stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    Thermal_Strain_Gauss_Point( &strain_np1[0], Temp, stress, strain );
}

void ElasticDamageThermalStrainModel::postNonlinearReset()
{
    d_EquilibriumStress = d_tmp1Stress;
    d_EquilibriumStrain = d_tmp1Strain;

    d_EquilibriumTemperature     = d_tmp1Temperature;
    d_EquilibriumTau             = d_tmp1Tau;
    d_EquilibriumDamage          = d_tmp1Damage;
    d_EquilibriumDamageThreshold = d_tmp1DamageThreshold;

    if ( d_checkCladOrPellet == true ) {
        if ( d_useMaterialsLibrary == true ) {
            d_EquilibriumThermalStrain_Axial  = d_tmp1ThermalStrain_Axial;
            d_EquilibriumThermalStrain_Radial = d_tmp1ThermalStrain_Radial;
        }
    }
}

void ElasticDamageThermalStrainModel::nonlinearJacobianGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    if ( d_useMaterialsLibrary == true ) {
        computeEvalv( strain );

        double pass_E  = d_E[d_gaussPtCnt];
        double pass_Nu = d_Nu[d_gaussPtCnt];

        constructConstitutiveMatrix( pass_E, pass_Nu );

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                d_initialConstitutiveMatrix[i][j] = d_constitutiveMatrix[i][j];
            }
        }

        double dam = d_tmp1Damage[d_gaussPtCnt];
        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                d_constitutiveMatrix[i][j] = ( 1.0 - dam ) * d_initialConstitutiveMatrix[i][j];
            }
        }
    }
}

void ElasticDamageThermalStrainModel::getInternalStress(
    const std::vector<std::vector<double>> &strain, double *&stress )
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

    Thermal_Strain_Gauss_Point( &strain_np1[0], Temp, stress, strain );
}

void ElasticDamageThermalStrainModel::getExternalStress( double *&stress )
{
    AMP_INSIST( ( d_Is_Source == true ), "Inconsistent options!" );

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    // Thermal_Strain_Gauss_Point( strain_np1, Temp, stress );
}

void ElasticDamageThermalStrainModel::getConstitutiveMatrix( double *&constitutiveMatrix )
{
    // Consistent Tangent

    if ( d_useMaterialsLibrary == false ) {
        constitutiveMatrix = &( d_constitutiveMatrix[0][0] );
    }

    if ( d_useMaterialsLibrary == true ) {
        /*      double pass_E = d_E[d_gaussPtCnt];
              double pass_Nu = d_Nu[d_gaussPtCnt];

              constructConstitutiveMatrix(pass_E, pass_Nu);
        */
        constitutiveMatrix = &( d_constitutiveMatrix[0][0] );
    }
}

void ElasticDamageThermalStrainModel::constructConstitutiveMatrix( const double passed_E,
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

void ElasticDamageThermalStrainModel::computeEvalv( const std::vector<std::vector<double>> &strain )
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
        std::vector<double> TEC_1( 1 );
        std::vector<double> TEC_2( 1 );

        std::map<std::string, std::vector<double> &> args = { { "temperature", tempVec },
                                                              { "concentration", oxygenVec },
                                                              { "burnup", burnupVec } };

        d_material->property( "YoungsModulus" )->evalv( YM, {}, args );
        d_material->property( "PoissonRatio" )->evalv( PR, {}, args );
        if ( d_checkCladOrPellet == false ) {
            d_material->property( "ThermalExpansion" )->evalv( TEC, {}, args );
        } else {
            d_material->property( "ThermalExpansionAxial" )->evalv( TEC, {}, args );
            d_material->property( "ThermalExpansion" )->evalv( TEC_2, {}, args );
        }

        if ( !( strain[Mechanics::TEMPERATURE].empty() ) ) {
            tempVec[0] = d_EquilibriumTemperature[d_gaussPtCnt];
        }
        d_material->property( "ThermalExpansion" )->evalv( TEC_1, {}, args );

        d_E[d_gaussPtCnt]  = YM[0];
        d_Nu[d_gaussPtCnt] = PR[0];
        if ( d_checkCladOrPellet == false ) {
            d_alpha[d_gaussPtCnt] = ( TEC[0] + TEC_1[0] ) / 2.0;
        } else {
            d_tmp1ThermalStrain_Axial[d_gaussPtCnt]  = TEC[0];
            d_tmp1ThermalStrain_Radial[d_gaussPtCnt] = TEC_2[0];
        }
    }
}

void ElasticDamageThermalStrainModel::Thermal_Strain_Gauss_Point(
    const double *stra_np1,
    const double Temp,
    double *stre_np1,
    const std::vector<std::vector<double>> &strain )
{
    if ( d_useMaterialsLibrary == true ) {
        computeEvalv( strain );

        double pass_E  = d_E[d_gaussPtCnt];
        double pass_Nu = d_Nu[d_gaussPtCnt];

        constructConstitutiveMatrix( pass_E, pass_Nu );

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                d_initialConstitutiveMatrix[i][j] = d_constitutiveMatrix[i][j];
            }
        }
    }

    double thermal_expansion_coefficient = d_alpha[d_gaussPtCnt];

    const double *stre_n = &( d_EquilibriumStress[6 * d_gaussPtCnt] );
    const double *stra_n = &( d_EquilibriumStrain[6 * d_gaussPtCnt] );

    double delta_thermal_stress[6], delta_thermal_strain[6], delta_thermal_stress_1[6];

    double Temperature_n   = ( d_EquilibriumTemperature[d_gaussPtCnt] );
    double Temperature_np1 = Temp;

    if ( d_iDebugPrintInfoLevel > 11 ) {
        for ( int i = 0; i < 6; i++ ) {
            std::cout << "stra_np1[" << i << "]=" << stra_np1[i] << std::endl;
            std::cout << "stra_n[" << i << "]=" << stra_n[i] << std::endl;
        }
    }

    for ( int i = 0; i < 6; i++ ) {
        delta_thermal_stress[i]   = 0.0;
        delta_thermal_stress_1[i] = 0.0;
        delta_thermal_strain[i]   = 0.0;
    }

    if ( d_checkCladOrPellet == false ) {
        delta_thermal_strain[0] +=
            thermal_expansion_coefficient * ( Temperature_np1 - Temperature_n );
        delta_thermal_strain[1] +=
            thermal_expansion_coefficient * ( Temperature_np1 - Temperature_n );
        delta_thermal_strain[2] +=
            thermal_expansion_coefficient * ( Temperature_np1 - Temperature_n );
    } else {
        if ( d_useMaterialsLibrary == true ) {
            delta_thermal_strain[0] += ( d_tmp1ThermalStrain_Radial[d_gaussPtCnt] -
                                         d_EquilibriumThermalStrain_Radial[d_gaussPtCnt] );
            delta_thermal_strain[1] += ( d_tmp1ThermalStrain_Radial[d_gaussPtCnt] -
                                         d_EquilibriumThermalStrain_Radial[d_gaussPtCnt] );
            delta_thermal_strain[2] += ( d_tmp1ThermalStrain_Axial[d_gaussPtCnt] -
                                         d_EquilibriumThermalStrain_Axial[d_gaussPtCnt] );
        }
    }

    double C0_eph[6], eph_C0_eph = 0.0;
    double E_curr = d_E[d_gaussPtCnt];

    for ( int i = 0; i < 6; i++ ) {
        C0_eph[i] = 0.0;
        for ( int j = 0; j < 6; j++ ) {
            C0_eph[i] += ( d_initialConstitutiveMatrix[i][j] * stra_np1[j] );
        }
    }

    for ( int i = 0; i < 6; i++ ) {
        eph_C0_eph += ( stra_np1[i] * C0_eph[i] );
    }

    AMP_INSIST( ( eph_C0_eph >= 0.0 ), "The energy is less than zero, which is wrong." );
    d_tmp1Tau[d_gaussPtCnt] = std::sqrt( eph_C0_eph / ( E_curr / 100.0 ) );

    if ( d_tmp1Tau[d_gaussPtCnt] <= d_EquilibriumDamageThreshold[d_gaussPtCnt] ) {
        d_tmp1DamageThreshold[d_gaussPtCnt] = d_EquilibriumDamageThreshold[d_gaussPtCnt];
        d_tmp1Damage[d_gaussPtCnt]          = d_EquilibriumDamage[d_gaussPtCnt];
        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                d_constitutiveMatrix[i][j] = d_initialConstitutiveMatrix[i][j];
            }
        }
    } else {
        d_tmp1DamageThreshold[d_gaussPtCnt] = d_tmp1Tau[d_gaussPtCnt];
        d_tmp1Damage[d_gaussPtCnt]          = d_EquilibriumDamage[d_gaussPtCnt] +
                                     ( d_tmp1Tau[d_gaussPtCnt] - d_EquilibriumTau[d_gaussPtCnt] );
        if ( d_tmp1Damage[d_gaussPtCnt] > d_CriticalDamageVec[d_gaussPtCnt] )
            d_tmp1Damage[d_gaussPtCnt] = d_CriticalDamageVec[d_gaussPtCnt];
        double dam = d_tmp1Damage[d_gaussPtCnt];
        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                d_constitutiveMatrix[i][j] = ( 1.0 - dam ) * d_initialConstitutiveMatrix[i][j];
            }
        }
    }

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
                delta_thermal_stress[i] += d_constitutiveMatrix[i][j] *
                                           ( stra_np1[j] - stra_n[j] - delta_thermal_strain[j] );
                delta_thermal_stress_1[i] += d_initialConstitutiveMatrix[i][j] *
                                             ( stra_np1[j] - stra_n[j] - delta_thermal_strain[j] );
            }
        }

        if ( d_iDebugPrintInfoLevel > 15 ) {
            double vec_norm = 0.0, vec_norm_1 = 0.0;
            for ( int i = 0; i < 6; i++ ) {
                vec_norm += ( delta_thermal_stress[i] * delta_thermal_stress[i] );
                vec_norm_1 += ( delta_thermal_stress_1[i] * delta_thermal_stress_1[i] );
            }
            std::cout << "vec_norm = " << vec_norm << "    vec_norm_1 = " << vec_norm_1
                      << std::endl;
        }

        for ( int i = 0; i < 6; i++ ) {
            stre_np1[i] = stre_n[i] + delta_thermal_stress[i];
        }
    }
}
} // namespace AMP::Operator
