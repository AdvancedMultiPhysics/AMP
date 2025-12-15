#include "AMP/operators/mechanics/GeneralCladThermalCreepPlasticModel.h"
#include "AMP/materials/Property.h"
#include "AMP/operators/mechanics/MechanicsConstants.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Operator {

GeneralCladThermalCreepPlasticModel::GeneralCladThermalCreepPlasticModel(
    std::shared_ptr<MechanicsMaterialModelParameters> params )
    : MechanicsMaterialModel( params ), d_constitutiveMatrix{ { 0 } }
{
    AMP_INSIST( ( ( params.get() ) != nullptr ), "NULL parameter" );

    AMP_INSIST( ( ( ( params->d_db ).get() ) != nullptr ), "NULL database" );

    AMP_INSIST( ( d_useMaterialsLibrary == true ),
                "For GeneralCladThermalCreepPlasticModel, the materials library must be used." );

    d_H = params->d_db->getWithDefault<double>( "Linear_Strain_Hardening", 5.0e8 );

    d_n = params->d_db->getWithDefault<double>( "Plastic_Strain_Exponent", 1.0 );

    default_E = params->d_db->getWithDefault<double>( "Youngs_Modulus", 2.08e11 );

    default_Nu = params->d_db->getWithDefault<double>( "Poissons_Ratio", 0.23 );

    default_alpha = params->d_db->getWithDefault<double>( "THERMAL_EXPANSION_COEFFICIENT", 2.0e-6 );

    d_Sig0 = params->d_db->getWithDefault<double>( "Elastic_Yield_Stress", 3.45e8 );

    default_TEMPERATURE = params->d_db->getWithDefault<double>( "Default_Temperature", 310.0 );

    default_BURNUP = params->d_db->getWithDefault<double>( "Default_Burnup", 0.0 );

    default_OXYGEN_CONCENTRATION =
        params->d_db->getWithDefault<double>( "Default_Oxygen_Concentration", 0.0 );

    d_UseThermalStrain = params->d_db->getWithDefault<bool>( "Use_Thermal_Strain", true );

    AMP_INSIST( ( d_UseThermalStrain == true ), "Thermal_Strain must be used inside the clad." );

    d_UseCreepStrain = params->d_db->getWithDefault<bool>( "Use_Creep_Strain", false );

    d_Is_Init_Called = false;

    for ( auto &elem : d_constitutiveMatrix ) {
        for ( double &j : elem )
            j = 0.;
    }
    d_Delta_Time                 = 0.;
    d_gaussPtCnt                 = 0;
    Total_Gauss_Point            = 0;
    Plastic_Gauss_Point          = 0;
    d_resetReusesRadialReturn    = false;
    d_jacobianReusesRadialReturn = false;
}

void GeneralCladThermalCreepPlasticModel::preNonlinearInit( bool resetReusesRadialReturn,
                                                            bool jacobianReusesRadialReturn )
{
    d_resetReusesRadialReturn    = resetReusesRadialReturn;
    d_jacobianReusesRadialReturn = jacobianReusesRadialReturn;

    d_E.clear();
    d_Nu.clear();
    d_alpha.clear();
    d_StrengthCoeff.clear();
    d_PlasticExponent.clear();

    d_Lambda.clear();
    d_ElPl.clear();

    d_EquilibriumStress.clear();
    d_EquilibriumStrain.clear();
    d_EquilibriumYieldStress.clear();
    d_EquilibriumEffectivePlasticStrain.clear();
    d_EquilibriumTemperature.clear();
    d_EquilibriumCreepStrain.clear();
    d_EquilibriumThermalStrain.clear();

    d_tmp1Stress.clear();
    d_tmp1Strain.clear();
    d_tmp1YieldStress.clear();
    d_tmp1EffectivePlasticStrain.clear();
    d_tmp1Temperature.clear();
    d_tmp1CreepStrain.clear();
    d_tmp1ThermalStrain.clear();

    d_tmp2Stress.clear();
    d_tmp2YieldStress.clear();
    d_tmp2EffectivePlasticStrain.clear();

    Total_Gauss_Point = 0;

    d_Is_Init_Called = true;
}

void GeneralCladThermalCreepPlasticModel::nonlinearInitGaussPointOperation(
    double tempAtGaussPoint )
{
    d_Lambda.push_back( 0 );
    d_ElPl.push_back( 0 );

    d_E.push_back( default_E );
    d_Nu.push_back( default_Nu );
    d_alpha.push_back( default_alpha );

    d_StrengthCoeff.push_back( d_H );
    d_PlasticExponent.push_back( d_n );

    for ( int i = 0; i < 6; i++ ) {
        d_EquilibriumStress.push_back( 0 );
        d_EquilibriumStrain.push_back( 0 );

        d_tmp1Stress.push_back( 0 );
        d_tmp1Strain.push_back( 0 );
    } // end for i

    d_EquilibriumYieldStress.push_back( d_Sig0 );
    d_EquilibriumEffectivePlasticStrain.push_back( 0 );
    d_EquilibriumTemperature.push_back( tempAtGaussPoint );
    d_EquilibriumThermalStrain.push_back( 0.0 );

    d_tmp1YieldStress.push_back( d_Sig0 );
    d_tmp1EffectivePlasticStrain.push_back( 0 );
    d_tmp1Temperature.push_back( tempAtGaussPoint );
    d_tmp1ThermalStrain.push_back( 0.0 );

    d_EquilibriumCreepStrain.push_back( 0 );
    d_tmp1CreepStrain.push_back( 0 );

    if ( !d_jacobianReusesRadialReturn ) {
        for ( int i = 0; i < 6; i++ ) {
            d_tmp2Stress.push_back( 0 );
        } // end for i

        d_tmp2YieldStress.push_back( d_Sig0 );
        d_tmp2EffectivePlasticStrain.push_back( 0 );
    }

    Total_Gauss_Point++;
}

void GeneralCladThermalCreepPlasticModel::globalReset()
{
    AMP_INSIST( ( d_resetReusesRadialReturn == true ), "Inconsistent options!" );

    AMP_INSIST( ( d_Is_Init_Called == true ), "Init must be called before globalReset!" );

    d_EquilibriumStress                 = d_tmp1Stress;
    d_EquilibriumStrain                 = d_tmp1Strain;
    d_EquilibriumYieldStress            = d_tmp1YieldStress;
    d_EquilibriumEffectivePlasticStrain = d_tmp1EffectivePlasticStrain;
    d_EquilibriumTemperature            = d_tmp1Temperature;
    d_EquilibriumCreepStrain            = d_tmp1CreepStrain;
    d_EquilibriumThermalStrain          = d_tmp1ThermalStrain;
}

void GeneralCladThermalCreepPlasticModel::nonlinearResetGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    AMP_INSIST( ( d_Is_Init_Called == true ),
                "Init must be called before nonlinearResetGaussPointOperation!" );

    AMP_INSIST( ( strain[Mechanics::TEMPERATURE].empty() == false ),
                "Temperature must be an active variable." );

    computeEvalv( strain );

    double stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    d_tmp1Temperature[d_gaussPtCnt] = strain[Mechanics::TEMPERATURE][0];

    double Temp_np1 = strain[Mechanics::TEMPERATURE][0];

    double net_stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        net_stra_np1[i] = stra_np1[i];
    }

    if ( d_UseThermalStrain == true ) {
        double thermal_strain =
            d_tmp1ThermalStrain[d_gaussPtCnt] - d_EquilibriumThermalStrain[d_gaussPtCnt];

        for ( int i = 0; i < 3; i++ ) {
            net_stra_np1[i] = net_stra_np1[i] - thermal_strain;
        }
    }

    if ( d_UseCreepStrain == true ) {
        double stress_n[6], delta_creep_strain[6];

        d_Delta_Time = d_currentTime - d_previousTime;

        double creep_strain_prev = d_EquilibriumCreepStrain[d_gaussPtCnt];

        for ( int i = 0; i < 6; i++ ) {
            stress_n[i] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + i];
        }

        computeCreepStrain(
            Temp_np1, stress_n, creep_strain_prev, delta_creep_strain, net_stra_np1 );

        for ( int i = 0; i < 6; i++ ) {
            net_stra_np1[i] = net_stra_np1[i] - delta_creep_strain[i];
        }
    }

    double *stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    radialReturn( &net_stra_np1[0],
                  stress,
                  &( d_tmp1YieldStress[d_gaussPtCnt] ),
                  &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] ) );
}

void GeneralCladThermalCreepPlasticModel::nonlinearJacobianGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    AMP_INSIST( ( d_Is_Init_Called == true ),
                "Init must be called before nonlinearJacobianGaussPointOperation!" );

    AMP_INSIST( ( strain[Mechanics::TEMPERATURE].empty() == false ),
                "Temperature must be an active variable." );

    computeEvalv( strain );

    double stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    double Temp_np1 = strain[Mechanics::TEMPERATURE][0];

    double net_stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        net_stra_np1[i] = stra_np1[i];
    }

    if ( d_UseThermalStrain == true ) {
        double thermal_strain =
            d_tmp1ThermalStrain[d_gaussPtCnt] - d_EquilibriumThermalStrain[d_gaussPtCnt];

        for ( int i = 0; i < 3; i++ ) {
            net_stra_np1[i] = net_stra_np1[i] - thermal_strain;
        }
    }

    if ( d_UseCreepStrain == true ) {
        double stress_n[6], delta_creep_strain[6];

        d_Delta_Time = d_currentTime - d_previousTime;

        double creep_strain_prev = d_EquilibriumCreepStrain[d_gaussPtCnt];

        for ( int i = 0; i < 6; i++ ) {
            stress_n[i] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + i];
        }

        computeCreepStrain(
            Temp_np1, stress_n, creep_strain_prev, delta_creep_strain, net_stra_np1 );

        for ( int i = 0; i < 6; i++ ) {
            net_stra_np1[i] = net_stra_np1[i] - delta_creep_strain[i];
        }
    }

    double *stress = &( d_tmp2Stress[6 * d_gaussPtCnt] );

    radialReturn( &net_stra_np1[0],
                  stress,
                  &( d_tmp2YieldStress[d_gaussPtCnt] ),
                  &( d_tmp2EffectivePlasticStrain[d_gaussPtCnt] ) );
}

void GeneralCladThermalCreepPlasticModel::postNonlinearReset()
{
    AMP_INSIST( ( d_Is_Init_Called == true ), "Init must be called before postNonlinearReset!" );

    d_EquilibriumStress                 = d_tmp1Stress;
    d_EquilibriumStrain                 = d_tmp1Strain;
    d_EquilibriumYieldStress            = d_tmp1YieldStress;
    d_EquilibriumEffectivePlasticStrain = d_tmp1EffectivePlasticStrain;
    d_EquilibriumTemperature            = d_tmp1Temperature;
    d_EquilibriumCreepStrain            = d_tmp1CreepStrain;
    d_EquilibriumThermalStrain          = d_tmp1ThermalStrain;
}

void GeneralCladThermalCreepPlasticModel::getInternalStress(
    const std::vector<std::vector<double>> &strain, double *&stress )
{
    AMP_INSIST( ( d_Is_Init_Called == true ), "Init must be called before getInternalStress!" );

    AMP_INSIST( ( strain[Mechanics::TEMPERATURE].empty() == false ),
                "Temperature must be an active variable." );

    computeEvalv( strain );

    double stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    d_tmp1Temperature[d_gaussPtCnt] = strain[Mechanics::TEMPERATURE][0];

    double Temp_np1 = strain[Mechanics::TEMPERATURE][0];

    double net_stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        net_stra_np1[i] = stra_np1[i];
    }

    if ( d_UseThermalStrain == true ) {
        double thermal_strain =
            d_tmp1ThermalStrain[d_gaussPtCnt] - d_EquilibriumThermalStrain[d_gaussPtCnt];

        for ( int i = 0; i < 3; i++ ) {
            net_stra_np1[i] = net_stra_np1[i] - thermal_strain;
        }
    }

    if ( d_UseCreepStrain == true ) {
        double stress_n[6], delta_creep_strain[6];

        d_Delta_Time = d_currentTime - d_previousTime;

        double creep_strain_prev = d_EquilibriumCreepStrain[d_gaussPtCnt];

        for ( int i = 0; i < 6; i++ ) {
            stress_n[i] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + i];
        }

        computeCreepStrain(
            Temp_np1, stress_n, creep_strain_prev, delta_creep_strain, net_stra_np1 );

        for ( int i = 0; i < 6; i++ ) {
            net_stra_np1[i] = net_stra_np1[i] - delta_creep_strain[i];
        }
    }

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    radialReturn( &net_stra_np1[0],
                  stress,
                  &( d_tmp1YieldStress[d_gaussPtCnt] ),
                  &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] ) );
}

double GeneralCladThermalCreepPlasticModel::help_compute_E1( const double Temp_np1,
                                                             double effective_stress )
{
    double Q = 15027.0, R = 1.987, A = 1.0e-12, n = 2.0;
    double dt           = d_Delta_Time;
    double scale_stress = 1.0, scale_strain = 1.0;

    effective_stress = effective_stress * scale_stress;

    double e_dot =
        A * std::pow( effective_stress, n ) * exp( -Q / ( R * Temp_np1 ) ) * scale_strain;
    double delta_e = e_dot * dt;

    return ( delta_e );
}

int GeneralCladThermalCreepPlasticModel::det_sign( double a ) { return a >= 0.0 ? 1 : -1; }

double GeneralCladThermalCreepPlasticModel::compute_dE1_dsig_e( double effective_stress_trial,
                                                                double effective_stress,
                                                                double Temp_np1,
                                                                double G )
{
    const double Q = 15027.0, R = 1.987, A = 1.0e-12, n = 2.0;
    const double dt           = d_Delta_Time;
    const double scale_stress = 1.0, scale_strain = 1.0;

    const double delta_e = help_compute_E1( Temp_np1, effective_stress );
    const double term1   = effective_stress_trial / ( effective_stress + ( 3.0 * G * delta_e ) );
    const double term2   = ( effective_stress * effective_stress_trial ) /
                         ( ( effective_stress + ( 3.0 * G * delta_e ) ) *
                           ( effective_stress + ( 3.0 * G * delta_e ) ) );

    effective_stress               = effective_stress * scale_stress;
    const double d_delta_e_d_sig_e = A * dt * exp( -Q / ( R * Temp_np1 ) ) * n *
                                     std::pow( effective_stress, ( n - 1.0 ) ) * scale_strain;
    const double term3      = 1.0 + ( 3.0 * G * d_delta_e_d_sig_e );
    const double dE1_dsig_e = 1.0 - term1 + ( term2 * term3 );

    return dE1_dsig_e;
}

void GeneralCladThermalCreepPlasticModel::computeCreepStrain( const double Temp_np1,
                                                              const double stress_n[6],
                                                              const double creep_strain_prev,
                                                              double delta_creep_strain[6],
                                                              double net_stra_np1[6] )
{
    double one3   = 1.0 / 3.0;
    double three2 = 3.0 / 2.0;
    double cr_n   = creep_strain_prev;
    double cr_np1;
    double tol = 1.0e-8;
    double stre_n[6], deviatoric_stress[6];
    double effective_stress, hydrostatic_stress;
    double effective_stress_trial, deviatoric_stress_trial[6];
    double delta_strain[6], hydrostatic_delta_strain, deviatoric_delta_strain[6];
    const double *stra_n = &( d_EquilibriumStrain[6 * d_gaussPtCnt] );
    double G, pass_E = d_E[d_gaussPtCnt], pass_Nu = d_Nu[d_gaussPtCnt];
    int bisection_converged = 0, newton_converged = 0;

    if ( d_iDebugPrintInfoLevel > 9 ) {
        std::cout << "currentTime = " << d_currentTime << " previousTime = " << d_previousTime
                  << std::endl;
    }

    G = pass_E / ( 2.0 * ( 1.0 + pass_Nu ) );

    for ( int i = 0; i < 6; i++ ) {
        delta_creep_strain[i] = 0.0;
        stre_n[i]             = stress_n[i];
    }

    for ( int i = 0; i < 3; i++ ) {
        delta_strain[i]     = net_stra_np1[i] - stra_n[i];
        delta_strain[i + 3] = 0.5 * ( net_stra_np1[i + 3] - stra_n[i + 3] );
    }

    hydrostatic_delta_strain = delta_strain[0] + delta_strain[1] + delta_strain[2];
    for ( int i = 0; i < 3; i++ ) {
        deviatoric_delta_strain[i]     = delta_strain[i] - ( one3 * hydrostatic_delta_strain );
        deviatoric_delta_strain[i + 3] = delta_strain[i + 3];
    }

    hydrostatic_stress = stre_n[0] + stre_n[1] + stre_n[2];
    for ( int i = 0; i < 3; i++ ) {
        deviatoric_stress[i]     = stre_n[i] - ( one3 * hydrostatic_stress );
        deviatoric_stress[i + 3] = stre_n[i + 3];
    }

    for ( int i = 0; i < 6; i++ ) {
        deviatoric_stress_trial[i] =
            deviatoric_stress[i] + ( 2.0 * G * deviatoric_delta_strain[i] );
    }

    effective_stress_trial =
        std::sqrt( three2 * ( ( deviatoric_stress_trial[0] * deviatoric_stress_trial[0] ) +
                              ( deviatoric_stress_trial[1] * deviatoric_stress_trial[1] ) +
                              ( deviatoric_stress_trial[2] * deviatoric_stress_trial[2] ) +
                              ( 2.0 * deviatoric_stress_trial[3] * deviatoric_stress_trial[3] ) +
                              ( 2.0 * deviatoric_stress_trial[4] * deviatoric_stress_trial[4] ) +
                              ( 2.0 * deviatoric_stress_trial[5] * deviatoric_stress_trial[5] ) ) );

    effective_stress = tol;

    for ( int jkl = 0; jkl < 1000; jkl++ ) {
        double delta_e = help_compute_E1( Temp_np1, effective_stress );
        double E1      = effective_stress - ( effective_stress_trial /
                                         ( 1.0 + ( ( 3.0 * G * delta_e ) / effective_stress ) ) );
        double dE1_dsig_e =
            compute_dE1_dsig_e( effective_stress_trial, effective_stress, Temp_np1, G );
        double delta_effective_stress = E1 / dE1_dsig_e;
        effective_stress              = effective_stress - ( E1 / dE1_dsig_e );

        if ( fabs( effective_stress ) < tol ) {
            newton_converged = 1;
            break;
        }
        if ( fabs( delta_effective_stress / effective_stress ) < tol ) {
            newton_converged = 1;
            break;
        }
    }

    if ( newton_converged == 0 ) {

        double effective_stress_1 = effective_stress_trial + ( 1.0e2 );
        double effective_stress_2 = tol;

        for ( int jkl = 0; jkl < 1000; jkl++ ) {
            double delta_e_1 = help_compute_E1( Temp_np1, effective_stress_1 );
            double E1_1 =
                effective_stress_1 - ( effective_stress_trial /
                                       ( 1.0 + ( ( 3.0 * G * delta_e_1 ) / effective_stress_1 ) ) );

            double delta_e_2 = help_compute_E1( Temp_np1, effective_stress_2 );
            double E1_2 =
                effective_stress_2 - ( effective_stress_trial /
                                       ( 1.0 + ( ( 3.0 * G * delta_e_2 ) / effective_stress_2 ) ) );

            if ( det_sign( E1_1 ) == det_sign( E1_2 ) ) {
                AMP_ERROR( "Condition did not satisfy" );
            }

            effective_stress = ( effective_stress_1 + effective_stress_2 ) / 2.0;
            double delta_e   = help_compute_E1( Temp_np1, effective_stress );
            double E1 =
                effective_stress -
                ( effective_stress_trial / ( 1.0 + ( ( 3.0 * G * delta_e ) / effective_stress ) ) );

            if ( ( ( det_sign( E1 ) * det_sign( E1_1 ) ) < 0 ) &&
                 ( ( det_sign( E1 ) * det_sign( E1_2 ) ) > 0 ) ) {
                effective_stress_2 = effective_stress;
            }

            if ( ( ( det_sign( E1 ) * det_sign( E1_2 ) ) < 0 ) &&
                 ( ( det_sign( E1 ) * det_sign( E1_1 ) ) > 0 ) ) {
                effective_stress_1 = effective_stress;
            }

            if ( fabs( E1 / effective_stress ) < tol ) {
                bisection_converged = 1;
                break;
            }
        }
    }

    if ( ( bisection_converged == 0 ) && ( newton_converged == 0 ) ) {
        AMP_ERROR( "Creep code did not converge for both newton and bisection iterations" );
    }

    double delta_e = help_compute_E1( Temp_np1, effective_stress );

    cr_np1 = cr_n + delta_e;

    d_tmp1CreepStrain[d_gaussPtCnt] = cr_np1;

    for ( int i = 0; i < 6; i++ ) {
        if ( effective_stress > tol ) {
            deviatoric_stress[i] = ( effective_stress * deviatoric_stress_trial[i] ) /
                                   ( effective_stress + ( 3.0 * G * delta_e ) );
        } else {
            deviatoric_stress[i] = 0.0;
        }
    }

    for ( int i = 0; i < 6; i++ ) {
        if ( effective_stress < tol ) {
            delta_creep_strain[i] = 0.0;
        } else {
            delta_creep_strain[i] = three2 * delta_e * deviatoric_stress[i] / effective_stress;
        }
    }
}

void GeneralCladThermalCreepPlasticModel::computeEvalv(
    const std::vector<std::vector<double>> &strain )
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

        d_E[d_gaussPtCnt]                 = YM[0];
        d_Nu[d_gaussPtCnt]                = PR[0];
        d_alpha[d_gaussPtCnt]             = TEC[0];
        d_tmp1ThermalStrain[d_gaussPtCnt] = d_alpha[d_gaussPtCnt];

        tempVec[0] = d_EquilibriumTemperature[d_gaussPtCnt];
        d_material->property( tecString )->evalv( TEC, {}, args );
        d_EquilibriumThermalStrain[d_gaussPtCnt] = TEC[0];
    }
}

void GeneralCladThermalCreepPlasticModel::getConstitutiveMatrix( double *&constitutiveMatrix )
{
    AMP_INSIST( ( d_Is_Init_Called == true ), "Init must be called before getConstitutiveMatrix!" );

    // Consistent Tangent

    double E  = d_E[d_gaussPtCnt];
    double Nu = d_Nu[d_gaussPtCnt];
    double H  = d_StrengthCoeff[d_gaussPtCnt];
    double n  = d_PlasticExponent[d_gaussPtCnt];
    // double Sig0 = d_Sig0;

    double lambda = d_Lambda[d_gaussPtCnt];
    int el_or_pl  = d_ElPl[d_gaussPtCnt];

    const double *stre_np1;
    double ystre_np1;
    double eph_bar_plas_np1;

    if ( d_jacobianReusesRadialReturn ) {
        stre_np1         = &( d_tmp1Stress[6 * d_gaussPtCnt] );
        ystre_np1        = d_tmp1YieldStress[d_gaussPtCnt];
        eph_bar_plas_np1 = d_tmp1EffectivePlasticStrain[d_gaussPtCnt];
    } else {
        stre_np1         = &( d_tmp2Stress[6 * d_gaussPtCnt] );
        ystre_np1        = d_tmp2YieldStress[d_gaussPtCnt];
        eph_bar_plas_np1 = d_tmp2EffectivePlasticStrain[d_gaussPtCnt];
    }

    constitutiveMatrix = &( d_constitutiveMatrix[0][0] );

    double sig_np1[6];
    double ephbp_np1, sigy_np1;
    double G, K, Ep, sq23;
    double one3, two3;
    // double three2 = 1.5, one2 = 0.5;
    double sig_dev[6], n_dir[6];
    double q_np1, sig_np1_kk, q_trial, lam;
    double beta, gamma, gamma_bar, term1, term2;

    one3 = 1.0 / 3.0;
    two3 = 2.0 / 3.0;
    sq23 = std::sqrt( two3 );

    // If the stress is within the elastic range.
    // Only the elastic tangent is computed.
    if ( el_or_pl == 0 ) {
        G = E / ( 2.0 * ( 1.0 + Nu ) );
        K = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) );

        // Initializing the tangent matrix as zero.
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
                d_constitutiveMatrix[i][j] += ( K - ( two3 * G ) );
            }
        }

        return;
    }

    // Stress inside the plastic range : The elasto-plastic tangent is calculated.
    ephbp_np1 = eph_bar_plas_np1; // Effective plastic strain.
    sigy_np1  = ystre_np1;        // Yield stress.
    lam       = lambda;
    G         = E / ( 2.0 * ( 1.0 + Nu ) );           // of Elastic
    K         = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) ); // and other
    Ep        = H;                                    // constants.
    for ( int i = 0; i < 6; i++ ) {
        sig_np1[i] = stre_np1[i]; // Stress.
    }

    // Hydrostatic component of the stress tensor.
    sig_np1_kk = sig_np1[0] + sig_np1[1] + sig_np1[2];
    // Deviatoric component of the stress tensor.
    for ( int i = 0; i < 3; i++ ) {
        sig_dev[i]     = sig_np1[i] - ( one3 * sig_np1_kk );
        sig_dev[i + 3] = sig_np1[i + 3];
    }

    // The effective stress.
    q_np1 = std::sqrt( ( sig_dev[0] * sig_dev[0] ) + ( sig_dev[1] * sig_dev[1] ) +
                       ( sig_dev[2] * sig_dev[2] ) + ( 2.0 * sig_dev[3] * sig_dev[3] ) +
                       ( 2.0 * sig_dev[4] * sig_dev[4] ) + ( 2.0 * sig_dev[5] * sig_dev[5] ) );

    // The normal direction.
    for ( int i = 0; i < 6; i++ ) {
        n_dir[i] = sig_dev[i] / q_np1;
    }

    // The trial effective stress.
    q_trial = q_np1 + ( 2.0 * G * lam );

    beta               = sq23 * ( sigy_np1 / q_trial );
    double kappa_prime = sq23 * n * Ep * std::pow( ephbp_np1, ( n - 1.0 ) );
    gamma              = 3.0 * G / ( ( 3.0 * G ) + kappa_prime );
    gamma_bar          = gamma - ( 1.0 - beta );
    term1              = 2.0 * G * beta;
    term2              = 2.0 * G * gamma_bar;

    // Initiaization of the constitutive matrix.
    for ( auto &elem : d_constitutiveMatrix ) {
        for ( double &j : elem ) {
            j = 0.0;
        }
    }

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            d_constitutiveMatrix[i][j] += ( K - ( one3 * term1 ) );
        }
    }

    for ( int i = 0; i < 3; i++ ) {
        d_constitutiveMatrix[i][i] += term1;
        d_constitutiveMatrix[i + 3][i + 3] += ( 0.5 * term1 );
    }

    for ( int i = 0; i < 6; i++ ) {
        for ( int j = 0; j < 6; j++ ) {
            d_constitutiveMatrix[i][j] -= ( term2 * n_dir[i] * n_dir[j] );
        }
    }
}

void GeneralCladThermalCreepPlasticModel::radialReturn( const double *stra_np1,
                                                        double *stre_np1,
                                                        double *ystre_np1,
                                                        double *eph_bar_plas_np1 )
{
    double E    = d_E[d_gaussPtCnt];
    double Nu   = d_Nu[d_gaussPtCnt];
    double H    = d_StrengthCoeff[d_gaussPtCnt];
    double n    = d_PlasticExponent[d_gaussPtCnt];
    double Sig0 = d_Sig0;

    const double *stre_n  = &( d_EquilibriumStress[6 * d_gaussPtCnt] );
    const double *stra_n  = &( d_EquilibriumStrain[6 * d_gaussPtCnt] );
    double ystre_n        = d_EquilibriumYieldStress[d_gaussPtCnt];
    double eph_bar_plas_n = d_EquilibriumEffectivePlasticStrain[d_gaussPtCnt];

    double *lambda = &( d_Lambda[d_gaussPtCnt] );
    int *el_or_pl  = &( d_ElPl[d_gaussPtCnt] );

    double sig_n[6], d_strain[6];
    double ephbp_n, ephbp_np1, sigy_n, sigy_np1, lam;
    double G, K, Ep, sq23;
    double one3, two3, tol = 0.000000001;
    double deph_dev[6], sig_dev[6], sig_trial_dev[6], n_dir[6];
    double deph_kk, sig_kk, sig_trial_kk, q_trial, twoG, phi;

    double dstra[6];

    for ( int i = 0; i < 3; i++ ) {
        dstra[i]     = stra_np1[i] - stra_n[i];
        dstra[i + 3] = 0.5 * ( stra_np1[i + 3] - stra_n[i + 3] );
        // std::cout << "dstra[" << i << "]=" << dstra[i] << std::endl;
    }

    one3    = 1.0 / 3.0;
    two3    = 2.0 / 3.0;
    sq23    = std::sqrt( two3 );
    ephbp_n = eph_bar_plas_n;             // Effective plastic strain at the previous time step.
    sigy_n  = ystre_n;                    // Yield stress at the previous time step.
    G       = E / ( 2.0 * ( 1.0 + Nu ) ); // of Elastic
    K       = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) ); // and other
    Ep      = H;                                    // constants.
    for ( int i = 0; i < 6; i++ ) {
        sig_n[i]    = stre_n[i]; // Stress at the previous time step.
        d_strain[i] = dstra[i];  // Change in strain (strain increment).
    }

    // Compute Traces
    deph_kk = d_strain[0] + d_strain[1] + d_strain[2];
    sig_kk  = sig_n[0] + sig_n[1] + sig_n[2];

    // Compute deviatoric components.
    for ( int i = 0; i < 3; i++ ) {
        deph_dev[i]     = d_strain[i] - ( one3 * deph_kk );
        sig_dev[i]      = sig_n[i] - ( one3 * sig_kk );
        deph_dev[i + 3] = d_strain[i + 3];
        sig_dev[i + 3]  = sig_n[i + 3];
    }

    // Trial stress calculation.
    twoG = 2.0 * G;
    for ( int i = 0; i < 6; i++ ) {
        sig_trial_dev[i] = sig_dev[i] + ( twoG * deph_dev[i] );
    }
    sig_trial_kk = sig_kk + ( 3.0 * K * deph_kk );

    // Compute the trial effective stress.
    q_trial = std::sqrt(
        ( sig_trial_dev[0] * sig_trial_dev[0] ) + ( sig_trial_dev[1] * sig_trial_dev[1] ) +
        ( sig_trial_dev[2] * sig_trial_dev[2] ) + ( 2.0 * sig_trial_dev[3] * sig_trial_dev[3] ) +
        ( 2.0 * sig_trial_dev[4] * sig_trial_dev[4] ) +
        ( 2.0 * sig_trial_dev[5] * sig_trial_dev[5] ) );

    ephbp_np1 = ephbp_n;                       // Initializing the equivalent plastic strain
    sigy_np1  = sigy_n;                        // Initializing the yield stress
    phi       = q_trial - ( sq23 * sigy_np1 ); // Computing the value of the yield function.
    // std::cout << "phi=" << phi << " q_trial=" << q_trial << " sigy_np1=" << sigy_np1 <<
    // std::endl;

    // Stress within the elastic range.
    if ( phi < 0.0 ) {
        *el_or_pl = 0;
        for ( int i = 0; i < 3; i++ ) {
            stre_np1[i]     = sig_trial_dev[i] + ( one3 * sig_trial_kk );
            stre_np1[i + 3] = sig_trial_dev[i + 3];
        }
        *ystre_np1        = ystre_n;
        *eph_bar_plas_np1 = eph_bar_plas_n;
        *lambda           = 0.0;
        return;
    }

    // std::cout << "Entered the Plasticity." << std::endl;
    // Stress outside the elastic range, inside the plastic range.

    Plastic_Gauss_Point++;

    *el_or_pl = 1;
    for ( int i = 0; i < 6; i++ ) {
        n_dir[i] = sig_trial_dev[i] / q_trial; // Computing the normal direction.
    }

    // Updating the plastic parameters.
    lam = 0.0;
    for ( int i = 0; i < 1000; i++ ) {
        ephbp_np1        = ephbp_n + ( sq23 * lam );
        double kappa_np1 = Sig0 + ( Ep * std::pow( ephbp_np1, n ) );
        double g_np1     = q_trial - ( twoG * lam ) - ( sq23 * kappa_np1 );
        double g_prime   = -( twoG + ( two3 * Ep * n * std::pow( ephbp_np1, ( n - 1.0 ) ) ) );
        double delta_lam = g_np1 / g_prime;
        lam              = lam - delta_lam;
        if ( fabs( delta_lam / lam ) < tol )
            break;
        if ( i > 995 ) {
            AMP_ERROR( "The radial return did not converge" );
        }
    }
    ephbp_np1 = ephbp_n + ( sq23 * lam );
    sigy_np1  = Sig0 + ( Ep * std::pow( ephbp_np1, n ) );

    // Updating the stress.
    for ( int i = 0; i < 6; i++ ) {
        sig_dev[i] = sig_trial_dev[i] - ( twoG * lam * n_dir[i] );
    }

    for ( int i = 0; i < 3; i++ ) {
        stre_np1[i]     = sig_dev[i] + ( one3 * sig_trial_kk );
        stre_np1[i + 3] = sig_dev[i + 3];
    }

    *ystre_np1        = sigy_np1;
    *eph_bar_plas_np1 = ephbp_np1;
    *lambda           = lam;
}

void GeneralCladThermalCreepPlasticModel::postNonlinearAssembly()
{
    if ( Total_Gauss_Point == 0 ) {
        std::cout << "Total number of gauss points are zero." << std::endl;
    } else {
        double Plastic_Fraction = ( (double) Plastic_Gauss_Point ) / ( (double) Total_Gauss_Point );
        Plastic_Fraction        = Plastic_Fraction * 100.0;
        if ( d_iDebugPrintInfoLevel > 5 ) {
            std::cout << "Fraction = " << Plastic_Fraction << "% Plastic = " << Plastic_Gauss_Point
                      << " Total = " << Total_Gauss_Point << " Gauss Points." << std::endl;
        }
    }
}
} // namespace AMP::Operator
