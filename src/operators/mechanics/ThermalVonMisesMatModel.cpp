#include "AMP/operators/mechanics/ThermalVonMisesMatModel.h"
#include "AMP/materials/Property.h"
#include "AMP/operators/mechanics/MechanicsConstants.h"
#include "AMP/utils/Utilities.h"


namespace AMP::Operator {

ThermalVonMisesMatModel::ThermalVonMisesMatModel(
    std::shared_ptr<MechanicsMaterialModelParameters> params )
    : MechanicsMaterialModel( params ), d_constitutiveMatrix{ { 0 } }
{
    AMP_INSIST( ( ( params.get() ) != nullptr ), "NULL parameter" );

    AMP_INSIST( ( ( ( params->d_db ).get() ) != nullptr ), "NULL database" );

    if ( d_useMaterialsLibrary == false ) {

        AMP_INSIST( params->d_db->keyExists( "Youngs_Modulus" ), "Missing key: Youngs_Modulus" );

        AMP_INSIST( params->d_db->keyExists( "Poissons_Ratio" ), "Missing key: Poissons_Ratio" );

        AMP_INSIST( params->d_db->keyExists( "THERMAL_EXPANSION_COEFFICIENT" ),
                    "Missing key: THERMAL_EXPANSION_COEFFICIENT" );
    }

    AMP_INSIST( params->d_db->keyExists( "Linear_Strain_Hardening" ),
                "Missing key: Linear_Strain_Hardening" );

    AMP_INSIST( params->d_db->keyExists( "Elastic_Yield_Stress" ),
                "Missing key: Elastic_Yield_Stress" );

    if ( d_useMaterialsLibrary == false ) {

        default_E = params->d_db->getScalar<double>( "Youngs_Modulus" );

        default_Nu = params->d_db->getScalar<double>( "Poissons_Ratio" );

        default_alpha = params->d_db->getScalar<double>( "THERMAL_EXPANSION_COEFFICIENT" );
    }

    d_H = params->d_db->getScalar<double>( "Linear_Strain_Hardening" );

    d_Sig0 = params->d_db->getScalar<double>( "Elastic_Yield_Stress" );

    default_TEMPERATURE = params->d_db->getWithDefault<double>( "Default_Temperature", 310.0 );

    default_BURNUP = params->d_db->getWithDefault<double>( "Default_Burnup", 0.0 );

    default_OXYGEN_CONCENTRATION =
        params->d_db->getWithDefault<double>( "Default_Oxygen_Concentration", 0.0 );

    d_Is_Init_Called = false;

    for ( auto &elem : d_constitutiveMatrix ) {
        for ( double &j : elem )
            j = 0.;
    }
    d_gaussPtCnt                 = 0;
    Total_Gauss_Point            = 0;
    Plastic_Gauss_Point          = 0;
    d_resetReusesRadialReturn    = false;
    d_jacobianReusesRadialReturn = false;
}

void ThermalVonMisesMatModel::preNonlinearInit( bool resetReusesRadialReturn,
                                                bool jacobianReusesRadialReturn )
{
    d_resetReusesRadialReturn    = resetReusesRadialReturn;
    d_jacobianReusesRadialReturn = jacobianReusesRadialReturn;

    d_E.clear();
    d_Nu.clear();
    d_alpha.clear();

    d_Lambda.clear();
    d_ElPl.clear();

    d_EquilibriumStress.clear();
    d_EquilibriumStrain.clear();
    d_EquilibriumYieldStress.clear();
    d_EquilibriumEffectivePlasticStrain.clear();
    d_EquilibriumTemperature.clear();

    d_tmp1Stress.clear();
    d_tmp1Strain.clear();
    d_tmp1YieldStress.clear();
    d_tmp1EffectivePlasticStrain.clear();
    d_tmp1Temperature.clear();

    d_tmp2Stress.clear();
    d_tmp2YieldStress.clear();
    d_tmp2EffectivePlasticStrain.clear();

    Total_Gauss_Point = 0;

    d_Is_Init_Called = true;
}

void ThermalVonMisesMatModel::nonlinearInitGaussPointOperation( double tempAtGaussPoint )
{
    d_Lambda.push_back( 0 );
    d_ElPl.push_back( 0 );

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

    d_EquilibriumYieldStress.push_back( d_Sig0 );
    d_EquilibriumEffectivePlasticStrain.push_back( 0 );
    d_EquilibriumTemperature.push_back( tempAtGaussPoint );

    d_tmp1YieldStress.push_back( d_Sig0 );
    d_tmp1EffectivePlasticStrain.push_back( 0 );
    d_tmp1Temperature.push_back( tempAtGaussPoint );

    if ( !d_jacobianReusesRadialReturn ) {
        for ( int i = 0; i < 6; i++ ) {
            d_tmp2Stress.push_back( 0 );
        } // end for i

        d_tmp2YieldStress.push_back( d_Sig0 );
        d_tmp2EffectivePlasticStrain.push_back( 0 );
    }

    Total_Gauss_Point++;
}

void ThermalVonMisesMatModel::globalReset()
{
    AMP_INSIST( ( d_resetReusesRadialReturn == true ), "Inconsistent options!" );

    AMP_INSIST( ( d_Is_Init_Called == true ), "Init must be called before globalReset!" );

    d_EquilibriumStress                 = d_tmp1Stress;
    d_EquilibriumStrain                 = d_tmp1Strain;
    d_EquilibriumYieldStress            = d_tmp1YieldStress;
    d_EquilibriumEffectivePlasticStrain = d_tmp1EffectivePlasticStrain;
    d_EquilibriumTemperature            = d_tmp1Temperature;
}

void ThermalVonMisesMatModel::nonlinearResetGaussPointOperation(
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

    double Temp_n = d_EquilibriumTemperature[d_gaussPtCnt];

    double net_stra_np1[6];

    for ( int i = 0; i < 3; i++ ) {
        net_stra_np1[i] = stra_np1[i] - ( d_alpha[d_gaussPtCnt] * ( Temp_np1 - Temp_n ) );

        net_stra_np1[i + 3] = stra_np1[i + 3];
    }

    double *stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    radialReturn( &net_stra_np1[0],
                  stress,
                  &( d_tmp1YieldStress[d_gaussPtCnt] ),
                  &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] ) );
}

void ThermalVonMisesMatModel::nonlinearJacobianGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    AMP_INSIST( ( d_Is_Init_Called == true ),
                "Init must be called before nonlinearJacobianGaussPointOperation!" );

    AMP_INSIST( ( strain[Mechanics::TEMPERATURE].empty() == false ),
                "Temperature must be an active variable." );

    computeEvalv( strain );

    double *stress = &( d_tmp2Stress[6 * d_gaussPtCnt] );

    double stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    double Temp_np1 = strain[Mechanics::TEMPERATURE][0];

    double Temp_n = d_EquilibriumTemperature[d_gaussPtCnt];

    double net_stra_np1[6];

    for ( int i = 0; i < 3; i++ ) {
        net_stra_np1[i] = stra_np1[i] - ( d_alpha[d_gaussPtCnt] * ( Temp_np1 - Temp_n ) );

        net_stra_np1[i + 3] = stra_np1[i + 3];
    }

    radialReturn( &net_stra_np1[0],
                  stress,
                  &( d_tmp2YieldStress[d_gaussPtCnt] ),
                  &( d_tmp2EffectivePlasticStrain[d_gaussPtCnt] ) );
}

void ThermalVonMisesMatModel::postNonlinearReset()
{
    AMP_INSIST( ( d_Is_Init_Called == true ), "Init must be called before postNonlinearReset!" );

    d_EquilibriumStress                 = d_tmp1Stress;
    d_EquilibriumStrain                 = d_tmp1Strain;
    d_EquilibriumYieldStress            = d_tmp1YieldStress;
    d_EquilibriumEffectivePlasticStrain = d_tmp1EffectivePlasticStrain;
    d_EquilibriumTemperature            = d_tmp1Temperature;
}

void ThermalVonMisesMatModel::getInternalStress( const std::vector<std::vector<double>> &strain,
                                                 double *&stress )
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

    double Temp_n = d_EquilibriumTemperature[d_gaussPtCnt];

    double net_stra_np1[6];

    // double change_T = Temp_np1 - Temp_n;

    // double alpha = d_alpha[d_gaussPtCnt];

    // std::cout << "Delta_Temperature=" << change_T << " TEC=" << alpha << std::endl;

    for ( int i = 0; i < 3; i++ ) {
        net_stra_np1[i] = stra_np1[i] - ( d_alpha[d_gaussPtCnt] * ( Temp_np1 - Temp_n ) );

        net_stra_np1[i + 3] = stra_np1[i + 3];

        // std::cout<<"stra_np1["<<i<<"]="<<stra_np1[i]<<"
        // net_stra_np1["<<i<<"]="<<net_stra_np1[i]<<std::endl;
    }

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    radialReturn( &net_stra_np1[0],
                  stress,
                  &( d_tmp1YieldStress[d_gaussPtCnt] ),
                  &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] ) );
}

void ThermalVonMisesMatModel::computeEvalv( const std::vector<std::vector<double>> &strain )
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

void ThermalVonMisesMatModel::getConstitutiveMatrix( double *&constitutiveMatrix )
{
    AMP_INSIST( ( d_Is_Init_Called == true ), "Init must be called before getConstitutiveMatrix!" );

    // Consistent Tangent

    double E  = d_E[d_gaussPtCnt];
    double Nu = d_Nu[d_gaussPtCnt];
    double H  = d_H;
    // double Sig0 = d_Sig0;

    double lambda = d_Lambda[d_gaussPtCnt];
    int el_or_pl  = d_ElPl[d_gaussPtCnt];

    const double *stre_np1;
    double ystre_np1;

    if ( d_jacobianReusesRadialReturn ) {
        stre_np1  = &( d_tmp1Stress[6 * d_gaussPtCnt] );
        ystre_np1 = d_tmp1YieldStress[d_gaussPtCnt];
    } else {
        stre_np1  = &( d_tmp2Stress[6 * d_gaussPtCnt] );
        ystre_np1 = d_tmp2YieldStress[d_gaussPtCnt];
    }

    constitutiveMatrix = &( d_constitutiveMatrix[0][0] );

    double sig_np1[6];
    double sigy_np1;
    double G, K, Ep, sq23;
    double one3, two3;
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
            d_constitutiveMatrix[i + 3][i + 3] += G;
        }

        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                d_constitutiveMatrix[i][j] += ( K - ( two3 * G ) );
            }
        }

        return;
    }

    // Stress inside the plastic range : The elasto-plastic tangent is calculated.
    sigy_np1 = ystre_np1; // Yield stress.
    lam      = lambda;
    G        = E / ( 2.0 * ( 1.0 + Nu ) );           // of Elastic
    K        = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) ); // and other
    Ep       = H;                                    // constants.
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

    beta      = sq23 * ( sigy_np1 / q_trial );
    gamma     = 3.0 * G / ( ( 3.0 * G ) + Ep );
    gamma_bar = gamma - ( 1.0 - beta );
    term1     = 2.0 * G * beta;
    term2     = 2.0 * G * gamma_bar;

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

void ThermalVonMisesMatModel::radialReturn( const double *stra_np1,
                                            double *stre_np1,
                                            double *ystre_np1,
                                            double *eph_bar_plas_np1 )
{
    double E  = d_E[d_gaussPtCnt];
    double Nu = d_Nu[d_gaussPtCnt];
    double H  = d_H;
    // double Sig0 = d_Sig0;

    const double *stre_n  = &( d_EquilibriumStress[6 * d_gaussPtCnt] );
    const double *stra_n  = &( d_EquilibriumStrain[6 * d_gaussPtCnt] );
    double ystre_n        = d_EquilibriumYieldStress[d_gaussPtCnt];
    double eph_bar_plas_n = d_EquilibriumEffectivePlasticStrain[d_gaussPtCnt];

    double *lambda = &( d_Lambda[d_gaussPtCnt] );
    int *el_or_pl  = &( d_ElPl[d_gaussPtCnt] );

    double sig_n[6], d_strain[6];
    double ephbp_n, ephbp_np1, sigy_n, sigy_np1, lam;
    double G, K, Ep, sq23;
    double one3, two3;
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
    lam       = ( q_trial - ( sq23 * sigy_n ) ) / ( twoG + ( two3 * Ep ) );
    ephbp_np1 = ephbp_n + ( sq23 * lam );
    sigy_np1  = sigy_n + ( sq23 * Ep * lam );

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
} // namespace AMP::Operator
