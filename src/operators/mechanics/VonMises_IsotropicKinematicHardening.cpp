#include "AMP/operators/mechanics/VonMises_IsotropicKinematicHardening.h"
#include "AMP/materials/Property.h"
#include "AMP/operators/mechanics/MechanicsConstants.h"
#include "AMP/utils/Utilities.h"

#include <iostream>


namespace AMP::Operator {

VonMises_IsotropicKinematicHardening::VonMises_IsotropicKinematicHardening(
    std::shared_ptr<MechanicsMaterialModelParameters> params )
    : MechanicsMaterialModel( params ), d_constitutiveMatrix{ { 0 } }
{
    AMP_INSIST( ( ( params.get() ) != nullptr ), "NULL parameter" );

    AMP_INSIST( ( ( ( params->d_db ).get() ) != nullptr ), "NULL database" );

    if ( d_useMaterialsLibrary == false ) {
        AMP_INSIST( params->d_db->keyExists( "Youngs_Modulus" ), "Missing key: Youngs_Modulus" );

        AMP_INSIST( params->d_db->keyExists( "Poissons_Ratio" ), "Missing key: Poissons_Ratio" );

        default_E = params->d_db->getScalar<double>( "Youngs_Modulus" );

        default_Nu = params->d_db->getScalar<double>( "Poissons_Ratio" );
    }

    AMP_INSIST( params->d_db->keyExists( "Cook_Membrane_Plasticity_Test" ),
                "Missing key: Cook_Membrane_Plasticity_Test" );

    AMP_INSIST( params->d_db->keyExists( "Thick_Walled_Cylinder_Plasticity_Test" ),
                "Missing key: Thick_Walled_Cylinder_Plasticity_Test" );

    d_CM_Test = params->d_db->getScalar<bool>( "Cook_Membrane_Plasticity_Test" );

    d_TW_Test = params->d_db->getScalar<bool>( "Thick_Walled_Cylinder_Plasticity_Test" );

    AMP_INSIST( ( ( ( d_CM_Test == true ) && ( d_TW_Test == false ) ) ||
                  ( ( d_CM_Test == false ) && ( d_TW_Test == true ) ) ),
                "Options are wrong for the combination of Cook_Membrane_Plasticity_Test and "
                "Thick_Walled_Cylinder_Plasticity_Test" );

    if ( d_TW_Test == true ) {
        AMP_INSIST( params->d_db->keyExists( "Linear_Strain_Hardening" ),
                    "Missing key: Linear_Strain_Hardening" );

        AMP_INSIST( params->d_db->keyExists( "Exponent_Delta" ), "Missing key: Exponent_Delta" );

        AMP_INSIST( params->d_db->keyExists( "Value_K_0" ), "Missing key: Value_K_0" );

        AMP_INSIST( params->d_db->keyExists( "K_Infinity" ), "Missing key: K_Infinity" );

        AMP_INSIST( params->d_db->keyExists( "Fraction_Beta" ), "Missing key: Fraction_Beta" );

        d_H = params->d_db->getScalar<double>( "Linear_Strain_Hardening" );

        d_delta = params->d_db->getScalar<double>( "Exponent_Delta" );

        d_K_0 = params->d_db->getScalar<double>( "Value_K_0" );

        d_K_inf = params->d_db->getScalar<double>( "K_Infinity" );

        d_beta = params->d_db->getScalar<double>( "Fraction_Beta" );
    }

    if ( d_CM_Test == true ) {
        AMP_INSIST( params->d_db->keyExists( "Isotropic_Linear_Hardening" ),
                    "Missing key: Isotropic_Linear_Hardening" );

        AMP_INSIST( params->d_db->keyExists( "Kinematic_Linear_Hardening" ),
                    "Missing key: Kinematic_Linear_Hardening" );

        AMP_INSIST( params->d_db->keyExists( "Initial_Yield_Strength" ),
                    "Missing key: Initial_Yield_Strength" );

        d_Ep = params->d_db->getScalar<double>( "Isotropic_Linear_Hardening" );

        d_Kin = params->d_db->getScalar<double>( "Kinematic_Linear_Hardening" );

        d_Sig_0 = params->d_db->getScalar<double>( "Initial_Yield_Strength" );
    }

    default_TEMPERATURE = params->d_db->getWithDefault<double>( "Default_Temperature", 310.0 );

    default_BURNUP = params->d_db->getWithDefault<double>( "Default_Burnup", 0.0 );

    default_OXYGEN_CONCENTRATION =
        params->d_db->getWithDefault<double>( "Default_Oxygen_Concentration", 0.0 );

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

void VonMises_IsotropicKinematicHardening::preNonlinearInit( bool resetReusesRadialReturn,
                                                             bool jacobianReusesRadialReturn )
{
    d_resetReusesRadialReturn    = resetReusesRadialReturn;
    d_jacobianReusesRadialReturn = jacobianReusesRadialReturn;

    d_Lambda.clear();
    d_ElPl.clear();

    d_EquilibriumStress.clear();
    d_EquilibriumBackStress.clear();
    d_EquilibriumStrain.clear();
    d_EquilibriumYieldStress.clear();
    d_EquilibriumEffectivePlasticStrain.clear();

    d_E.clear();
    d_Nu.clear();

    d_tmp1Stress.clear();
    d_tmp1BackStress.clear();
    d_tmp1Strain.clear();
    d_tmp1YieldStress.clear();
    d_tmp1EffectivePlasticStrain.clear();

    d_tmp2Stress.clear();
    d_tmp2BackStress.clear();
    d_tmp2YieldStress.clear();
    d_tmp2EffectivePlasticStrain.clear();

    Total_Gauss_Point = 0;
}

void VonMises_IsotropicKinematicHardening::nonlinearInitGaussPointOperation( double )
{
    if ( d_useMaterialsLibrary == false ) {
        d_E.push_back( default_E );
        d_Nu.push_back( default_Nu );
    } else {
        d_E.push_back( 0.0 );
        d_Nu.push_back( 0.0 );
    }

    d_Lambda.push_back( 0 );
    d_ElPl.push_back( 0 );

    for ( int i = 0; i < 6; i++ ) {
        d_EquilibriumStress.push_back( 0 );
        d_EquilibriumBackStress.push_back( 0 );
        d_EquilibriumStrain.push_back( 0 );

        d_tmp1Stress.push_back( 0 );
        d_tmp1BackStress.push_back( 0 );
        d_tmp1Strain.push_back( 0 );
    } // end for i

    if ( d_TW_Test == true ) {
        d_EquilibriumYieldStress.push_back( d_K_0 );
    }
    if ( d_CM_Test == true ) {
        d_EquilibriumYieldStress.push_back( d_Sig_0 );
    }
    d_EquilibriumEffectivePlasticStrain.push_back( 0 );

    if ( d_TW_Test == true ) {
        d_tmp1YieldStress.push_back( d_K_0 );
    }
    if ( d_CM_Test == true ) {
        d_tmp1YieldStress.push_back( d_Sig_0 );
    }
    d_tmp1EffectivePlasticStrain.push_back( 0 );

    if ( !d_jacobianReusesRadialReturn ) {
        for ( int i = 0; i < 6; i++ ) {
            d_tmp2Stress.push_back( 0 );
            d_tmp2BackStress.push_back( 0 );
        } // end for i

        if ( d_TW_Test == true ) {
            d_tmp2YieldStress.push_back( d_K_0 );
        }
        if ( d_CM_Test == true ) {
            d_tmp2YieldStress.push_back( d_Sig_0 );
        }
        d_tmp2EffectivePlasticStrain.push_back( 0 );
    }

    Total_Gauss_Point++;
}

void VonMises_IsotropicKinematicHardening::globalReset()
{
    AMP_INSIST( ( d_resetReusesRadialReturn == true ), "Inconsistent options!" );

    d_EquilibriumStress                 = d_tmp1Stress;
    d_EquilibriumBackStress             = d_tmp1BackStress;
    d_EquilibriumStrain                 = d_tmp1Strain;
    d_EquilibriumYieldStress            = d_tmp1YieldStress;
    d_EquilibriumEffectivePlasticStrain = d_tmp1EffectivePlasticStrain;
}

void VonMises_IsotropicKinematicHardening::nonlinearResetGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    double stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    double *stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    double *back_stress = &( d_tmp1BackStress[6 * d_gaussPtCnt] );

    radialReturn( &stra_np1[0],
                  stress,
                  back_stress,
                  &( d_tmp1YieldStress[d_gaussPtCnt] ),
                  &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] ),
                  strain );
}

void VonMises_IsotropicKinematicHardening::nonlinearJacobianGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    double *stress = &( d_tmp2Stress[6 * d_gaussPtCnt] );

    double *back_stress = &( d_tmp2BackStress[6 * d_gaussPtCnt] );

    double stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    radialReturn( &stra_np1[0],
                  stress,
                  back_stress,
                  &( d_tmp2YieldStress[d_gaussPtCnt] ),
                  &( d_tmp2EffectivePlasticStrain[d_gaussPtCnt] ),
                  strain );
}

void VonMises_IsotropicKinematicHardening::postNonlinearReset()
{
    d_EquilibriumStress                 = d_tmp1Stress;
    d_EquilibriumBackStress             = d_tmp1BackStress;
    d_EquilibriumStrain                 = d_tmp1Strain;
    d_EquilibriumYieldStress            = d_tmp1YieldStress;
    d_EquilibriumEffectivePlasticStrain = d_tmp1EffectivePlasticStrain;
}

void VonMises_IsotropicKinematicHardening::getInternalStress(
    const std::vector<std::vector<double>> &strain, double *&stress )
{
    double stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    double *back_stress = &( d_tmp1BackStress[6 * d_gaussPtCnt] );

    radialReturn( &stra_np1[0],
                  stress,
                  back_stress,
                  &( d_tmp1YieldStress[d_gaussPtCnt] ),
                  &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] ),
                  strain );
}

void VonMises_IsotropicKinematicHardening::computeEvalv(
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

        std::string ymString = "YoungsModulus";
        std::string prString = "PoissonRatio";

        std::map<std::string, std::vector<double> &> args = { { "temperature", tempVec },
                                                              { "concentration", oxygenVec },
                                                              { "burnup", burnupVec } };

        d_material->property( ymString )->evalv( YM, {}, args );
        d_material->property( prString )->evalv( PR, {}, args );

        d_E[d_gaussPtCnt]  = YM[0];
        d_Nu[d_gaussPtCnt] = PR[0];
    }
}

void VonMises_IsotropicKinematicHardening::getConstitutiveMatrix( double *&constitutiveMatrix )
{
    // Consistent Tangent

    double E  = d_E[d_gaussPtCnt];
    double Nu = d_Nu[d_gaussPtCnt];
    double H  = 1.0;
    if ( d_TW_Test == true ) {
        H = d_H;
    }
    // double Sig0 = d_Sig0;
    double tol = 1.0E-8;

    double lambda = d_Lambda[d_gaussPtCnt];
    int el_or_pl  = d_ElPl[d_gaussPtCnt];

    const double *stre_np1;
    const double *alpha_np1;
    const double *alpha_n;
    double ystre_np1;
    double eph_bar_plas_np1;

    if ( d_jacobianReusesRadialReturn ) {
        stre_np1         = &( d_tmp1Stress[6 * d_gaussPtCnt] );
        alpha_np1        = &( d_tmp1BackStress[6 * d_gaussPtCnt] );
        ystre_np1        = d_tmp1YieldStress[d_gaussPtCnt];
        eph_bar_plas_np1 = d_tmp1EffectivePlasticStrain[d_gaussPtCnt];
    } else {
        stre_np1         = &( d_tmp2Stress[6 * d_gaussPtCnt] );
        alpha_np1        = &( d_tmp2BackStress[6 * d_gaussPtCnt] );
        ystre_np1        = d_tmp2YieldStress[d_gaussPtCnt];
        eph_bar_plas_np1 = d_tmp2EffectivePlasticStrain[d_gaussPtCnt];
    }
    alpha_n = &( d_EquilibriumBackStress[6 * d_gaussPtCnt] );

    constitutiveMatrix = &( d_constitutiveMatrix[0][0] );

    double sig_np1[6];
    double ephbp_np1 = 1.0, sigy_np1 = 1.0;
    double G = 1.0, K = 1.0, Ep = 1.0, sq23 = 1.0;

    double sig_dev[6], n_dir[6];
    // double q_np1 = 1.0, sig_np1_kk = 1.0, lam = 1.0;
    double sig_np1_kk = 1.0, lam = 1.0;
    double beta = 1.0, gamma = 1.0, gamma_bar = 1.0, term1 = 1.0, term2 = 1.0, kappa_prime = 1.0;
    double delta = 1.0, k_0 = 1.0, k_inf = 1.0, beta_1 = 1.0;
    if ( d_TW_Test == true ) {
        delta  = d_delta;
        k_0    = d_K_0;
        k_inf  = d_K_inf;
        beta_1 = d_beta;
    }

    double a_np1[6], delta_H_alpha = 1.0, H_alpha_prime = 1.0, xi_dev[6], xi_np1 = 1.0;
    double a_n[6], xi_trial_eff = 1.0, xi_trial[6];
    // double a_np1[6], alpha_kk = 1.0, delta_H_alpha = 1.0, H_alpha_prime = 1.0, xi_dev[6], xi_np1
    // = 1.0;
    // double a_n[6], xi_trial_eff = 1.0, xi_trial[6], alpha_n_kk = 1.0;

    const double one3 = 1.0 / 3.0;
    const double two3 = 2.0 / 3.0;
    sq23              = std::sqrt( two3 );

    // std::cout << "sig_0 = " << sig_0 << " sig_inf = " << sig_inf << std::endl;

    /*for(int i = 0; i < 6; i++) {
      std::cout << "stre_np1[" << i << "] = " << stre_np1[i] << std::endl;
    }*/
    // std::cout << "el_or_pl = " << el_or_pl << std::endl;
    // If the stress is within the elastic range.
    // Only the elastic tangent is computed.
    if ( el_or_pl == 0 ) {
        term1 = 2.0 * ( 1.0 + Nu );
        term2 = 3.0 * ( 1.0 - ( 2.0 * Nu ) );
        AMP_INSIST( term1 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 234" );
        G = E / ( 2.0 * ( 1.0 + Nu ) );
        AMP_INSIST( term2 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 236" );
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
    term1     = 2.0 * ( 1.0 + Nu );
    term2     = 3.0 * ( 1.0 - ( 2.0 * Nu ) );
    AMP_INSIST( term1 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 265" );
    G = E / ( 2.0 * ( 1.0 + Nu ) ); // of Elastic and other constants.
    AMP_INSIST( term2 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 267" );
    K = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) );
    if ( d_TW_Test == true ) {
        Ep = H; // For thick walled cylinder test.
    }
    if ( d_CM_Test == true ) {
        Ep = d_Ep; // For Cook's Membrane test.
    }

    // for(int i = 0; i < 6; i++) {
    // std::cout << "stre_np1[" << i << "] = " << stre_np1[i] << std::endl;
    //}

    for ( int i = 0; i < 6; i++ ) {
        sig_np1[i] = stre_np1[i]; // Stress.
    }

    // Hydrostatic component of the stress tensor.
    sig_np1_kk = sig_np1[0] + sig_np1[1] + sig_np1[2];
    // alpha_kk = alpha_np1[0] + alpha_np1[1] + alpha_np1[2];
    // alpha_n_kk = alpha_n[0] + alpha_n[1] + alpha_n[2];
    // Deviatoric component of the stress tensor.
    for ( int i = 0; i < 3; i++ ) {
        sig_dev[i]     = sig_np1[i] - ( one3 * sig_np1_kk );
        sig_dev[i + 3] = sig_np1[i + 3];
        // a_np1[i] = alpha_np1[i] - (one3 * alpha_kk);
        a_np1[i]     = alpha_np1[i];
        a_np1[i + 3] = alpha_np1[i + 3];
        // a_n[i] = alpha_n[i] - (one3 * alpha_n_kk);
        a_n[i]     = alpha_n[i];
        a_n[i + 3] = alpha_n[i + 3];
    }

    for ( int i = 0; i < 6; i++ ) {
        xi_dev[i] = sig_dev[i] - a_np1[i];
    }

    /*std::cout << "q_np1 = " << q_np1 << std::endl;
    for(int i = 0; i < 6; i++) {
      std::cout << "sig_dev[" << i << "] = " << sig_dev[i] << std::endl;
    }*/

    // The effective stress.
    /*
        q_np1 = std::sqrt((sig_dev[0] * sig_dev[0]) +
            (sig_dev[1] * sig_dev[1]) +
            (sig_dev[2] * sig_dev[2]) +
            (2.0 * sig_dev[3] * sig_dev[3]) +
            (2.0 * sig_dev[4] * sig_dev[4]) +
            (2.0 * sig_dev[5] * sig_dev[5]));
    */
    xi_np1 = std::sqrt( ( xi_dev[0] * xi_dev[0] ) + ( xi_dev[1] * xi_dev[1] ) +
                        ( xi_dev[2] * xi_dev[2] ) + ( 2.0 * xi_dev[3] * xi_dev[3] ) +
                        ( 2.0 * xi_dev[4] * xi_dev[4] ) + ( 2.0 * xi_dev[5] * xi_dev[5] ) );

    // The normal direction.
    for ( int i = 0; i < 6; i++ ) {
        AMP_INSIST( xi_np1 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 302" );
        n_dir[i] = xi_dev[i] / xi_np1;
    }

    for ( int i = 0; i < 6; i++ ) {
        xi_trial[i] = sig_dev[i] + ( 2.0 * G * lam * n_dir[i] ) - a_n[i];
    }

    xi_trial_eff =
        std::sqrt( ( xi_trial[0] * xi_trial[0] ) + ( xi_trial[1] * xi_trial[1] ) +
                   ( xi_trial[2] * xi_trial[2] ) + ( 2.0 * xi_trial[3] * xi_trial[3] ) +
                   ( 2.0 * xi_trial[4] * xi_trial[4] ) + ( 2.0 * xi_trial[5] * xi_trial[5] ) );

    // The trial effective stress.
    // q_trial = q_np1 + (2.0 * G * lam);
    if ( d_TW_Test == true ) {
        double h_alpha = 1.0, h_alpha_prime = 1.0;
        h_alpha = k_inf - ( ( k_inf - k_0 ) * exp( -delta * ephbp_np1 ) ) + ( Ep * ephbp_np1 );
        h_alpha_prime = ( ( delta * ( k_inf - k_0 ) ) * exp( -delta * ephbp_np1 ) ) + Ep;
        H_alpha_prime = ( 1.0 - beta_1 ) * h_alpha_prime;
        sigy_np1      = beta_1 * h_alpha;
    }
    if ( d_CM_Test == true ) {
        H_alpha_prime = d_Kin;
        sigy_np1      = d_Sig_0 + ( Ep * ephbp_np1 );
    }

    delta_H_alpha = sq23 * H_alpha_prime * lam;

    AMP_INSIST( xi_trial_eff > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 309" );
    beta = sq23 * ( ( sigy_np1 + delta_H_alpha ) / xi_trial_eff );
    if ( d_TW_Test == true ) {
        kappa_prime = beta_1 * ( ( ( delta * ( k_inf - k_0 ) ) * exp( -delta * ephbp_np1 ) ) + Ep );
    }
    if ( d_CM_Test == true ) {
        kappa_prime = Ep;
    }
    AMP_INSIST( ( ( 3.0 * G ) + kappa_prime + H_alpha_prime ) > tol,
                "Divide by zero in VonMisesElastoPlasticModel. Line 312" );
    gamma     = 3.0 * G / ( ( 3.0 * G ) + kappa_prime + H_alpha_prime );
    gamma_bar = gamma - ( 1.0 - beta );
    term1     = 2.0 * G * beta;
    term2     = 2.0 * G * gamma_bar;

    // Initiaization of the constitutive matrix.
    for ( auto &elem : d_constitutiveMatrix ) {
        for ( double &j : elem ) {
            j = 0.0;
        }
    }

    if ( d_useContinuumTangent == false ) {
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
    } else {
        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                d_constitutiveMatrix[i][j] += ( K - ( one3 * 2.0 * G ) );
            }
        }

        for ( int i = 0; i < 3; i++ ) {
            d_constitutiveMatrix[i][i] += ( 2.0 * G );
            d_constitutiveMatrix[i + 3][i + 3] += ( 1.0 * G );
        }

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                d_constitutiveMatrix[i][j] -= ( 2.0 * G * gamma * n_dir[i] * n_dir[j] );
            }
        }
    }
}

void VonMises_IsotropicKinematicHardening::radialReturn(
    const double *stra_np1,
    double *stre_np1,
    double *alpha_np1,
    double *ystre_np1,
    double *eph_bar_plas_np1,
    const std::vector<std::vector<double>> &strain )
{
    if ( d_useMaterialsLibrary == true ) {
        computeEvalv( strain );
    }

    double E  = d_E[d_gaussPtCnt];
    double Nu = d_Nu[d_gaussPtCnt];
    double H  = 1.0;
    if ( d_TW_Test == true ) {
        H = d_H;
    }
    // double Sig0 = d_Sig0;

    const double *stre_n  = &( d_EquilibriumStress[6 * d_gaussPtCnt] );
    const double *stra_n  = &( d_EquilibriumStrain[6 * d_gaussPtCnt] );
    const double *alpha_n = &( d_EquilibriumBackStress[6 * d_gaussPtCnt] );
    double ystre_n        = d_EquilibriumYieldStress[d_gaussPtCnt];
    double eph_bar_plas_n = d_EquilibriumEffectivePlasticStrain[d_gaussPtCnt];

    double *lambda = &( d_Lambda[d_gaussPtCnt] );
    int *el_or_pl  = &( d_ElPl[d_gaussPtCnt] );
    int counter;

    double sig_n[6], d_strain[6];
    // double ephbp_n = 1.0, ephbp_np1 = 1.0, sigy_n = 1.0, sigy_np1 = 1.0, lam = 1.0;
    double ephbp_n = 1.0, ephbp_np1 = 1.0, sigy_np1 = 1.0, lam = 1.0;
    double G = 1.0, K = 1.0, Ep = 1.0, sq23 = 1.0, term1 = 1.0, term2 = 1.0;
    double tol = 1.0E-8;
    double deph_dev[6], sig_dev[6], sig_trial_dev[6], n_dir[6];
    // double deph_kk = 1.0, sig_kk = 1.0, sig_trial_kk = 1.0, q_trial = 1.0, twoG = 1.0, phi = 1.0;
    double deph_kk = 1.0, sig_kk = 1.0, sig_trial_kk = 1.0, twoG = 1.0, phi = 1.0;
    double g_lam = 1.0, Dg_lam = 1.0, kappa_np1 = 1.0, kappa_prime = 1.0, h_alpha = 1.0,
           h_alpha_prime = 1.0;
    double delta = 1.0, d_lam = 1.0, k_0 = 1.0, k_inf = 1.0, beta_1 = 1.0;
    if ( d_TW_Test == true ) {
        k_0   = d_K_0;
        k_inf = d_K_inf;
    }
    double H_alpha_prime = 1.0;
    double a_n[6], xi_trial, xi_trial_dev[6], alpha_n_kk;

    double dstra[6];

    for ( int i = 0; i < 3; i++ ) {
        dstra[i]     = stra_np1[i] - stra_n[i];
        dstra[i + 3] = 0.5 * ( stra_np1[i + 3] - stra_n[i + 3] );
    }

    /*for(int i = 0; i < 6; i++) {
      std::cout << "dstra[" << i << "] = " << dstra[i] << std::endl;
    }*/

    constexpr double one3 = 1.0 / 3.0;
    constexpr double two3 = 2.0 / 3.0;
    sq23                  = std::sqrt( two3 );
    ephbp_n               = eph_bar_plas_n; // Effective plastic strain at the previous time step.
    // sigy_n = ystre_n;           //Yield stress at the previous time step.

    if ( d_TW_Test == true ) {
        delta  = d_delta;
        beta_1 = d_beta;
    }

    term1 = 2.0 * ( 1.0 + Nu );
    term2 = 3.0 * ( 1.0 - ( 2.0 * Nu ) );
    AMP_INSIST( term1 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 421" );
    G = E / ( 2.0 * ( 1.0 + Nu ) ); // Computation of Elastic and other constants.
    AMP_INSIST( term2 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 423" );
    K = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) );
    if ( d_TW_Test == true ) {
        Ep = H; // Thick walled cyliinder test.
    }
    if ( d_CM_Test == true ) {
        Ep = d_Ep; // Cook's Membrane test.
    }
    for ( int i = 0; i < 6; i++ ) {
        sig_n[i]    = stre_n[i]; // Stress at the previous time step.
        d_strain[i] = dstra[i];  // Change in strain (strain increment).
    }

    // Compute Traces
    deph_kk    = d_strain[0] + d_strain[1] + d_strain[2];
    sig_kk     = sig_n[0] + sig_n[1] + sig_n[2];
    alpha_n_kk = alpha_n[0] + alpha_n[1] + alpha_n[2];

    // Compute deviatoric components.
    for ( int i = 0; i < 3; i++ ) {
        deph_dev[i]     = d_strain[i] - ( one3 * deph_kk );
        sig_dev[i]      = sig_n[i] - ( one3 * sig_kk );
        deph_dev[i + 3] = d_strain[i + 3];
        sig_dev[i + 3]  = sig_n[i + 3];
        a_n[i]          = alpha_n[i] - ( one3 * alpha_n_kk );
        // a_n[i] = alpha_n[i];
        a_n[i + 3] = alpha_n[i + 3];
    }

    // Trial stress calculation.
    twoG = 2.0 * G;
    for ( int i = 0; i < 6; i++ ) {
        sig_trial_dev[i] = sig_dev[i] + ( twoG * deph_dev[i] );
        xi_trial_dev[i]  = sig_trial_dev[i] - a_n[i];
    }
    sig_trial_kk = sig_kk + ( 3.0 * K * deph_kk );

    // Compute the trial effective stress.
    /*
        q_trial = std::sqrt((sig_trial_dev[0] * sig_trial_dev[0]) +
            (sig_trial_dev[1] * sig_trial_dev[1]) +
            (sig_trial_dev[2] * sig_trial_dev[2]) +
            (2.0 * sig_trial_dev[3] * sig_trial_dev[3]) +
            (2.0 * sig_trial_dev[4] * sig_trial_dev[4]) +
            (2.0 * sig_trial_dev[5] * sig_trial_dev[5]));
    */
    xi_trial = std::sqrt(
        ( xi_trial_dev[0] * xi_trial_dev[0] ) + ( xi_trial_dev[1] * xi_trial_dev[1] ) +
        ( xi_trial_dev[2] * xi_trial_dev[2] ) + ( 2.0 * xi_trial_dev[3] * xi_trial_dev[3] ) +
        ( 2.0 * xi_trial_dev[4] * xi_trial_dev[4] ) + ( 2.0 * xi_trial_dev[5] * xi_trial_dev[5] ) );

    ephbp_np1 = ephbp_n; // Initializing the equivalent plastic strain
    if ( d_TW_Test == true ) {
        h_alpha  = k_inf - ( ( k_inf - k_0 ) * exp( -delta * ephbp_np1 ) ) + ( Ep * ephbp_np1 );
        sigy_np1 = beta_1 * h_alpha; // Initializing the yield stress
    }
    if ( d_CM_Test == true ) {
        sigy_np1 = d_Sig_0 + ( Ep * ephbp_np1 ); // Initializing the yield stress
    }

    phi = xi_trial - ( sq23 * sigy_np1 ); // Computing the value of the yield function.

    // std::cout << "phi = " << phi << "q_trial = " << q_trial << "sigy_np1 = " << sigy_np1 <<
    // "sigy_n = " << sigy_n <<
    // std::endl;

    // Stress within the elastic range.
    if ( phi < 0.0 ) {
        *el_or_pl = 0;
        for ( int i = 0; i < 3; i++ ) {
            stre_np1[i]      = sig_trial_dev[i] + ( one3 * sig_trial_kk );
            stre_np1[i + 3]  = sig_trial_dev[i + 3];
            alpha_np1[i]     = alpha_n[i];
            alpha_np1[i + 3] = alpha_n[i + 3];
        }
        *ystre_np1        = ystre_n;
        *eph_bar_plas_np1 = eph_bar_plas_n;
        *lambda           = 0.0;

        // std::cout << "Response : Elastic." << std::endl;

        return;
    }

    // std::cout << "Response : Plastic." << std::endl;
    // std::cout << "sig_0 = " << sig_0 << " sig_inf = " << sig_inf << std::endl;

    // Stress outside the elastic range, inside the plastic range.
    *el_or_pl = 1;

    Plastic_Gauss_Point++;

    for ( int i = 0; i < 6; i++ ) {
        AMP_INSIST( xi_trial > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 480" );
        n_dir[i] = xi_trial_dev[i] / xi_trial; // Computing the normal direction.
    }

    // Initializing the plastic parameters.
    if ( d_TW_Test == true ) {
        h_alpha = k_inf - ( ( k_inf - k_0 ) * exp( -delta * ephbp_np1 ) ) + ( Ep * ephbp_np1 );
        h_alpha_prime = ( ( delta * ( k_inf - k_0 ) ) * exp( -delta * ephbp_np1 ) ) + Ep;
        // H_alpha_prime = (gam * (k_inf - k_0) * exp(-gam * ephbp_np1)) + k_bar;
        H_alpha_prime = ( 1.0 - beta_1 ) * h_alpha_prime;
        // kappa_np1 = sig_inf - ((sig_inf - sig_0) * exp(-delta * ephbp_np1)) + (Ep * ephbp_np1);
        kappa_np1 = beta_1 * h_alpha;
        // kappa_prime = ((delta * (sig_inf - sig_0)) * exp(-delta * ephbp_np1)) + Ep;
        kappa_prime = ( ( ( delta * ( k_inf - k_0 ) ) * exp( -delta * ephbp_np1 ) ) + Ep ) * beta_1;
    }
    if ( d_CM_Test == true ) {
        H_alpha_prime = d_Kin;
        kappa_np1     = d_Sig_0 + ( Ep * ephbp_np1 );
        kappa_prime   = Ep;
    }
    g_lam  = xi_trial - ( sq23 * kappa_np1 );
    Dg_lam = -twoG * ( 1.0 + ( ( 1.0 / ( 3.0 * G ) ) * ( kappa_prime + H_alpha_prime ) ) );
    AMP_INSIST( fabs( Dg_lam ) > tol,
                "Divide by zero in term Dg_lam in "
                "VonMisesElastoPlasticModel_NonlinearStrainHardening.cc. "
                "Line 489" );
    d_lam = -g_lam / Dg_lam;
    lam   = d_lam;
    // std::cout << "Dg_lam = " << Dg_lam << " g_lam = " << g_lam << " lam = " << lam << std::endl;

    // Updating the plastic parameters.
    for ( counter = 0; counter < 100; counter++ ) {
        ephbp_np1 = ephbp_n + ( sq23 * lam );
        // kappa_np1 = sig_inf - ((sig_inf - sig_0) * exp(-delta * ephbp_np1)) + (Ep * ephbp_np1);
        // kappa_prime = ((delta * (sig_inf - sig_0)) * exp(-delta * ephbp_np1)) + Ep;
        // H_alpha_prime = (gam * (k_inf - k_0) * exp(-gam * ephbp_np1)) + k_bar;

        if ( d_TW_Test == true ) {
            h_alpha = k_inf - ( ( k_inf - k_0 ) * exp( -delta * ephbp_np1 ) ) + ( Ep * ephbp_np1 );
            h_alpha_prime = ( ( delta * ( k_inf - k_0 ) ) * exp( -delta * ephbp_np1 ) ) + Ep;
            H_alpha_prime = ( 1.0 - beta_1 ) * h_alpha_prime;
            kappa_np1     = beta_1 * h_alpha;
            kappa_prime =
                ( ( ( delta * ( k_inf - k_0 ) ) * exp( -delta * ephbp_np1 ) ) + Ep ) * beta_1;
        }
        if ( d_CM_Test == true ) {
            H_alpha_prime = d_Kin;
            kappa_np1     = d_Sig_0 + ( Ep * ephbp_np1 );
            kappa_prime   = Ep;
        }

        g_lam  = xi_trial - ( ( twoG + ( two3 * H_alpha_prime ) ) * lam ) - ( sq23 * kappa_np1 );
        Dg_lam = -twoG * ( 1.0 + ( ( 1.0 / ( 3.0 * G ) ) * ( kappa_prime + H_alpha_prime ) ) );
        AMP_INSIST( fabs( Dg_lam ) > tol,
                    "Divide by zero in term Dg_lam in "
                    "VonMisesElastoPlasticModel_NonlinearStrainHardening.cc. "
                    "Line 500" );
        d_lam = -g_lam / Dg_lam;
        lam   = lam + d_lam;
        // std::cout << "Dg_lam = " << Dg_lam << " g_lam = " << g_lam << " lam = " << lam <<
        // std::endl;
        if ( fabs( d_lam ) < tol ) {
            break;
        }
    }

    if ( ( counter == 99 ) || ( counter == 100 ) ) {
        AMP_ERROR( "The lambda value did not converge" );
    }

    // std::cout << "Code converged with i = " << counter+1 << std::endl;

    /*term1 = twoG + (two3 * Ep);
    AMP_INSIST(term1 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 516");
    lam = (q_trial - (sq23 * sigy_n)) / (twoG + (two3 * Ep));
    ephbp_np1 = ephbp_n + (sq23 * lam);
    sigy_np1 = sigy_n + (sq23 * Ep * lam);*/

    ephbp_np1 = ephbp_n + ( sq23 * lam );

    if ( d_TW_Test == true ) {
        // H_alpha_prime = (gam * (k_inf - k_0) * exp(-gam * ephbp_np1)) + k_bar;
        // sigy_np1 = sig_inf - ((sig_inf - sig_0) * exp(-delta * ephbp_np1)) + (Ep * ephbp_np1);
        h_alpha = k_inf - ( ( k_inf - k_0 ) * exp( -delta * ephbp_np1 ) ) + ( Ep * ephbp_np1 );
        h_alpha_prime = ( ( delta * ( k_inf - k_0 ) ) * exp( -delta * ephbp_np1 ) ) + Ep;
        H_alpha_prime = ( 1.0 - beta_1 ) * h_alpha_prime;
        sigy_np1      = beta_1 * h_alpha;
    }
    if ( d_CM_Test == true ) {
        H_alpha_prime = d_Kin;
        sigy_np1      = d_Sig_0 + ( Ep * ephbp_np1 );
    }

    // Updating the back-stress.
    for ( int i = 0; i < 6; i++ ) {
        alpha_np1[i] = alpha_n[i] + ( two3 * H_alpha_prime * lam * n_dir[i] );
    }

    // Updating the stress.
    for ( int i = 0; i < 6; i++ ) {
        sig_dev[i] = alpha_np1[i] + ( sq23 * sigy_np1 * n_dir[i] );
    }

    for ( int i = 0; i < 3; i++ ) {
        stre_np1[i]     = sig_dev[i] + ( one3 * sig_trial_kk );
        stre_np1[i + 3] = sig_dev[i + 3];
    }
    /*for(int i = 0; i < 6; i++) {
      std::cout << "stre_np1[" << i << "] = " << stre_np1[i] << std::endl;
    }*/

    *ystre_np1        = sigy_np1;
    *eph_bar_plas_np1 = ephbp_np1;
    *lambda           = lam;
}

void VonMises_IsotropicKinematicHardening::postNonlinearAssembly()
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
