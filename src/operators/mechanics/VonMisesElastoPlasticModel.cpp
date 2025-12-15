#include "AMP/operators/mechanics/VonMisesElastoPlasticModel.h"
#include "AMP/materials/Property.h"
#include "AMP/operators/mechanics/MechanicsConstants.h"
#include "AMP/utils/Utilities.h"

#include <iostream>


namespace AMP::Operator {


VonMisesElastoPlasticModel::VonMisesElastoPlasticModel(
    std::shared_ptr<MechanicsMaterialModelParameters> params )
    : MechanicsMaterialModel( params ), d_constitutiveMatrix{ { 0 } }
{
    AMP_INSIST( params, "NULL parameter" );
    AMP_INSIST( params->d_db, "NULL database" );
    if ( d_useMaterialsLibrary == false ) {
        AMP_INSIST( params->d_db->keyExists( "Youngs_Modulus" ), "Missing key: Youngs_Modulus" );
        AMP_INSIST( params->d_db->keyExists( "Poissons_Ratio" ), "Missing key: Poissons_Ratio" );
    }
    AMP_INSIST( params->d_db->keyExists( "Linear_Strain_Hardening" ),
                "Missing key: Linear_Strain_Hardening" );

    if ( d_useMaterialsLibrary == false ) {
        default_E  = params->d_db->getScalar<double>( "Youngs_Modulus" );
        default_Nu = params->d_db->getScalar<double>( "Poissons_Ratio" );
    }

    d_H = params->d_db->getScalar<double>( "Linear_Strain_Hardening" );

    if ( d_useMaterialsLibrary == false ) {
        AMP_INSIST( params->d_db->keyExists( "Elastic_Yield_Stress" ),
                    "Missing key: Elastic_Yield_Stress" );

        d_Sig0 = params->d_db->getScalar<double>( "Elastic_Yield_Stress" );
    }

    if ( d_useMaterialsLibrary == true ) {
        d_Sig0 = params->d_db->getWithDefault<double>( "Elastic_Yield_Stress", 1000000.0 );
    }

    mat_name = 0;

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


void VonMisesElastoPlasticModel::preNonlinearInit( bool resetReusesRadialReturn,
                                                   bool jacobianReusesRadialReturn )
{
    d_resetReusesRadialReturn    = resetReusesRadialReturn;
    d_jacobianReusesRadialReturn = jacobianReusesRadialReturn;

    d_Lambda.clear();
    d_ElPl.clear();

    d_EquilibriumStress.clear();
    d_EquilibriumStrain.clear();
    d_EquilibriumYieldStress.clear();
    d_EquilibriumEffectivePlasticStrain.clear();

    d_E.clear();
    d_Nu.clear();
    d_detULF.clear();

    d_tmp1Stress.clear();
    d_tmp1Strain.clear();
    d_tmp1YieldStress.clear();
    d_tmp1EffectivePlasticStrain.clear();

    d_tmp2Stress.clear();
    d_tmp2YieldStress.clear();
    d_tmp2EffectivePlasticStrain.clear();

    Total_Gauss_Point = 0;
}


void VonMisesElastoPlasticModel::nonlinearInitGaussPointOperation( double )
{
    if ( d_useMaterialsLibrary == false ) {
        d_E.push_back( default_E );
        d_Nu.push_back( default_Nu );
    } else {
        d_E.push_back( 0.0 );
        d_Nu.push_back( 0.0 );
    }

    d_detULF.push_back( 1.0 );

    d_Lambda.push_back( 0 );
    d_ElPl.push_back( 0 );

    for ( int i = 0; i < 6; i++ ) {
        d_EquilibriumStress.push_back( 0 );
        d_EquilibriumStrain.push_back( 0 );

        d_tmp1Stress.push_back( 0 );
        d_tmp1Strain.push_back( 0 );
    } // end for i

    d_EquilibriumYieldStress.push_back( d_Sig0 );
    d_EquilibriumEffectivePlasticStrain.push_back( 0 );

    d_tmp1YieldStress.push_back( d_Sig0 );
    d_tmp1EffectivePlasticStrain.push_back( 0 );

    if ( !d_jacobianReusesRadialReturn ) {
        for ( int i = 0; i < 6; i++ ) {
            d_tmp2Stress.push_back( 0 );
        } // end for i

        d_tmp2YieldStress.push_back( d_Sig0 );
        d_tmp2EffectivePlasticStrain.push_back( 0 );
    }

    Total_Gauss_Point++;
}


void VonMisesElastoPlasticModel::globalReset()
{
    AMP_INSIST( ( d_resetReusesRadialReturn == true ), "Inconsistent options!" );

    d_EquilibriumStress                 = d_tmp1Stress;
    d_EquilibriumStrain                 = d_tmp1Strain;
    d_EquilibriumYieldStress            = d_tmp1YieldStress;
    d_EquilibriumEffectivePlasticStrain = d_tmp1EffectivePlasticStrain;
}


void VonMisesElastoPlasticModel::nonlinearResetGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    double stra_np1[6], Identity[3][3];

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            Identity[i][j] = 0.0;
        }
        Identity[i][i] = 1.0;
    }

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    double *stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    radialReturn( &stra_np1[0],
                  stress,
                  &( d_tmp1YieldStress[d_gaussPtCnt] ),
                  &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] ),
                  strain,
                  Identity,
                  Identity );
}


void VonMisesElastoPlasticModel::nonlinearResetGaussPointOperation_UL(
    const std::vector<std::vector<double>> &strain, double R_n[3][3], double R_np1[3][3] )
{
    double stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] += strain[Mechanics::DISPLACEMENT][i];

        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    double *stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    radialReturn( &stra_np1[0],
                  stress,
                  &( d_tmp1YieldStress[d_gaussPtCnt] ),
                  &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] ),
                  strain,
                  R_n,
                  R_np1 );
}


void VonMisesElastoPlasticModel::nonlinearJacobianGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    double *stress = &( d_tmp2Stress[6 * d_gaussPtCnt] );

    double stra_np1[6], Identity[3][3];

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            Identity[i][j] = 0.0;
        }
        Identity[i][i] = 1.0;
    }

    for ( int i = 0; i < 6; i++ ) {
        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    radialReturn( &stra_np1[0],
                  stress,
                  &( d_tmp2YieldStress[d_gaussPtCnt] ),
                  &( d_tmp2EffectivePlasticStrain[d_gaussPtCnt] ),
                  strain,
                  Identity,
                  Identity );
}


void VonMisesElastoPlasticModel::nonlinearJacobianGaussPointOperation_UL(
    const std::vector<std::vector<double>> &strain, double R_n[3][3], double R_np1[3][3] )
{
    double *stress = &( d_tmp2Stress[6 * d_gaussPtCnt] );

    double stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    radialReturn( &stra_np1[0],
                  stress,
                  &( d_tmp2YieldStress[d_gaussPtCnt] ),
                  &( d_tmp2EffectivePlasticStrain[d_gaussPtCnt] ),
                  strain,
                  R_n,
                  R_np1 );
}


void VonMisesElastoPlasticModel::postNonlinearReset()
{
    d_EquilibriumStress                 = d_tmp1Stress;
    d_EquilibriumStrain                 = d_tmp1Strain;
    d_EquilibriumYieldStress            = d_tmp1YieldStress;
    d_EquilibriumEffectivePlasticStrain = d_tmp1EffectivePlasticStrain;
}


void VonMisesElastoPlasticModel::getInternalStress( const std::vector<std::vector<double>> &strain,
                                                    double *&stress )
{
    double stra_np1[6], Identity[3][3];

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            Identity[i][j] = 0.0;
        }
        Identity[i][i] = 1.0;
    }

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];

        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
    }

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    radialReturn( &stra_np1[0],
                  stress,
                  &( d_tmp1YieldStress[d_gaussPtCnt] ),
                  &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] ),
                  strain,
                  Identity,
                  Identity );
}


void VonMisesElastoPlasticModel::getInternalStress_UL(
    const std::vector<std::vector<double>> &strain,
    double *&stress,
    double R_n[3][3],
    double R_np1[3][3],
    double detF )
{
    double stra_np1[6];

    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] += strain[Mechanics::DISPLACEMENT][i];

        stra_np1[i] = strain[Mechanics::DISPLACEMENT][i];
        // AMP::pout<<"stra_np1["<<i<<"]="<<stra_np1[i]<<std::endl;
    }

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    radialReturn( &stra_np1[0],
                  stress,
                  &( d_tmp1YieldStress[d_gaussPtCnt] ),
                  &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] ),
                  strain,
                  R_n,
                  R_np1 );

    d_detULF[d_gaussPtCnt] = detF;

    if ( d_useJaumannRate == true ) {
        for ( int i = 0; i < 6; i++ )
            stress[i] /= detF;
    }
}


void VonMisesElastoPlasticModel::getEffectiveStress( double *& )
{
    AMP_ERROR( "Redesign this function so we do not assign a pointer to a temporary variable" );
    /*
    double stress[6], eff_stress;
    for(int i = 0; i < 6; i++) {
        stress[i] = d_tmp1Stress[(6*d_gaussPtCnt)+i];
    }
    eff_stress = std::sqrt((stress[0] * stress[0]) + (stress[1] * stress[1]) +
        (stress[2] * stress[2]) + (2.0 * stress[3] * stress[3]) +
        (2.0 * stress[4] * stress[4]) + (2.0 * stress[5] * stress[5]));
    sigma_e = &(eff_stress);*/
}


void VonMisesElastoPlasticModel::getEquivalentStrain( double *&epsilon_e )
{
    epsilon_e = &( d_tmp1EffectivePlasticStrain[d_gaussPtCnt] );
}


void VonMisesElastoPlasticModel::getConstitutiveMatrix( double *&constitutiveMatrix )
{
    constructConstitutiveMatrix();
    constitutiveMatrix = &( d_constitutiveMatrix[0][0] );
}


void VonMisesElastoPlasticModel::getConstitutiveMatrixUpdatedLagrangian(
    double constitutiveMatrix[6][6], double R_np1[3][3] )
{
    constructConstitutiveMatrix();

    if ( d_useJaumannRate == true ) {
        for ( auto &elem : d_constitutiveMatrix ) {
            for ( double &j : elem ) {
                j /= d_detULF[d_gaussPtCnt];
            }
        }

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                constitutiveMatrix[i][j] = d_constitutiveMatrix[i][j];
            }
        }
    }

    if ( d_useJaumannRate == false ) {
        double rotatedConstitutiveMatrix[6][6];
        pushforwardCorotationalStiffness( R_np1, d_constitutiveMatrix, rotatedConstitutiveMatrix );

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                constitutiveMatrix[i][j] = rotatedConstitutiveMatrix[i][j];
            }
        }
    }
}


void VonMisesElastoPlasticModel::constructConstitutiveMatrix()
{
    // Consistent Tangent

    double E  = d_E[d_gaussPtCnt];
    double Nu = d_Nu[d_gaussPtCnt];
    double H  = d_H;
    // double Sig0 = d_Sig0;
    double tol = 1.0E-6;

    double lambda = d_Lambda[d_gaussPtCnt];
    int el_or_pl  = d_ElPl[d_gaussPtCnt];

    const double *stre_np1;
    double ystre_np1;
    // double eph_bar_plas_np1;

    if ( d_jacobianReusesRadialReturn ) {
        stre_np1  = &( d_tmp1Stress[6 * d_gaussPtCnt] );
        ystre_np1 = d_tmp1YieldStress[d_gaussPtCnt];
        // eph_bar_plas_np1 = d_tmp1EffectivePlasticStrain[d_gaussPtCnt];
    } else {
        stre_np1  = &( d_tmp2Stress[6 * d_gaussPtCnt] );
        ystre_np1 = d_tmp2YieldStress[d_gaussPtCnt];
        // eph_bar_plas_np1 = d_tmp2EffectivePlasticStrain[d_gaussPtCnt];
    }

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

    /*for(int i = 0; i < 6; i++) {
    std::cout << "stre_np1[" << i << "] = " << stre_np1[i] << std::endl;
    }*/
    // std::cout << "el_or_pl = " << el_or_pl << std::endl;
    // If the stress is within the elastic range.
    // Only the elastic tangent is computed.
    if ( el_or_pl == 0 ) {
        term1 = 2.0 * ( 1.0 + Nu );
        term2 = 3.0 * ( 1.0 - ( 2.0 * Nu ) );
        AMP_INSIST( term1 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 225" );
        G = E / ( 2.0 * ( 1.0 + Nu ) );
        AMP_INSIST( term2 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 227" );
        K = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) );

        // Initializing the tangent matrix as zero.
        for ( auto &elem : d_constitutiveMatrix ) {
            for ( double &j : elem ) {
                j = 0.0;
            }
        }

        for ( int i = 0; i < 3; i++ ) {
            d_constitutiveMatrix[i][i] += ( 2.0 * G );
        }

        // this if block has identical components.
        // if(d_useUpdatedLagrangian == true) {
        for ( int i = 3; i < 6; i++ )
            d_constitutiveMatrix[i][i] += ( 1.0 * G );
        //} else {
        //  for(int i = 3; i < 6; i++)
        //    d_constitutiveMatrix[i][i] += (1.0 * G);
        //}

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
    term1    = 2.0 * ( 1.0 + Nu );
    term2    = 3.0 * ( 1.0 - ( 2.0 * Nu ) );
    AMP_INSIST( term1 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 256" );
    G = E / ( 2.0 * ( 1.0 + Nu ) ); // of Elastic
    AMP_INSIST( term2 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 258" );
    K  = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) ); // and other
    Ep = H;                                    // constants.

    // for(int i = 0; i < 6; i++) {
    // std::cout << "stre_np1[" << i << "] = " << stre_np1[i] << std::endl;
    //}

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

    /*std::cout << "q_np1 = " << q_np1 << std::endl;
    for(int i = 0; i < 6; i++) {
    std::cout << "sig_dev[" << i << "] = " << sig_dev[i] << std::endl;
    }*/

    // The effective stress.
    q_np1 = std::sqrt( ( sig_dev[0] * sig_dev[0] ) + ( sig_dev[1] * sig_dev[1] ) +
                       ( sig_dev[2] * sig_dev[2] ) + ( 2.0 * sig_dev[3] * sig_dev[3] ) +
                       ( 2.0 * sig_dev[4] * sig_dev[4] ) + ( 2.0 * sig_dev[5] * sig_dev[5] ) );

    // The normal direction.
    for ( int i = 0; i < 6; i++ ) {
        AMP_INSIST( q_np1 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 293" );
        n_dir[i] = sig_dev[i] / q_np1;
    }

    // The trial effective stress.
    q_trial = q_np1 + ( 2.0 * G * lam );

    AMP_INSIST( q_trial > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 300" );
    beta = sq23 * ( sigy_np1 / q_trial );
    AMP_INSIST( ( ( 3.0 * G ) + Ep ) > tol,
                "Divide by zero in VonMisesElastoPlasticModel. Line 302" );
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

    if ( d_useContinuumTangent == false ) {
        for ( int i = 0; i < 3; i++ ) {
            for ( int j = 0; j < 3; j++ ) {
                d_constitutiveMatrix[i][j] += ( K - ( one3 * term1 ) );
            }
        }

        for ( int i = 0; i < 3; i++ ) {
            d_constitutiveMatrix[i][i] += term1;
            // this if block has identical components.
            // if(d_useUpdatedLagrangian == true) {
            // d_constitutiveMatrix[i + 3][i + 3] += (G + (2.0 * G * (beta - 1.0)));
            d_constitutiveMatrix[i + 3][i + 3] += ( 0.5 * term1 );
            //} else {
            //  d_constitutiveMatrix[i + 3][i + 3] += (0.5 * term1);
            //}
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
            if ( d_useUpdatedLagrangian == true ) {
                d_constitutiveMatrix[i + 3][i + 3] += G;
            } else {
                d_constitutiveMatrix[i + 3][i + 3] += ( 1.0 * G );
            }
        }

        for ( int i = 0; i < 6; i++ ) {
            for ( int j = 0; j < 6; j++ ) {
                d_constitutiveMatrix[i][j] -= ( 2.0 * G * gamma * n_dir[i] * n_dir[j] );
            }
        }
    }
}


void VonMisesElastoPlasticModel::radialReturn( const double *stra_np1,
                                               double *stre_np1,
                                               double *ystre_np1,
                                               double *eph_bar_plas_np1,
                                               const std::vector<std::vector<double>> &strain,
                                               double R_n[3][3],
                                               double R_np1[3][3] )
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

        std::string scString   = "StrengthCoefficient";
        std::string sheString  = "StrainHardeningExponent";
        std::string srseString = "StrainRateSensitivityExponent";

        bool scExist   = d_material->hasProperty( scString );
        bool sheExist  = d_material->hasProperty( sheString );
        bool srseExist = d_material->hasProperty( srseString );

        if ( scExist && sheExist && srseExist ) {
            mat_name = 1;

            std::vector<double> SC( 1 );
            std::vector<double> SHE( 1 );
            std::vector<double> SRSE( 1 );

            d_material->property( scString )->evalv( SC, {}, args );
            d_material->property( sheString )->evalv( SHE, {}, args );
            d_material->property( srseString )->evalv( SRSE, {}, args );

            double K        = SC[0];
            double n        = SHE[0];
            double m        = SRSE[0];
            double str_term = 0.01;
            double E        = d_E[d_gaussPtCnt];
            double exp_term = 1.0 / ( 1.0 - n );

            d_Sig0 = std::pow( ( ( K / std::pow( E, n ) ) * std::pow( str_term, m ) ), exp_term );
        }
    }

    double E    = d_E[d_gaussPtCnt];
    double Nu   = d_Nu[d_gaussPtCnt];
    double H    = d_H;
    double Sig0 = d_Sig0;

    double S[3][3], Sr[3][3], stre_n[6];
    stre_n[0] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 0];
    stre_n[1] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 1];
    stre_n[2] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 2];
    stre_n[3] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 3];
    stre_n[4] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 4];
    stre_n[5] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 5];

    if ( d_useUpdatedLagrangian == true ) {
        if ( d_useJaumannRate == false ) {
            if ( d_iDebugPrintInfoLevel > 11 ) {
                for ( int i = 0; i < 6; i++ ) {
                    AMP::pout << "d_EquilibriumStress[" << i
                              << "]=" << d_EquilibriumStress[( 6 * d_gaussPtCnt ) + i] << std::endl;
                }
            }
            S[0][0] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 0];
            S[1][1] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 1];
            S[2][2] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 2];
            S[1][2] = S[2][1] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 3];
            S[0][2] = S[2][0] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 4];
            S[0][1] = S[1][0] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 5];
            pullbackCorotational( R_n, S, Sr );
            stre_n[0] = Sr[0][0];
            stre_n[1] = Sr[1][1];
            stre_n[2] = Sr[2][2];
            stre_n[3] = ( 0.5 * ( Sr[1][2] + Sr[2][1] ) );
            stre_n[4] = ( 0.5 * ( Sr[0][2] + Sr[2][0] ) );
            stre_n[5] = ( 0.5 * ( Sr[0][1] + Sr[1][0] ) );
            if ( d_iDebugPrintInfoLevel > 11 ) {
                for ( int i = 0; i < 6; i++ ) {
                    AMP::pout << "stre_n[" << i << "]=" << stre_n[i] << std::endl;
                }
            }
        }

        if ( d_useJaumannRate == true ) {
            S[0][0] = d_tmp1Stress[( 6 * d_gaussPtCnt ) + 0];
            S[1][1] = d_tmp1Stress[( 6 * d_gaussPtCnt ) + 1];
            S[2][2] = d_tmp1Stress[( 6 * d_gaussPtCnt ) + 2];
            S[1][2] = S[2][1] = d_tmp1Stress[( 6 * d_gaussPtCnt ) + 3];
            S[0][2] = S[2][0] = d_tmp1Stress[( 6 * d_gaussPtCnt ) + 4];
            S[0][1] = S[1][0] = d_tmp1Stress[( 6 * d_gaussPtCnt ) + 5];
            jaumannToCauchy( R_n, S );
            stre_n[0] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 0];
            stre_n[1] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 1];
            stre_n[2] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 2];
            stre_n[3] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 3];
            stre_n[4] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 4];
            stre_n[5] = d_EquilibriumStress[( 6 * d_gaussPtCnt ) + 5];
        }
    }

    const double *stra_n  = &( d_EquilibriumStrain[6 * d_gaussPtCnt] );
    double ystre_n        = d_EquilibriumYieldStress[d_gaussPtCnt];
    double eph_bar_plas_n = d_EquilibriumEffectivePlasticStrain[d_gaussPtCnt];

    double *lambda = &( d_Lambda[d_gaussPtCnt] );
    int *el_or_pl  = &( d_ElPl[d_gaussPtCnt] );

    double sig_n[6], d_strain[6];
    double ephbp_n, ephbp_np1, sigy_n, sigy_np1, lam;
    double G, K, Ep, sq23, term1, term2;
    double tol = 1.0E-6, one3, two3;
    double deph_dev[6], sig_dev[6], sig_trial_dev[6], n_dir[6];
    double deph_kk, sig_kk, sig_trial_kk, q_trial, twoG, phi;
    double one2 = 1.0 / 2.0;

    double dstra[6];

    if ( d_useUpdatedLagrangian == false ) {
        for ( int i = 0; i < 3; i++ ) {
            dstra[i]     = stra_np1[i] - stra_n[i];
            dstra[i + 3] = one2 * ( stra_np1[i + 3] - stra_n[i + 3] );
        }
    } else {
        for ( int i = 0; i < 3; i++ ) {
            dstra[i]     = stra_np1[i];
            dstra[i + 3] = one2 * stra_np1[i + 3];
        }
    }

    /* for(int i = 0; i < 6; i++) {
          std::cout << "dstra[" << i << "] = " << dstra[i] << std::endl;
       }
    */
    one3    = 1.0 / 3.0;
    two3    = 2.0 / 3.0;
    sq23    = std::sqrt( two3 );
    ephbp_n = eph_bar_plas_n; // Effective plastic strain at the previous time step.
    sigy_n  = ystre_n;        // Yield stress at the previous time step.
    term1   = 2.0 * ( 1.0 + Nu );
    term2   = 3.0 * ( 1.0 - ( 2.0 * Nu ) );
    AMP_INSIST( term1 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 405" );
    G = E / ( 2.0 * ( 1.0 + Nu ) ); // of Elastic
    AMP_INSIST( term2 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 407" );
    K  = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) ); // and other
    Ep = H;                                    // constants.
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

    ephbp_np1 = ephbp_n; // Initializing the equivalent plastic strain
    sigy_np1  = sigy_n;  // Initializing the yield stress
    if ( mat_name == 1 ) {
        sigy_np1 = Sig0 + ( Ep * ephbp_np1 );
    }
    phi = q_trial - ( sq23 * sigy_np1 ); // Computing the value of the yield function.

    // std::cout << "phi = " << phi << "q_trial = " << q_trial << "sigy_np1 = " << sigy_np1 <<
    // "sigy_n = " << sigy_n <<
    // std::endl;

    // Stress within the elastic range.
    if ( phi < 0.0 ) {
        *el_or_pl = 0;
        for ( int i = 0; i < 3; i++ ) {
            stre_np1[i]     = sig_trial_dev[i] + ( one3 * sig_trial_kk );
            stre_np1[i + 3] = sig_trial_dev[i + 3];
        }

        if ( d_useUpdatedLagrangian == true ) {
            if ( d_useJaumannRate == false ) {
                S[0][0] = stre_np1[0];
                S[1][1] = stre_np1[1];
                S[2][2] = stre_np1[2];
                S[1][2] = S[2][1] = stre_np1[3];
                S[0][2] = S[2][0] = stre_np1[4];
                S[0][1] = S[1][0] = stre_np1[5];
                pushforwardCorotational( R_np1, S, Sr );
                stre_np1[0] = Sr[0][0];
                stre_np1[1] = Sr[1][1];
                stre_np1[2] = Sr[2][2];
                stre_np1[3] = 0.5 * ( Sr[1][2] + Sr[2][1] );
                stre_np1[4] = 0.5 * ( Sr[0][2] + Sr[2][0] );
                stre_np1[5] = 0.5 * ( Sr[0][1] + Sr[1][0] );
            } else {
                stre_np1[0] += S[0][0];
                stre_np1[1] += S[1][1];
                stre_np1[2] += S[2][2];
                stre_np1[3] += ( 0.5 * ( S[1][2] + S[2][1] ) );
                stre_np1[4] += ( 0.5 * ( S[0][2] + S[2][0] ) );
                stre_np1[5] += ( 0.5 * ( S[0][1] + S[1][0] ) );
            }
        }

        if ( d_iDebugPrintInfoLevel > 11 ) {
            AMP::pout << "The current Gauss point is - " << d_gaussPtCnt << std::endl;
            for ( int i = 0; i < 6; i++ ) {
                AMP::pout << "delta_strain[" << i << "] = " << dstra[i] << std::endl;
            }
            for ( int i = 0; i < 6; i++ ) {
                std::cout << "stre_n[" << i << "] = " << stre_n[i] << std::endl;
            }
            for ( int i = 0; i < 6; i++ ) {
                AMP::pout << "stre_np1[" << i << "] = " << stre_np1[i] << std::endl;
            }
            AMP::pout << "\n" << std::endl;
        }

        *ystre_np1 = ystre_n;
        if ( mat_name == 1 ) {
            *ystre_np1 = Sig0 + ( Ep * ephbp_np1 );
        }
        *eph_bar_plas_np1 = eph_bar_plas_n;
        *lambda           = 0.0;
        return;
    }

    if ( d_iDebugPrintInfoLevel > 11 ) {
        AMP::pout << "The current gauss point is - " << d_gaussPtCnt << std::endl;
        for ( int i = 0; i < 6; i++ ) {
            AMP::pout << "dstra[" << i << "] = " << dstra[i] << std::endl;
        }
    }

    // Stress outside the elastic range, inside the plastic range.
    *el_or_pl = 1;
    for ( int i = 0; i < 6; i++ ) {
        AMP_INSIST( q_trial > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 462" );
        n_dir[i] = sig_trial_dev[i] / q_trial; // Computing the normal direction.
    }

    Plastic_Gauss_Point++;

    // Updating the plastic parameters.
    term1 = twoG + ( two3 * Ep );
    AMP_INSIST( term1 > tol, "Divide by zero in VonMisesElastoPlasticModel. Line 468" );
    lam = ( q_trial - ( sq23 * sigy_n ) ) / ( twoG + ( two3 * Ep ) );
    if ( mat_name == 1 ) {
        lam = ( q_trial - ( sq23 * ( Sig0 + ( Ep * ephbp_n ) ) ) ) / ( twoG + ( two3 * Ep ) );
    }
    ephbp_np1 = ephbp_n + ( sq23 * lam );
    sigy_np1  = sigy_n + ( sq23 * Ep * lam );
    if ( mat_name == 1 ) {
        sigy_np1 = Sig0 + ( Ep * ephbp_np1 );
    }

    // Updating the stress.
    for ( int i = 0; i < 6; i++ ) {
        sig_dev[i] = sig_trial_dev[i] - ( twoG * lam * n_dir[i] );
    }

    for ( int i = 0; i < 3; i++ ) {
        stre_np1[i]     = sig_dev[i] + ( one3 * sig_trial_kk );
        stre_np1[i + 3] = sig_dev[i + 3];
    }

    if ( d_useUpdatedLagrangian == true ) {
        if ( d_useJaumannRate == false ) {
            S[0][0] = stre_np1[0];
            S[1][1] = stre_np1[1];
            S[2][2] = stre_np1[2];
            S[1][2] = S[2][1] = stre_np1[3];
            S[0][2] = S[2][0] = stre_np1[4];
            S[0][1] = S[1][0] = stre_np1[5];
            pushforwardCorotational( R_np1, S, Sr );
            stre_np1[0] = Sr[0][0];
            stre_np1[1] = Sr[1][1];
            stre_np1[2] = Sr[2][2];
            stre_np1[3] = 0.5 * ( Sr[1][2] + Sr[2][1] );
            stre_np1[4] = 0.5 * ( Sr[0][2] + Sr[2][0] );
            stre_np1[5] = 0.5 * ( Sr[0][1] + Sr[1][0] );
        } else {
            stre_np1[0] += S[0][0];
            stre_np1[1] += S[1][1];
            stre_np1[2] += S[2][2];
            stre_np1[3] += ( 0.5 * ( S[1][2] + S[2][1] ) );
            stre_np1[4] += ( 0.5 * ( S[0][2] + S[2][0] ) );
            stre_np1[5] += ( 0.5 * ( S[0][1] + S[1][0] ) );
        }
    }

    if ( d_iDebugPrintInfoLevel > 11 ) {
        for ( int i = 0; i < 6; i++ ) {
            AMP::pout << "delta_strain[" << i << "] = " << dstra[i] << std::endl;
        }
        for ( int i = 0; i < 6; i++ ) {
            std::cout << "stre_n[" << i << "] = " << stre_n[i] << std::endl;
        }
        for ( int i = 0; i < 6; i++ ) {
            std::cout << "stre_np1[" << i << "] = " << stre_np1[i] << std::endl;
        }
        AMP::pout << "\n" << std::endl;
    }

    *ystre_np1        = sigy_np1;
    *eph_bar_plas_np1 = ephbp_np1;
    *lambda           = lam;
}


void VonMisesElastoPlasticModel::postNonlinearAssembly()
{
    if ( Total_Gauss_Point == 0 ) {
        std::cout << "Total number of gauss points are zero." << std::endl;
    } else {
        double Plastic_Fraction = ( (double) Plastic_Gauss_Point ) / ( (double) Total_Gauss_Point );
        Plastic_Fraction        = Plastic_Fraction * 100.0;
        if ( d_iDebugPrintInfoLevel > 1 ) {
            std::cout << "Fraction = " << Plastic_Fraction << "% Plastic = " << Plastic_Gauss_Point
                      << " Total = " << Total_Gauss_Point << " Gauss Points."
                      << "  " << d_iDebugPrintInfoLevel << std::endl;
        }
    }
}
} // namespace AMP::Operator
