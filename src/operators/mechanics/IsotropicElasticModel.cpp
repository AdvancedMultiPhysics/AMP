#include "AMP/operators/mechanics/IsotropicElasticModel.h"
#include "AMP/materials/Property.h"
#include "AMP/operators/mechanics/MechanicsConstants.h"
#include "AMP/utils/Utilities.h"

#include <iostream>


using namespace AMP::Operator::UpdatedLagrangianUtils;


namespace AMP::Operator {


IsotropicElasticModel::IsotropicElasticModel(
    std::shared_ptr<MechanicsMaterialModelParameters> params )
    : MechanicsMaterialModel( params )
{
    // Zero d_constitutiveMatrix and d_constitutiveMatrix_UL
    // Must be initialized before call to constructConstitutiveMatrix
    for ( size_t i = 0; i < 6; i++ ) {
        for ( size_t j = 0; j < 6; j++ ) {
            d_constitutiveMatrix[i][j]    = 0.;
            d_constitutiveMatrix_UL[i][j] = 0.;
        }
    }
    d_gaussPtCnt                 = 0;
    d_resetReusesRadialReturn    = false;
    d_jacobianReusesRadialReturn = false;

    if ( d_useMaterialsLibrary == false ) {
        AMP_INSIST( ( ( params.get() ) != nullptr ), "NULL parameter" );
        AMP_INSIST( ( ( ( params->d_db ).get() ) != nullptr ), "NULL database" );
        AMP_INSIST( params->d_db->keyExists( "Youngs_Modulus" ), "Missing key: Youngs_Modulus" );
        AMP_INSIST( params->d_db->keyExists( "Poissons_Ratio" ), "Missing key: Poissons_Ratio" );

        default_E  = params->d_db->getScalar<double>( "Youngs_Modulus" );
        default_Nu = params->d_db->getScalar<double>( "Poissons_Ratio" );

        constructConstitutiveMatrix( default_E, default_Nu );
        constructConstitutiveMatrixUpdatedLagrangian( default_E, default_Nu );

    } // end d_useMaterialsLibrary == false

    default_TEMPERATURE = params->d_db->getWithDefault<double>( "Default_Temperature", 310.0 );

    default_BURNUP = params->d_db->getWithDefault<double>( "Default_Burnup", 0.0 );

    default_OXYGEN_CONCENTRATION =
        params->d_db->getWithDefault<double>( "Default_Oxygen_Concentration", 0.0 );

    // fout = fopen("Stress-Strain-Response.txt","w");
    // fprintf(fout,"%le %le %le %le %le %le\n",0.0,0.0,0.0,0.0,0.0,0.0);
}


void IsotropicElasticModel::preNonlinearInit( bool resetReusesRadialReturn,
                                              bool jacobianReusesRadialReturn )
{
    d_resetReusesRadialReturn    = resetReusesRadialReturn;
    d_jacobianReusesRadialReturn = jacobianReusesRadialReturn;

    d_E.clear();
    d_Nu.clear();
    d_detULF.clear();

    d_EquilibriumStress.clear();
    d_EquilibriumStrain.clear();

    d_tmp1Stress.clear();
    d_tmp1Strain.clear();
}


void IsotropicElasticModel::nonlinearInitGaussPointOperation( double )
{
    if ( d_useMaterialsLibrary == false ) {
        d_E.push_back( default_E );
        d_Nu.push_back( default_Nu );
    } else {
        d_E.push_back( 0.0 );
        d_Nu.push_back( 0.0 );
    }

    d_detULF.push_back( 1.0 );

    for ( int i = 0; i < 6; i++ ) {
        d_EquilibriumStress.push_back( 0 );
        d_EquilibriumStrain.push_back( 0 );

        d_tmp1Stress.push_back( 0 );
        d_tmp1Strain.push_back( 0 );
    } // end for i
}


void IsotropicElasticModel::globalReset()
{
    AMP_INSIST( ( d_resetReusesRadialReturn == true ), "Inconsistent options!" );

    d_EquilibriumStress = d_tmp1Stress;
    d_EquilibriumStrain = d_tmp1Strain;

    // double gaussPtCnt = 3;
    // fprintf(fout,"%le %le %le %le %le %le\n",d_EquilibriumStrain[(6 * gaussPtCnt) +
    // 1],d_EquilibriumStress[(6 *
    // gaussPtCnt) + 1],d_EquilibriumStrain[(6 * gaussPtCnt) + 2],d_EquilibriumStress[(6 *
    // gaussPtCnt) +
    // 2],d_EquilibriumStrain[(6 * gaussPtCnt) + 3],d_EquilibriumStress[(6 * gaussPtCnt) + 3]);
}


void IsotropicElasticModel::nonlinearResetGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];
    }

    double *stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    calculateStress( strain, stress );
}


void IsotropicElasticModel::nonlinearResetGaussPointOperation_UL(
    const std::vector<std::vector<double>> &strain, double R_n[3][3], double R_np1[3][3] )
{
    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] =
            d_EquilibriumStrain[( 6 * d_gaussPtCnt ) + i] + strain[Mechanics::DISPLACEMENT][i];
    }

    double *stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    /*
    double Identity[3][3];
    // Constructing an identity matrix.
    for(unsigned int i = 0; i < 3; i++) {
      for(unsigned int j = 0; j < 3; j++) {
        Identity[i][j] = 0.0;
      }
      Identity[i][i] = 1.0;
    }
    */

    // calculateStress(strain, stress, Om, Identity);
    calculateStress( strain, stress, R_n, R_np1 );
}


void IsotropicElasticModel::nonlinearJacobianGaussPointOperation(
    const std::vector<std::vector<double>> &strain )
{
    if ( d_useMaterialsLibrary == true ) {
        computeEvalv( strain );
    }
}


void IsotropicElasticModel::nonlinearJacobianGaussPointOperation_UL(
    const std::vector<std::vector<double>> &strain, double[3][3], double[3][3] )
{
    if ( d_useMaterialsLibrary == true ) {
        computeEvalv( strain );
    }
}


void IsotropicElasticModel::postNonlinearReset()
{
    d_EquilibriumStress = d_tmp1Stress;
    d_EquilibriumStrain = d_tmp1Strain;
}


void IsotropicElasticModel::getConstitutiveMatrix( double *&constitutiveMatrix )
{
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


void IsotropicElasticModel::getConstitutiveMatrixUpdatedLagrangian( double constitutiveMatrix[6][6],
                                                                    double[3][3] )
{
    // if(d_useMaterialsLibrary == false) {
    //  constitutiveMatrix = &(d_constitutiveMatrix_UL[0][0]);
    //}

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


void IsotropicElasticModel::getInternalStress( const std::vector<std::vector<double>> &strain,
                                               double *&stress )
{
    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] = strain[Mechanics::DISPLACEMENT][i];
    }

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    calculateStress( strain, stress );
}


void IsotropicElasticModel::getInternalStress_UL( const std::vector<std::vector<double>> &strain,
                                                  double *&stress,
                                                  double R_n[3][3],
                                                  double R_np1[3][3],
                                                  double detF )
{
    for ( int i = 0; i < 6; i++ ) {
        d_tmp1Strain[( 6 * d_gaussPtCnt ) + i] =
            d_EquilibriumStrain[( 6 * d_gaussPtCnt ) + i] + strain[Mechanics::DISPLACEMENT][i];
    }

    stress = &( d_tmp1Stress[6 * d_gaussPtCnt] );

    if ( d_iDebugPrintInfoLevel > 11 ) {
        for ( int i = 0; i < 6; i++ ) {
            AMP::pout << "strain[" << i << "] = " << strain[Mechanics::DISPLACEMENT][i]
                      << std::endl;
        }
    }

    calculateStress( strain, stress, R_n, R_np1 );

    d_detULF[d_gaussPtCnt] = detF;

    if ( d_useJaumannRate == true ) {
        for ( int i = 0; i < 6; i++ ) {
            stress[i] /= detF;
        }
    }
}


void IsotropicElasticModel::constructConstitutiveMatrix( const double E, const double Nu )
{

    double G = E / ( 2.0 * ( 1.0 + Nu ) );

    // double c = 2.0*G;
    double c = G;
    double a = 2.0 * c * ( 1.0 - Nu ) / ( 1.0 - ( 2.0 * Nu ) );
    double b = 2.0 * c * Nu / ( 1.0 - ( 2.0 * Nu ) );

    for ( auto &elem : d_constitutiveMatrix ) {
        for ( double &j : elem ) {
            j = 0.0;
        } // end for j
    }     // end for i

    d_constitutiveMatrix[0][0] = a;
    d_constitutiveMatrix[1][1] = a;
    d_constitutiveMatrix[2][2] = a;

    d_constitutiveMatrix[0][1] = b;
    d_constitutiveMatrix[0][2] = b;
    d_constitutiveMatrix[1][0] = b;
    d_constitutiveMatrix[1][2] = b;
    d_constitutiveMatrix[2][0] = b;
    d_constitutiveMatrix[2][1] = b;

    d_constitutiveMatrix[3][3] = c;
    d_constitutiveMatrix[4][4] = c;
    d_constitutiveMatrix[5][5] = c;

    /*
       double K = E/(3.0*(1.0 - (2.0*Nu)));

       for(int i = 0; i < 6; i++) {
       for(int j = 0; j < 6; j++) {
       d_constitutiveMatrix[i][j] = 0.0;
       }
       }

       for(int i = 0; i < 6; i++) {
       d_constitutiveMatrix[i][i] += (2.0 * G);
       }

       for(int i = 0; i < 3; i++) {
       for(int j = 0; j < 3; j++) {
       d_constitutiveMatrix[i][j] += (K - ((2.0/3.0) * G));
       }//end for j
       }//end for i
       */
}


void IsotropicElasticModel::constructConstitutiveMatrixUpdatedLagrangian( const double E,
                                                                          const double Nu )
{
    double G          = E / ( 2.0 * ( 1.0 + Nu ) );
    double K          = E / ( 3.0 * ( 1.0 - ( 2.0 * Nu ) ) );
    double twoG_three = 2.0 * G / 3.0;

    for ( auto &elem : d_constitutiveMatrix_UL ) {
        for ( double &j : elem ) {
            j = 0.0;
        } // end for j
    }     // end for i

    for ( int i = 0; i < 3; i++ ) {
        for ( int j = 0; j < 3; j++ ) {
            d_constitutiveMatrix_UL[i][j] += ( K - twoG_three );
        }
        d_constitutiveMatrix_UL[i][i] += ( 2.0 * G );
        d_constitutiveMatrix_UL[i + 3][i + 3] += ( 1.0 * G );
    }
}


void IsotropicElasticModel::computeEvalv( const std::vector<std::vector<double>> &strain )
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


void IsotropicElasticModel::calculateStress( const std::vector<std::vector<double>> &strain,
                                             double *&stress )
{
    if ( d_useMaterialsLibrary == true ) {

        computeEvalv( strain );

        double pass_E  = d_E[d_gaussPtCnt];
        double pass_Nu = d_Nu[d_gaussPtCnt];

        constructConstitutiveMatrix( pass_E, pass_Nu );
    }

    for ( int i = 0; i < 6; i++ ) {
        stress[i] = 0.0;
        for ( int j = 0; j < 6; j++ ) {
            stress[i] += ( d_constitutiveMatrix[i][j] * strain[Mechanics::DISPLACEMENT][j] );
        } // end for j
    }     // end for i

    // for(int i=0; i<6; i++) std::cout<<"stress["<<i<<"]="<<stress[i]<<std::endl;
}


void IsotropicElasticModel::calculateStress( const std::vector<std::vector<double>> &strain,
                                             double *&stress,
                                             double R_n[3][3],
                                             double R_np1[3][3] )
{
    if ( d_useMaterialsLibrary == true ) {
        computeEvalv( strain );
        double pass_E  = d_E[d_gaussPtCnt];
        double pass_Nu = d_Nu[d_gaussPtCnt];
        constructConstitutiveMatrixUpdatedLagrangian( pass_E, pass_Nu );
    } else {
        double pass_E  = d_E[d_gaussPtCnt];
        double pass_Nu = d_Nu[d_gaussPtCnt];
        constructConstitutiveMatrixUpdatedLagrangian( pass_E, pass_Nu );
    }

    /*
    double Identity[3][3];
    for(int i = 0; i < 3; i++) {
      for(int j = 0; j < 3; j++) {
        Identity[i][j] = 0.0;
      }
      Identity[i][i] = 1.0;
    }
    */

    double stress_n[6], S[3][3], Sr[3][3], stress_np1[6];

    if ( d_useJaumannRate == false ) {
        if ( d_iDebugPrintInfoLevel > 11 ) {
            for ( int i = 0; i < 6; i++ ) {
                std::cout << "d_EquilibriumStress[" << i
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
        stress_n[0] = Sr[0][0];
        stress_n[1] = Sr[1][1];
        stress_n[2] = Sr[2][2];
        stress_n[3] = ( 0.5 * ( Sr[1][2] + Sr[2][1] ) );
        stress_n[4] = ( 0.5 * ( Sr[0][2] + Sr[2][0] ) );
        stress_n[5] = ( 0.5 * ( Sr[0][1] + Sr[1][0] ) );
        if ( d_iDebugPrintInfoLevel > 11 ) {
            for ( int i = 0; i < 6; i++ ) {
                std::cout << "stress_n[" << i << "]=" << stress_n[i] << std::endl;
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

    for ( int i = 0; i < 6; i++ ) {
        stress_np1[i] = stress_n[i];
        for ( int j = 0; j < 6; j++ ) {
            stress_np1[i] += ( d_constitutiveMatrix_UL[i][j] * strain[Mechanics::DISPLACEMENT][j] );
        } // end for j
    }     // end for i

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

    if ( d_iDebugPrintInfoLevel > 11 ) {
        AMP::pout << "The current Gauss point is - " << d_gaussPtCnt << std::endl;
        for ( int i = 0; i < 6; i++ ) {
            AMP::pout << "stress_np1[" << i << "] = " << stress_np1[i] << std::endl;
        }
        AMP::pout << "\n" << std::endl;
    }

    for ( int i = 0; i < 6; i++ ) {
        stress[i] = stress_np1[i];
    }
    // for(int i=0; i<6; i++) std::cout<<"stress["<<i<<"]="<<stress[i]<<std::endl;
}
} // namespace AMP::Operator
