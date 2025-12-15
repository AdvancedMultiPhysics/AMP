#include "AMP/IO/PIO.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/operators/mechanics/VonMisesElastoPlasticModel.h"
#include "AMP/operators/mechanics/VonMises_IsotropicKinematicHardening.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"

#include <iostream>
#include <memory>
#include <string>


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    auto matModelDatabase = input_db->getDatabase( "MechanicsMaterialModel" );
    auto elementPhysicsModel =
        AMP::Operator::ElementPhysicsModelFactory::createElementPhysicsModel( matModelDatabase );
    auto vmikModel = std::dynamic_pointer_cast<AMP::Operator::VonMises_IsotropicKinematicHardening>(
        elementPhysicsModel );

    vmikModel->preNonlinearInit( true, true );
    vmikModel->nonlinearInitGaussPointOperation( 500.0 );
    vmikModel->preNonlinearAssembly();

    int max_load = 250;
    int max_num  = 3 * max_load;

    double *stress;
    double *C;
    // double sigy = 0.2, E = 70.0, Ep = 0.135, H_a = 1.5, nu = 0.3333333, diff_norm;
    double sigy = 0.2, E = 70.0, H_a = 1.5, nu = 0.3333333, diff_norm;
    double eph11[1000], sig11[1000], sig11_init[1000], slope[1000], sig11p[1000], eph11p[1000],
        slope_p[1000];
    std::vector<std::vector<double>> strain( 1 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );

    for ( int i = 0; i < max_load; i++ ) {
        sig11_init[i] = ( (double) ( i + 1 ) ) * 0.001;
        if ( sig11_init[i] <= sigy ) {
            strain[0][0] = sig11_init[i] / E;
            strain[0][1] = -nu * ( sig11_init[i] / E );
            strain[0][2] = -nu * ( sig11_init[i] / E );
        } else {
            strain[0][0] = ( sig11_init[i] / E ) + ( ( sig11_init[i] - sigy ) / H_a );
            strain[0][1] = ( -nu * ( sig11_init[i] / E ) ) -
                           ( ( 1.0 / 2.0 ) * ( sig11_init[i] - sigy ) / H_a );
            strain[0][2] = ( -nu * ( sig11_init[i] / E ) ) -
                           ( ( 1.0 / 2.0 ) * ( sig11_init[i] - sigy ) / H_a );
            // strain[0][0] = (sig11_init[i] / E) + ((sig11_init[i] - sigy) / Ep);
            // strain[0][1] = (-nu * (sig11_init[i] / E)) - ((1.0 / 2.0) * (sig11_init[i] - sigy) /
            // Ep);
            // strain[0][2] = (-nu * (sig11_init[i] / E)) - ((1.0 / 2.0) * (sig11_init[i] - sigy) /
            // Ep);
        }
        eph11[i] = strain[0][0];
        eph11p[i] =
            strain[0][0] - ( ( 1.0 / 3.0 ) * ( strain[0][0] + strain[0][1] + strain[0][2] ) );
        vmikModel->getInternalStress( strain, stress );

        sig11[i]  = stress[0];
        sig11p[i] = stress[0] - ( ( 1.0 / 3.0 ) * ( stress[0] + stress[1] + stress[2] ) );
        vmikModel->globalReset();

        if ( i == 0 ) {
            slope[0]   = 0.0;
            slope_p[0] = 0.0;
        } else {
            slope[i]   = ( sig11[i] - sig11[i - 1] ) / ( eph11[i] - eph11[i - 1] );
            slope_p[i] = ( sig11p[i] - sig11p[i - 1] ) / ( eph11p[i] - eph11p[i - 1] );
        }
    }

    double sigy1             = ( ( 2.0 * sigy ) - sig11_init[max_load - 1] );
    double plastic_strain_11 = ( ( sig11_init[max_load - 1] - sigy ) / H_a );
    double plastic_strain_22 = -( ( 1.0 / 2.0 ) * ( sig11_init[max_load - 1] - sigy ) / H_a );
    double plastic_strain_33 = -( ( 1.0 / 2.0 ) * ( sig11_init[max_load - 1] - sigy ) / H_a );

    for ( int i = 0; i < ( 2 * max_load ); i++ ) {
        sig11_init[max_load + i] = sig11_init[max_load - 1] - ( ( (double) ( i + 1 ) ) * 0.001 );
        if ( -sig11_init[max_load + i] <= sigy1 ) {
            strain[0][0] = ( plastic_strain_11 ) + ( sig11_init[max_load + i] / E );
            strain[0][1] = ( plastic_strain_22 ) + ( -nu * ( sig11_init[max_load + i] / E ) );
            strain[0][2] = ( plastic_strain_33 ) + ( -nu * ( sig11_init[max_load + i] / E ) );
        } else {
            strain[0][0] = ( plastic_strain_11 ) + ( sig11_init[max_load + i] / E ) +
                           ( ( sig11_init[max_load + i] + sigy1 ) / H_a );
            strain[0][1] = ( plastic_strain_22 ) + ( -nu * ( sig11_init[max_load + i] / E ) ) -
                           ( ( 1.0 / 2.0 ) * ( sig11_init[max_load + i] + sigy1 ) / H_a );
            strain[0][2] = ( plastic_strain_33 ) + ( -nu * ( sig11_init[max_load + i] / E ) ) -
                           ( ( 1.0 / 2.0 ) * ( sig11_init[max_load + i] + sigy1 ) / H_a );
        }
        eph11[max_load + i] = strain[0][0];
        eph11p[max_load + i] =
            strain[0][0] - ( ( 1.0 / 3.0 ) * ( strain[0][0] + strain[0][1] + strain[0][2] ) );
        vmikModel->getInternalStress( strain, stress );

        sig11[max_load + i] = stress[0];
        sig11p[max_load + i] =
            stress[0] - ( ( 1.0 / 3.0 ) * ( stress[0] + stress[1] + stress[2] ) );
        vmikModel->globalReset();

        slope[max_load + i] = ( sig11[max_load + i] - sig11[max_load + i - 1] ) /
                              ( eph11[max_load + i] - eph11[max_load + i - 1] );
        slope_p[max_load + i] = ( sig11p[max_load + i] - sig11p[max_load + i - 1] ) /
                                ( eph11p[max_load + i] - eph11p[max_load + i - 1] );
    }

    FILE *fout;
    fout = fopen( "vmik_stress_strain_results.txt", "w" );
    for ( int i = 0; i < max_num; i++ ) {
        fprintf( fout,
                 "%15.8f%15.8f%15.8f%15.8f%15.8f%15.8f%15.8f\n",
                 eph11[i],
                 sig11[i],
                 slope[i],
                 eph11p[i],
                 sig11p[i],
                 slope_p[i],
                 sig11_init[i] );
    }
    fclose( fout );

    // for(int i = 0; i < 6; i++) {
    //  printf("stress[%d]=%lf\n",i,stress[i]);
    //}

    // vmikModel->preNonlinearJacobian();
    // vmikModel->nonlinearJacobianGaussPointOperation(strain);
    // vmikModel->postNonlinearJacobianGaussPointOperation();

    vmikModel->preLinearAssembly();
    vmikModel->getConstitutiveMatrix( C );
    vmikModel->postLinearGaussPointOperation();

    // for(int i = 0; i < 6; i++) {
    //  for(int j = 0; j < 6; j++) {
    //    printf("C[%d][%d] = %lf\n",i,j,C[(i * 6) + j]);
    //  }
    //}

    diff_norm = 0.0;
    for ( int i = 0; i < max_num; i++ ) {
        diff_norm += ( ( sig11_init[i] - sig11[i] ) * ( sig11_init[i] - sig11[i] ) );
    }
    diff_norm = std::sqrt( diff_norm ) / sig11[max_num - 1];
    if ( fabs( diff_norm ) > ( 1.0e-8 ) ) {
        ut->failure( "Gauss point test for Linear Isotropic Hardening" );
    } else {
        ut->passes( "Gauss point test for Linear Isotropic Hardening" );
    }

    std::cout << "diff_norm = " << fabs( diff_norm ) << std::endl;

    ut->passes( exeName );
}

int testMaterialModels_VonMisesPlasticity_2( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testMaterialModels_VonMisesPlasticity-2" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
