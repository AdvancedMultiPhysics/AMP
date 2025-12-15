#include "AMP/IO/PIO.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/IsotropicElasticModel.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/operators/mechanics/VonMisesElastoPlasticModel.h"
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

    std::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel;
    std::shared_ptr<AMP::Operator::VonMisesElastoPlasticModel> vmepModel;

    auto matModelDatabase = input_db->getDatabase( "MechanicsMaterialModel" );
    elementPhysicsModel =
        AMP::Operator::ElementPhysicsModelFactory::createElementPhysicsModel( matModelDatabase );
    vmepModel =
        std::dynamic_pointer_cast<AMP::Operator::VonMisesElastoPlasticModel>( elementPhysicsModel );
    // vmepModel = std::dynamic_pointer_cast<AMP::IsotropicElasticModel>(elementPhysicsModel);

    vmepModel->preNonlinearInit( true, true );
    vmepModel->nonlinearInitGaussPointOperation( 500.0 );
    vmepModel->preNonlinearAssembly();

    int max_num = 250;

    double *stress;
    double *C;
    double sigy = 0.243, E = 70.0, Ep = 0.135, nu = 0.3333333, diff_norm;
    double eph11[500], sig11[500], sig11_init[500], slope[500], sig11p[500], eph11p[500],
        slope_p[500];
    std::vector<std::vector<double>> strain( 1 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );

    for ( int i = 0; i < max_num; i++ ) {
        sig11_init[i] = ( (double) ( i + 1 ) ) * 0.001;
        if ( sig11_init[i] <= sigy ) {
            strain[0][0] = sig11_init[i] / E;
            strain[0][1] = -nu * ( sig11_init[i] / E );
            strain[0][2] = -nu * ( sig11_init[i] / E );
        } else {
            strain[0][0] = ( sig11_init[i] / E ) + ( ( sig11_init[i] - sigy ) / Ep );
            strain[0][1] =
                ( -nu * ( sig11_init[i] / E ) ) - ( ( 1.0 / 2.0 ) * ( sig11_init[i] - sigy ) / Ep );
            strain[0][2] =
                ( -nu * ( sig11_init[i] / E ) ) - ( ( 1.0 / 2.0 ) * ( sig11_init[i] - sigy ) / Ep );
        }
        eph11[i] = strain[0][0];
        eph11p[i] =
            strain[0][0] - ( ( 1.0 / 3.0 ) * ( strain[0][0] + strain[0][1] + strain[0][2] ) );
        vmepModel->getInternalStress( strain, stress );
        sig11[i]  = stress[0];
        sig11p[i] = stress[0] - ( ( 1.0 / 3.0 ) * ( stress[0] + stress[1] + stress[2] ) );
        vmepModel->globalReset();

        if ( i == 0 ) {
            slope[0]   = 0.0;
            slope_p[0] = 0.0;
        } else {
            slope[i]   = ( sig11[i] - sig11[i - 1] ) / ( eph11[i] - eph11[i - 1] );
            slope_p[i] = ( sig11p[i] - sig11p[i - 1] ) / ( eph11p[i] - eph11p[i - 1] );
        }
    }

    FILE *fout;
    fout = fopen( "stress_stain_results.xls", "w" );
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

    /*  for(int i = 0; i < 6; i++) {
        printf("stress[%d]=%lf\n",i,stress[i]);
      }*/

    // vmepModel->preNonlinearJacobian();
    // vmepModel->nonlinearJacobianGaussPointOperation(strain);
    // vmepModel->postNonlinearJacobianGaussPointOperation();

    vmepModel->preLinearAssembly();
    vmepModel->getConstitutiveMatrix( C );
    vmepModel->postLinearGaussPointOperation();

    /*  for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 6; j++) {
          printf("C[%d][%d] = %lf\n",i,j,C[(i * 6) + j]);
        }
      }*/

    diff_norm = 0.0;
    for ( int i = 0; i < max_num; i++ ) {
        diff_norm += ( ( sig11_init[i] - sig11[i] ) * ( sig11_init[i] - sig11[i] ) );
    }
    diff_norm = std::sqrt( diff_norm ) / sig11[max_num - 1];
    if ( diff_norm > ( 1.0e-8 ) ) {
        ut->failure( "Gauss point test for Linear Isotropic Hardening" );
    } else {
        ut->passes( "Gauss point test for Linear Isotropic Hardening" );
    }

    std::cout << "diff_norm = " << diff_norm << std::endl;

    ut->passes( exeName );
}

int testMaterialModels_VonMisesPlasticity_1( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testMaterialModels_VonMisesPlasticity-1" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
