#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ElementPhysicsModelFactory.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/ElasticDamageThermalStrainModel.h"
#include "AMP/operators/mechanics/IsotropicElasticModel.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iostream>
#include <memory>
#include <string>

#define ITFAILS ut.failure( __LINE__ );
#define UNIT_TEST( a ) \
    if ( !( a ) )      \
        ut.failure( __LINE__ );


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
    auto edtsModel = std::dynamic_pointer_cast<AMP::Operator::ElasticDamageThermalStrainModel>(
        elementPhysicsModel );
    // vmepModel = std::dynamic_pointer_cast<AMP::IsotropicElasticModel>(elementPhysicsModel);

    edtsModel->preNonlinearInit( true, true );
    edtsModel->nonlinearInitGaussPointOperation( 300.0 );
    edtsModel->preNonlinearAssembly();

    int max_num = 100;

    double *stress;
    double *C;
    // double sigy = 0.243, E = 70.0, Ep = 0.135, nu = 0.3333333, diff_norm, alp = 1.0;
    double alp = 1.0;
    // double eph11[500], sig11[500], sig11_init[500], slope[500], sig11p[500], eph11p[500],
    // slope_p[500], d_strain[6],
    // damage_param[500];
    double eph11[500], sig11[500], slope[500], d_strain[6], damage_param[500];
    std::vector<std::vector<double>> strain( 2 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );
    strain[0].push_back( 0.0 );

    strain[1].push_back( 301.0 );

    d_strain[0] = 0.001;
    d_strain[1] = 0.0;
    d_strain[2] = 0.0;
    d_strain[3] = 0.0;
    d_strain[4] = 0.0;
    d_strain[5] = 0.0;

    for ( int i = 0; i < max_num; i++ ) {
        for ( int j = 0; j < 6; j++ ) {
            strain[0][j] += ( alp * d_strain[j] );
        }
        eph11[i] = strain[0][0];
        // eph11p[i] = strain[0][0] - ((1.0/3.0) * (strain[0][0] + strain[0][1] + strain[0][2]));
        edtsModel->getInternalStress( strain, stress );
        sig11[i] = stress[0];
        // sig11p[i] = stress[0] - ((1.0/3.0) * (stress[0] + stress[1] + stress[2]));
        edtsModel->globalReset();

        damage_param[i] = edtsModel->d_EquilibriumDamage[0];

        if ( i == 0 ) {
            slope[0] = 0.0;
            // slope_p[0] = 0.0;
        } else {
            slope[i] = ( sig11[i] - sig11[i - 1] ) / ( eph11[i] - eph11[i - 1] );
            // slope_p[i] = (sig11p[i] - sig11p[i-1]) / (eph11p[i] - eph11p[i-1]);
        }
    }

    FILE *fout;
    fout = fopen( "edts_stress_stain_results.xls_6", "w" );
    for ( int i = 0; i < max_num; i++ ) {
        // fprintf(fout,"%15.8lf%15.8lf%15.8lf%15.8lf%15.8lf%15.8lf%15.8lf\n",eph11[i],sig11[i],slope[i],eph11p[i],sig11p[i],slope_p[i],sig11_init[i]);
        fprintf(
            fout, "%15.8f%15.8f%15.8f%15.8f\n", eph11[i], sig11[i], slope[i], damage_param[i] );
    }
    fclose( fout );

    /*  for(int i = 0; i < 6; i++) {
        printf("stress[%d]=%lf\n",i,stress[i]);
      }*/

    // vmepModel->preNonlinearJacobian();
    // vmepModel->nonlinearJacobianGaussPointOperation(strain);
    // vmepModel->postNonlinearJacobianGaussPointOperation();

    edtsModel->preLinearAssembly();
    edtsModel->getConstitutiveMatrix( C );
    edtsModel->postLinearGaussPointOperation();

    /*  for(int i = 0; i < 6; i++) {
        for(int j = 0; j < 6; j++) {
          printf("C[%d][%d] = %lf\n",i,j,C[(i * 6) + j]);
        }
      }*/

    /*  diff_norm = 0.0;
      for(int i = 0; i < max_num; i++) {
        diff_norm += ((sig11_init[i] - sig11[i]) * (sig11_init[i] - sig11[i]));
      }
      diff_norm = std::sqrt(diff_norm) / sig11[max_num - 1];
      if(diff_norm > (1.0e-8)) {
        ut.numFails++;
        ut.failure("Gauss point test for Linear Isotropic Hardening");
      } else {
        ut.passes("Gauss point test for Linear Isotropic Hardening");
      }

      std::cout << "diff_norm = " << diff_norm << std::endl;
    */
    ut->passes( exeName );
}

int testMaterialModels_ElasticDamageThermalStrain( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testMaterialModels_ElasticDamageThermalStrain-1" );

    for ( auto &exeName : exeNames )
        myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
