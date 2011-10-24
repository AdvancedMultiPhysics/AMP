
#ifndef included_AMP_ThermalVonMisesMatModel
#define included_AMP_ThermalVonMisesMatModel

#include "MechanicsMaterialModel.h"

#include "boost/shared_ptr.hpp"

#include <vector>

namespace AMP {
namespace Operator {

  class ThermalVonMisesMatModel : public MechanicsMaterialModel 
  {
    public :

      ThermalVonMisesMatModel(const boost::shared_ptr<MechanicsMaterialModelParameters>& );

      ~ThermalVonMisesMatModel() { }

      void getConstitutiveMatrix(double*& ); 

      void getInternalStress(const std::vector<std::vector<double> >& , double*& );

      void preLinearAssembly() {
        d_gaussPtCnt = 0;
      }

      void postLinearGaussPointOperation() {
        d_gaussPtCnt++;
      }

      void preNonlinearInit(bool, bool);

      void nonlinearInitGaussPointOperation(double); 

      void preNonlinearAssembly() {

        Plastic_Gauss_Point = 0;

        d_gaussPtCnt = 0;
      }

      void postNonlinearAssembly() {
        
        if(Total_Gauss_Point == 0) {
          std::cout << "Total number of gauss points are zero." << std::endl;
        } else {
          double Plastic_Fraction = ((double)Plastic_Gauss_Point) / ((double)Total_Gauss_Point);
          Plastic_Fraction = Plastic_Fraction * 100.0;
          std::cout << "Fraction = " << Plastic_Fraction << "% Plastic = " << Plastic_Gauss_Point << " Total = " << Total_Gauss_Point << " Gauss Points." << std::endl;
        }

      }

      void postNonlinearAssemblyGaussPointOperation() {
        d_gaussPtCnt++;
      }

      void preNonlinearReset() {
        d_gaussPtCnt = 0;
      }

      void postNonlinearResetGaussPointOperation() {
        d_gaussPtCnt++;
      }

      void nonlinearResetGaussPointOperation(const std::vector<std::vector<double> >&); 

      void globalReset();

      void postNonlinearReset();

      void preNonlinearJacobian() {
        d_gaussPtCnt = 0;
      }

      void postNonlinearJacobianGaussPointOperation() {
        d_gaussPtCnt++;
      }

      void nonlinearJacobianGaussPointOperation(const std::vector<std::vector<double> >&); 

    protected :

      void computeEvalv(const std::vector<std::vector<double> >& );

      void radialReturn(const double* stra_np1, double* stre_np1,
          double *ystre_np1, double *eph_bar_plas_np1 );

      double default_TEMPERATURE;

      double default_OXYGEN_CONCENTRATION;

      double default_BURNUP;

      std::vector<double> d_E;

      double default_E;

      std::vector<double> d_Nu;

      double default_Nu;

      double d_H;

      double d_Sig0;

      std::vector<double> d_alpha;

      double default_alpha;

      unsigned int d_gaussPtCnt;

      unsigned int Total_Gauss_Point;

      unsigned int Plastic_Gauss_Point;

      double d_constitutiveMatrix[6][6];

      std::vector<double> d_EquilibriumStress;

      std::vector<double> d_EquilibriumStrain;

      std::vector<double> d_EquilibriumYieldStress;

      std::vector<double> d_EquilibriumEffectivePlasticStrain;

      std::vector<double> d_EquilibriumTemperature;

      std::vector<double> d_Lambda;

      std::vector<int> d_ElPl;

      std::vector<double> d_tmp1Stress;

      std::vector<double> d_tmp1Strain;

      std::vector<double> d_tmp1YieldStress;

      std::vector<double> d_tmp1EffectivePlasticStrain;

      std::vector<double> d_tmp1Temperature;

      std::vector<double> d_tmp2Stress;

      std::vector<double> d_tmp2YieldStress;

      std::vector<double> d_tmp2EffectivePlasticStrain;

      bool d_resetReusesRadialReturn;

      bool d_jacobianReusesRadialReturn;

      bool d_Is_Init_Called;

    private :

  };

}
}

#endif

