#ifndef included_AMP_ThermalVonMisesMatModel
#define included_AMP_ThermalVonMisesMatModel

#include "AMP/operators/mechanics/MechanicsMaterialModel.h"

#include <memory>
#include <vector>


namespace AMP::Operator {

class ThermalVonMisesMatModel : public MechanicsMaterialModel
{
public:
    explicit ThermalVonMisesMatModel( std::shared_ptr<MechanicsMaterialModelParameters> );

    virtual ~ThermalVonMisesMatModel() {}

    void getConstitutiveMatrix( double *& ) override;

    void getInternalStress( const std::vector<std::vector<double>> &, double *& ) override;

    void preLinearAssembly() override { d_gaussPtCnt = 0; }

    void postLinearGaussPointOperation() override { d_gaussPtCnt++; }

    void preNonlinearInit( bool, bool ) override;

    void nonlinearInitGaussPointOperation( double ) override;

    void preNonlinearAssembly() override
    {

        Plastic_Gauss_Point = 0;

        d_gaussPtCnt = 0;
    }

    void postNonlinearAssembly() override;

    void postNonlinearAssemblyGaussPointOperation() override { d_gaussPtCnt++; }

    void preNonlinearReset() override { d_gaussPtCnt = 0; }

    void postNonlinearResetGaussPointOperation() override { d_gaussPtCnt++; }

    void nonlinearResetGaussPointOperation( const std::vector<std::vector<double>> & ) override;

    void globalReset() override;

    void postNonlinearReset() override;

    void preNonlinearJacobian() override { d_gaussPtCnt = 0; }

    void postNonlinearJacobianGaussPointOperation() override { d_gaussPtCnt++; }

    void nonlinearJacobianGaussPointOperation( const std::vector<std::vector<double>> & ) override;

protected:
    void computeEvalv( const std::vector<std::vector<double>> & );

    void radialReturn( const double *stra_np1,
                       double *stre_np1,
                       double *ystre_np1,
                       double *eph_bar_plas_np1 );

    bool d_resetReusesRadialReturn;
    bool d_jacobianReusesRadialReturn;
    bool d_Is_Init_Called;

    unsigned int d_gaussPtCnt;
    unsigned int Total_Gauss_Point;
    unsigned int Plastic_Gauss_Point;

    double default_TEMPERATURE;
    double default_OXYGEN_CONCENTRATION;
    double default_BURNUP;

    double default_E;
    double default_Nu;
    double default_alpha;

    double d_H;
    double d_Sig0;
    double d_constitutiveMatrix[6][6];

    std::vector<int> d_ElPl;
    std::vector<double> d_E;
    std::vector<double> d_Nu;
    std::vector<double> d_alpha;
    std::vector<double> d_EquilibriumStress;
    std::vector<double> d_EquilibriumStrain;
    std::vector<double> d_EquilibriumYieldStress;
    std::vector<double> d_EquilibriumEffectivePlasticStrain;
    std::vector<double> d_EquilibriumTemperature;
    std::vector<double> d_Lambda;
    std::vector<double> d_tmp1Stress;
    std::vector<double> d_tmp1Strain;
    std::vector<double> d_tmp1YieldStress;
    std::vector<double> d_tmp1EffectivePlasticStrain;
    std::vector<double> d_tmp1Temperature;
    std::vector<double> d_tmp2Stress;
    std::vector<double> d_tmp2YieldStress;
    std::vector<double> d_tmp2EffectivePlasticStrain;

private:
};
} // namespace AMP::Operator

#endif
