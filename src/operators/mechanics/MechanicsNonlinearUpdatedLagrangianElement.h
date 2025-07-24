#ifndef included_AMP_MechanicsNonlinearUpdatedLagrangianElement
#define included_AMP_MechanicsNonlinearUpdatedLagrangianElement

#include "AMP/operators/mechanics/MechanicsConstants.h"
#include "AMP/operators/mechanics/MechanicsElement.h"
#include "AMP/operators/mechanics/MechanicsNonlinearElement.h"
#include "AMP/operators/mechanics/UpdatedLagrangianUtils.h"

#include <memory>
#include <vector>


namespace AMP::Operator {


/**
A class for representing the element level computation performed within a
nonlinear updated lagrangian finite element operator for modelling solid mechanics.
*/
class MechanicsNonlinearUpdatedLagrangianElement : public MechanicsElement
{
public:
    //! Constructor.
    explicit MechanicsNonlinearUpdatedLagrangianElement(
        std::shared_ptr<const ElementOperationParameters> params )
        : MechanicsElement( params ), d_elementOutputVector( nullptr )
    {
        d_JxW = &( d_fe->get_JxW() );

        d_dphi = &( d_fe->get_dphi() );

        d_phi = &( d_fe->get_phi() );

        d_xyz = &( d_fe->get_xyz() );

        d_onePointShearIntegration = false;

        d_gaussPtCnt = 0;
    }

    //! Destructor.
    virtual ~MechanicsNonlinearUpdatedLagrangianElement() {}

    /**
      This function is used by MechanicsNonlinearFEOperator to pass the address
      of the element Input and Output vector to this class.
      @param [in] elementInputVectors Element input vector
      @param [in] elementInputVectors_pre Element input vector for previous time
      @param [in] elementOutputVector Element residual vector
     */
    void setElementVectors( const std::vector<std::vector<double>> &elementInputVectors,
                            const std::vector<std::vector<double>> &elementInputVectors_pre,
                            std::vector<double> &elementOutputVector )
    {
        d_elementInputVectors     = elementInputVectors;
        d_elementInputVectors_pre = elementInputVectors_pre;
        d_elementOutputVector     = &( elementOutputVector );
    }

    /**
      Element residual vector computation.
     */
    void apply() override
    {
        if ( d_useReducedIntegration ) {
            apply_Reduced();
        } else {
            apply_Normal();
        }
    }

    /**
      This function is used to initialize the material model at all Gauss points within the current
      element.
      Some material models require the reference values for some fields such as temperature and so
      these values are also passed to the material model through this function.
      Currently, the only field required by the mechanics material models during initialization is
      the reference
      temperature.
      @param [in] initTempVector Reference temperature at the nodes of the current element. This can
      be
      empty if the material model does not require a reference temperature.
     */
    void initMaterialModel( const std::vector<double> &initTempVector );

    /**
      This function is used to update the equilibrium values of stress, strain,
      temperature, burnup etc that are stored in the material model class.
      This is typically required at the end of each loading step.
      This function is typically  called from within the MechanicsNonlinearFEOperator's
      reset function, which is evaluated at the end of each loading step.
      This function also gets called from within the MechanicsNonlinearFEOperator's
      getJacobianParameters function if the jacobian is evaluated at a state
      (displacement, temperature etc) different from that of the last call to
      MechanicsNonlinearFEOperator's apply function.
      @param [in] elementInputVectors Field (Displacement, Temperature, Burnup etc) values at the
         nodes of the current element.
      @param [in] elementInputVectors_pre Field (Displacement, Temperature, Burnup etc) values at
         the nodes of the current element at the previous time-step.
     */
    void updateMaterialModel( MechanicsNonlinearElement::MaterialUpdateType type,
                              const std::vector<std::vector<double>> &elementInputVectors,
                              const std::vector<std::vector<double>> &elementInputVectors_pre );

    /**
      Writes the stess and strain values at the Gauss points within the current element to the file
      The 6 components of stress and strain at each Gauss point are arranged in the order:
      xx, yy, zz, yz, xz and  xy.
      @param [in] fp File pointer
      @param [in] inputVec Input vector (Displacement, Temperature, Burnup etc) at the nodes of the
      current element.
     */
    void printStressAndStrain( FILE *fp, const std::vector<std::vector<double>> &inputVec );

    /**
      Computes the stress and strain values at the Gauss points within the current element
      The 6 components of stress and strain at each Gauss point are arranged in the order:
      xx, yy, zz, yz, xz and  xy.
      @param [in] inputVec Input vector (Displacement, Temperature, Burnup etc) at the nodes of the
      current element.
      @param [out] stressVec Stresses at the Gauss points of the current element.
      @param [out] strainVec Strains at the Gauss points of the current element.
     */
    void computeStressAndStrain( const std::vector<std::vector<double>> &inputVec,
                                 std::vector<double> &stressVec,
                                 std::vector<double> &strainVec );

    /**
      Computes the deformation gradient at n-th, (n+1)-th and (n+1/2)-th time step.
     */
    void computeDeformationGradient( const std::vector<std::vector<libMesh::RealGradient>> &dphi,
                                     const std::vector<libMesh::Point> &xyz,
                                     unsigned int num_nodes,
                                     unsigned int qp,
                                     double F[3][3] );

    /**
      Initializes the reference x, y and z coordinates.
     */
    void initializeReferenceXYZ( std::vector<double> &elementRefXYZ );

    /**
      Assign the reference x, y and z coordinates for the current element.
     */
    void assignReferenceXYZ( const std::vector<double> &elementRefXYZ )
    {
        d_elementRefXYZ = elementRefXYZ;
    }

    void preNonlinearElementInit();

    void zeroOutGaussPointCount() { d_gaussPtCnt = 0; }

    void resetElementInfo()
    {
        if ( ( d_useJaumannRate == false ) && ( d_useFlanaganTaylorElem == true ) ) {
            d_leftStretchV_n = d_leftStretchV_np1;
            d_rotationR_n    = d_rotationR_np1;
        }
    }

protected:
    void materialModelPreNonlinearElementOperation( MechanicsNonlinearElement::MaterialUpdateType );

    void materialModelPreNonlinearGaussPointOperation(
        MechanicsNonlinearElement::MaterialUpdateType );

    void materialModelNonlinearGaussPointOperation( MechanicsNonlinearElement::MaterialUpdateType,
                                                    const std::vector<std::vector<double>> &,
                                                    double[3][3],
                                                    double[3][3] );

    void materialModelPostNonlinearGaussPointOperation(
        MechanicsNonlinearElement::MaterialUpdateType );

    void
        materialModelPostNonlinearElementOperation( MechanicsNonlinearElement::MaterialUpdateType );

    /**
      Element residual vector computation using normal integration scheme.
     */
    void apply_Normal();

    /**
      Element residual vector computation using reduced integration scheme.
     */
    void apply_Reduced();

    const std::vector<libMesh::Real> *d_JxW; /**< Product of the determinant of Jacobian and the
                                    quadrature weight at the Gauss points in the current element. */

    const std::vector<std::vector<libMesh::RealGradient>>
        *d_dphi; /**< Spatial Derivatives of the shape functions at
                  the Gauss points in the current element. */

    const std::vector<std::vector<libMesh::Real>>
        *d_phi; /**< Shape functions at
                         the Gauss points in the current element. */

    const std::vector<libMesh::Point>
        *d_xyz; /**< Locations of the Gauss points in the current element. */

    std::vector<std::vector<double>> d_elementInputVectors; /**< Element input vectors
                                                                   (Displacement, temperature,
                                                               burnup etc). */

    std::vector<std::vector<double>>
        d_elementInputVectors_pre; /**< Element input vectors at the previous time step
                                       (Displacement, temperature, burnup etc). */

    std::vector<double> *d_elementOutputVector; /**< Element residual vector */

    bool d_onePointShearIntegration;

    std::vector<double> d_elementRefXYZ;

    std::vector<double> d_leftStretchV_n;

    std::vector<double> d_leftStretchV_np1;

    std::vector<double> d_rotationR_n;

    std::vector<double> d_rotationR_np1;

    int d_gaussPtCnt;

private:
};


} // namespace AMP::Operator

#endif
