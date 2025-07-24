#ifndef included_AMP_MechanicsNonlinearElement
#define included_AMP_MechanicsNonlinearElement

#include "AMP/operators/mechanics/MechanicsConstants.h"
#include "AMP/operators/mechanics/MechanicsElement.h"

#include <memory>
#include <vector>


namespace AMP::Operator {


/**
  A class for representing the element level computation performed within a
  nonlinear finite element operator for modelling solid mechanics.
 */
class MechanicsNonlinearElement : public MechanicsElement
{
public:
    /**
      This is primarily for use with the updateMaterialModel() function.
      */
    enum MaterialUpdateType { RESET, JACOBIAN };

    //! Constructor.
    explicit MechanicsNonlinearElement( std::shared_ptr<const ElementOperationParameters> params )
        : MechanicsElement( params ), d_elementOutputVector( nullptr )
    {
        d_JxW = &( d_fe->get_JxW() );

        d_dphi = &( d_fe->get_dphi() );

        d_phi = &( d_fe->get_phi() );

        d_xyz = &( d_fe->get_xyz() );

        AMP_INSIST( ( d_useJaumannRate == false ),
                    "Jaumann rate with small strain does not make any sense." );
    }

    //! Destructor.
    virtual ~MechanicsNonlinearElement() {}

    /**
      This function is used by MechanicsNonlinearFEOperator to pass the address
      of the element Input and Output vector to this class.
      @param [in] elementInputVectors Element input vector
      @param [in] elementOutputVector Element residual vector
     */
    void setElementVectors( const std::vector<std::vector<double>> &elementInputVectors,
                            std::vector<double> &elementOutputVector )
    {
        d_elementInputVectors = elementInputVectors;
        d_elementOutputVector = &( elementOutputVector );
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
      This function is used to update the equilibrium values of stress, strain, temperature, burnup
      etc that are
      stored in the material model class. This is typically required at the end of each loading
      step. This function is
      typically
      called from within the MechanicsNonlinearFEOperator's reset function, which is evaluated at
      the end of each
      loading step.
      This function also gets called from within the MechanicsNonlinearFEOperator's
      getJacobianParameters
      function if the jacobian is evaluated at a state (displacement, temperature etc) different
      from
      that of the last call to MechanicsNonlinearFEOperator's apply function.
      @param [in] elementInputVectors Field (Displacement, Temperature, Burnup etc) values at the
      nodes of the current element.
     */
    void updateMaterialModel( MaterialUpdateType type,
                              const std::vector<std::vector<double>> &elementInputVectors );

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

protected:
    void materialModelPreNonlinearElementOperation( MaterialUpdateType );

    void materialModelPreNonlinearGaussPointOperation( MaterialUpdateType );

    void materialModelNonlinearGaussPointOperation( MaterialUpdateType,
                                                    const std::vector<std::vector<double>> & );

    void materialModelPostNonlinearGaussPointOperation( MaterialUpdateType );

    void materialModelPostNonlinearElementOperation( MaterialUpdateType );

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

    std::vector<double> *d_elementOutputVector; /**< Element residual vector */

private:
};


} // namespace AMP::Operator

#endif
