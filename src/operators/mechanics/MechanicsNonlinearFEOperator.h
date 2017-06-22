
#ifndef included_AMP_MechanicsNonlinearFEOperator
#define included_AMP_MechanicsNonlinearFEOperator

/* AMP files */
#include "ampmesh/MeshElement.h"
#include "discretization/DOF_Manager.h"
#include "operators/libmesh/NonlinearFEOperator.h"
#include "operators/mechanics/MechanicsConstants.h"
#include "operators/mechanics/MechanicsNonlinearElement.h"
#include "operators/mechanics/MechanicsNonlinearFEOperatorParameters.h"
#include "operators/mechanics/MechanicsNonlinearUpdatedLagrangianElement.h"
#include "vectors/MultiVariable.h"
#include "vectors/Variable.h"

#include <vector>

namespace AMP {
namespace Operator {

/**
  A class used for representing the nonlinear mechanics operator.
  This class can be used to compute the finite element (FE) residual
  vector corresponding to the mechanical equilibrium equations for a
  solid body. This class only deals with the volume integration,
  the boundary conditions are handled separately by the boundary operators.
  */
class MechanicsNonlinearFEOperator : public NonlinearFEOperator
{
public:
    /**
      Constructor. This reads the values for the following keys from the database object contained
      in
      the parameter object, params:
      1) RESET_REUSES_RADIAL_RETURN (true by default) - Can we assume that the apply() function is
      called before a call
      to
      the reset() function and can we reuse the results of the radial return algorithm computed in
      the apply()
      function in the reset() function? Note, this is typically true unless the reset() function
      is evaluated at a different state from the last call to apply().
      2) JACOBIAN_REUSES_RADIAL_RETURN (true by default) - Can we assume that the apply() function
      is called before a
      call to
      the getJacobianParameters() function and can we reuse the results of the radial return
      algorithm computed in the
      apply()
      function in the getJacobianParameters() function? Note, this is typically true unless the
      getJacobianParameters()
      function
      is evaluated at a different state from the last call to apply().
      3) ActiveInputVariables (No default value) - List of active input variables names. The
      supported variable types
      are:
      DISPLACEMENT, TEMPERATURE, BURNUP, OXYGEN_CONCENTRATION and LHGR. DISPLACEMENT must be active.
      4) FREEZE_TEMPERATURE/FREEZE_BURNUP/FREEZE_OXYGEN_CONCENTRATION/FREEZE_LHGR (true by default)
      - Are these
      variables frozen? This
      will be ignored if the corresponding variable is not active.
      5) OutputVariable (No default value) - Name of the output variable
      */
    explicit MechanicsNonlinearFEOperator(
        const AMP::shared_ptr<MechanicsNonlinearFEOperatorParameters> &params );

    /**
      Destructor.
      */
    virtual ~MechanicsNonlinearFEOperator() {}

    /**
      This is used to update the operator between successive solves with the operator.
      */
    void reset( const AMP::shared_ptr<OperatorParameters> & ) override;

    /**
      This function is used to set the reference temperature when using temperature dependent
      material models.
      @param [in] refTemp Reference temperature
      */
    void setReferenceTemperature( AMP::LinearAlgebra::Vector::const_shared_ptr refTemp );

    /**
      This function is used to set frozen vectors in this operator. This is used when some of the
      variables are solved for in an uncoupled manner.
      @param [in] id Variable Identifier - One of
      AMP::Mechanics::DISPLACEMENT/TEMPERATURE/BURNUP/OXYGEN_CONCENTRATION/LHGR
      @param [in] frozenVec Frozen vector
      @see MechanicsConstants.h
      */
    void setVector( unsigned int id, AMP::LinearAlgebra::Vector::const_shared_ptr frozenVec );

    /**
      @return The variable for the specified component of the input vector. If varId is equal to -1,
      it
      returns the multivariable for the entire vector.
      */
    AMP::LinearAlgebra::Variable::shared_ptr getInputVariable() override { return d_inpVariables; }

    /**
      @return The variable for the output vector
      */
    AMP::LinearAlgebra::Variable::shared_ptr getOutputVariable() override { return d_outVariable; }

    /**
      Writes the stress and strain at each Gauss point to a file.
      The 6 components of stress and strain at each Gauss point are arranged in the order:
      xx, yy, zz, yz, xz and  xy.
      @param [in] u Input vector
      @param [in] fname Name of the output file
      */
    void printStressAndStrain( AMP::LinearAlgebra::Vector::const_shared_ptr u, const std::string &fname );

    AMP::shared_ptr<MechanicsMaterialModel> getMaterialModel() { return d_materialModel; }

protected:
    /**
      This is used to compute the information required to reset the corresponding Linear (Jacobian)
      operator
      */
    AMP::shared_ptr<OperatorParameters>
        getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr ) override;

    /**
      This performs a dummy loop over the elements and gauss points so that the mechanics material
      model classes can
      allocate memory and/or initialize their data as required.
      */
    void init();

    /**
      This function is called at the beginning of the FE assembly. The output vector, r, is set to
      0.
      The values of the input vector, u, on the nodes shared between two or more processors are made
      consistent.
      @param [in] u  input vector
      @param [out] r output vector
      */
    void preAssembly( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                      AMP::shared_ptr<AMP::LinearAlgebra::Vector>
                          r ) override;

    /**
      This function is called at the end of the FE assembly.
      The values of the output vector on the nodes shared between two or more processors are made
      consistent.
      */
    void postAssembly() override;

    /**
      This function is called at the beginning of the element computation. The part of the
      input vector that is relevant for the computation in the current element is extracted
      and passed to MechanicsNonlinearElement.
      */
    void preElementOperation( const AMP::Mesh::MeshElement & ) override;

    /**
      This function is called at the end of the element computation. The entries of the
      element output vector are added to the corresponding entries of the global output vector.
      */
    void postElementOperation() override;

    AMP::LinearAlgebra::Vector::shared_ptr
    mySubsetVector( AMP::LinearAlgebra::Vector::shared_ptr vec,
                    AMP::LinearAlgebra::Variable::shared_ptr var );

    AMP::LinearAlgebra::Vector::const_shared_ptr
    mySubsetVector( AMP::LinearAlgebra::Vector::const_shared_ptr vec,
                    AMP::LinearAlgebra::Variable::shared_ptr var );

    template <MechanicsNonlinearElement::MaterialUpdateType updateType>
    void updateMaterialForElement( const AMP::Mesh::MeshElement & );

    template <MechanicsNonlinearUpdatedLagrangianElement::MaterialUpdateType updateType>
    void updateMaterialForUpdatedLagrangianElement( const AMP::Mesh::MeshElement & );

    void updateMaterialForElementCommonFunction( const AMP::Mesh::MeshElement &,
                                                 std::vector<std::vector<double>> &,
                                                 std::vector<std::vector<double>> & );

    void getDofIndicesForCurrentElement( int varId, std::vector<std::vector<size_t>> &dofIds );

    std::vector<double> d_elementOutputVector; /**< Element output vector. */

    AMP::shared_ptr<MechanicsNonlinearElement> d_mechNonlinElem; /**< Element operation. */

    AMP::shared_ptr<MechanicsNonlinearUpdatedLagrangianElement>
        d_mechNULElem; /**< Nonlinear Updated Lagrangian Element operation. */

    AMP::shared_ptr<MechanicsMaterialModel> d_materialModel; /**< Material model. */

    std::vector<AMP::LinearAlgebra::Vector::const_shared_ptr> d_inVec; /**< Input vector. */

    std::vector<AMP::LinearAlgebra::Vector::shared_ptr>
        d_inVec_pre; /**< Input vector. at nth (previous) configuration */

    AMP::LinearAlgebra::Vector::shared_ptr d_refXYZ; /**< Reference x, y and z coordinates. */

    AMP::LinearAlgebra::Vector::const_shared_ptr
        d_referenceTemperature; /**< Reference temperature. */

    AMP::LinearAlgebra::Vector::shared_ptr d_outVec; /**< Output vector. */

    bool d_resetReusesRadialReturn; /**< A flag that is true if the reset() function can reuse
                                      the results from the radial return computation
                                      in the last call to the apply() function and false otherwise.
                                      */

    bool d_jacobianReusesRadialReturn; /**< A flag that is true if the getJacobianParameters()
                                         function can reuse
                                         the results from the radial return computation
                                         in the last call to the apply() function and false
                                         otherwise. */

    std::vector<bool> d_isActive; /**< A list of flags to determine which variables are active. */

    std::vector<bool> d_isFrozen; /**< A list of flags to determine which variables are frozen. */

    bool d_useUpdatedLagrangian; /**< A flag that checks whether to use Updated Lagrangian
                                    formulation or not. */

    bool d_isInitialized; /**< A flag that is true if init() has been called and false otherwsie. */

    AMP::shared_ptr<AMP::LinearAlgebra::MultiVariable> d_inpVariables; /**< Input variables. */

    AMP::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable; /**< Output variable */

    AMP::shared_ptr<AMP::Discretization::DOFManager> d_dofMap[Mechanics::TOTAL_NUMBER_OF_VARIABLES];

    std::vector<AMP::Mesh::MeshElement> d_currNodes;

    std::vector<std::vector<size_t>> d_dofIndices; /**< Primary DOF indices */
};

template <MechanicsNonlinearElement::MaterialUpdateType updateType>
void MechanicsNonlinearFEOperator::updateMaterialForElement( const AMP::Mesh::MeshElement &elem )
{
    std::vector<std::vector<double>> elementInputVectors1( Mechanics::TOTAL_NUMBER_OF_VARIABLES );
    std::vector<std::vector<double>> elementInputVectors_pre1(
        Mechanics::TOTAL_NUMBER_OF_VARIABLES );

    updateMaterialForElementCommonFunction( elem, elementInputVectors1, elementInputVectors_pre1 );

    d_mechNonlinElem->updateMaterialModel<updateType>( elementInputVectors1 );
}

template <MechanicsNonlinearUpdatedLagrangianElement::MaterialUpdateType updateType>
void MechanicsNonlinearFEOperator::updateMaterialForUpdatedLagrangianElement(
    const AMP::Mesh::MeshElement &elem )
{
    std::vector<std::vector<double>> elementInputVectors2( Mechanics::TOTAL_NUMBER_OF_VARIABLES );
    std::vector<std::vector<double>> elementInputVectors_pre2(
        Mechanics::TOTAL_NUMBER_OF_VARIABLES );

    updateMaterialForElementCommonFunction( elem, elementInputVectors2, elementInputVectors_pre2 );

    d_mechNULElem->updateMaterialModel<updateType>( elementInputVectors2,
                                                    elementInputVectors_pre2 );
}
}
}

#endif
