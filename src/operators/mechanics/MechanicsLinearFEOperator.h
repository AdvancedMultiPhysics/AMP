#ifndef included_AMP_MechanicsLinearFEOperator
#define included_AMP_MechanicsLinearFEOperator

/* AMP files */
#include "AMP/operators/libmesh/LinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsConstants.h"
#include "AMP/operators/mechanics/MechanicsLinearElement.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperatorParameters.h"
#include "AMP/operators/mechanics/MechanicsLinearUpdatedLagrangianElement.h"
#include "AMP/vectors/Variable.h"

#include <vector>


namespace AMP::Operator {

/**
  A class for representing the linear operator for linear/nonlinear mechanics.
  In the case of nonlinear mechanics, this operator will result from the
  linearization (or some approximate linearization) of the nonlinear operator.
  This class can be used to compute the finite element (FE) stiffness
  matrix corresponding to the mechanical equilibrium equations for a
  solid body. This class only deals with the volume integration,
  the boundary conditions are handled separately by the boundary operators.
  */
class MechanicsLinearFEOperator : public LinearFEOperator
{
public:
    /**
      Constructor. This allocates memory for the stiffness matrix. This also computes the entries of
      the stiffness
      matrix unless
      (a) this operator is the jacobian of the nonlinear mechanics operator and (b) the nonlinear
      mechanics operator is
      not already
      initialized at the time of construction of this operator. This reads the values for the
      following keys from the
      database object contained in
      the parameter object, params:
      1) isAttachedToNonlinearOperator (false by default) - Is this a jacobian of the nonlinear
      mechanics operator?
      2) isNonlinearOperatorInitialized (false by default) - If this is a jacobian of the nonlinear
      mechanics
      operator, is the nonlinear mechanics operator already initialized at the time of construction
      of this operator?
      3) InputVariable (No default value) - Name of the input variable
      4) OutputVariable (No default value) - Name of the output variable
      */
    explicit MechanicsLinearFEOperator( std::shared_ptr<const OperatorParameters> params );

    /**
      Destructor
      */
    virtual ~MechanicsLinearFEOperator() {}

    /**
      This is called at the start of the FE assembly. The matrix is set to 0.
      */
    void preAssembly( std::shared_ptr<const OperatorParameters> params ) override;

    /**
      This is called at the end of the FE assembly. The entries of the matrix corresponding
      to nodes that are shared between two or more processors are made consistent.
      */
    void postAssembly() override;

    /**
      This function will be called once for each element, just before performing
      the element operation. This function extracts the local information from
      the global mesh objects (DOFMap, global vectors and matrices) and
      passes them to the element operation.
      */
    void preElementOperation( const AMP::Mesh::MeshElement & ) override;

    /**
      This function will be called once for each element, just after performing the element
      operation.
      The element stiffness matrix is added to the global stiffness matrix in this function.
      */
    void postElementOperation() override;

    /**
      Writes the stress and strain at each Gauss point to a file.
      The 6 components of stress and strain at each Gauss point are arranged in the order:
      xx, yy, zz, yz, xz and  xy.
      @param [in] disp Displacement vector
      @param [in] fname Name of the output file
      */
    void printStressAndStrain( AMP::LinearAlgebra::Vector::shared_ptr disp,
                               const std::string &fname );

    std::shared_ptr<MechanicsMaterialModel> getMaterialModel() { return d_materialModel; }

protected:
    void getDofIndicesForCurrentElement();
    MechanicsLinearFEOperator( std::shared_ptr<const MechanicsLinearFEOperatorParameters>, bool );

protected:
    std::vector<std::vector<size_t>> d_dofIndices; /**< DOF indices */

    std::vector<std::vector<double>> d_elementStiffnessMatrix; /**< Element stiffness matrix. */

    std::shared_ptr<MechanicsLinearElement> d_mechLinElem; /**< Element operation. */

    std::shared_ptr<MechanicsLinearUpdatedLagrangianElement>
        d_mechLinULElem; /**< Linear Updated Lagrangian Element operation. */

    std::shared_ptr<MechanicsMaterialModel> d_materialModel; /**< Material model. */

    AMP::LinearAlgebra::Vector::shared_ptr d_refXYZ; /**< Reference x, y and z coordinates. */

    AMP::LinearAlgebra::Vector::shared_ptr d_dispVec;

    bool d_useUpdatedLagrangian;
};
} // namespace AMP::Operator

#endif
