
#ifndef included_AMP_NeumannVectorCorrection
#define included_AMP_NeumannVectorCorrection

#include "AMP/operators/boundary/BoundaryOperator.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrectionParameters.h"
#include "AMP/operators/boundary/libmesh/RobinPhysicsModel.h"

#include "AMP/discretization/createLibmeshElements.h"

// Libmesh files
#include "libmesh/quadrature.h"

#include <string>

namespace AMP {
namespace Operator {

/**
  A class to impose Neumann (Flux) Boundary Conditions for both Linear and Nonlinear operator.
  For both the Linear/Nonlinear operator to impose these conditions involves adding the corrections
  to the RHS vector at the appropriate locations. When you do not impose these Neumann condition for
  the weak formulation, a natural condition is assumed.
  This class is also a base class for the Robin Boundary Operator.
  */
class NeumannVectorCorrection : public BoundaryOperator
{
public:
    //! Constructor. This function reads all the parameters required for surface elements.
    explicit NeumannVectorCorrection(
        const std::shared_ptr<NeumannVectorCorrectionParameters> &params );

    /**
      Set the variable for the vector that will used with this operator.
      */
    void setVariable( const AMP::LinearAlgebra::Variable::shared_ptr &var ) { d_variable = var; }

    /**
      Destructor
      */
    virtual ~NeumannVectorCorrection() {}

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    /**
      This function computes the surface integral for either constant or varrying flux values
      across the boundary.
      */
    void computeRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhsCorrection );

    /**
      This function reads parameters related to boundary Ids
      */
    void reset( const std::shared_ptr<OperatorParameters> &params ) override;

    /**
      Adds a vector to the RHS vector.
      */
    void addRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs ) override;

    /**
     * get a pointer to the cached parameters that were used to create this
     * operator
     */
    std::shared_ptr<OperatorParameters> getOperatorParameters() { return d_params; }

    void setVariableFlux( const AMP::LinearAlgebra::Vector::shared_ptr &flux );

    void setFrozenVector( AMP::LinearAlgebra::Vector::shared_ptr f );

    std::shared_ptr<RobinPhysicsModel> getRobinPhysicsModel() { return d_robinPhysicsModel; }

    std::vector<short int> getBoundaryIds() const { return d_boundaryIds; }

    std::vector<std::vector<unsigned int>> getDofIds() const { return d_dofIds; }

    AMP::LinearAlgebra::Variable::shared_ptr getOutputVariable() override { return d_variable; }

    AMP::LinearAlgebra::Variable::shared_ptr getInputVariable() override { return d_variable; }

protected:
    /**
      This function returns a parameter object that can be used to reset the corresponding
      NeumannVectorCorrection operator.
      */
    std::shared_ptr<OperatorParameters>
    getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u ) override;

    Discretization::createLibmeshElements d_libmeshElements;

    std::vector<short int> d_boundaryIds;

    std::vector<std::vector<double>> d_neumannValues;

    std::vector<std::vector<unsigned int>> d_dofIds;

    AMP::LinearAlgebra::Vector::shared_ptr d_rhsCorrectionAdd;

    // This must be a simple variable not a dual or multivariable
    AMP::LinearAlgebra::Variable::shared_ptr d_variable;

    bool d_isConstantFlux;

    AMP::LinearAlgebra::Vector::shared_ptr d_Frozen;

    AMP::LinearAlgebra::Vector::shared_ptr d_variableFlux;

    std::vector<bool> d_IsCoupledBoundary;
    bool d_isFluxGaussPtVector;

    int d_numBndIds;

    std::vector<short int> d_numDofIds;

    std::shared_ptr<NeumannVectorCorrectionParameters> d_params;

    std::shared_ptr<RobinPhysicsModel> d_robinPhysicsModel;

    std::shared_ptr<const libMesh::FEType> d_type;
    libMeshEnums::Order d_qruleOrder;
    libMeshEnums::QuadratureType d_qruleType;

    std::vector<AMP::Mesh::MeshElement> d_currNodes;

private:
};
} // namespace Operator
} // namespace AMP

#endif
