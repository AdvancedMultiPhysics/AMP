
#ifndef included_AMP_VolumeIntegralOperator
#define included_AMP_VolumeIntegralOperator

#include "AMP/matrices/Matrix.h"
#include "AMP/operators/libmesh/NonlinearFEOperator.h"
#include "AMP/operators/libmesh/SourceNonlinearElement.h"
#include "AMP/operators/libmesh/VolumeIntegralOperatorParameters.h"
#include "AMP/vectors/MultiVariable.h"

#include <vector>

namespace AMP::Operator {


/**
  This class is used to integrate any arbitrary functional over the entire domain. The
  evaluation of the functional at quadrature points and volume integration are its key
  basic operations.
  */
class VolumeIntegralOperator : public NonlinearFEOperator
{
public:
    /**
      Constructor. This reads the values for the following keys from the database object contained
      in the parameter object, params:
      1) Primary(Active)InputVariables - List of active input variables names. The supported
      variable types are:
      NodalScalar, Nodalvector, IntegrationPointScalar, IntegrationPointVector.
      2) AuxillaryInputVariables - List of auxillary input variables names. These are frozen
      variables and are temporarily not used in any formulation.
      3) OutputVariable - Name of the output variable
      */
    explicit VolumeIntegralOperator( std::shared_ptr<const OperatorParameters> params );

    /**
      Destructor.
      */
    virtual ~VolumeIntegralOperator() {}

    //! Return the name of the operator
    std::string type() const override { return "VolumeIntegralOperator"; }

    /**
      This is used to update the operator between successive solves with the operator.
      */
    void reset( std::shared_ptr<const OperatorParameters> ) override;

    void setAuxVariable( std::shared_ptr<AMP::LinearAlgebra::MultiVariable> var )
    {
        d_auxVariables = var;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_inpVariables;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_outVariable;
    }

    std::shared_ptr<SourcePhysicsModel> getSourcePhysicsModel() { return d_sourcePhysicsModel; }

    std::shared_ptr<SourceNonlinearElement> getSourceElement() { return d_srcNonlinElem; }

    /*
       std::shared_ptr<AMP::LinearAlgebra::Matrix> getLinearizedVolumeIntegralOperator(
       std::shared_ptr<OperatorParameters> params);
     */

protected:
    explicit VolumeIntegralOperator( std::shared_ptr<const VolumeIntegralOperatorParameters> params,
                                     bool );

    /**
      This is used to compute the information required to reset the corresponding Linear (Jacobian)
      operator
      */
    std::shared_ptr<OperatorParameters>
    getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u ) override;


    /**
      This function is called at the beginning of the FE assembly
      */
    void preAssembly( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                      AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    /**
      This function is called at the end of the FE assembly
      */
    void postAssembly() override;

    /**
      This function is called at the beginning of the element computation
      */
    void preElementOperation( const AMP::Mesh::MeshElement & ) override;

    /**
      This function is called at the end of the element computation
      */
    void postElementOperation() override;

    void getNodeDofIndicesForCurrentElement();

    void init( std::shared_ptr<const VolumeIntegralOperatorParameters> params );

    std::string d_isInputType;

    std::vector<double> d_elementOutputVector; /**< Output vector for the Element Operation. */

    std::shared_ptr<SourceNonlinearElement> d_srcNonlinElem; /**< Element Operation. */

    std::shared_ptr<SourcePhysicsModel> d_sourcePhysicsModel; /**< Source Physics Model. */

    std::vector<AMP::LinearAlgebra::Vector::const_shared_ptr>
        d_inVec; /**< Input vector for active variables. */

    std::vector<AMP::LinearAlgebra::Vector::const_shared_ptr>
        d_auxVec; /**< Input vector for auxillary variables. */

    AMP::LinearAlgebra::Vector::shared_ptr d_multiAuxPtr;

    AMP::LinearAlgebra::Vector::shared_ptr d_outVec; /**< Output vector. */

    std::vector<size_t> d_dofIndices; /**<Node dof Ids */

    std::shared_ptr<AMP::Discretization::DOFManager> d_elementDofMap;
    std::shared_ptr<AMP::Discretization::DOFManager> d_nodeDofMap;

    std::vector<std::unique_ptr<AMP::Mesh::MeshElement>> d_currNodes;

private:
    std::shared_ptr<AMP::LinearAlgebra::MultiVariable> d_inpVariables; // Input Active variable
    std::shared_ptr<AMP::LinearAlgebra::MultiVariable> d_auxVariables; // Input Auxillary variable
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable;       // Output variable
};


} // namespace AMP::Operator

#endif
