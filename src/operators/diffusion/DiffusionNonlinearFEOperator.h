#ifndef included_AMP_DiffusionNonlinearFEOperator
#define included_AMP_DiffusionNonlinearFEOperator

#include "AMP/operators/diffusion/DiffusionNonlinearElement.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperatorParameters.h"
#include "AMP/operators/libmesh/NonlinearFEOperator.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/Vector.h"

#include <vector>

namespace AMP::Operator {

class DiffusionNonlinearFEOperator : public NonlinearFEOperator
{
public:
    explicit DiffusionNonlinearFEOperator( std::shared_ptr<const OperatorParameters> params );

    virtual ~DiffusionNonlinearFEOperator() {}

    void reset( std::shared_ptr<const OperatorParameters> ) override;

    void setInputVariableName( const std::string &name, int varId = -1 );

    void setOutputVariableName( const std::string &name, int varId = -1 );

    std::shared_ptr<AMP::LinearAlgebra::Variable> createInputVariable( const std::string &name,
                                                                       int varId = -1 );

    std::shared_ptr<AMP::LinearAlgebra::Variable> createOutputVariable( const std::string &name,
                                                                        int varId = -1 );

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override;

    unsigned int numberOfDOFMaps() { return 1; }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getVariableForDOFMap( unsigned int id );

    const auto &getPrincipalVariable() { return d_PrincipalVariable; }

    std::vector<std::string> getNonPrincipalVariableIds();

    std::shared_ptr<DiffusionTransportModel> getTransportModel();

    std::map<std::string, AMP::LinearAlgebra::Vector::shared_ptr> getFrozen();

    /**
      This function is used to set frozen vectors in this operator. This is used when some of the
      variables are solved for in an uncoupled manner.
      @param [in] name      Variable Identifier
      @param [in] frozenVec Frozen vector
      */
    void setVector( const std::string &name, AMP::LinearAlgebra::Vector::shared_ptr frozenVec );

    /**
     * checks input to apply operator for satisfaction of range conditions
     */
    bool isValidVector( AMP::LinearAlgebra::Vector::const_shared_ptr u ) override;

protected:
    std::shared_ptr<OperatorParameters>
    getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u ) override;

    void preAssembly( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                      AMP::LinearAlgebra::Vector::shared_ptr r ) override;

    void postAssembly() override;

    void preElementOperation( const AMP::Mesh::MeshElement & ) override;

    void postElementOperation() override;

    void init( std::shared_ptr<const DiffusionNonlinearFEOperatorParameters> params );

    std::vector<double> d_elementOutputVector;

    std::shared_ptr<DiffusionNonlinearElement> d_diffNonlinElem;

    std::shared_ptr<DiffusionTransportModel> d_transportModel;

    AMP::Mesh::MeshElementVectorPtr d_currNodes;

    AMP::LinearAlgebra::Vector::shared_ptr d_outVec;

    std::shared_ptr<std::vector<double>> d_TransportGauss;
    AMP::LinearAlgebra::Vector::shared_ptr d_TransportNodal;

protected:
    struct InputVectorStruct {
        bool isFrozen = false;
        std::shared_ptr<const AMP::LinearAlgebra::Vector> vec;
        std::shared_ptr<const AMP::LinearAlgebra::Vector> frozen;
    };
    std::map<std::string, InputVectorStruct> d_active;

private:
    std::shared_ptr<AMP::LinearAlgebra::MultiVariable> d_inpVariables;

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable;

    std::string d_PrincipalVariable;

    void resetFrozen( std::shared_ptr<const DiffusionNonlinearFEOperatorParameters> params );

    explicit DiffusionNonlinearFEOperator(
        std::shared_ptr<const DiffusionNonlinearFEOperatorParameters> params, bool );
};
} // namespace AMP::Operator

#endif
