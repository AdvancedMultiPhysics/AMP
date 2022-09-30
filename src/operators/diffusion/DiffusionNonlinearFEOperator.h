
#ifndef included_AMP_DiffusionNonlinearFEOperator
#define included_AMP_DiffusionNonlinearFEOperator

#include "AMP/operators/diffusion/DiffusionConstants.h"
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
    explicit DiffusionNonlinearFEOperator(
        std::shared_ptr<const DiffusionNonlinearFEOperatorParameters> params );

    virtual ~DiffusionNonlinearFEOperator() {}

    void reset( std::shared_ptr<const OperatorParameters> ) override;

    void setInputVariableName( const std::string &name, int varId = -1 );

    void setOutputVariableName( const std::string &name, int varId = -1 );

    std::shared_ptr<AMP::LinearAlgebra::Variable> createInputVariable( const std::string &name,
                                                                       int varId = -1 );

    std::shared_ptr<AMP::LinearAlgebra::Variable> createOutputVariable( const std::string &name,
                                                                        int varId = -1 );

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() override;

    unsigned int numberOfDOFMaps();

    std::shared_ptr<AMP::LinearAlgebra::Variable> getVariableForDOFMap( unsigned int id );

    std::string getPrincipalVariable();

    std::vector<std::string> getNonPrincipalVariableIds();

    std::shared_ptr<DiffusionTransportModel> getTransportModel();

    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> getFrozen();

    /**
      This function is used to set frozen vectors in this operator. This is used when some of the
      variables are solved for in an uncoupled manner.
      @param [in] name      Variable Identifier
      @param [in] frozenVec Frozen vector
      @see DiffusionConstants.h
      */
    void setVector( const std::string &name, AMP::LinearAlgebra::Vector::shared_ptr frozenVec );

    /**
     * checks input to apply operator for satisfaction of range conditions
     */
    bool isValidInput( AMP::LinearAlgebra::Vector::const_shared_ptr u ) override;

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

    std::vector<AMP::Mesh::MeshElement> d_currNodes;

    AMP::LinearAlgebra::Vector::shared_ptr d_outVec;

    std::shared_ptr<std::vector<double>> d_TransportGauss;
    AMP::LinearAlgebra::Vector::shared_ptr d_TransportNodal;

protected:
    /*struct InputVectorStruct {
        bool active = false;
        bool frozen = false;
        AMP::LinearAlgebra::Vector::const_shared_ptr> vec;
    };
    std::map<std::string,InputVectorStruct> d_inVec;*/

    std::vector<bool> d_isActive;
    std::vector<bool> d_isFrozen;
    std::vector<AMP::LinearAlgebra::Vector::const_shared_ptr> d_inVec;

private:
    std::shared_ptr<AMP::LinearAlgebra::MultiVariable> d_inpVariables;

    std::shared_ptr<AMP::LinearAlgebra::Variable> d_outVariable;

    std::string d_PrincipalVariable;

    unsigned int d_numberActive;

    unsigned int d_numberFrozen;

    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> d_Frozen;

    void resetFrozen( std::shared_ptr<const DiffusionNonlinearFEOperatorParameters> params );
};
} // namespace AMP::Operator

#endif
