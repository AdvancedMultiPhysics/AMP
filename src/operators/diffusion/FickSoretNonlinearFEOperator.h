#ifndef included_AMP_FickSoretNonlinearFEOperator
#define included_AMP_FickSoretNonlinearFEOperator

/* AMP files */
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/diffusion/FickSoretNonlinearFEOperatorParameters.h"
#include <memory>

namespace AMP::Operator {

/**
  Class to add the output of the Fick and Soret operators.
  */
class FickSoretNonlinearFEOperator : public Operator
{
public:
    explicit FickSoretNonlinearFEOperator( std::shared_ptr<const OperatorParameters> params );

    virtual ~FickSoretNonlinearFEOperator() {}

    std::string type() const override { return "FickSoretNonlinearFEOperator"; }

    void reset( std::shared_ptr<const OperatorParameters> params ) override;

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    std::shared_ptr<OperatorParameters>
    getParameters( const std::string &type,
                   AMP::LinearAlgebra::Vector::const_shared_ptr u,
                   std::shared_ptr<OperatorParameters> params = nullptr ) override
    {
        return d_FickOperator->getParameters( type, u, params );
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_OutputVariable;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_FickOperator->getInputVariable();
    }

    //! checks input to apply operator for satisfaction of range conditions
    bool isValidVector( AMP::LinearAlgebra::Vector::const_shared_ptr u ) override
    {
        bool result;
        result = d_FickOperator->isValidVector( u ) and d_SoretOperator->isValidVector( u );
        return result;
    }

    inline auto getFickOperator() { return d_FickOperator; }
    inline auto getSoretOperator() { return d_SoretOperator; }

protected:
private:
    std::shared_ptr<DiffusionNonlinearFEOperator> d_FickOperator;
    std::shared_ptr<DiffusionNonlinearFEOperator> d_SoretOperator;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_OutputVariable;
    bool d_AddSoretTerm;
};
} // namespace AMP::Operator

#endif
