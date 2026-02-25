#ifndef included_AMP_NodeToGaussPointOperator
#define included_AMP_NodeToGaussPointOperator

#include "AMP/discretization/createLibmeshElements.h"
#include "AMP/operators/Operator.h"


namespace AMP::Operator {

class NodeToGaussPointOperator : public Operator
{
public:
    explicit NodeToGaussPointOperator( std::shared_ptr<const OperatorParameters> params );

    virtual ~NodeToGaussPointOperator() {}

    std::string type() const override { return "NodeToGaussPointOperator"; }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        return d_GaussPtVariable;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        return d_NodalVariable;
    }

protected:
    bool d_UseSurfaceElements;
    int d_dim;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_NodalVariable;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_GaussPtVariable;
    AMP::Mesh::MeshIterator d_iterator;
    Discretization::createLibmeshElements d_libmeshElements;
    std::vector<std::vector<AMP::Mesh::MeshElementID>> d_nodes;
    std::vector<unsigned short int> d_N_quad;
    std::vector<std::vector<double>> d_phi;
};
} // namespace AMP::Operator

#endif
