
#ifndef included_AMP_ContactResidualCorrection
#define included_AMP_ContactResidualCorrection

#include "AMP/mesh/MeshID.h"
#include "AMP/operators/Operator.h"
#include "AMP/vectors/MultiVariable.h"

namespace AMP::Operator {

typedef OperatorParameters ContactResidualCorrectionParameters;

class ContactResidualCorrection : public Operator
{
public:
    explicit ContactResidualCorrection(
        std::shared_ptr<const ContactResidualCorrectionParameters> params )
        : Operator( params )
    {
    }

    virtual ~ContactResidualCorrection() {}

    std::string type() const override { return "ContactResidualCorrection"; }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                AMP::LinearAlgebra::Vector::shared_ptr f ) override;

    void setMasterVariable( const std::shared_ptr<AMP::LinearAlgebra::Variable> &var )
    {
        d_masterVariable = var;
    }

    void setSlaveVariable( const std::shared_ptr<AMP::LinearAlgebra::Variable> &var )
    {
        d_slaveVariable = var;
    }

    void setMasterMesh( const std::shared_ptr<AMP::Mesh::Mesh> &mesh ) { d_Mesh = mesh; }

    void setSlaveMesh( const std::shared_ptr<AMP::Mesh::Mesh> &mesh ) { d_slaveMesh = mesh; }

    void setMasterNodes( const std::vector<AMP::Mesh::MeshElementID> &vec ) { d_masterNodes = vec; }

    void setSlaveNodes( const std::vector<AMP::Mesh::MeshElementID> &vec ) { d_slaveNodes = vec; }

    void setDofs( const std::vector<std::vector<unsigned int>> &vec ) { d_dofs = vec; }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getOutputVariable() const override
    {
        std::shared_ptr<AMP::LinearAlgebra::MultiVariable> retVariable(
            new AMP::LinearAlgebra::MultiVariable( "ContactVariable" ) );
        retVariable->add( d_masterVariable );
        retVariable->add( d_slaveVariable );
        return retVariable;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getInputVariable() const override
    {
        std::shared_ptr<AMP::LinearAlgebra::MultiVariable> retVariable(
            new AMP::LinearAlgebra::MultiVariable( "ContactVariable" ) );
        retVariable->add( d_masterVariable );
        retVariable->add( d_slaveVariable );
        return retVariable;
    }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getMasterVariable() { return d_masterVariable; }

    std::shared_ptr<AMP::LinearAlgebra::Variable> getSlaveVariable() { return d_slaveVariable; }

    const auto &getMasterNodes() { return d_masterNodes; }

    const auto &getSlaveNodes() { return d_slaveNodes; }

    const auto &getDofs() { return d_dofs; }

private:
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_masterVariable;
    std::shared_ptr<AMP::LinearAlgebra::Variable> d_slaveVariable;

    std::vector<AMP::Mesh::MeshElementID> d_masterNodes;
    std::vector<AMP::Mesh::MeshElementID> d_slaveNodes;

    std::vector<std::vector<unsigned int>> d_dofs;

    std::shared_ptr<AMP::Mesh::Mesh> d_slaveMesh;
};
} // namespace AMP::Operator

#endif
