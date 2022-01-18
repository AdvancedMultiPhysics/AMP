#ifndef included_AMP_OperatorFactory
#define included_AMP_OperatorFactory

/* AMP files */
#include "AMP/mesh/MeshManager.h"
#include "AMP/utils/Database.h"

namespace AMP::Operator {

template<typename OPERATOR>
class OperatorFactory
{
public:
    typedef OPERATOR Operator_t;
    typedef typename Operator_t::Parameters OperatorParameters;
    typedef typename Operator_t::Jacobian Jacobian_t;
    typedef typename Jacobian_t::Parameters JacobianParameters;

    static Operator::shared_ptr getOperator( std::shared_ptr<AMP::Database> input_db,
                                             AMP::Mesh::MeshManager::Adapter::shared_ptr mesh =
                                                 AMP::Mesh::MeshManager::Adapter::shared_ptr( 0 ) );

    static Operator::shared_ptr getJacobian( Operator::shared_ptr oper,
                                             const AMP::LinearAlgebra::Vector::shared_ptr &vec,
                                             AMP::Mesh::MeshManager::Adapter::shared_ptr mesh =
                                                 AMP::Mesh::MeshManager::Adapter::shared_ptr( 0 ) );
};


template<typename OPERATOR>
Operator::shared_ptr
OperatorFactory<OPERATOR>::getOperator( std::shared_ptr<AMP::Database> input_db,
                                        AMP::Mesh::MeshManager::Adapter::shared_ptr mesh )
{
    std::shared_ptr<OperatorParameters> params(
        new OperatorParameters( input_db->getDatabase( Operator_t::DBName() ) ) );
    params->d_MeshAdapter = mesh;
    Operator::shared_ptr retVal( new Operator_t( params ) );
    retVal->setInputVariable( std::shared_ptr<AMP::LinearAlgebra::Variable>(
        new typename Operator_t::InputVariable( "factory input" ) ) );
    retVal->setOutputVariable( std::shared_ptr<AMP::LinearAlgebra::Variable>(
        new typename Operator_t::OutputVariable( "factory output" ) ) );
    return retVal;
}

template<typename OPERATOR>
Operator::shared_ptr
OperatorFactory<OPERATOR>::getJacobian( Operator::shared_ptr oper,
                                        const AMP::LinearAlgebra::Vector::shared_ptr &vec,
                                        AMP::Mesh::MeshManager::Adapter::shared_ptr mesh )
{
    std::shared_ptr<JacobianParameters> params =
        std::dynamic_pointer_cast<JacobianParameters>( oper->getJacobianParameters( vec ) );
    params->d_MeshAdapter = mesh;
    Operator::shared_ptr retVal( new Jacobian_t( params ) );
    retVal->setInputVariable( std::shared_ptr<AMP::LinearAlgebra::Variable>(
        new typename Jacobian_t::InputVariable( "factory jacobian input" ) ) );
    retVal->setOutputVariable( std::shared_ptr<AMP::LinearAlgebra::Variable>(
        new typename Jacobian_t::OutputVariable( "factory jacobian output" ) ) );
    return retVal;
}
} // namespace AMP::Operator

#endif
