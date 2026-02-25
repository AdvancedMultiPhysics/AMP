#include "AMP/operators/IdentityOperator.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"


namespace AMP::Operator {


IdentityOperator::IdentityOperator() : LinearOperator() {}

IdentityOperator::IdentityOperator( std::shared_ptr<const OperatorParameters> params )
    : LinearOperator( params )
{
    reset( params );
}

void IdentityOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    if ( params->d_db ) {
        if ( params->d_db->keyExists( "InputVariable" ) ) {
            std::string inpVar = params->d_db->getString( "InputVariable" );
            d_inputVariable.reset( new AMP::LinearAlgebra::Variable( inpVar ) );
        }
        if ( params->d_db->keyExists( "OutputVariable" ) ) {
            std::string outVar = params->d_db->getString( "OutputVariable" );
            d_outputVariable.reset( new AMP::LinearAlgebra::Variable( outVar ) );
        }
        d_localSize = params->d_db->getWithDefault<size_t>( "localSize", 10 );
    }
}

void IdentityOperator::setMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> )
{
    AMP_ERROR( "setMatrix is invalid for the Identity operator" );
}


void IdentityOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                              AMP::LinearAlgebra::Vector::shared_ptr r )
{
    AMP_INSIST( u, "NULL Solution Vector" );
    AMP_INSIST( r, "NULL Residual Vector" );

    auto uInternal = subsetInputVector( u );
    auto rInternal = subsetOutputVector( r );

    AMP_INSIST( uInternal, "uInternal is NULL" );
    AMP_INSIST( rInternal, "rInternal is NULL" );

    rInternal->copyVector( uInternal );

    rInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}

std::shared_ptr<AMP::LinearAlgebra::Vector> IdentityOperator::createInputVector() const
{
    return AMP::LinearAlgebra::createSimpleVector<double>(
        d_localSize, d_inputVariable, AMP_COMM_WORLD );
}

std::shared_ptr<AMP::LinearAlgebra::Vector> IdentityOperator::createOutputVector() const
{
    return AMP::LinearAlgebra::createSimpleVector<double>(
        d_localSize, d_outputVariable, AMP_COMM_WORLD );
}

std::shared_ptr<OperatorParameters>
IdentityOperator::getParameters( const std::string &type,
                                 std::shared_ptr<const AMP::LinearAlgebra::Vector>,
                                 std::shared_ptr<OperatorParameters> )
{
    std::shared_ptr<OperatorParameters> params;
    if ( type == "Jacobian" ) {
        std::shared_ptr<Database> db = AMP::Database::create( "name", "IdentityOperator" );
        Operator::setMemoryAndBackendParameters( db );

        params = std::make_shared<OperatorParameters>( db );
        if ( d_inputVariable )
            db->putScalar<std::string>( "InputVariable", d_inputVariable->getName() );
        if ( d_outputVariable )
            db->putScalar<std::string>( "OutputVariable", d_outputVariable->getName() );
    } else {
        // Derived class should implement this
        AMP_ERROR( "Unknown OperatorParameters type specified" );
    }
    return params;
}

} // namespace AMP::Operator
