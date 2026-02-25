#include "AMP/operators/boundary/ColumnBoundaryOperator.h"
#include "AMP/operators/OperatorFactory.h"


namespace AMP::Operator {


ColumnBoundaryOperator::ColumnBoundaryOperator( std::shared_ptr<const OperatorParameters> params )
    : BoundaryOperator( params )
{
    auto columnOpParams =
        std::dynamic_pointer_cast<const ColumnBoundaryOperatorParameters>( params );
    if ( columnOpParams ) {
        for ( auto p : columnOpParams->d_OperatorParameters ) {
            std::shared_ptr<Operator> op = AMP::Operator::OperatorFactory::create( p );
            auto boundaryOp              = std::dynamic_pointer_cast<BoundaryOperator>( op );
            AMP_ASSERT( boundaryOp );
            d_Operators.push_back( boundaryOp );
        }
    }
}


void ColumnBoundaryOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                    AMP::LinearAlgebra::Vector::shared_ptr r )
{
    for ( auto &elem : d_Operators ) {
        elem->apply( u, r );
    }
}

std::shared_ptr<OperatorParameters>
ColumnBoundaryOperator::getParameters( const std::string &type,
                                       AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                       std::shared_ptr<OperatorParameters> params )
{
    auto db = std::make_shared<Database>();
    Operator::setMemoryAndBackendParameters( db );
    auto opParameters  = std::make_shared<ColumnBoundaryOperatorParameters>( db );
    opParameters->d_db = std::make_shared<AMP::Database>( "ColumnBoundaryOperator" );
    opParameters->d_db->putScalar( "name", "ColumnBoundaryOperator" );
    opParameters->d_OperatorParameters.resize( d_Operators.size() );
    for ( unsigned int i = 0; i < d_Operators.size(); i++ )
        opParameters->d_OperatorParameters[i] = d_Operators[i]->getParameters( type, u, params );
    return opParameters;
}

void ColumnBoundaryOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );
    auto columnParameters =
        std::dynamic_pointer_cast<const ColumnBoundaryOperatorParameters>( params );

    AMP_INSIST( columnParameters, "ColumnBoundaryOperator::reset parameter object is NULL" );

    AMP_INSIST( columnParameters->d_OperatorParameters.size() == d_Operators.size(),
                "std::vector sizes do not match!" );

    for ( unsigned int i = 0; i < d_Operators.size(); i++ )
        d_Operators[i]->reset( columnParameters->d_OperatorParameters[i] );
}

void ColumnBoundaryOperator::append( std::shared_ptr<BoundaryOperator> op )
{
    AMP_INSIST(
        op, "AMP::Operator::ColumnBoundaryOperator::appendRow input argument is a NULL operator" );
    d_Operators.push_back( op );
}

void ColumnBoundaryOperator::addRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs )
{
    for ( auto &elem : d_Operators )
        elem->addRHScorrection( rhs );
}

void ColumnBoundaryOperator::setRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs )
{
    for ( auto &elem : d_Operators )
        elem->setRHScorrection( rhs );
}

void ColumnBoundaryOperator::modifyInitialSolutionVector(
    AMP::LinearAlgebra::Vector::shared_ptr sol )
{
    for ( auto &elem : d_Operators )
        elem->modifyInitialSolutionVector( sol );
}

} // namespace AMP::Operator
