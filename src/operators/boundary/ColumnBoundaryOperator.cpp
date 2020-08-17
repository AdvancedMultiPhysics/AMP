
#include "ColumnBoundaryOperator.h"
#include "AMP/utils/Utilities.h"

namespace AMP {
namespace Operator {

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

    std::shared_ptr<AMP::Database> db;
    std::shared_ptr<ColumnBoundaryOperatorParameters> opParameters(
        new ColumnBoundaryOperatorParameters( db ) );

    ( opParameters->d_OperatorParameters ).resize( d_Operators.size() );

    for ( unsigned int i = 0; i < d_Operators.size(); i++ ) {
        ( opParameters->d_OperatorParameters )[i] =
            ( d_Operators[i]->getParameters( type, u, params ) );
    }

    return opParameters;
}

void ColumnBoundaryOperator::reset( const std::shared_ptr<OperatorParameters> &params )
{
    std::shared_ptr<ColumnBoundaryOperatorParameters> columnParameters =
        std::dynamic_pointer_cast<ColumnBoundaryOperatorParameters>( params );

    AMP_INSIST( ( columnParameters.get() != nullptr ),
                "ColumnBoundaryOperator::reset parameter object is NULL" );

    AMP_INSIST( ( ( ( columnParameters->d_OperatorParameters ).size() ) == ( d_Operators.size() ) ),
                " std::vector sizes do not match! " );

    for ( unsigned int i = 0; i < d_Operators.size(); i++ ) {
        d_Operators[i]->reset( ( columnParameters->d_OperatorParameters )[i] );
    }
}

void ColumnBoundaryOperator::append( std::shared_ptr<BoundaryOperator> op )
{
    AMP_INSIST(
        ( op.get() != nullptr ),
        "AMP::Operator::ColumnBoundaryOperator::appendRow input argument is a NULL operator" );

    d_Operators.push_back( op );
}

void ColumnBoundaryOperator::addRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs )
{
    for ( auto &elem : d_Operators ) {
        elem->addRHScorrection( rhs );
    } // end for i
}

void ColumnBoundaryOperator::setRHScorrection( AMP::LinearAlgebra::Vector::shared_ptr rhs )
{
    for ( auto &elem : d_Operators ) {
        elem->setRHScorrection( rhs );
    } // end for i
}

void ColumnBoundaryOperator::modifyInitialSolutionVector(
    AMP::LinearAlgebra::Vector::shared_ptr sol )
{
    for ( auto &elem : d_Operators ) {
        elem->modifyInitialSolutionVector( sol );
    } // end for i
}
} // namespace Operator
} // namespace AMP