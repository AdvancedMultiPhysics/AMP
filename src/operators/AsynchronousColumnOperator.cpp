#include "AMP/operators/AsynchronousColumnOperator.h"
#include "AMP/operators/AsynchronousOperator.h"

namespace AMP::Operator {


/********************************************************
 * Constructors                                          *
 ********************************************************/
AsynchronousColumnOperator::AsynchronousColumnOperator() : ColumnOperator() {}

AsynchronousColumnOperator::AsynchronousColumnOperator(
    std::shared_ptr<const OperatorParameters> params )
    : ColumnOperator( params )
{
}


/********************************************************
 * apply                                                 *
 ********************************************************/
void AsynchronousColumnOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                        AMP::LinearAlgebra::Vector::shared_ptr f )
{
    applyStart( u, f );
    applyFinish( u, f );
}
void AsynchronousColumnOperator::applyStart( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                             AMP::LinearAlgebra::Vector::shared_ptr f )
{
    for ( size_t i = 0; i != getNumberOfOperators(); i++ )
        std::dynamic_pointer_cast<AsynchronousOperator>( getOperator( i ) )->applyStart( u, f );
}
void AsynchronousColumnOperator::applyFinish( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                              AMP::LinearAlgebra::Vector::shared_ptr f )
{
    for ( size_t i = 0; i != getNumberOfOperators(); i++ )
        std::dynamic_pointer_cast<AsynchronousOperator>( getOperator( i ) )->applyFinish( u, f );
}


/********************************************************
 * append                                                *
 ********************************************************/
void AsynchronousColumnOperator::append( std::shared_ptr<Operator> op )
{
    if ( std::dynamic_pointer_cast<AsynchronousOperator>( op ) ) {
        ColumnOperator::append( op );
    } else if ( std::dynamic_pointer_cast<AsynchronousColumnOperator>( op ) ) {
        std::shared_ptr<AsynchronousColumnOperator> aco =
            std::dynamic_pointer_cast<AsynchronousColumnOperator>( op );
        for ( size_t i = 0; i != aco->getNumberOfOperators(); i++ ) {
            append( aco->getOperator( i ) );
        }
    }
}
} // namespace AMP::Operator
