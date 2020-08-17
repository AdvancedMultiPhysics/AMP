#include "AsynchronousColumnOperator.h"
#include "AsynchronousOperator.h"

namespace AMP {
namespace Operator {


AsynchronousColumnOperator::AsynchronousColumnOperator(
    const std::shared_ptr<OperatorParameters> &params )
    : ColumnOperator( params )
{
}


void AsynchronousColumnOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                        AMP::LinearAlgebra::Vector::shared_ptr f )
{
    applyStart( u, f );
    applyFinish( u, f );
}


// Initiate all applies in the column
void AsynchronousColumnOperator::applyStart( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                             AMP::LinearAlgebra::Vector::shared_ptr f )
{
    for ( size_t i = 0; i != getNumberOfOperators(); i++ )
        std::dynamic_pointer_cast<AsynchronousOperator>( getOperator( i ) )->applyStart( u, f );
}


// Finish all applies in the column
void AsynchronousColumnOperator::applyFinish( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                              AMP::LinearAlgebra::Vector::shared_ptr f )
{
    for ( size_t i = 0; i != getNumberOfOperators(); i++ )
        std::dynamic_pointer_cast<AsynchronousOperator>( getOperator( i ) )->applyFinish( u, f );
}


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
} // namespace Operator
} // namespace AMP