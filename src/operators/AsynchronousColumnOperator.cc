#include "AsynchronousColumnOperator.h"
#include "AsynchronousOperator.h"

namespace AMP {
namespace Operator {


AsynchronousColumnOperator::AsynchronousColumnOperator ( const boost::shared_ptr <OperatorParameters> &params )
    : ColumnOperator ( params )
{
}


void AsynchronousColumnOperator::apply(AMP::LinearAlgebra::Vector::const_shared_ptr f, AMP::LinearAlgebra::Vector::const_shared_ptr u,
      AMP::LinearAlgebra::Vector::shared_ptr r, const double a , const double b )
{
    applyStart( f, u, r, a, b );
    applyFinish( f, u, r, a, b );
}


// Initiate all applies in the column
void AsynchronousColumnOperator::applyStart(AMP::LinearAlgebra::Vector::const_shared_ptr f, AMP::LinearAlgebra::Vector::const_shared_ptr u,
      AMP::LinearAlgebra::Vector::shared_ptr r, const double a , const double b )
{
    for ( size_t i = 0 ; i != getNumberOfOperators() ; i++ )
        boost::dynamic_pointer_cast<AsynchronousOperator> ( getOperator ( i ) )->applyStart ( f , u , r , a , b );
}



// Finish all applies in the column
void AsynchronousColumnOperator::applyFinish(AMP::LinearAlgebra::Vector::const_shared_ptr f, AMP::LinearAlgebra::Vector::const_shared_ptr u,
      AMP::LinearAlgebra::Vector::shared_ptr r, const double a , const double b )
{
    for ( size_t i = 0 ; i != getNumberOfOperators() ; i++ )
        boost::dynamic_pointer_cast<AsynchronousOperator> ( getOperator ( i ) )->applyFinish ( f , u , r , a , b );
}



void AsynchronousColumnOperator::append(boost::shared_ptr< Operator > op)
{
    if ( boost::dynamic_pointer_cast<AsynchronousOperator> ( op ) )
    {
      ColumnOperator::append ( op );
    }
    else if ( boost::dynamic_pointer_cast<AsynchronousColumnOperator> ( op ) )
    {
      boost::shared_ptr<AsynchronousColumnOperator> aco = boost::dynamic_pointer_cast<AsynchronousColumnOperator> ( op );
      for ( size_t i = 0 ; i != aco->getNumberOfOperators() ; i++ )
      {
        append ( aco->getOperator ( i ) );
      }
    }
}


}
}

