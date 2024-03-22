
#include "AMP/operators/ConstraintsEliminationOperator.h"

#include <algorithm>
#include <cmath>

namespace AMP::Operator {

void ConstraintsEliminationOperator::reset( std::shared_ptr<const OperatorParameters> params )
{

    AMP_INSIST( ( params != nullptr ), "NULL parameter" );
    AMP_INSIST( ( ( params->d_db ) != nullptr ), "NULL database" );
}

std::shared_ptr<AMP::LinearAlgebra::Variable> ConstraintsEliminationOperator::getInputVariable()
{
    return d_InputVariable;
}

std::shared_ptr<AMP::LinearAlgebra::Variable> ConstraintsEliminationOperator::getOutputVariable()
{
    return d_OutputVariable;
}

void ConstraintsEliminationOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr,
                                            AMP::LinearAlgebra::Vector::shared_ptr f )
{
    addSlaveToMaster( f );
    setSlaveToZero( f );
}

void ConstraintsEliminationOperator::setSlaveToZero( AMP::LinearAlgebra::Vector::shared_ptr u )
{
    if ( !d_SlaveIndices.empty() ) {
        std::vector<double> zeroSlaveValues( d_SlaveIndices.size(), 0.0 );
        u->setLocalValuesByGlobalID(
            d_SlaveIndices.size(), &( d_SlaveIndices[0] ), &( zeroSlaveValues[0] ) );
    } // end if
    u->makeConsistent();
}

void ConstraintsEliminationOperator::addShiftToSlave( AMP::LinearAlgebra::Vector::shared_ptr u )
{
    AMP_ASSERT( d_SlaveShift.size() == d_SlaveIndices.size() );
    if ( !d_SlaveIndices.empty() ) {
        u->addLocalValuesByGlobalID(
            d_SlaveIndices.size(), &( d_SlaveIndices[0] ), &( d_SlaveShift[0] ) );
    } // end if
}

} // namespace AMP::Operator
