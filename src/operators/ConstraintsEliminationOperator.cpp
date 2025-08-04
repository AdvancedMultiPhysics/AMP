
#include "AMP/operators/ConstraintsEliminationOperator.h"

#include <algorithm>
#include <cmath>

namespace AMP::Operator {

std::shared_ptr<AMP::LinearAlgebra::Variable>
ConstraintsEliminationOperator::getInputVariable() const
{
    return d_InputVariable;
}

std::shared_ptr<AMP::LinearAlgebra::Variable>
ConstraintsEliminationOperator::getOutputVariable() const
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
        u->setValuesByGlobalID(
            d_SlaveIndices.size(), &( d_SlaveIndices[0] ), &( zeroSlaveValues[0] ) );
    } // end if
    u->makeConsistent();
}

void ConstraintsEliminationOperator::addShiftToSlave( AMP::LinearAlgebra::Vector::shared_ptr u )
{
    AMP_ASSERT( d_SlaveShift.size() == d_SlaveIndices.size() );
    if ( !d_SlaveIndices.empty() ) {
        u->addValuesByGlobalID(
            d_SlaveIndices.size(), &( d_SlaveIndices[0] ), &( d_SlaveShift[0] ) );
    } // end if
}

} // namespace AMP::Operator
