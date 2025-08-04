#include "AMP/operators/CustomConstraintsEliminationOperator.h"

namespace AMP::Operator {

CustomConstraintsEliminationOperator::CustomConstraintsEliminationOperator(
    std::shared_ptr<const OperatorParameters> params )
    : ConstraintsEliminationOperator( params )
{
}

void CustomConstraintsEliminationOperator::addSlaveToMaster(
    AMP::LinearAlgebra::Vector::shared_ptr u )
{
    if ( !d_MasterIndices.empty() ) {
        std::vector<double> slaveValues( d_SlaveIndices.size() );
        u->getValuesByGlobalID(
            d_SlaveIndices.size(), &( d_SlaveIndices[0] ), &( slaveValues[0] ) );
        std::vector<double> addToMasterValues;
        for ( size_t i = 0; i < d_SlaveIndices.size(); ++i ) {
            if ( !d_MasterIndices[i].empty() ) {
                addToMasterValues.resize( d_MasterIndices[i].size() );
                for ( size_t j = 0; j < d_MasterIndices[i].size(); ++j ) {
                    addToMasterValues[j] = d_MasterCoefficients[i][j] * slaveValues[i];
                } // end for j
                u->addValuesByGlobalID( d_MasterIndices[i].size(),
                                        &( d_MasterIndices[i][0] ),
                                        &( addToMasterValues[0] ) );
            } // end if
        }     // end for i
    }         // end if
}

void CustomConstraintsEliminationOperator::copyMasterToSlave(
    AMP::LinearAlgebra::Vector::shared_ptr u )
{
    if ( !d_SlaveIndices.empty() ) {
        std::vector<double> slaveValues( d_SlaveIndices.size(), 0.0 );
        if ( !d_MasterIndices.empty() ) {
            std::vector<double> masterValues;
            for ( size_t i = 0; i < d_SlaveIndices.size(); ++i ) {
                if ( !d_MasterIndices[i].empty() ) {
                    masterValues.resize( d_MasterIndices[i].size() );
                    u->getValuesByGlobalID( d_MasterIndices[i].size(),
                                            &( d_MasterIndices[i][0] ),
                                            &( masterValues[0] ) );
                    for ( size_t j = 0; j < d_MasterIndices[i].size(); ++j ) {
                        slaveValues[i] += d_MasterCoefficients[i][j] * masterValues[j];
                    } // end for j
                }     // end if
            }         // end for i
        }             // end if
        u->setValuesByGlobalID(
            d_SlaveIndices.size(), &( d_SlaveIndices[0] ), &( slaveValues[0] ) );
    } // end if
}

void CustomConstraintsEliminationOperator::initialize(
    std::vector<size_t> const &slaveIndices,
    std::vector<double> const &slaveShift,
    std::vector<std::vector<size_t>> const &masterIndices,
    std::vector<std::vector<double>> const &masterCoefficients )
{
    AMP_ASSERT( slaveIndices.size() == slaveShift.size() );
    AMP_ASSERT( masterIndices.size() == masterCoefficients.size() );
    if ( !masterIndices.empty() ) {
        AMP_ASSERT( slaveIndices.size() == masterIndices.size() );
        for ( size_t i = 0; i < slaveIndices.size(); ++i ) {
            AMP_ASSERT( masterIndices[i].size() == masterCoefficients[i].size() );
        } // end for i
    }     // end if
    d_SlaveIndices       = slaveIndices;
    d_SlaveShift         = slaveShift;
    d_MasterIndices      = masterIndices;
    d_MasterCoefficients = masterCoefficients;
}
} // namespace AMP::Operator
