#include "AMP/solvers/amg/UASolver.h"

namespace AMP::Solver::AMG {

std::shared_ptr<AMP::Database> UASolver::createInternalOperatorDb() const
{
    auto op_db = std::make_shared<Database>( type() + "::Internal" );
    if ( d_mem_loc == Utilities::MemoryType::host ) {
        op_db->putScalar<std::string>( "memory_location", "host" );
    } else {
        AMP_ERROR( type() + ": Only host memory is supported currently" );
    }
    return op_db;
}

PairwiseCoarsenSettings UASolver::getCoarsenSettings( size_t lvl ) const
{
    auto settings = d_pair_coarsen_settings;
    if ( lvl > 0 ) {
        settings.checkdd = false;
    }
    return settings;
}

} // namespace AMP::Solver::AMG
