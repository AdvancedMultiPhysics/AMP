#include "AMP/solvers/amg/Stats.hpp"

namespace AMP::Solver::AMG {

template void
print_summary( const std::string &, const std::vector<KCycleLevel> &, const SolverStrategy & );

template HierarchyStats
collect_statistics( const std::string &, const std::vector<KCycleLevel> &, const SolverStrategy & );

} // namespace AMP::Solver::AMG
