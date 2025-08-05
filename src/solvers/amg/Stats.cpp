#include "AMP/solvers/amg/Stats.hpp"

namespace AMP::Solver::AMG {

template void
print_summary( std::string, const std::vector<KCycleLevel> &, const SolverStrategy & );
}
