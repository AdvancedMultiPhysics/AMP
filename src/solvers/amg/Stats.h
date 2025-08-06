#ifndef included_AMP_AMG_STATS
#define included_AMP_AMG_STATS

#include <string>
#include <vector>

#include "AMP/solvers/amg/Cycle.h"

namespace AMP::Solver::AMG {

/**
   Print summary of an AMG Hierarchy.

   \param[in] amg_name name of solver containing hierarchy
   \param[in] ml multilevel hierarchy
   \param[in] cg_solver Coarse grid solver for hierarchy
 */
template<class T, std::enable_if_t<is_level_v<T>, bool> = true>
void print_summary( std::string amg_name,
                    const std::vector<T> &ml,
                    const SolverStrategy &cg_solver );

} // namespace AMP::Solver::AMG

#endif
