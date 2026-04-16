#ifndef included_AMP_AMG_STATS
#define included_AMP_AMG_STATS

#include <string>
#include <vector>

#include "AMP/solvers/amg/Cycle.h"

namespace AMP::Solver::AMG {

struct HierarchyStats {
    float operator_complexity;
    float grid_complexity;
    struct level_type {
        std::string solver_name;
        int comm_size;
        std::size_t nrows;
        std::size_t nnz;
        std::size_t max_local_rows;
        std::size_t min_local_rows;
    };
    std::vector<level_type> levels;
};


/**
   Gather set of statistics on an AMG Hierarchy.

   \param[in] amg_name name of solver containing hierarchy
   \param[in] ml multilevel hierarchy
   \param[in] cg_solver Coarse grid solver for hierarchy
   \return information on AMG Hierarchy as specified in HierarchyStats.
 */
template<class L>
HierarchyStats collect_statistics( const std::string &amg_name,
                                   const std::vector<L> &ml,
                                   const SolverStrategy &cg_solver );


/**
   Print summary of an AMG Hierarchy.

   \param[in] amg_name name of solver containing hierarchy
   \param[in] ml multilevel hierarchy
   \param[in] cg_solver Coarse grid solver for hierarchy
 */
template<class L>
void print_summary( const std::string &amg_name,
                    const std::vector<L> &ml,
                    const SolverStrategy &cg_solver );

} // namespace AMP::Solver::AMG

#endif
