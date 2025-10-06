#ifndef included_AMP_AMG_Cycle
#define included_AMP_AMG_Cycle

#include <memory>
#include <vector>

#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/amg/DeferConsistency.h"

namespace AMP::Solver::AMG {
struct LevelOperator : HasDeferConsistency<AMP::Operator::LinearOperator> {
    using base = HasDeferConsistency<AMP::Operator::LinearOperator>;
    using base::base;
    explicit LevelOperator( const AMP::Operator::LinearOperator &linop );

    virtual void apply( std::shared_ptr<const LinearAlgebra::Vector> u,
                        std::shared_ptr<LinearAlgebra::Vector> f ) override;
};

struct Level {
    std::shared_ptr<LevelOperator> A;
    std::shared_ptr<AMP::Operator::Operator> R, P;
    std::unique_ptr<AMP::Solver::SolverStrategy> pre_relaxation, post_relaxation;
    std::shared_ptr<LinearAlgebra::Vector> x, b, r, correction;
    mutable std::size_t nrelax = 0;
};

template<std::size_t N>
struct LevelWithWorkspace : Level {
    // extra work vectors used in cycling
    std::array<std::shared_ptr<LinearAlgebra::Vector>, N> work;
};

template<class T>
constexpr bool is_level_v = std::is_base_of_v<Level, T>;

/**
   Initialize workspace by cloning a vector

   \param[in] level Level with workspace to initialize
   \param[in] donor Vector to use as donor for workspace vector clones
*/
template<std::size_t N>
void clone_workspace( LevelWithWorkspace<N> &level, const LinearAlgebra::Vector &donor );

inline constexpr std::size_t num_work_kcycle = 5;
using KCycleLevel                            = LevelWithWorkspace<num_work_kcycle>;

void kappa_kcycle( size_t lvl,
                   std::shared_ptr<const LinearAlgebra::Vector> b,
                   std::shared_ptr<LinearAlgebra::Vector> x,
                   const std::vector<KCycleLevel> &levels,
                   SolverStrategy &coarse_solver,
                   size_t kappa,
                   float ktol,
                   bool comm_free_interp );

void kappa_kcycle( std::shared_ptr<const LinearAlgebra::Vector> b,
                   std::shared_ptr<LinearAlgebra::Vector> x,
                   const std::vector<KCycleLevel> &levels,
                   SolverStrategy &coarse_solver,
                   size_t kappa,
                   float ktol,
                   bool comm_free_interp = false );
} // namespace AMP::Solver::AMG

#endif
