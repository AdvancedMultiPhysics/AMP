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

/**
 * The Kappa K-cycle implements the Kylov-based multigrid cycle from
 * Notay, Y., & Vassilevski, P. S. (2008). Recursive Krylov‐based
 * multigrid cycles. Numerical Linear Algebra with Applications,
 * 15(5), 473-487. https://doi.org/10.1002/nla.542.
 *
 * augmented with Kappa cycling from
 * Avnat, O., & Yavneh, I. (2022). On the recursive structure of
 * multigrid cycles. SIAM Journal on Scientific Computing, 45(3),
 * S103-S126.  https://doi.org/10.1137/21M1433502.
 *
 * The K-cycle implements economical variants of FCG and GCR as described in
 * Notay, Y. (2010). An aggregation-based algebraic multigrid
 * method. Electron. Trans. Numer. Anal, 37(6), 123-146.
 */
struct KappaKCycle {
    using level = KCycleLevel;
    //! K-cycle variant.
    enum class krylov_type { fcg, gcr };
    //! Recover krylov_type from input string (either "fcg" or "gcr").
    static krylov_type parseType( const std::string &kcycle_type );

    //! Settings for the Kappa K-cycle.
    struct settings {
        /**
         * Kappa parameter for recursive cycle.
         * kappa = 1 corresponds to V-cycle.
         * kappa = 2 corresponds to Krylov-based cycle with F-cycle structure.
         * kappa >= cycle depth corresponds to standard K-cycle.
         */
        size_t kappa = 1;
        //! residual tolerance for FCG/GCR.
        float tol = 0;
        //! indicates interpolation will not use ghosts and communication may be avoided.
        bool comm_free_interp = false;
        //! Krylov method for cycle.
        krylov_type type = krylov_type::fcg;
    };

    KappaKCycle( const settings & );

    /**
     * Run cycle with system \f$Ax = b\f$.
     * @param[in] b : shared pointer to right hand side vector
     * @param[out] u : shared pointer to approximate computed solution.
     */
    void operator()( std::shared_ptr<const LinearAlgebra::Vector> b,
                     std::shared_ptr<LinearAlgebra::Vector> x,
                     const std::vector<level> &levels,
                     SolverStrategy &coarse_solver ) const;

private:
    void cycle( size_t lvl,
                size_t kappa,
                std::shared_ptr<const LinearAlgebra::Vector> b,
                std::shared_ptr<LinearAlgebra::Vector> x,
                const std::vector<level> &levels,
                SolverStrategy &coarse_solver ) const;
    settings d_settings;
    static inline std::map<std::string, krylov_type> type_map{ { "fcg", krylov_type::fcg },
                                                               { "gcr", krylov_type::gcr } };
};
} // namespace AMP::Solver::AMG

#endif
