#ifndef included_AMP_AMG_Relaxation
#define included_AMP_AMG_Relaxation

#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/Operator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/Matrix.h"

#include <cstddef>

namespace AMP::Solver::AMG {

struct RelaxationParameters : SolverStrategyParameters {
    explicit RelaxationParameters( std::shared_ptr<AMP::Database> db )
        : SolverStrategyParameters( db )
    {
    }

    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_matrix;
};

struct Relaxation : SolverStrategy {
public:
    enum class Sweep { forward, backward, symmetric };
    enum class Direction { forward, backward };
    explicit Relaxation( std::shared_ptr<const SolverStrategyParameters> params,
                         std::string name_,
                         std::string short_name_ );

    virtual std::string type() const override { return "Relaxation"; }

    void getFromInput( std::shared_ptr<AMP::Database> );

    void apply( std::shared_ptr<const LinearAlgebra::Vector> b,
                std::shared_ptr<LinearAlgebra::Vector> x ) override;

protected:
    virtual void relax_visit( std::shared_ptr<const LinearAlgebra::Vector> b,
                              std::shared_ptr<LinearAlgebra::Vector> x ) = 0;

    const std::string name, short_name;
    bool need_norms;
    Sweep d_sweep;
    size_t d_num_sweeps;
    std::shared_ptr<AMP::LinearAlgebra::Matrix> d_matrix;
};

struct HybridGS : Relaxation {
    explicit HybridGS( std::shared_ptr<const SolverStrategyParameters> params );

    ~HybridGS();

    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return std::make_unique<HybridGS>( params );
    }

    std::string type() const override { return "Hybrid Gauss-Seidel"; }

    void registerOperator( std::shared_ptr<AMP::Operator::Operator> ) override;

protected:
    std::byte *d_ghost_vals;
    size_t d_num_ghosts;
    size_t d_num_ghost_bytes;

    void deallocateGhosts();

    void relax_visit( std::shared_ptr<const LinearAlgebra::Vector> b,
                      std::shared_ptr<LinearAlgebra::Vector> x ) override;

    template<typename Config>
    void relax( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                std::shared_ptr<const LinearAlgebra::Vector> b,
                std::shared_ptr<LinearAlgebra::Vector> x );

    template<typename Config>
    void sweep( const Relaxation::Direction relax_dir,
                LinearAlgebra::CSRMatrix<Config> &A,
                const LinearAlgebra::Vector &bvec,
                LinearAlgebra::Vector &xvec );
};

struct JacobiL1 : Relaxation {
    explicit JacobiL1( std::shared_ptr<const SolverStrategyParameters> params );

    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return std::make_unique<JacobiL1>( params );
    }

    std::string type() const override { return "Jacobi L1"; }

    void registerOperator( std::shared_ptr<AMP::Operator::Operator> ) override;

protected:
    void relax_visit( std::shared_ptr<const LinearAlgebra::Vector> b,
                      std::shared_ptr<LinearAlgebra::Vector> x ) override;

    float d_spec_lower;
    std::shared_ptr<LinearAlgebra::Vector> d_diag;
    template<typename Config>
    void relax( std::shared_ptr<LinearAlgebra::CSRMatrix<Config>> A,
                std::shared_ptr<const LinearAlgebra::Vector> b,
                std::shared_ptr<LinearAlgebra::Vector> x );
};

} // namespace AMP::Solver::AMG
#endif
