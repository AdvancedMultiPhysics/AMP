#ifndef included_AMP_SASolver_H_
#define included_AMP_SASolver_H_

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/data/CSRLocalMatrixData.h"
#include "AMP/matrices/data/CSRMatrixData.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/amg/Aggregation.h"
#include "AMP/solvers/amg/AggregationSettings.h"
#include "AMP/solvers/amg/Aggregator.h"
#include "AMP/solvers/amg/Cycle.h"
#include "AMP/solvers/amg/Relaxation.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/Variable.h"

#include <memory>
#include <vector>

namespace AMP::Solver::AMG {

struct SASolver : SolverStrategy {
public:
    explicit SASolver( std::shared_ptr<SolverStrategyParameters> );

    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> params )
    {
        return std::make_unique<SASolver>( params );
    }

    std::string type() const override { return "SASolver"; }

    void registerOperator( std::shared_ptr<Operator::Operator> ) override;

    void getFromInput( std::shared_ptr<Database> );

    void apply( std::shared_ptr<const LinearAlgebra::Vector> f,
                std::shared_ptr<LinearAlgebra::Vector> u ) override;

protected:
    // **** settings applicable to solver as a whole **** //
    //! Memory location for all given operator and all internally created operators
    Utilities::MemoryType d_mem_loc;
    //! Maximum depth of AMG hierarchy
    size_t d_max_levels;
    //! Smallest number of locally owned rows allowed in operator, terminates hierarchy
    int d_min_coarse_local;
    //! Smallest number of global rows allowed in operator, terminates hierarchy
    size_t d_min_coarse_global;
    //! Cycle type, tolerance, kappa value
    KappaKCycle::settings d_cycle_settings;

    // **** settings that can change level-by-level **** //
    //! Number of smoothing steps applied to tentative prolongator
    int d_num_smooth_prol;
    //! Lower bound on spectrum for prolongator smoother, must be in (0,1)
    float d_prol_spec_lower;
    //! Truncation parameter for pruning smoothed prolongator, removes [-trunc,trunc]
    float d_prol_trunc;
    //! Aggregation method for producing tentative prolongator
    std::string d_agg_type;
    //! Settings applicable to all coarsening methods
    CoarsenSettings d_coarsen_settings;
    //! Settings specific to pairwise coarsening method
    PairwiseCoarsenSettings d_pair_coarsen_settings;
    //! Database defining pre-cycle relaxation solver
    std::shared_ptr<AMP::Database> d_pre_relax_db;
    //! Database defining post-cycle relaxation solver
    std::shared_ptr<AMP::Database> d_post_relax_db;

    //! storage for all levels in hierarchy
    std::vector<AMG::KCycleLevel> d_levels;
    //! Aggregator, may be replaced as level-by-level settings are updated
    std::shared_ptr<AMG::Aggregator> d_aggregator;
    //! Parameters read from d_pre_relax_db
    std::shared_ptr<AMG::RelaxationParameters> d_pre_relax_params;
    //! Parameters read from d_post_relax_db
    std::shared_ptr<AMG::RelaxationParameters> d_post_relax_params;
    //! Parameters defining coarse level solver
    std::shared_ptr<SolverStrategyParameters> d_coarse_solver_params;
    //! Coarse level solver
    std::unique_ptr<SolverStrategy> d_coarse_solver;

    //! reset level-by-level options to overall defaults from outer DB
    void resetLevelOptions();
    //! set level-by-level options to match specific level DB if found
    void setLevelOptions( const size_t lvl );

    void setup( std::shared_ptr<LinearAlgebra::Variable> xVar,
                std::shared_ptr<LinearAlgebra::Variable> bVar );

    void makeCoarseSolver();

    std::unique_ptr<SolverStrategy>
    createRelaxation( size_t lvl,
                      std::shared_ptr<Operator::Operator> A,
                      std::shared_ptr<AMG::RelaxationParameters> params );

    void smoothP_JacobiL1( std::shared_ptr<LinearAlgebra::Matrix> A,
                           std::shared_ptr<LinearAlgebra::Matrix> &P ) const;
};

} // namespace AMP::Solver::AMG

#endif
