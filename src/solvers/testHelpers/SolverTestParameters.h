#ifndef included_SolverTestParameters_h
#define included_SolverTestParameters_h

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UtilityMacros.h"
#include "AMP/vectors/Vector.h"

#include <memory>
#include <string>


namespace AMP::Solver::Test {


struct SolverParameters {

    static std::unique_ptr<AMP::Database> getParameters( const std::string &solver,
                                                         bool use_nested );

    static std::unique_ptr<AMP::Database> getCGParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getGMRESParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getFGMRESParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getBiCGSTABParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getTFQMRParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getPetscFGMRESParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getHyprePCGParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getHypreGMRESParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getHypreBiCGSTABParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getBoomerAMGParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getMLParameters( bool use_nested );

    static std::unique_ptr<AMP::Database> getMueLuParameters( bool use_nested );
};


std::shared_ptr<AMP::Solver::SolverStrategy>
buildSolver( const std::string &solver_name,
             std::shared_ptr<AMP::Database> input_db,
             const AMP::AMP_MPI &comm,
             std::shared_ptr<AMP::LinearAlgebra::Vector> initialGuess,
             std::shared_ptr<AMP::Operator::Operator> op );


} // namespace AMP::Solver::Test

#endif
