#ifndef included_AMP_SolverFactory_H
#define included_AMP_SolverFactory_H

#include "AMP/utils/FactoryStrategy.hpp"


namespace AMP::Solver {

class SolverStrategy;
class SolverStrategyParameters;


//! Solver factory class
class SolverFactory :
    public FactoryStrategy<SolverStrategy, std::shared_ptr<SolverStrategyParameters>>
{
public:
    static std::unique_ptr<SolverStrategy>
    create( std::shared_ptr<SolverStrategyParameters> parameters );
};


} // namespace AMP::Solver

#endif
