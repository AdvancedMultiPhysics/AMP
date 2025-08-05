#ifndef included_AMP_AMG_Cycle_hpp
#define included_AMP_AMG_Cycle_hpp

#include "AMP/solvers/amg/Cycle.h"

namespace AMP::Solver::AMG {

namespace detail {
template<std::size_t... I>
std::array<std::shared_ptr<LinearAlgebra::Vector>, sizeof...( I )>
clone( const LinearAlgebra::Vector &donor, std::index_sequence<I...> )
{
    return { [&]( std::size_t ) { return donor.clone(); }( I )... };
}
} // namespace detail

template<std::size_t N>
void clone_workspace( LevelWithWorkspace<N> &level, const LinearAlgebra::Vector &donor )
{
    level.work = detail::clone( donor, std::make_index_sequence<N>() );
}

} // namespace AMP::Solver::AMG
#endif
