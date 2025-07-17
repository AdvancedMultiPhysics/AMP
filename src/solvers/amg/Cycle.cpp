#include "AMP/solvers/amg/Cycle.h"

namespace AMP::Solver::AMG {

namespace {

template<size_t... I>
std::array<std::shared_ptr<LinearAlgebra::Vector>, sizeof...( I )>
clone( const LinearAlgebra::Vector &x, std::index_sequence<I...> )
{
    return { [&]( size_t ) { return x.clone(); }( I )... };
}

template<size_t N>
auto make_clones( const LinearAlgebra::Vector &x )
{
    return clone( x, std::make_index_sequence<N>() );
}

} // namespace

void kappa_kcycle( size_t lvl,
                   std::shared_ptr<const LinearAlgebra::Vector> b,
                   std::shared_ptr<LinearAlgebra::Vector> x,
                   const std::vector<Level> &ml,
                   SolverStrategy &coarse_solver,
                   size_t kappa,
                   float ktol )
{
    auto &flevel = ml[lvl];
    auto &clevel = ml[lvl + 1];
    auto &A      = flevel.A;

    flevel.pre_relaxation->apply( b, x );
    ++flevel.nrelax;

    auto r = b->clone();
    A->residual( b, x, r );

    auto coarse_b = clevel.b;
    auto coarse_x = clevel.x;

    clevel.R->apply( r, coarse_b );
    coarse_x->zero();
    if ( lvl + 1 == ml.size() - 1 ) {
        coarse_solver.apply( coarse_b, coarse_x );
        ++clevel.nrelax;
    } else {
        if ( kappa > 1 ) {
            auto Ac                   = clevel.A;
            auto [c, v, btilde, d, w] = make_clones<5>( *coarse_b );
            c->zero();
            kappa_kcycle( lvl + 1, coarse_b, c, ml, coarse_solver, kappa, ktol );
            Ac->apply( c, v );
            auto rho1   = c->dot( *v );
            auto alpha1 = c->dot( *coarse_b );
            auto tau1   = alpha1 / rho1;
            btilde->axpy( -tau1, *v, *coarse_b );
            if ( ktol > 0 && btilde->L2Norm() < ktol * coarse_b->L2Norm() ) {
                coarse_x->axpy( tau1, *c, *coarse_x );
            } else {
                d->zero();
                kappa_kcycle( lvl + 1, btilde, d, ml, coarse_solver, kappa - 1, ktol );
                Ac->apply( d, w );
                auto gamma  = d->dot( *v );
                auto beta   = d->dot( *w );
                auto alpha2 = d->dot( *btilde );
                auto rho2   = beta - ( ( gamma * gamma ) / rho1 );
                auto tau2   = tau1 - ( gamma * alpha2 ) / ( rho1 * rho2 );
                auto tau3   = alpha2 / rho2;
                coarse_x->linearSum( tau2, *c, tau3, *d );
                coarse_x->makeConsistent();
            }
        } else {
            kappa_kcycle( lvl + 1, coarse_b, coarse_x, ml, coarse_solver, kappa, ktol );
        }
    }

    auto correction = b->clone();
    clevel.P->apply( coarse_x, correction );
    x->add( *x, *correction );
    x->makeConsistent();
    flevel.post_relaxation->apply( b, x );
    ++flevel.nrelax;
}

void kappa_kcycle( std::shared_ptr<const LinearAlgebra::Vector> b,
                   std::shared_ptr<LinearAlgebra::Vector> x,
                   const std::vector<Level> &ml,
                   SolverStrategy &coarse_solver,
                   size_t kappa,
                   float ktol )
{
    if ( ml.size() == 1 )
        coarse_solver.apply( b, x );
    else
        kappa_kcycle( 0, b, x, ml, coarse_solver, kappa, ktol );
}

} // namespace AMP::Solver::AMG
