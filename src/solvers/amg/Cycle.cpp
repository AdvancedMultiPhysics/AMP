#include "AMP/solvers/amg/Cycle.hpp"

namespace AMP::Solver::AMG {

template void clone_workspace( LevelWithWorkspace<num_work_kcycle> &,
                               const LinearAlgebra::Vector & );

void kappa_kcycle( size_t lvl,
                   std::shared_ptr<const LinearAlgebra::Vector> b,
                   std::shared_ptr<LinearAlgebra::Vector> x,
                   const std::vector<KCycleLevel> &ml,
                   SolverStrategy &coarse_solver,
                   size_t kappa,
                   float ktol,
                   bool comm_free_interp )
{
    auto &flevel = ml[lvl];
    auto &clevel = ml[lvl + 1];
    auto &A      = flevel.A;

    flevel.pre_relaxation->apply( b, x );
    ++flevel.nrelax;

    auto r = flevel.r;
    A->residual( b, x, r );
    const auto r_norm = static_cast<double>( r->L2Norm() );
    // AMP::pout << "On level " << lvl << " pre-rel res norm = " << r_norm << std::endl;

    auto coarse_b = clevel.b;
    auto coarse_x = clevel.x;

    clevel.R->apply( r, coarse_b );
    coarse_x->zero();
    if ( lvl + 1 == ml.size() - 1 ) {
        // AMP::pout << "Calling coarse solver" << std::endl;
        coarse_solver.apply( coarse_b, coarse_x );
        ++clevel.nrelax;
    } else {
        if ( kappa > 1 ) {
            auto Ac                   = clevel.A;
            auto [c, v, btilde, d, w] = clevel.work;
            c->zero();
            kappa_kcycle( lvl + 1, coarse_b, c, ml, coarse_solver, kappa, ktol, comm_free_interp );
            Ac->apply( c, v );
            auto rho1 = c->dot( *v );
            // AMP::pout << "rho1 = " << rho1 << " at level " << lvl << std::endl;
            auto alpha1 = c->dot( *coarse_b );
            auto tau1   = alpha1 / rho1;
            btilde->axpy( -tau1, *v, *coarse_b );
            if ( ktol > 0 && btilde->L2Norm() < ktol * coarse_b->L2Norm() ) {
                coarse_x->axpy( tau1, *c, *coarse_x );
            } else {
                d->zero();
                kappa_kcycle(
                    lvl + 1, btilde, d, ml, coarse_solver, kappa - 1, ktol, comm_free_interp );
                Ac->apply( d, w );
                auto gamma  = d->dot( *v );
                auto beta   = d->dot( *w );
                auto alpha2 = d->dot( *btilde );
                auto rho2   = beta - ( ( gamma * gamma ) / rho1 );
                // AMP::pout << "rho2 = " << rho2 << " at level " << lvl << std::endl;
                auto tau2 = tau1 - ( gamma * alpha2 ) / ( rho1 * rho2 );
                auto tau3 = alpha2 / rho2;
                coarse_x->linearSum( tau2, *c, tau3, *d );
                if ( !comm_free_interp )
                    coarse_x->makeConsistent();
            }
        } else {
            kappa_kcycle(
                lvl + 1, coarse_b, coarse_x, ml, coarse_solver, kappa, ktol, comm_free_interp );
        }
    }

    auto correction    = flevel.correction;
    const auto cx_norm = static_cast<double>( coarse_x->L2Norm() );
    clevel.P->apply( coarse_x, correction );
    const auto c_norm = static_cast<double>( correction->L2Norm() );
    // AMP::pout << "On level " << lvl << " cx_norm = " << cx_norm << " c_norm = " << c_norm
    //           << std::endl;
    x->add( *x, *correction );
    x->makeConsistent( LinearAlgebra::ScatterType::CONSISTENT_SET );
    flevel.post_relaxation->apply( b, x );
    ++flevel.nrelax;
}

void kappa_kcycle( std::shared_ptr<const LinearAlgebra::Vector> b,
                   std::shared_ptr<LinearAlgebra::Vector> x,
                   const std::vector<KCycleLevel> &ml,
                   SolverStrategy &coarse_solver,
                   size_t kappa,
                   float ktol,
                   bool comm_free_interp )
{
    if ( ml.size() == 1 )
        coarse_solver.apply( b, x );
    else
        kappa_kcycle( 0, b, x, ml, coarse_solver, kappa, ktol, comm_free_interp );
}

} // namespace AMP::Solver::AMG
