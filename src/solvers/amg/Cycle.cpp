#include "AMP/solvers/amg/Cycle.hpp"

#define AMP_AMG_CYCLE_PROFILE

namespace AMP::Solver::AMG {


LevelOperator::LevelOperator( const AMP::Operator::LinearOperator &linop ) : base( linop ) {}


void LevelOperator::apply( std::shared_ptr<const LinearAlgebra::Vector> u,
                           std::shared_ptr<LinearAlgebra::Vector> f )
{
    AMP_INSIST( u, "NULL Solution Vector" );
    AMP_INSIST( f, "NULL Residual Vector" );
    AMP_INSIST( d_matrix, "NULL Matrix" );

    AMP_DEBUG_INSIST( u->getVariable(), "LevelOperator::apply: u must have a variable" );
    AMP_DEBUG_INSIST( f->getVariable(), "LevelOperator::apply: f must have a variable" );
    AMP_DEBUG_INSIST( d_inputVariable,
                      "LevelOperator::apply: operator must have an input variable" );
    AMP_DEBUG_INSIST( d_outputVariable,
                      "LevelOperator::apply: operator must have an output variable" );

    AMP_INSIST( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED,
                "Input vector is in an inconsistent state" );

    auto uInternal = subsetInputVector( u );
    auto fInternal = subsetOutputVector( f );

    AMP_INSIST( uInternal, "uInternal is NULL" );
    AMP_INSIST( fInternal, "fInternal is NULL" );

    d_matrix->mult( uInternal, fInternal );

    if ( !defer_consistency )
        fInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


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
    {
#ifdef AMP_AMG_CYCLE_PROFILE
        PROFILE( "Kcycle::matvec" );
#endif
        A->residual( b, x, r );
    }

    if ( !comm_free_interp ) {
        r->makeConsistent();
    }

    auto coarse_b = clevel.b;
    auto coarse_x = clevel.x;

    {
#ifdef AMP_AMG_CYCLE_PROFILE
        PROFILE( "Kcycle::intergrid" );
#endif
        clevel.R->apply( r, coarse_b );
    }
    coarse_x->zero();
    if ( lvl + 1 == ml.size() - 1 ) {
        coarse_solver.apply( coarse_b, coarse_x );
        ++clevel.nrelax;
    } else {
        if ( kappa > 1 ) {
            auto Ac                   = clevel.A;
            auto [c, v, btilde, d, w] = clevel.work;
            c->zero();
            kappa_kcycle( lvl + 1, coarse_b, c, ml, coarse_solver, kappa, ktol, comm_free_interp );
            {
#ifdef AMP_AMG_CYCLE_PROFILE
                PROFILE( "Kcycle::matvec" );
#endif
                Ac->applyDeferConsistency( c, v );
            }
            AMP::Scalar rho1, alpha1, tau1;
            {
#ifdef AMP_AMG_CYCLE_PROFILE
                PROFILE( "Kcycle::dot" );
#endif
                rho1   = c->dot( *v );
                alpha1 = c->dot( *coarse_b );
                tau1   = alpha1 / rho1;
                btilde->axpy( -tau1, *v, *coarse_b );
            }
            if ( ktol > 0 && btilde->L2Norm() < ktol * coarse_b->L2Norm() ) {
                coarse_x->axpy( tau1, *c, *coarse_x );
            } else {
                d->zero();
                kappa_kcycle(
                    lvl + 1, btilde, d, ml, coarse_solver, kappa - 1, ktol, comm_free_interp );
                {
#ifdef AMP_AMG_CYCLE_PROFILE
                    PROFILE( "Kcycle::matvec" );
#endif
                    Ac->applyDeferConsistency( d, w );
                }
                {
#ifdef AMP_AMG_CYCLE_PROFILE
                    PROFILE( "Kcycle::dot" );
#endif
                    auto gamma  = d->dot( *v );
                    auto beta   = d->dot( *w );
                    auto alpha2 = d->dot( *btilde );
                    auto rho2   = beta - ( ( gamma * gamma ) / rho1 );
                    auto tau2   = tau1 - ( gamma * alpha2 ) / ( rho1 * rho2 );
                    auto tau3   = alpha2 / rho2;
                    coarse_x->linearSum( tau2, *c, tau3, *d );
                }
                if ( !comm_free_interp )
                    coarse_x->makeConsistent();
            }
        } else {
            kappa_kcycle(
                lvl + 1, coarse_b, coarse_x, ml, coarse_solver, kappa, ktol, comm_free_interp );
        }
    }

    auto correction = flevel.correction;
    {
#ifdef AMP_AMG_CYCLE_PROFILE
        PROFILE( "Kcycle::intergrid" );
#endif
        clevel.P->apply( coarse_x, correction );
        x->add( *x, *correction );
        x->makeConsistent( LinearAlgebra::ScatterType::CONSISTENT_SET );
    }
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
