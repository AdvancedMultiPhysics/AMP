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

KappaKCycle::KappaKCycle( const settings &s ) : d_settings( s ) {}

void KappaKCycle::operator()( std::shared_ptr<const LinearAlgebra::Vector> b,
                              std::shared_ptr<LinearAlgebra::Vector> x,
                              const std::vector<level> &ml,
                              SolverStrategy &coarse_solver ) const
{
    if ( ml.size() == 1 )
        coarse_solver.apply( b, x );
    else
        cycle( 0, d_settings.kappa, b, x, ml, coarse_solver );
}

void KappaKCycle::cycle( size_t lvl,
                         size_t kappa,
                         std::shared_ptr<const LinearAlgebra::Vector> b,
                         std::shared_ptr<LinearAlgebra::Vector> x,
                         const std::vector<level> &ml,
                         SolverStrategy &coarse_solver ) const
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

    if ( !d_settings.comm_free_interp ) {
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
            cycle( lvl + 1, kappa, coarse_b, c, ml, coarse_solver );
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
                switch ( d_settings.type ) {
                case krylov_type::fcg:
                    rho1   = c->dot( *v );
                    alpha1 = c->dot( *coarse_b );
                    break;
                case krylov_type::gcr:
                    rho1   = v->dot( *v );
                    alpha1 = v->dot( *coarse_b );
                    break;
                }
                tau1 = alpha1 / rho1;
                btilde->axpy( -tau1, *v, *coarse_b );
            }
            if ( d_settings.tol > 0 && btilde->L2Norm() < d_settings.tol * coarse_b->L2Norm() ) {
                coarse_x->axpy( tau1, *c, *coarse_x );
            } else {
                d->zero();
                cycle( lvl + 1, kappa - 1, btilde, d, ml, coarse_solver );
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
                    AMP::Scalar gamma, beta, alpha2;
                    switch ( d_settings.type ) {
                    case krylov_type::fcg:
                        gamma  = d->dot( *v );
                        beta   = d->dot( *w );
                        alpha2 = d->dot( *btilde );
                        break;
                    case krylov_type::gcr:
                        gamma  = w->dot( *v );
                        beta   = w->dot( *w );
                        alpha2 = w->dot( *btilde );
                        break;
                    }
                    auto rho2 = beta - ( ( gamma * gamma ) / rho1 );
                    auto tau2 = tau1 - ( gamma * alpha2 ) / ( rho1 * rho2 );
                    auto tau3 = alpha2 / rho2;
                    coarse_x->linearSum( tau2, *c, tau3, *d );
                }
                if ( !d_settings.comm_free_interp )
                    coarse_x->makeConsistent();
            }
        } else {
            cycle( lvl + 1, kappa, coarse_b, coarse_x, ml, coarse_solver );
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

KappaKCycle::krylov_type KappaKCycle::parseType( const std::string &kcycle_type )
{
    auto it = type_map.find( kcycle_type );
    AMP_INSIST( it != type_map.end(), "KappaKCycle: invalid kcycle_type (" + kcycle_type + ")." );
    return it->second;
}
} // namespace AMP::Solver::AMG
