#ifndef included_AMP_DeferConsistency
#define included_AMP_DeferConsistency

#include "AMP/vectors/Vector.h"

namespace AMP::Solver::AMG {

/**
 * A helper type to provide deferred consistency call for apply to an
 * operator.  Operators inherit from this type passing their base
 * operator as the template parameter.  They then have access to a
 * bool valued "defer_consistency" that can be queried to condionally
 * call makeConsistent on the output vector.
 *
 * @tparam P Parent operator to inherit from.
 */
template<class P>
struct HasDeferConsistency : P {
    using P::P;
    // Enable copy construction directly from P
    explicit HasDeferConsistency( const P &p ) : P( p ) {}
    struct defer_guard {
        explicit defer_guard( HasDeferConsistency<P> &tgt )
            : d_target( tgt ),
              d_inner_scope( d_target.defer_consistency ) // inner scope of another guard
        {
            d_target.defer_consistency = true;
        }
        ~defer_guard() noexcept
        {
            if ( !d_inner_scope ) // give precedence to outer guard
                d_target.defer_consistency = false;
        }

        defer_guard( const defer_guard & ) = delete;
        defer_guard( defer_guard && )      = delete;
        defer_guard &operator=( const defer_guard & ) = delete;
        defer_guard &operator=( defer_guard && ) = delete;

    private:
        HasDeferConsistency<P> &d_target;
        bool d_inner_scope = false;
    };

    /**
     * The apply function for this operator without ensuring output vector consistency.
     * @param [in] u input vector.
     * @param [out] f residual/output vector.
     */
    void applyDeferConsistency( std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                                std::shared_ptr<AMP::LinearAlgebra::Vector> f )
    {
        ( defer_guard{ *this }, static_cast<P *>( this )->apply( u, f ) );
    }


    /**
     * Residual wrapper to avoid makeConsistent when it subsequently calls apply.
     * @param [in] u input vector.
     * @param [out] f residual/output vector.
     */
    void residual( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                   std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                   std::shared_ptr<AMP::LinearAlgebra::Vector> r ) final override
    {
        ( defer_guard{ *this }, P::residual( f, u, r ) );
    }

protected:
    // used to indicate to derived type P that output vector consistency should be deferred.
    bool defer_consistency = false;
};


} // namespace AMP::Solver::AMG

#endif
