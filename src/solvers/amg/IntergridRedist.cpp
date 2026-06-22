#include "AMP/solvers/amg/IntergridRedist.h"
#include "AMP/vectors/VectorHelpers.h"

#include <utility>

namespace AMP::Solver::AMG {

IntergridRedist::IntergridRedist( std::shared_ptr<AMP::Operator::OperatorParameters> params,
                                  direction dir,
                                  const redist_context &ctx )
    : AMP::Operator::LinearOperator( params ), d_direction{ dir }, d_redist_context{ ctx }
{
}

IntergridRedist::IntergridRedist( std::shared_ptr<AMP::Operator::OperatorParameters> params,
                                  direction dir,
                                  const redist_context &ctx,
                                  std::shared_ptr<AMP::Operator::Operator> transfer )
    : AMP::Operator::LinearOperator( params ),
      d_direction{ dir },
      d_redist_context{ ctx },
      d_transfer{ std::move( transfer ) }
{
}

void IntergridRedist::apply( std::shared_ptr<const LinearAlgebra::Vector> u,
                             std::shared_ptr<LinearAlgebra::Vector> f )
{
    switch ( d_direction ) {
    case direction::down: {
        AMP_INSIST( u, "NULL Solution Vector" );

        auto uInternal = subsetInputVector( u );
        AMP_INSIST( uInternal, "uInternal is NULL" );

        if ( d_redist_context.isActive() ) {
            AMP_INSIST( f, "NULL Residual Vector" );
            AMP_INSIST( d_transfer || d_matrix, "NULL Transfer Operator" );
            auto fInternal = subsetOutputVector( f );
            AMP_INSIST( fInternal, "fInternal is NULL" );

            if ( !tmp ) {
                tmp = d_transfer ? d_transfer->createInputVector() : d_matrix->createInputVector();
                AMP_INSIST( tmp, "IntergridRedist: transfer input vector is null" );
            }
            LinearAlgebra::VectorHelpers::redistribute( d_redist_context, uInternal, tmp );
            if ( d_transfer ) {
                d_transfer->apply( tmp, fInternal );
            } else {
                d_matrix->mult( tmp, fInternal );
            }
            fInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        } else {
            LinearAlgebra::VectorHelpers::redistribute( d_redist_context, uInternal, nullptr );
        }
        break;
    }
    case direction::up: {
        AMP_INSIST( f, "NULL Residual Vector" );
        auto fInternal = subsetOutputVector( f );
        AMP_INSIST( fInternal, "fInternal is NULL" );

        if ( d_redist_context.isActive() ) {
            AMP_INSIST( u, "NULL Solution Vector" );
            AMP_INSIST( d_transfer || d_matrix, "NULL Transfer Operator" );
            if ( !d_transfer ) {
                AMP_INSIST( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED,
                            "Input vector is in an inconsistent state" );
            }

            auto uInternal = subsetInputVector( u );
            AMP_INSIST( uInternal, "uInternal is NULL" );

            if ( !tmp ) {
                tmp =
                    d_transfer ? d_transfer->createOutputVector() : d_matrix->createOutputVector();
                AMP_INSIST( tmp, "IntergridRedist: transfer output vector is null" );
            }
            if ( d_transfer ) {
                d_transfer->apply( uInternal, tmp );
            } else {
                d_matrix->mult( uInternal, tmp );
            }
            LinearAlgebra::VectorHelpers::scatterRedistributed( d_redist_context, tmp, fInternal );
        } else {
            LinearAlgebra::VectorHelpers::scatterRedistributed(
                d_redist_context, nullptr, fInternal );
        }
        fInternal->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        break;
    }
    }
}

} // namespace AMP::Solver::AMG
