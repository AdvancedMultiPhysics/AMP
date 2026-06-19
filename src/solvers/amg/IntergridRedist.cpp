#include "AMP/solvers/amg/IntergridRedist.h"
#include "AMP/vectors/VectorHelpers.h"

namespace AMP::Solver::AMG {

IntergridRedist::IntergridRedist( std::shared_ptr<AMP::Operator::OperatorParameters> params,
                                  direction dir,
                                  const redist_context &ctx )
    : AMP::Operator::LinearOperator( params ), d_direction{ dir }, d_redist_context{ ctx }
{
}

void IntergridRedist::apply( std::shared_ptr<const LinearAlgebra::Vector> u,
                             std::shared_ptr<LinearAlgebra::Vector> f )
{
    switch ( d_direction ) {
    case direction::down: {
        AMP_INSIST( u, "NULL Solution Vector" );
        AMP_INSIST( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED,
                    "Input vector is in an inconsistent state" );

        auto uInternal = subsetInputVector( u );
        AMP_INSIST( uInternal, "uInternal is NULL" );

        if ( d_redist_context.isActive() ) {
            AMP_INSIST( f, "NULL Residual Vector" );
            AMP_INSIST( d_matrix, "NULL Matrix" );
            auto fInternal = subsetOutputVector( f );
            AMP_INSIST( fInternal, "fInternal is NULL" );

            if ( !tmp ) {
                tmp = d_matrix->createInputVector();
            }
            LinearAlgebra::VectorHelpers::redistribute( d_redist_context, uInternal, tmp );
            d_matrix->mult( tmp, fInternal );
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
            AMP_INSIST( d_matrix, "NULL Matrix" );
            AMP_INSIST( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED,
                        "Input vector is in an inconsistent state" );

            auto uInternal = subsetInputVector( u );
            AMP_INSIST( uInternal, "uInternal is NULL" );

            if ( !tmp ) {
                tmp = d_matrix->createOutputVector();
            }
            d_matrix->mult( uInternal, tmp );
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
