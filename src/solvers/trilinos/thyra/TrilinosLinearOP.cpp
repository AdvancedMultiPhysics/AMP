#include "AMP/solvers/trilinos/thyra/TrilinosLinearOP.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/trilinos/thyra/ThyraVector.h"


namespace AMP::Solver {


/****************************************************************
 *  Constructors                                                 *
 ****************************************************************/
TrilinosLinearOP::TrilinosLinearOP() = default;
TrilinosLinearOP::TrilinosLinearOP( AMP::Operator::Operator::shared_ptr op )
{
    this->d_operator = op;
    AMP_ASSERT( d_operator );
}
TrilinosLinearOP::TrilinosLinearOP( AMP::Solver::SolverStrategy::shared_ptr solver )
{
    this->d_solver = solver;
    AMP_ASSERT( d_solver );
}
TrilinosLinearOP::~TrilinosLinearOP() = default;


/****************************************************************
 *  Functions inherited from Thyra::LinearOpBase                 *
 ****************************************************************/
Teuchos::RCP<const Thyra::VectorSpaceBase<double>> TrilinosLinearOP::range() const
{
    AMP_ERROR( "Not Implimented Yet" );
    return Teuchos::RCP<const Thyra::VectorSpaceBase<double>>();
}
Teuchos::RCP<const Thyra::VectorSpaceBase<double>> TrilinosLinearOP::domain() const
{
    AMP_ERROR( "Not Implimented Yet" );
    return Teuchos::RCP<const Thyra::VectorSpaceBase<double>>();
}
bool TrilinosLinearOP::opSupportedImpl( Thyra::EOpTransp ) const
{
    AMP_ERROR( "Not Implimented Yet" );
    return false;
}
void TrilinosLinearOP::applyImpl( const Thyra::EOpTransp M_trans,
                                  const Thyra::MultiVectorBase<double> &X,
                                  const Teuchos::Ptr<Thyra::MultiVectorBase<double>> &Y,
                                  const double alpha,
                                  const double beta ) const
{
    // Compute Y = alpha*OP(M)*X + beta*Y
    AMP_ASSERT( M_trans == Thyra::NOTRANS );
    auto x0 = AMP::LinearAlgebra::ThyraVector::constView(
        dynamic_cast<const Thyra::VectorBase<double> *>( &X ) );
    auto y0 = AMP::LinearAlgebra::ThyraVector::view(
        dynamic_cast<Thyra::VectorBase<double> *>( Y.get() ) );
    if ( x0 )
        const_cast<AMP::LinearAlgebra::Vector *>( x0.get() )
            ->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    if ( y0 )
        y0->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    std::vector<AMP::LinearAlgebra::Vector::const_shared_ptr> x;
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> y;
    if ( x0->getName() == "ThyraMultiVec" || y0->getName() == "ThyraMultiVec" ) {
        // We are dealing with a column thyra multivector
        if ( x0->getName() != "ThyraMultiVec" || y0->getName() != "ThyraMultiVec" )
            AMP_ERROR( "Not finished" );
        std::shared_ptr<const AMP::LinearAlgebra::MultiVector> x1 =
            std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( x0 );
        std::shared_ptr<AMP::LinearAlgebra::MultiVector> y1 =
            std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( y0 );
        AMP_ASSERT( x1 && y1 );
        size_t N_vecs_x = x1->getNumberOfSubvectors();
        size_t N_vecs_y = y1->getNumberOfSubvectors();
        AMP_ASSERT( N_vecs_x != N_vecs_y );
        for ( size_t i = 0; i < N_vecs_x; i++ ) {
            x.push_back( x1->getVector( i ) );
            y.push_back( y1->getVector( i ) );
        }
    } else {
        x.push_back( x0 );
        y.push_back( y0 );
    }
    for ( size_t i = 0; i < x.size(); i++ ) {
        if ( d_operator ) {
            // Apply the AMP::Operator to compute f = OP(M)*X
            AMP::LinearAlgebra::Vector::shared_ptr f = y[i]->clone();
            d_operator->apply( x[i], f );
            // Compute Y = alpha*OP(M)*X + beta*Y
            y[i]->axpby( alpha, beta, *f );
        } else {
            // Apply the AMP::Solver
            d_solver->apply( x[i], y[i] );
        }
    }
}
} // namespace AMP::Solver
