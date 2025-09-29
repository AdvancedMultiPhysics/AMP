#include "AMP/operators/RowOperator.h"
#include "AMP/operators/ColumnOperatorParameters.h"


namespace AMP::Operator {


// Constructor
RowOperator::RowOperator( std::shared_ptr<const OperatorParameters> params ) : Operator()
{
    (void) params;
    getAllJacobian = false;
    d_paramsize    = 1;
}


// Apply
void RowOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                         AMP::LinearAlgebra::Vector::shared_ptr r )
{
    d_OutputVariable = ( d_Operators[0] )->getOutputVariable();

    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> rInternal( d_Operators.size() );
    auto rOriginal = r->subsetVectorForVariable( d_OutputVariable );

    rOriginal->zero();

    for ( unsigned int i = 0; i < d_Operators.size(); i++ ) {
        rInternal[i] = rOriginal->clone();
        rInternal[i]->zero();
        d_Operators[i]->apply( u, rInternal[i] );
        rInternal[i]->scale( scalea[i] );
    }

    for ( unsigned int i = 0; i < d_Operators.size(); i++ ) {
        rOriginal->add( *rOriginal, *( rInternal[i] ) );
    }
}


// Reset
void RowOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    AMP_ASSERT( params );
    auto fParams = std::dynamic_pointer_cast<const ColumnOperatorParameters>( params );

    AMP_INSIST( fParams, "RowOperator::reset parameter object is NULL" );

    AMP_INSIST( fParams->d_OperatorParameters.size() == d_Operators.size(),
                " std::vector sizes do not match! " );

    for ( unsigned int i = 0; i < d_Operators.size(); i++ ) {
        d_Operators[i]->reset( ( fParams->d_OperatorParameters )[i] );
    }
}


// getParameters
void RowOperator::append( std::shared_ptr<Operator> op, double a )
{
    AMP_INSIST( ( op ), "AMP::RowOperator::appendRow input argument is a NULL operator" );
    d_Operators.push_back( op );
    scalea.push_back( a );
}


// getParameters
std::shared_ptr<OperatorParameters>
RowOperator::getParameters( const std::string &type,
                            AMP::LinearAlgebra::Vector::const_shared_ptr u,
                            std::shared_ptr<OperatorParameters> params )
{
    auto db = std::make_shared<Database>();
    Operator::setMemoryAndBackendParameters( db );
    auto opParameters = std::make_shared<ColumnOperatorParameters>( db );

    auto rtParameters = std::make_shared<OperatorParameters>( db );

    if ( type == "Jacobian" ) {
        if ( getAllJacobian ) {
            ( opParameters->d_OperatorParameters ).resize( d_Operators.size() );

            for ( unsigned int i = 0; i < d_Operators.size(); i++ ) {
                opParameters->d_OperatorParameters[i] =
                    d_Operators[i]->getParameters( type, u, params );
            }

            rtParameters = std::dynamic_pointer_cast<OperatorParameters>( opParameters );
        } else {
            opParameters->d_OperatorParameters.resize( d_paramsize );

            for ( int i = 0; i < d_paramsize; i++ ) {
                opParameters->d_OperatorParameters[i] =
                    d_Operators[i]->getParameters( type, u, params );
            }

            rtParameters = std::dynamic_pointer_cast<OperatorParameters>( opParameters );
            // rtParameters = (d_Operators[0]->getJacobianParameters(u));
        }
    } else {
        AMP_ERROR( "Unknown type requested" );
    }

    return rtParameters;
}


} // namespace AMP::Operator
