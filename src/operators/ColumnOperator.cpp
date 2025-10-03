#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/MultiVariable.h"

#include "ProfilerApp.h"


namespace AMP::Operator {


static inline AMP::LinearAlgebra::UpdateState
getState( AMP::LinearAlgebra::Vector::const_shared_ptr u )
{
    if ( u )
        return u->getUpdateStatus();
    return AMP::LinearAlgebra::UpdateState::UNCHANGED;
}
static inline void checkState( AMP::LinearAlgebra::UpdateState initial,
                               AMP::LinearAlgebra::Vector::const_shared_ptr u,
                               std::shared_ptr<const Operator> op )
{
    auto UNCHANGED = AMP::LinearAlgebra::UpdateState::UNCHANGED;
    if ( initial == UNCHANGED && u )
        AMP_INSIST( u->getUpdateStatus() == UNCHANGED,
                    op->type() + " left vector in an inconsistent state" );
}


/********************************************************
 * Constructors                                          *
 ********************************************************/
ColumnOperator::ColumnOperator() : Operator() {}
ColumnOperator::ColumnOperator( std::shared_ptr<const OperatorParameters> params )
    : Operator( params )
{
    auto columnOpParams = std::dynamic_pointer_cast<const ColumnOperatorParameters>( params );
    if ( columnOpParams ) {
        for ( auto p : columnOpParams->d_OperatorParameters )
            d_operators.push_back( AMP::Operator::OperatorFactory::create( p ) );
    }
}


/********************************************************
 * Apply/residual                                        *
 ********************************************************/
void ColumnOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                            AMP::LinearAlgebra::Vector::shared_ptr f )
{
    AMP_ASSERT( getState( u ) == AMP::LinearAlgebra::UpdateState::UNCHANGED );
    auto state = getState( f );
    for ( auto &op : d_operators ) {
        AMP_INSIST( op, "ColumnOperator::operator component is NULL" );
        op->apply( u, f );
        checkState( state, f, op );
    }
}
void ColumnOperator::residual( AMP::LinearAlgebra::Vector::const_shared_ptr f,
                               AMP::LinearAlgebra::Vector::const_shared_ptr u,
                               AMP::LinearAlgebra::Vector::shared_ptr r )
{
    AMP_ASSERT( getState( u ) == AMP::LinearAlgebra::UpdateState::UNCHANGED );
    for ( auto &op : d_operators ) {
        AMP_INSIST( op, "ColumnOperator::operator component is NULL" );
        op->residual( f, u, r );
    }
}


/********************************************************
 * getParameters                                         *
 ********************************************************/
std::shared_ptr<OperatorParameters>
ColumnOperator::getParameters( const std::string &type,
                               AMP::LinearAlgebra::Vector::const_shared_ptr u,
                               std::shared_ptr<OperatorParameters> params )
{
    auto db = std::make_shared<Database>();
    Operator::setMemoryAndBackendParameters( db );
    auto opParameters    = std::make_shared<ColumnOperatorParameters>( db );
    opParameters->d_Mesh = d_Mesh;
    opParameters->d_db   = std::make_shared<AMP::Database>( "ColumnOperator" );
    opParameters->d_db->putScalar( "name", "ColumnOperator" );
    opParameters->d_OperatorParameters.resize( d_operators.size() );
    for ( unsigned int i = 0; i < d_operators.size(); i++ ) {
        opParameters->d_OperatorParameters[i] = d_operators[i]->getParameters( type, u, params );
    }
    return opParameters;
}


/********************************************************
 * reset                                                  *
 ********************************************************/
void ColumnOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    auto columnParameters = std::dynamic_pointer_cast<const ColumnOperatorParameters>( params );
    AMP_INSIST( ( columnParameters ), "ColumnOperator::reset parameter object is NULL" );
    AMP_INSIST( ( ( ( columnParameters->d_OperatorParameters ).size() ) == ( d_operators.size() ) ),
                " std::vector sizes do not match! " );
    for ( size_t i = 0; i < d_operators.size(); i++ ) {
        d_operators[i]->reset( ( columnParameters->d_OperatorParameters )[i] );
    }
}


/********************************************************
 * Add an operator                                       *
 ********************************************************/
void ColumnOperator::append( std::shared_ptr<Operator> op )
{
    AMP_INSIST( ( op ), "AMP::ColumnOperator::appendRow input argument is a NULL operator" );
    d_operators.push_back( op );
}


/********************************************************
 * Get the variables                                     *
 ********************************************************/
std::shared_ptr<AMP::LinearAlgebra::Variable> ColumnOperator::getInputVariable() const
{
    auto retVariable = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "ColumnVariable" );
    for ( auto &elem : d_operators ) {
        auto opVar = elem->getInputVariable();
        if ( opVar )
            retVariable->add( opVar );
    }
    retVariable->removeDuplicateVariables();
    return retVariable;
}
std::shared_ptr<AMP::LinearAlgebra::Variable> ColumnOperator::getOutputVariable() const
{
    auto retVariable = std::make_shared<AMP::LinearAlgebra::MultiVariable>( "ColumnVariable" );
    for ( auto &elem : d_operators ) {
        auto opVar = elem->getOutputVariable();
        if ( opVar )
            retVariable->add( opVar );
    }
    retVariable->removeDuplicateVariables();
    return retVariable;
}


/********************************************************
 * Check the input                                       *
 ********************************************************/
bool ColumnOperator::isValidVector( std::shared_ptr<const AMP::LinearAlgebra::Vector> u )
{
    bool bRetVal = true;
    for ( auto &elem : d_operators ) {
        bRetVal = bRetVal && elem->isValidVector( u );
    }
    return bRetVal;
}


/********************************************************
 * find                                                  *
 ********************************************************/
std::vector<std::shared_ptr<Operator>> ColumnOperator::find( const std::string &name )
{
    std::vector<std::shared_ptr<Operator>> ops;
    for ( auto op : d_operators ) {
        if ( op->type() == name )
            ops.push_back( op );
    };
    return ops;
}


} // namespace AMP::Operator
