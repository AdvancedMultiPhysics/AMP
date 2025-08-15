#include "AMP/operators/MemorySpaceMigrationOperator.h"
#include "AMP/vectors/VectorBuilder.h"

namespace AMP::Operator {

MemorySpaceMigrationOperator::MemorySpaceMigrationOperator(
    std::shared_ptr<const OperatorParameters> params )
    : Operator( params )
{
    AMP_INSIST(
        params->d_pOperator,
        "MemorySpaceMigrationOperator is required to be initialized with another Operator" );
    d_pOperator    = params->d_pOperator;
    d_migrate_data = ( d_memory_location != d_pOperator->getMemoryLocation() );
}

void MemorySpaceMigrationOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    d_pOperator->reset( params );
}
void MemorySpaceMigrationOperator::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                                          std::shared_ptr<AMP::LinearAlgebra::Vector> f )
{
    if ( d_migrate_data ) {
        const auto op_memory_location = d_pOperator->getMemoryLocation();
        if ( !d_inputVec ) {
            d_inputVec = AMP::LinearAlgebra::createVector( u, op_memory_location );
        }
        if ( !d_outputVec ) {
            d_outputVec = AMP::LinearAlgebra::createVector( f, op_memory_location );
        }
        d_inputVec->copyVector( u );
        d_outputVec->copyVector( f );
        d_inputVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->apply( d_inputVec, d_outputVec );
        f->copyVector( d_outputVec );
    } else {
        d_pOperator->apply( u, f );
    }
}
void MemorySpaceMigrationOperator::residual( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                             std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                                             std::shared_ptr<AMP::LinearAlgebra::Vector> r )
{
    if ( d_migrate_data ) {
        const auto op_memory_location = d_pOperator->getMemoryLocation();
        if ( !d_inputVec ) {
            d_inputVec = AMP::LinearAlgebra::createVector( u, op_memory_location );
        }
        if ( !d_outputVec ) {
            d_outputVec = AMP::LinearAlgebra::createVector( f, op_memory_location );
        }
        if ( !d_resVec ) {
            d_resVec = AMP::LinearAlgebra::createVector( r, op_memory_location );
        }
        d_inputVec->copyVector( u );
        d_outputVec->copyVector( f );
        d_inputVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->residual( d_inputVec, d_outputVec, d_resVec );
        r->copyVector( d_resVec );
    } else {
        d_pOperator->residual( f, u, r );
    }
}
std::shared_ptr<OperatorParameters>
MemorySpaceMigrationOperator::getParameters( const std::string &type,
                                             std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                                             std::shared_ptr<OperatorParameters> params )
{
    if ( d_migrate_data ) {
        d_inputVec->copyVector( u );
        return d_pOperator->getParameters( type, d_inputVec, params );
    } else {
        return d_pOperator->getParameters( type, u, params );
    }
}
void MemorySpaceMigrationOperator::setDebugPrintInfoLevel( int level )
{
    return d_pOperator->setDebugPrintInfoLevel( level );
}
std::shared_ptr<AMP::LinearAlgebra::Variable>
MemorySpaceMigrationOperator::getOutputVariable() const
{
    return d_pOperator->getOutputVariable();
}
std::shared_ptr<AMP::LinearAlgebra::Variable> MemorySpaceMigrationOperator::getInputVariable() const
{
    return d_pOperator->getInputVariable();
}
std::shared_ptr<AMP::LinearAlgebra::Vector> MemorySpaceMigrationOperator::createInputVector() const
{
    if ( d_migrate_data ) {
        return AMP::LinearAlgebra::createVector( d_inputVec, d_memory_location );
    } else {
        return d_pOperator->createInputVector();
    }
}
std::shared_ptr<AMP::LinearAlgebra::Vector> MemorySpaceMigrationOperator::createOutputVector() const
{
    if ( d_migrate_data ) {
        return AMP::LinearAlgebra::createVector( d_outputVec, d_memory_location );
    } else {
        return d_pOperator->createOutputVector();
    }
}
std::shared_ptr<AMP::LinearAlgebra::VectorSelector>
MemorySpaceMigrationOperator::selectOutputVector() const
{
    return d_pOperator->selectOutputVector();
}
std::shared_ptr<AMP::LinearAlgebra::VectorSelector>
MemorySpaceMigrationOperator::selectInputVector() const
{
    return d_pOperator->selectInputVector();
}
bool MemorySpaceMigrationOperator::isValidVector(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> v )
{
    if ( d_migrate_data ) {
        d_inputVec->copyVector( v );
        return d_pOperator->isValidVector( d_inputVec );
    } else {
        return d_pOperator->isValidVector( v );
    }
}
void MemorySpaceMigrationOperator::makeConsistent( std::shared_ptr<AMP::LinearAlgebra::Vector> vec )
{
    if ( d_migrate_data ) {
        d_inputVec->copyVector( vec );
        d_pOperator->makeConsistent( d_inputVec );
        vec->getVectorData()->copyGhostValues( *( d_inputVec->getVectorData() ) );
    } else {
        d_pOperator->makeConsistent( vec );
    }
}
void MemorySpaceMigrationOperator::reInitializeVector(
    std::shared_ptr<AMP::LinearAlgebra::Vector> v )
{
    if ( d_migrate_data ) {
        d_inputVec->copyVector( v );
        d_pOperator->reInitializeVector( d_inputVec );
        v->copyVector( d_inputVec );
    } else {
        d_pOperator->reInitializeVector( v );
    }
}

} // namespace AMP::Operator
