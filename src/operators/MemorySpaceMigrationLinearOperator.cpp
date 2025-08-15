#include "AMP/operators/MemorySpaceMigrationLinearOperator.h"
#include "AMP/vectors/VectorBuilder.h"

namespace AMP::Operator {

MemorySpaceMigrationLinearOperator::MemorySpaceMigrationLinearOperator(
    std::shared_ptr<const OperatorParameters> params )
    : LinearOperator( params )
{
    AMP_INSIST(
        params->d_pOperator,
        "MemorySpaceMigrationLinearOperator is required to be initialized with another Operator" );
    d_pOperator    = params->d_pOperator;
    d_migrate_data = ( d_memory_location != d_pOperator->getMemoryLocation() );
    if ( d_migrate_data ) {
        d_inputVec  = d_pOperator->createInputVector();
        d_outputVec = d_pOperator->createOutputVector();
    }
}

void MemorySpaceMigrationLinearOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    d_pOperator->reset( params );
}
void MemorySpaceMigrationLinearOperator::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                                                std::shared_ptr<AMP::LinearAlgebra::Vector> f )
{
    if ( d_migrate_data ) {
        d_inputVec->copyVector( u );
        d_outputVec->copyVector( f );
        d_inputVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->apply( d_inputVec, d_outputVec );
        f->copyVector( d_outputVec );
    } else {
        d_pOperator->apply( u, f );
    }
}
void MemorySpaceMigrationLinearOperator::residual(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
    std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
    std::shared_ptr<AMP::LinearAlgebra::Vector> r )
{
    if ( d_migrate_data ) {
        d_inputVec->copyVector( u );
        d_outputVec->copyVector( f );
        if ( !d_resVec )
            d_resVec = d_pOperator->createOutputVector();
        d_inputVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        d_pOperator->residual( d_outputVec, d_inputVec, d_resVec );
        r->copyVector( d_resVec );
    } else {
        d_pOperator->residual( f, u, r );
    }
}
std::shared_ptr<OperatorParameters> MemorySpaceMigrationLinearOperator::getParameters(
    const std::string &type,
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
std::shared_ptr<AMP::LinearAlgebra::Matrix> MemorySpaceMigrationLinearOperator::getMatrix()
{
    return std::dynamic_pointer_cast<LinearOperator>( d_pOperator )->getMatrix();
}

void MemorySpaceMigrationLinearOperator::setDebugPrintInfoLevel( int level )
{
    return d_pOperator->setDebugPrintInfoLevel( level );
}
std::shared_ptr<AMP::LinearAlgebra::Variable>
MemorySpaceMigrationLinearOperator::getOutputVariable() const
{
    return d_pOperator->getOutputVariable();
}
std::shared_ptr<AMP::LinearAlgebra::Variable>
MemorySpaceMigrationLinearOperator::getInputVariable() const
{
    return d_pOperator->getInputVariable();
}
std::shared_ptr<AMP::LinearAlgebra::Vector>
MemorySpaceMigrationLinearOperator::createInputVector() const
{
    if ( d_migrate_data ) {
        return AMP::LinearAlgebra::createVector( d_inputVec, d_memory_location );
    } else {
        return d_pOperator->createInputVector();
    }
}
std::shared_ptr<AMP::LinearAlgebra::Vector>
MemorySpaceMigrationLinearOperator::createOutputVector() const
{
    if ( d_migrate_data ) {
        return AMP::LinearAlgebra::createVector( d_outputVec, d_memory_location );
    } else {
        return d_pOperator->createOutputVector();
    }
}
std::shared_ptr<AMP::LinearAlgebra::VectorSelector>
MemorySpaceMigrationLinearOperator::selectOutputVector() const
{
    return d_pOperator->selectOutputVector();
}
std::shared_ptr<AMP::LinearAlgebra::VectorSelector>
MemorySpaceMigrationLinearOperator::selectInputVector() const
{
    return d_pOperator->selectInputVector();
}
bool MemorySpaceMigrationLinearOperator::isValidVector(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> v )
{
    if ( d_migrate_data ) {
        d_inputVec->copyVector( v );
        return d_pOperator->isValidVector( d_inputVec );
    } else {
        return d_pOperator->isValidVector( v );
    }
}
void MemorySpaceMigrationLinearOperator::makeConsistent(
    std::shared_ptr<AMP::LinearAlgebra::Vector> vec )
{
    if ( d_migrate_data ) {
        d_inputVec->copyVector( vec );
        d_pOperator->makeConsistent( d_inputVec );
        vec->getVectorData()->copyGhostValues( *( d_inputVec->getVectorData() ) );
    } else {
        d_pOperator->makeConsistent( vec );
    }
}
void MemorySpaceMigrationLinearOperator::reInitializeVector(
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
