#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include <stdexcept>


namespace AMP::Operator {


NonlinearBVPOperator::NonlinearBVPOperator( std::shared_ptr<const OperatorParameters> params )
    : Operator( params )
{
    auto parmeters = std::dynamic_pointer_cast<const BVPOperatorParameters>( params );
    AMP_ASSERT( parmeters );
    d_volumeOperator   = parmeters->d_volumeOperator;
    d_boundaryOperator = parmeters->d_boundaryOperator;
    d_Mesh             = d_volumeOperator->getMesh();
}

void NonlinearBVPOperator::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u,
                                  AMP::LinearAlgebra::Vector::shared_ptr r )
{
    PROFILE( "apply" );

    if ( u )
        AMP_ASSERT( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );

    AMP_INSIST( r, "NULL Residual Vector" );

    auto rInternal = subsetOutputVector( r );
    AMP_INSIST( rInternal, "NULL Internal Residual Vector" );

    if ( d_iDebugPrintInfoLevel > 3 ) {
        AMP::pout << "L2 Norm of u in NonlinearBVPOperator volumeOperator::apply is : "
                  << u->L2Norm() << std::endl;
    }

    d_volumeOperator->apply( u, r );

    if ( d_iDebugPrintInfoLevel > 3 ) {
        AMP::pout << "L2 Norm of r in NonlinearBVPOperator volumeOperator::apply is : "
                  << r->L2Norm() << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 3 ) {
        AMP::pout << "L2 Norm of rInternal in NonlinearBVPOperator volumeOperator::apply is : "
                  << rInternal->L2Norm() << std::endl;
    }

    d_boundaryOperator->apply( u, r );

    if ( d_iDebugPrintInfoLevel > 3 ) {
        AMP::pout << "L2 Norm of r in NonlinearBVPOperator boundaryOperator::apply is : "
                  << r->L2Norm() << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "L2 Norm of output of NonlinearBVPOperator :: apply is : "
                  << rInternal->L2Norm() << std::endl;
    }
}

void NonlinearBVPOperator::reset( std::shared_ptr<const OperatorParameters> params )
{
    PROFILE( "reset" );
    AMP_ASSERT( params );

    auto inParams = std::dynamic_pointer_cast<const BVPOperatorParameters>( params );

    AMP_INSIST( ( inParams ), "NonlinearBVPOperator :: reset Null parameter" );

    d_volumeOperator->reset( inParams->d_volumeOperatorParams );
    d_boundaryOperator->reset( inParams->d_boundaryOperatorParams );
}

std::shared_ptr<OperatorParameters>
NonlinearBVPOperator::getParameters( const std::string &type,
                                     std::shared_ptr<const AMP::LinearAlgebra::Vector> u,
                                     std::shared_ptr<OperatorParameters> params )
{
    PROFILE( "getParameters" );
    auto db = std::make_shared<Database>();
    db->putScalar( "name", "LinearBVPOperator" );
    Operator::setMemoryAndBackendParameters( db );
    auto outParams                      = std::make_shared<BVPOperatorParameters>( db );
    outParams->d_Mesh                   = d_Mesh;
    outParams->d_volumeOperatorParams   = d_volumeOperator->getParameters( type, u, params );
    outParams->d_boundaryOperatorParams = d_boundaryOperator->getParameters( type, u, params );
    return outParams;
}

void NonlinearBVPOperator::modifyRHSvector( AMP::LinearAlgebra::Vector::shared_ptr rhs )
{
    getBoundaryOperator()->addRHScorrection( rhs );
    getBoundaryOperator()->setRHScorrection( rhs );
}

void NonlinearBVPOperator::modifyInitialSolutionVector( AMP::LinearAlgebra::Vector::shared_ptr sol )
{
    getBoundaryOperator()->modifyInitialSolutionVector( sol );
}


} // namespace AMP::Operator
