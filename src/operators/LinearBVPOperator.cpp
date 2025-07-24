#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/operators/boundary/ColumnBoundaryOperator.h"
#include "AMP/operators/boundary/LinearBoundaryOperatorParameters.h"
#include "AMP/utils/Utilities.h"

#include <stdexcept>


namespace AMP::Operator {


template<class TYPE>
static std::shared_ptr<TYPE> createOperator( std::shared_ptr<OperatorParameters> params )
{
    std::shared_ptr<Operator> op = OperatorFactory::create( params );
    return std::dynamic_pointer_cast<TYPE>( op );
}


static void addMatrix( std::shared_ptr<OperatorParameters> p,
                       std::shared_ptr<AMP::LinearAlgebra::Matrix> mat )
{
    auto linearOpParams = std::dynamic_pointer_cast<LinearBoundaryOperatorParameters>( p );
    auto columnOpParams = std::dynamic_pointer_cast<ColumnBoundaryOperatorParameters>( p );
    if ( linearOpParams )
        linearOpParams->d_inputMatrix = mat;
    if ( columnOpParams ) {
        for ( auto p2 : columnOpParams->d_OperatorParameters )
            addMatrix( p2, mat );
    }
}


LinearBVPOperator::LinearBVPOperator( std::shared_ptr<const OperatorParameters> inParams )
    : LinearOperator( inParams )
{
    AMP_ASSERT( inParams );
    auto params        = std::dynamic_pointer_cast<const BVPOperatorParameters>( inParams );
    d_volumeOperator   = std::dynamic_pointer_cast<LinearOperator>( params->d_volumeOperator );
    d_boundaryOperator = params->d_boundaryOperator;
    if ( !d_volumeOperator && params->d_volumeOperatorParams )
        d_volumeOperator = createOperator<LinearOperator>( params->d_volumeOperatorParams );
    if ( !d_boundaryOperator && params->d_boundaryOperatorParams ) {
        addMatrix( params->d_boundaryOperatorParams, d_volumeOperator->getMatrix() );
        d_boundaryOperator = createOperator<BoundaryOperator>( params->d_boundaryOperatorParams );
    }
    if ( d_volumeOperator ) {
        d_Mesh   = d_volumeOperator->getMesh();
        d_matrix = d_volumeOperator->getMatrix();
    }
}


void LinearBVPOperator::reset( std::shared_ptr<const OperatorParameters> inParams )
{
    if ( inParams ) {
        auto params = std::dynamic_pointer_cast<const BVPOperatorParameters>( inParams );

        AMP_INSIST( params, "LinearBVPOperator :: reset Null parameter" );

        d_volumeOperator->reset( params->d_volumeOperatorParams );

        // first case - single linear boundary operator parameter object
        // This logic does not work with NeumannVectorCorrection boundary operator.
        // As Neumann does not do a matrix correction and its params is not derived
        //    from LinearBoundaryOperatorParameters
        auto linearBoundaryParams = std::dynamic_pointer_cast<LinearBoundaryOperatorParameters>(
            std::const_pointer_cast<OperatorParameters>( params->d_boundaryOperatorParams ) );

        if ( linearBoundaryParams ) {
            linearBoundaryParams->d_inputMatrix = d_volumeOperator->getMatrix();
            d_boundaryOperator->reset( linearBoundaryParams );
        } else {
            auto columnBoundaryParams =
                std::dynamic_pointer_cast<const ColumnBoundaryOperatorParameters>(
                    params->d_boundaryOperatorParams );

            AMP_ASSERT( columnBoundaryParams );

            for ( auto cparams : columnBoundaryParams->d_OperatorParameters ) {

                auto linearColBoundaryParams =
                    std::dynamic_pointer_cast<LinearBoundaryOperatorParameters>(
                        std::const_pointer_cast<OperatorParameters>( cparams ) );
                if ( linearColBoundaryParams ) {
                    linearColBoundaryParams->d_inputMatrix = d_volumeOperator->getMatrix();
                }
            }
            d_boundaryOperator->reset( columnBoundaryParams );
        }

        d_matrix = d_volumeOperator->getMatrix();
    }
}


void LinearBVPOperator::modifyRHSvector( AMP::LinearAlgebra::Vector::shared_ptr rhs )
{
    this->getBoundaryOperator()->addRHScorrection( rhs );
    this->getBoundaryOperator()->setRHScorrection( rhs );
}


} // namespace AMP::Operator
