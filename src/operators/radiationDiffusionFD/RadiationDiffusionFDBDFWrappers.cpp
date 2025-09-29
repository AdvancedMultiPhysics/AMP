#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDBDFWrappers.h"

namespace AMP::Operator {

/** --------------------------------------------------------------------------------
 *                        Implementation of BDFRadDifOpPJacData
 *  ------------------------------------------------------------------------------ */
/** Overwrite the incoming data to BDF-based data. That is, overwrite
 *  [ d_E 0   ] + [ diag(r_EE) diag(r_ET) ]
 *  [ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]
 * to create
 *  ([I 0]         [ d_E 0   ])         [ diag(r_EE) diag(r_ET) ]
 *  ([0 I] + gamma*[ 0   d_T ]) + gamma*[ diag(r_TE) diag(r_TT) ]
 *      ==
 *  [ d_E_BDF  0    ]   [ diag(r_EE_BDF) diag(r_ET_BDF) ]
 *  [ 0      d_T_BDF] + [ diag(r_TE_BDF) diag(r_TT_BDF) ]
 */
BDFRadDifOpPJacData::BDFRadDifOpPJacData( std::shared_ptr<RadDifOpPJacData> data, double gamma )
{
    AMP_INSIST( data, "Non-null data required" );

    // Unpack reaction vectors
    r_EE_BDF = data->get_r_EE();
    r_ET_BDF = data->get_r_ET();
    r_TE_BDF = data->get_r_TE();
    r_TT_BDF = data->get_r_TT();
    // Scale them by gamma
    r_EE_BDF->scale( gamma );
    r_ET_BDF->scale( gamma );
    r_TE_BDF->scale( gamma );
    r_TT_BDF->scale( gamma );
    // Make consistent
    r_EE_BDF->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_ET_BDF->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_TE_BDF->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_TT_BDF->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Unpack diffusion matrices
    d_E_BDF = data->get_d_E();
    d_T_BDF = data->get_d_T();
    // Scale them by gamma
    d_E_BDF->scale( gamma );
    d_T_BDF->scale( gamma );
    // Add identity perturbations
    auto DOFMan_E = d_E_BDF->getRightDOFManager();
    for ( auto dof = DOFMan_E->beginDOF(); dof != DOFMan_E->endDOF(); dof++ ) {
        d_E_BDF->addValueByGlobalID( dof, dof, 1.0 );
    }
    auto DOFMan_T = d_T_BDF->getRightDOFManager();
    for ( auto dof = DOFMan_T->beginDOF(); dof != DOFMan_T->endDOF(); dof++ ) {
        d_T_BDF->addValueByGlobalID( dof, dof, 1.0 );
    }
    // Make consistent
    d_E_BDF->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    d_T_BDF->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
};


/** --------------------------------------------------------------------------------
 *                        Implementation of BDFRadDifOp
 * ------------------------------------------------------------------------------ */
BDFRadDifOp::BDFRadDifOp( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ )
    : AMP::Operator::Operator( params_ )
{
    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "BDFRadDifOp::BDFRadDifOp() " << std::endl;
    }

    // Create my RadDifOp
    d_RadDifOp = std::make_shared<RadDifOp>( params_ );
}

void BDFRadDifOp::setGamma( AMP::Scalar gamma_ )
{
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BDFRadDifOp::setGamma() " << std::endl;
    }
    d_gamma = double( gamma_ );
}

void BDFRadDifOp::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                         AMP::LinearAlgebra::Vector::shared_ptr r )
{
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BDFRadDifOp::apply() " << std::endl;
    }
    AMP_INSIST( d_RadDifOp, "RadDifOp not set!" );


    AMP::LinearAlgebra::Vector::const_shared_ptr u;
    // Scale components of incoming vector if required
    if ( d_pSolutionScaling ) {
        AMP_ASSERT( d_pFunctionScaling );
        if ( !d_pScratchSolVector ) {
            d_pScratchSolVector = u_in->clone();
        }
        d_pScratchSolVector->multiply( *u_in, *d_pSolutionScaling );
        d_pScratchSolVector->makeConsistent();
        u = d_pScratchSolVector;
    } else {
        u = u_in;
    }

    d_RadDifOp->apply( u_in, r );
    r->axpby( 1.0, d_gamma, *u_in ); // r <- 1.0*u + d_gamma * r

    // Scale components of outgoing vector if required
    if ( d_pFunctionScaling ) {
        r->divide( *r, *d_pFunctionScaling );
    }
}

void BDFRadDifOp::setComponentScalings( std::shared_ptr<AMP::LinearAlgebra::Vector> s,
                                        std::shared_ptr<AMP::LinearAlgebra::Vector> f )
{
    d_pSolutionScaling = s;
    d_pFunctionScaling = f;
}

std::shared_ptr<AMP::Operator::OperatorParameters>
BDFRadDifOp::getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in )
{

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BDFRadDifOp::getJacobianParameters() " << std::endl;
    }

    // Get RadDifOp parameters
    auto jacOpParams = d_RadDifOp->getParameters( "Jacobian", u_in );

    // Update the name in these parameters from "RadDifOpPJac"
    jacOpParams->d_db->setDefaultAddKeyBehavior( AMP::Database::Check::Overwrite, true );
    jacOpParams->d_db->putScalar<std::string>( "name", "BDFRadDifOpPJac" );

    // Set the time-step size
    jacOpParams->d_db->putScalar<double>( "gamma", d_gamma );

    return jacOpParams;
}


/** --------------------------------------------------------------------------------
 *                        Implementation of BDFRadDifOpPJac
 *  ----------------------------------------------------------------------------- */
BDFRadDifOpPJac::BDFRadDifOpPJac( std::shared_ptr<AMP::Operator::OperatorParameters> params )
    : AMP::Operator::LinearOperator( params )
{

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "BDFRadDifOpPJac::BDFRadDifOpPJac() " << std::endl;
    }

    // Create my RadDifOpPJac
    d_RadDifOpPJac = std::make_shared<RadDifOpPJac>( params );

    // Set the time step-size
    d_gamma = params->d_db->getScalar<double>( "gamma" );

    // Set data for needed for storing block 2x2 matrix
    setData();
}

void BDFRadDifOpPJac::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                             AMP::LinearAlgebra::Vector::shared_ptr r )
{
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BDFRadDifOpPJac::apply() " << std::endl;
    }
    AMP_INSIST( d_RadDifOpPJac, "RadDifOpPJac not set!" );

    // We have modified the Jacobian data of our RadDifOpPJac operator in such a way that its apply
    // will in fact be an apply of a BDFRadDifOpPJac, but we need to explicitly let it know that we
    // really do intend to do an apply with the current state of its data
    d_RadDifOpPJac->applyWithOverwrittenDataIsValid();
    d_RadDifOpPJac->apply( u_in, r );
}


std::shared_ptr<AMP::LinearAlgebra::Vector> BDFRadDifOpPJac::createInputVector() const
{
    return d_RadDifOpPJac->createInputVector();
};


void BDFRadDifOpPJac::reset( std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{

    if ( d_iDebugPrintInfoLevel > 1 )
        AMP::pout << "BDFRadDifOpPJac::reset()  " << std::endl;

    // Calls with empty parameters are to be ignored
    if ( !params ) {
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "Called with empty parameters...  not resetting anything" << std::endl;
        return;
    }

    AMP::Operator::LinearOperator::reset( params );

    // Reset my RadDifOpPJac
    d_RadDifOpPJac->reset( params );

    // Reset time step-size
    d_gamma = params->d_db->getScalar<double>( "gamma" );

    // Reset my data
    setData();
}

void BDFRadDifOpPJac::setData()
{
    d_data =
        std::make_shared<AMP::Operator::BDFRadDifOpPJacData>( d_RadDifOpPJac->d_data, d_gamma );
}

} // namespace AMP::Operator
