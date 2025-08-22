#include "RadiationDiffusionFDBEWrappers.h"


/** --------------------------------------------------------------------------------
 *                        Implementation of BERadDifOpPJacData 
 *  ------------------------------------------------------------------------------ */
/** Overwrite the incoming data to BE-based data. That is, overwrite
 *  [ d_E 0   ] + [ diag(r_EE) diag(r_ET) ]
 *  [ 0   d_T ] + [ diag(r_TE) diag(r_TT) ]
 * to create
 *  ([I 0]         [ d_E 0   ])         [ diag(r_EE) diag(r_ET) ]
 *  ([0 I] + gamma*[ 0   d_T ]) + gamma*[ diag(r_TE) diag(r_TT) ]
 *      ==
 *  [ d_E_BE  0    ]   [ diag(r_EE_BE) diag(r_ET_BE) ]
 *  [ 0      d_T_BE] + [ diag(r_TE_BE) diag(r_TT_BE) ]
 */
BERadDifOpPJacData::BERadDifOpPJacData( std::shared_ptr<RadDifOpPJacData> data, double gamma ) {
    AMP_INSIST( data, "Non-null data required" );

    // Unpack reaction vectors
    r_EE_BE = data->get_r_EE();
    r_ET_BE = data->get_r_ET();
    r_TE_BE = data->get_r_TE();
    r_TT_BE = data->get_r_TT();
    // Scale them by gamma
    r_EE_BE->scale( gamma );
    r_ET_BE->scale( gamma );
    r_TE_BE->scale( gamma );
    r_TT_BE->scale( gamma );
    // Make everything consistent
    r_EE_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_ET_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_TE_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    r_TT_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Unpack diffusion matrices
    d_E_BE  = data->get_d_E();
    d_T_BE  = data->get_d_T();
    // Scale them by gamma
    d_E_BE->scale( gamma );
    d_T_BE->scale( gamma );
    // Add identity perturbations 
    auto DOFMan_E = d_E_BE->getRightDOFManager(  );
    for ( auto dof = DOFMan_E->beginDOF(); dof != DOFMan_E->endDOF(); dof++ ) {
        d_E_BE->addValueByGlobalID( dof, dof, 1.0 );
    }
    auto DOFMan_T = d_T_BE->getRightDOFManager(  );
    for ( auto dof = DOFMan_T->beginDOF(); dof != DOFMan_T->endDOF(); dof++ ) {
        d_T_BE->addValueByGlobalID( dof, dof, 1.0 );
    }
    // Make everything consistent
    d_E_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    d_T_BE->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
};


/** --------------------------------------------------------------------------------
 *                        Implementation of BERadDifOp 
 * ------------------------------------------------------------------------------ */
BERadDifOp::BERadDifOp( std::shared_ptr<const AMP::Operator::OperatorParameters> params_ ) : 
        AMP::Operator::Operator( params_ ) 
{
    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "BERadDifOp::BERadDifOp() " << std::endl;   
    }

    // Create my RadDifOp
    d_RadDifOp = std::make_shared<RadDifOp>( params_ );
}

void BERadDifOp::setGamma( AMP::Scalar gamma_ )
{ 
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BERadDifOp::setGamma() " << std::endl;
    }
    d_gamma = double( gamma_ );
}

void BERadDifOp::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
            AMP::LinearAlgebra::Vector::shared_ptr r )
{
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BERadDifOp::apply() " << std::endl;  
    }
    AMP_INSIST( d_RadDifOp, "RadDifOp not set!" );
    d_RadDifOp->apply( u_in, r );
    r->axpby(1.0, d_gamma, *u_in); // r <- 1.0*u + d_gamma * r
}

std::shared_ptr<AMP::Operator::OperatorParameters> BERadDifOp::getJacobianParameters( AMP::LinearAlgebra::Vector::const_shared_ptr u_in ) {

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BERadDifOp::getJacobianParameters() " << std::endl;  
    }

    // Get RadDifOp parameters
    auto jacOpParams = d_RadDifOp->getJacobianParameters( u_in );

    // Update the name in these parameters from "RadDifOpPJac"
    jacOpParams->d_db->setDefaultAddKeyBehavior( AMP::Database::Check::Overwrite, true );
    jacOpParams->d_db->putScalar<std::string>( "name", "BERadDifOpPJac" );

    // Set the time-step size
    jacOpParams->d_db->putScalar<double>( "gamma", d_gamma );

    return jacOpParams;
}


/** --------------------------------------------------------------------------------
 *                        Implementation of BERadDifOpPJac 
 *  ----------------------------------------------------------------------------- */
BERadDifOpPJac::BERadDifOpPJac( std::shared_ptr<AMP::Operator::OperatorParameters> params ) : AMP::Operator::LinearOperator( params ) {

    if ( d_iDebugPrintInfoLevel > 0 ) {
        AMP::pout << "BERadDifOpPJac::BERadDifOpPJac() " << std::endl;
    }

    // Create my RadDifOpPJac
    d_RadDifOpPJac = std::make_shared<RadDifOpPJac>( params );

    // Set the time step-size
    d_gamma = params->d_db->getScalar<double>( "gamma" );   

    // Set data for needed for storing block 2x2 matrix
    setData();

    // 
    //setLocalPermutationArrays();
}

void BERadDifOpPJac::apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r )
{
    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BERadDifOpPJac::apply() " << std::endl; 
    }
    AMP_INSIST( d_RadDifOpPJac, "RadDifOpPJac not set!" );

    // We have modified the Jacobian data of our RadDifOpPJac operator in such a way that its apply will in fact be an apply of a BERadDifOpPJac, but we need to explicitly let it know that we really do intend to do an apply with the current state of its data
    d_RadDifOpPJac->applyWithOverwrittenDataIsValid();
    d_RadDifOpPJac->apply( u_in, r );

    // d_RadDifOpPJac->apply( u_in, r );
    // r->axpby (1.0, d_gamma, *u_in); // r <- 1.0*u + d_gamma * r

    #if 0
    /* Test apply to check nodal ordering...
        Seems to be working in 1D and 2D....
    */
    auto r_test = d_RadDifOpPJac->createInputVector();
    applyNodalToVariableVectors( u_in, r_test );
    std::cout << "+++Apply comparison: Variable vs nodal ordering\n";
    auto dof = d_RadDifOpPJac->d_RadDifOp->d_multiDOFMan; 
    for ( auto i = dof->beginDOF(); i != dof->endDOF(); i++ ) {
        auto ri = r->getValueByGlobalID( i );
        auto riTemp = r_test->getValueByGlobalID( i );
        std::cout << "dof=" << i << ": dr=" << 1e-16+std::fabs(ri - riTemp)/std::fabs( ri ) << ",\tr=" << ri << ",rt=" << riTemp << "\n";
    }
    //AMP_ERROR( "Halt...." );
    #endif
}

std::shared_ptr<AMP::LinearAlgebra::Vector> BERadDifOpPJac::createInputVector() const { return d_RadDifOpPJac->createInputVector( ); };

void BERadDifOpPJac::reset(std::shared_ptr<const AMP::Operator::OperatorParameters> params) {

    if ( d_iDebugPrintInfoLevel > 1 )
        AMP::pout << "BERadDifOpPJac::reset()  " << std::endl;

    // Calls with empty parameters are to be ignored
    if ( !params ) { 
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "  Called with empty parameters...  not resetting anything" << std::endl;
        return; 
    }

    AMP::Operator::LinearOperator::reset( params );

    // Reset my RadDifOpPJac
    d_RadDifOpPJac->reset( params );

    // Reset time step-size
    d_gamma = params->d_db->getScalar<double>( "gamma" );  

    // Reset my data
    setData( );

    // Reset my monolithic Jacobian
    //d_JNodal = createMonolithicJac( );
}  