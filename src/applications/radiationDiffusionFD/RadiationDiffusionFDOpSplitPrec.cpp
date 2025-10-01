#include "AMP/applications/radiationDiffusionFD/RadiationDiffusionFDOpSplitPrec.h"

namespace AMP::Solver {


BDFRadDifOpPJacOpSplitPrec::BDFRadDifOpPJacOpSplitPrec(
    std::shared_ptr<AMP::Solver::SolverStrategyParameters> params )
    : SolverStrategy( params )
{

    if ( d_iDebugPrintInfoLevel > 1 )
        AMP::pout << "BDFRadDifOpPJacOpSplitPrec::BDFRadDifOpPJacOpSplitPrec() " << std::endl;

    // Ensure DiffusionBlocks Database was parsed.
    AMP_INSIST( d_db->getDatabase( "DiffusionBlocks" ),
                "Solver requires a 'DiffusionBlocks' database" );
};


void BDFRadDifOpPJacOpSplitPrec::reset( std::shared_ptr<AMP::Solver::SolverStrategyParameters> )
{

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "BDFRadDifOpPJacOpSplitPrec::reset() " << std::endl;
    }

    // Reset solvers for diffusion blocks
    setDiffusionSolvers();
}


void BDFRadDifOpPJacOpSplitPrec::scalar2x2Solve(
    double a, double b, double c, double d, double e, double f, double &x, double &y ) const
{
    double det = a * d - b * c;
    AMP_INSIST( fabs( det ) > SINGULAR_TOL, "2x2 linear system is singular" );
    x = ( e * d - b * f ) / det;
    y = ( a * f - e * c ) / det;
}


void BDFRadDifOpPJacOpSplitPrec::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> bET_,
                                        std::shared_ptr<AMP::LinearAlgebra::Vector> ET_ )
{

    PROFILE( "BDFRadDifOpPJacOpSplitPrec::apply" );

    // A zero initial guess is hard-coded to avoid an apply of A for computing the initial residual
    AMP_INSIST( d_bUseZeroInitialGuess, "Zero initial guess is hard coded!" );

    // Current implementation only supports a fixed number of iterations, ignores tolerances
    AMP_INSIST( d_dAbsoluteTolerance == 0.0 && d_dRelativeTolerance == 0.0,
                "Non-zero tolerances not implemented; only fixed number of iterations" );

    // Get underlying operator
    auto op = std::dynamic_pointer_cast<AMP::Operator::BDFRadDifOpPJac>( this->getOperator() );
    AMP_INSIST( op, "Operator must be of BDFRadDifOpPJac type" );

    // Ensure diffusion solvers exists
    AMP_INSIST( d_difSolverE && d_difSolverT, "Solvers for diffusion blocks are null" );

    // Unpack components of input vectors
    auto bET = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( bET_ );
    auto ET  = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( ET_ );
    AMP_INSIST( bET, "bET downcast to MultiVector unsuccessful" );
    AMP_INSIST( ET, "ET downcast to MultiVector unsuccessful" );
    auto bE = bET->getVector( 0 );
    auto bT = bET->getVector( 1 );
    auto E  = ET->getVector( 0 );
    auto T  = ET->getVector( 1 );

    // Create a residual vector and upack it
    auto rET_ = op->createInputVector();
    auto rET  = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( rET_ );
    AMP_INSIST( rET, "rET downcast to MultiVector unsuccessful" );
    auto rE = rET->getVector( 0 );
    auto rT = rET->getVector( 1 );
    // Create a correction vector and upack it
    auto dET_ = op->createInputVector();
    auto dET  = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( dET_ );
    AMP_INSIST( dET, "dET downcast to MultiVector unsuccessful" );
    auto dE = dET->getVector( 0 );
    auto dT = dET->getVector( 1 );

    // Iterate the stationary linear iteration from zero initial guess
    ET->zero();
    for ( auto iter = 0; iter < d_iMaxIterations; iter++ ) {

        // Compute residual
        if ( iter == 0 ) {
            rET->copyVector( bET ); // Zero initial iterate means r0 = b - A*x0 = b
        } else {
            op->residual( bET, ET, rET );
        }
        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "BDFRadDifOpPJacOpSplitPrec::apply(): iteration " << iter << ":"
                      << ", residual norm=" << rET->L2Norm() << std::endl;
        }

        // Solve P_dif * [dE, dT] = [rE, rT], with residuals r
        diffusionSolve( rE, rT, dE, dT );
        // Solve P_react * [rE, rT] = [dE, dT] (here we abuse variable names)
        reactionSolve( dE, dT, rE, rT );

        // Update solution: ET <- ET + rET (abusing variable names)
        ET->axpby( 1.0, 1.0, *rET ); // ET <- 1.0*ET + 1.0*rET
        ET->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }

    // Display final residual; note that this residual calculation is unnecessary because the final
    // iterate is calculated already, hence the flag
    if ( d_bComputeResidual ) {
        op->residual( bET, ET, rET );
        AMP::pout << "BDFRadDifOpPJacOpSplitPrec::apply(): final residual norm=" << rET->L2Norm()
                  << std::endl;
    }
}


void BDFRadDifOpPJacOpSplitPrec::setDiffusionSolvers()
{

    PROFILE( "BDFRadDifOpPJacOpSplitPrec::setDiffusionSolvers" );

    // Get underlying operator
    auto op = std::dynamic_pointer_cast<AMP::Operator::BDFRadDifOpPJac>( this->getOperator() );
    AMP_INSIST( op, "Operator must be of BDFRadDifOpPJac type" );

    // Wrap diffusion matrices as LinearOperators
    // E
    auto E_db    = AMP::Database::create( "name", "EOperator", "print_info_level", 0 );
    auto EParams = std::make_shared<AMP::Operator::OperatorParameters>( std::move( E_db ) );
    auto E       = std::make_shared<AMP::Operator::LinearOperator>( EParams );
    AMP_INSIST( op->d_data->d_E_BDF, "E diffusion matrix is null" );
    E->setMatrix( op->d_data->d_E_BDF );

    // T
    auto T_db    = AMP::Database::create( "name", "TOperator", "print_info_level", 0 );
    auto TParams = std::make_shared<AMP::Operator::OperatorParameters>( std::move( T_db ) );
    auto T       = std::make_shared<AMP::Operator::LinearOperator>( TParams );
    T->setMatrix( op->d_data->d_T_BDF );
    AMP_INSIST( T->getMatrix(), "T diffusion matrix is null" );

    // Create solver parameters
    auto comm      = op->getMesh()->getComm();
    auto solver_db = d_db->getDatabase( "DiffusionBlocks" );
    // E
    auto ESolverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( solver_db );
    ESolverParams->d_pOperator = E;
    ESolverParams->d_comm      = comm;
    // T
    auto TSolverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( solver_db );
    TSolverParams->d_pOperator = T;
    TSolverParams->d_comm      = comm;

    // Create solvers from Factories
    d_difSolverE = AMP::Solver::SolverFactory::create( ESolverParams );
    d_difSolverT = AMP::Solver::SolverFactory::create( TSolverParams );
    // Ensure zero initial guess is used
    d_difSolverE->setZeroInitialGuess( true );
    d_difSolverT->setZeroInitialGuess( true );
}


void BDFRadDifOpPJacOpSplitPrec::diffusionSolve(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> bE,
    std::shared_ptr<const AMP::LinearAlgebra::Vector> bT,
    std::shared_ptr<AMP::LinearAlgebra::Vector> E,
    std::shared_ptr<AMP::LinearAlgebra::Vector> T ) const
{

    PROFILE( "BDFRadDifOpPJacOpSplitPrec::diffusionSolve" );

    AMP_INSIST( d_difSolverE, "Null diffusion solver for E block" );
    AMP_INSIST( d_difSolverT, "Null diffusion solver for T block" );
    d_difSolverE->apply( bE, E );
    d_difSolverT->apply( bT, T );
}


void BDFRadDifOpPJacOpSplitPrec::reactionSolve(
    std::shared_ptr<const AMP::LinearAlgebra::Vector> bE,
    std::shared_ptr<const AMP::LinearAlgebra::Vector> bT,
    std::shared_ptr<AMP::LinearAlgebra::Vector> E,
    std::shared_ptr<AMP::LinearAlgebra::Vector> T )
{

    PROFILE( "BDFRadDifOpPJacOpSplitPrec::reactionSolve" );

    // Get underlying operator
    auto op = std::dynamic_pointer_cast<AMP::Operator::BDFRadDifOpPJac>( this->getOperator() );
    AMP_INSIST( op, "Operator must be of BDFRadDifOpPJac type" );

    // Get Operator's data
    auto data = op->d_data;

    // Constants used to describe 2x2 linear systems
    double a, b, c, d, e, f, x, y;

    // Iterate through all local rows
    auto DOFMan = data->r_EE_BDF->getDOFManager();
    for ( auto dof = DOFMan->beginDOF(); dof != DOFMan->endDOF(); dof++ ) {
        // LHS matrix
        a = data->r_EE_BDF->getValueByGlobalID<double>( dof ) + 1.0; // Add identity
        b = data->r_ET_BDF->getValueByGlobalID<double>( dof );
        c = data->r_TE_BDF->getValueByGlobalID<double>( dof );
        d = data->r_TT_BDF->getValueByGlobalID<double>( dof ) + 1.0; // Add identity
        // RHS
        e = bE->getValueByGlobalID<double>( dof );
        f = bT->getValueByGlobalID<double>( dof );
        // Solve linear system
        scalar2x2Solve( a, b, c, d, e, f, x, y );
        // Pack results into solution vectors
        E->setValueByGlobalID<double>( dof, x );
        T->setValueByGlobalID<double>( dof, y );
    }
}

} // namespace AMP::Solver