#ifndef RAD_DIF_OP_SPLIT_PREC
#define RAD_DIF_OP_SPLIT_PREC

#include "AMP/IO/PIO.h"
#include "AMP/IO/AsciiWriter.h"
#include "AMP/utils/AMPManager.h"

#include "AMP/vectors/CommunicationList.h"
#include "AMP/matrices/petsc/NativePetscMatrix.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/vectors/MultiVariable.h"
#include "AMP/vectors/data/VectorData.h"
#include "AMP/vectors/data/VectorDataNull.h"
#include "AMP/vectors/operations/default/VectorOperationsDefault.h"
#include "AMP/vectors/VectorBuilder.h"

#include "AMP/discretization/boxMeshDOFManager.h"
#include "AMP/discretization/MultiDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshID.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/mesh/MeshElement.h"
#include "AMP/mesh/structured/BoxMesh.h"

#include "AMP/matrices/CSRMatrix.h"
#include "AMP/matrices/MatrixBuilder.h"

#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/petsc/PetscMatrixShellOperator.h"
#include "AMP/operators/OperatorFactory.h"

#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"

#include "RDUtils.h"
#include "RadiationDiffusionFDBEWrappers.h"
#include "RadiationDiffusionFDDiscretization.h"

#include <iostream>
#include <iomanip>

class BERadDifOpPJac;




/* An operator-split, block-based preconditioner for the LinearOperator 
        BERadDifOpPJac = A == I + gamma*D + gamma*R,
    where D is a block-diagonal diffusion matrix, and R is a matrix with diagonal blocks

    The preconditioner matrix arises in factored form as
        P = P_dif * P_react = ( I + gamma*D ) * ( I + gamma*R )
    And is implemeneted as a stationary linear iteration to solve the linear system
        A*[E,T] = [bE,bT]
    The preconditioner is implemented as a stationary linear iteration of the form:
        [E,T] <- [E,T] + [dE,dT], where P*[dE,dT]=r, for residual r = [bE,bT]-A*[E,T].

    Note that one iteration of the preconditioner is equivalent to returning
        [E,T] = P^{-1} * [bE,bT].

    The inverse of ( I + gamma*D ) is carried out approximately by applying individual solvers to each of its diagonal blocks. The incoming Database must contain a 'DiffusionBlocks' Database with sufficient information for the corresponding solvers to be created from a Solverfactory.
*/
class BERadDifOpPJacOpSplitPrec : public AMP::Solver::SolverStrategy {

public:

    // Keep a pointer to this to save having to down cast more than once 
    std::shared_ptr<BERadDifOpPJac> d_BERadDifOpPJac = nullptr;

    // Solvers for inverting diagonal diffusion blocks
    std::shared_ptr<AMP::Solver::SolverStrategy> d_dif_E_solver = nullptr;
    std::shared_ptr<AMP::Solver::SolverStrategy> d_dif_T_solver = nullptr;

    // The base class has the following data:
    // d_iMaxIterations       = "max_iterations"
    // d_iDebugPrintInfoLevel = "print_info_level"
    // d_bUseZeroInitialGuess = "zero_initial_guess"
    // d_dAbsoluteTolerance   = "absolute_tolerance"
    // d_dRelativeTolerance   = "relative_tolerance"
    // d_bComputeResidual     = "compute_residual"

    BERadDifOpPJacOpSplitPrec( std::shared_ptr<AMP::Solver::SolverStrategyParameters> params )
    : SolverStrategy( params ) {

        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOpPJacOpSplitPrec::BERadDifOpPJacOpSplitPrec() " << std::endl;
        
        // Ensure DiffusionBlocks Database was parsed.
        AMP_INSIST(  d_db->getDatabase( "DiffusionBlocks" ), "Preconditioner requires a 'DiffusionBlocks' database" );
    };  

    // Used by SolverFactory to create a BERadDifOpPJacOpSplitPrec
    static std::unique_ptr<AMP::Solver::SolverStrategy> create( std::shared_ptr<AMP::Solver::SolverStrategyParameters> params ) {  
        return std::make_unique<BERadDifOpPJacOpSplitPrec>( params ); };

    // Implementation of pure virtual function
    std::string type() const { return "BERadDifOpPJacOpSplitPrec"; };

    // Apply preconditioner 
    // On p. 26 of Bobby's paper P = P_dif * P_react, so we first invert P_dif then P_react
    void apply(std::shared_ptr<const AMP::LinearAlgebra::Vector> bET_, std::shared_ptr< AMP::LinearAlgebra::Vector> ET_) override {

        // I don't think it makes sense to use a non-zero initial guess, does it?
        AMP_INSIST( d_bUseZeroInitialGuess, "Zero initial guess is hard coded!" );

        // Current implementation only supports a fixed number of iterations, ignores tolerances
        AMP_INSIST( d_dAbsoluteTolerance == 0.0 && d_dRelativeTolerance == 0.0, "Non-zero tolerances not implemented; only fixed number of iterations" );

        AMP_INSIST( this->getOperator(), "Apply requires an operator to be registered" );
        // Sometimes the d_pOperator variable is referenced directly rather than calling registerOperator... This call ensures our overridden registerOperator gets called
        registerOperator( this->getOperator() );

        // Assemble solvers for diffusion blocks
        setDiffusionSolvers();

        // --- Downcast input Vectors to MultiVectors ---
        auto bET = std::dynamic_pointer_cast<const AMP::LinearAlgebra::MultiVector>( bET_ );
        auto  ET = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( ET_ );
        AMP_INSIST( bET, "bET downcast to MultiVector unsuccessful" );
        AMP_INSIST(  ET, "ET downcast to MultiVector unsuccessful" );
        // Unpack scalar vectors from multivectors
        auto bE = bET->getVector(0);
        auto bT = bET->getVector(1);
        auto E  = ET->getVector(0);
        auto T  = ET->getVector(1);

        // Create a residual vector and upack it
        auto rET_ = d_BERadDifOpPJac->createInputVector();
        auto rET  = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( rET_ );
        AMP_INSIST( rET, "rET downcast to MultiVector unsuccessful" );
        auto rE   = rET->getVector(0);
        auto rT   = rET->getVector(1); 
        // Create a correction vector and upack it
        auto dET_ = d_BERadDifOpPJac->createInputVector();
        auto dET  = std::dynamic_pointer_cast<AMP::LinearAlgebra::MultiVector>( dET_ );
        AMP_INSIST( dET, "dET downcast to MultiVector unsuccessful" );
        auto dE   = dET->getVector(0);
        auto dT   = dET->getVector(1); 
        
        // --- Iterate ---
        // Set initial iterate to zero
        ET->zero( );
        for ( auto iter = 0; iter < d_iMaxIterations; iter++ ) {

            // Compute residual
            if ( iter == 0 ) {
                rET->copyVector( bET ); // Zero initial iterate means r0 = b - A*x0 = b
            } else {
                d_BERadDifOpPJac->residual( bET, ET, rET );
            }
            if ( d_iDebugPrintInfoLevel > 1 ) {
                AMP::pout << "BERadDifOpPJacOpSplitPrec::apply(): iteration " << iter << ":" << std::endl;
                AMP_ERROR( "no residual norm implemented..." );
                // auto rnorms = getDiscreteNorms( 1.0, rET );
                // AMP::pout << "||r||=(" << rnorms[0] << "," << rnorms[1] << "," << rnorms[2] << ")" << std::endl;
            }

            // Solve P_dif * [dE, dT] = [rE, rT], with residuals r
            diffusionSolve( rE, rT, dE, dT );
            // Solve P_react * [rE, rT] = [dE, dT] (here we abuse variable names)
            reactionSolve( dE, dT, rE, rT );

            // Update solution: ET <- ET + rET
            ET->axpby( 1.0, 1.0, *rET ); // ET <- 1.0*ET + 1.0*rET 
            ET->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
        }

        // Display final residual; note that this residual calculation is unnecessary because the final iterate is calculated already, hence the flag 
        if ( d_bComputeResidual ) {
            d_BERadDifOpPJac->residual( bET, ET, rET );
            AMP::pout << "BERadDifOpPJacOpSplitPrec::apply(): final residual" << ":" << std::endl;
                AMP_ERROR( "no residual norm implemented..." );
                // auto rnorms = getDiscreteNorms( 1.0, rET );
                // AMP::pout << "||r||=(" << rnorms[0] << "," << rnorms[1] << "," << rnorms[2] << ")" << std::endl;
        }
    }

    

    /* The base class' reset() does: 
        1. Call's it's operator's reset() with null OperatorParameters
        2. Call's it's nested solver's reset with the incoming params. 
        
    In this reset() we're not going to do anything because: 1. The underlying Operator is reset by other means, with appropriate parameters. The only data we store that would need to actually be reset are the solvers for the diffusion blocks, but these are fully reconstructed every time there's a call to apply(), since this is guaranteed to have the most up to date information anyway. */
    void reset( std::shared_ptr<AMP::Solver::SolverStrategyParameters> params ) override {
        //AMP_WARNING( "BERadDifOpPJacOpSplitPrec::reset() doesn't do anything... What should it do?" );

        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOpPJacOpSplitPrec::reset() " << std::endl;

        d_dif_T_solver = nullptr;
        d_dif_E_solver = nullptr;
    }

    void registerOperator( std::shared_ptr<AMP::Operator::Operator> op ) override {
        if ( d_iDebugPrintInfoLevel > 1 )
            AMP::pout << "BERadDifOpPJacOpSplitPrec::registerOperator() " << std::endl;

        AMP_INSIST( op, "A null operator cannot be registered" );

        d_pOperator = op;
        auto myBEOp = std::dynamic_pointer_cast<BERadDifOpPJac>( op );
        AMP_INSIST( myBEOp, "Operator must be of BERadDifOpPJac type" );
        d_BERadDifOpPJac = myBEOp;
    }


private:

    /* Solve the diffusion system P_dif * [E,T] = [bE, bT] for [E,T] */
    void diffusionSolve(  
    std::shared_ptr<const AMP::LinearAlgebra::Vector> bE,
    std::shared_ptr<const AMP::LinearAlgebra::Vector> bT,
    std::shared_ptr<      AMP::LinearAlgebra::Vector>  E,
    std::shared_ptr<      AMP::LinearAlgebra::Vector>  T ) {
        AMP_INSIST( d_dif_E_solver, "Null dif_E_solver" );
        AMP_INSIST( d_dif_T_solver, "Null dif_T_solver" );
        d_dif_E_solver->apply( bE, E );
        d_dif_T_solver->apply( bT, T );
    }
    
    /* This solves the reaction system B*[E,T] = [bE, bT] for [E,T]
        Here we directly apply B^{-1}, where B == I + R_BE
            where R_BE = [ diag(r_EE_BE) diag(r_ET_BE) ]
                         [ diag(r_TE_BE) diag(r_TT_BE) ] 
    is a subset of the data stored in our opertor's d_data variable */
    void reactionSolve(  
        std::shared_ptr<const AMP::LinearAlgebra::Vector> bE,
        std::shared_ptr<const AMP::LinearAlgebra::Vector> bT,
        std::shared_ptr<      AMP::LinearAlgebra::Vector>  E,
        std::shared_ptr<      AMP::LinearAlgebra::Vector>  T ) {

        // Get Operator's data
        auto data = d_BERadDifOpPJac->d_data;

        // Constants used to describe 2x2 linear systems
        double a, b, c, d, e, f, x, y;

        // Iterate through all local rows
        auto scalarDOFMan = d_BERadDifOpPJac->d_RadDifOpPJac->d_RadDifOp->d_scalarDOFMan;
        for (auto dof = scalarDOFMan->beginDOF(); dof != scalarDOFMan->endDOF(); dof++) {
            // LHS matrix
            a = data->r_EE_BE->getValueByGlobalID<double>( dof ) + 1.0; // Add identity 
            b = data->r_ET_BE->getValueByGlobalID<double>( dof );
            c = data->r_TE_BE->getValueByGlobalID<double>( dof );
            d = data->r_TT_BE->getValueByGlobalID<double>( dof ) + 1.0; // Add identity 
            // RHS
            e = bE->getValueByGlobalID<double>( dof );
            f = bT->getValueByGlobalID<double>( dof );
            // Solve linear system
            twoByTwoSolve( a, b, c, d, e, f, x, y );
            // Pack results into solution vectors
            E->setValueByGlobalID<double>( dof, x );
            T->setValueByGlobalID<double>( dof, y );
        }
    }

    /* Solve the linear system
        ax+by=e
        cx+dy=f 
    for x and y. */
    void twoByTwoSolve( double a, double b, double c, double d, double e, double f, double &x, double &y ) {
        double det = a*d - b*c;
        //AMP::pout << det << " = " << a << " x " << d << " - " << b << " x " << c << std::endl;
        AMP_INSIST( fabs( det ) > 1e-12, "2x2 linear system is singular" );
        x = (e*d - b*f) / det;
        y = (a*f - e*c) / det;
    }


    // Create solvers for diffusion blocks if they don't exist already
    void setDiffusionSolvers( ) {

        if ( d_dif_E_solver && d_dif_T_solver )
            return;

        // Wrap diffusion matrices as LinearOperators 
        // E
        auto E_db    = AMP::Database::create( "name", "EOperator", "print_info_level", 0 );
        auto EParams = std::make_shared<AMP::Operator::OperatorParameters>( std::move(E_db) );
        auto E       = std::make_shared<AMP::Operator::LinearOperator>( EParams );
        AMP_INSIST( d_BERadDifOpPJac->d_data->d_E_BE, "E diffusion matrix is null" );
        E->setMatrix( d_BERadDifOpPJac->d_data->d_E_BE );
        
        // T
        auto T_db    = AMP::Database::create( "name", "TOperator", "print_info_level", 0 );
        auto TParams = std::make_shared<AMP::Operator::OperatorParameters>( std::move(T_db) );
        auto T       = std::make_shared<AMP::Operator::LinearOperator>( TParams );
        T->setMatrix( d_BERadDifOpPJac->d_data->d_T_BE );
        AMP_INSIST( T->getMatrix(), "T diffusion matrix is null" );

        // Create solver parameters
        auto comm        = d_BERadDifOpPJac->getMesh()->getComm(); 
        auto solver_db   = d_db->getDatabase( "DiffusionBlocks" );
        // E
        auto ESolverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( solver_db );
        ESolverParams->d_pOperator = E;
        ESolverParams->d_comm      = comm;
        // T
        auto TSolverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( solver_db );
        TSolverParams->d_pOperator = T;
        TSolverParams->d_comm      = comm;

        // auto A = d_TOp->getMatrix();
        // AMP::IO::AsciiWriter matWriter;
        // matWriter.registerMatrix( A );
        // matWriter.writeFile( "Aout", 0  );

        // Create solvers from Factories
        d_dif_E_solver = AMP::Solver::SolverFactory::create( ESolverParams );
        d_dif_T_solver = AMP::Solver::SolverFactory::create( TSolverParams );
        // Ensure zero initial guess is used
        d_dif_E_solver->setZeroInitialGuess( true );
        d_dif_T_solver->setZeroInitialGuess( true );
    }

};




#endif