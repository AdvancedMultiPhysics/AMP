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

#include "AMP/operators/radiationDiffusionFD/RDUtils.h" // oktodo: delete this
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDBEWrappers.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDDiscretization.h"

#include <iostream>
#include <iomanip>


// oktodo: clean up above includes 

// oktodo: do i need this declaration if header is included?
//class BERadDifOpPJac;

namespace AMP::Solver {


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
    std::shared_ptr<AMP::Operator::BERadDifOpPJac> d_BERadDifOpPJac = nullptr;

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

    BERadDifOpPJacOpSplitPrec( std::shared_ptr<AMP::Solver::SolverStrategyParameters> params );
    

    // Used by SolverFactory to create a BERadDifOpPJacOpSplitPrec
    static std::unique_ptr<AMP::Solver::SolverStrategy> create( std::shared_ptr<AMP::Solver::SolverStrategyParameters> params ) {  
        return std::make_unique<BERadDifOpPJacOpSplitPrec>( params ); };

    // Implementation of pure virtual function
    std::string type() const { return "BERadDifOpPJacOpSplitPrec"; };

    // Apply preconditioner 
    // On p. 26 of Bobby's paper P = P_dif * P_react, so we first invert P_dif then P_react
    void apply(std::shared_ptr<const AMP::LinearAlgebra::Vector> bET_, std::shared_ptr< AMP::LinearAlgebra::Vector> ET_) override;
    


    /* The base class' reset() does: 
        1. Call's it's operator's reset() with null OperatorParameters
        2. Call's it's nested solver's reset with the incoming params. 
        
    In this reset() we're not going to do anything because: 1. The underlying Operator is reset by other means, with appropriate parameters. The only data we store that would need to actually be reset are the solvers for the diffusion blocks, but these are fully reconstructed every time there's a call to apply(), since this is guaranteed to have the most up to date information anyway. */
    void reset( std::shared_ptr<AMP::Solver::SolverStrategyParameters> params ) override;
    

    void registerOperator( std::shared_ptr<AMP::Operator::Operator> op ) override; 

private:

    /* Solve the diffusion system P_dif * [E,T] = [bE, bT] for [E,T] */
    void diffusionSolve(  
    std::shared_ptr<const AMP::LinearAlgebra::Vector> bE,
    std::shared_ptr<const AMP::LinearAlgebra::Vector> bT,
    std::shared_ptr<      AMP::LinearAlgebra::Vector>  E,
    std::shared_ptr<      AMP::LinearAlgebra::Vector>  T ); 
    
    /* This solves the reaction system B*[E,T] = [bE, bT] for [E,T]
        Here we directly apply B^{-1}, where B == I + R_BE
            where R_BE = [ diag(r_EE_BE) diag(r_ET_BE) ]
                         [ diag(r_TE_BE) diag(r_TT_BE) ] 
    is a subset of the data stored in our opertor's d_data variable */
    void reactionSolve(  
        std::shared_ptr<const AMP::LinearAlgebra::Vector> bE,
        std::shared_ptr<const AMP::LinearAlgebra::Vector> bT,
        std::shared_ptr<      AMP::LinearAlgebra::Vector>  E,
        std::shared_ptr<      AMP::LinearAlgebra::Vector>  T );

    /* Solve the linear system
        ax+by=e
        cx+dy=f 
    for x and y. */
    void twoByTwoSolve( double a, double b, double c, double d, double e, double f, double &x, double &y ) const;


    // Create solvers for diffusion blocks if they don't exist already
    void setDiffusionSolvers( );
    

};


} // namespace AMP::Solver


#endif