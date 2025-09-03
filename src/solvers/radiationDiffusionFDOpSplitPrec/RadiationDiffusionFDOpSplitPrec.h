#ifndef RAD_DIF_OP_SPLIT_PREC
#define RAD_DIF_OP_SPLIT_PREC

#include "AMP/IO/PIO.h"
#include "AMP/IO/AsciiWriter.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/MultiVector.h"
#include "AMP/operators/Operator.h"
#include "AMP/operators/OperatorParameters.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDBEWrappers.h"
#include "AMP/operators/radiationDiffusionFD/RadiationDiffusionFDDiscretization.h"

#include <iostream>
#include <iomanip>

namespace AMP::Solver {


/** A solver based on an operator-split, block-based preconditioner for the LinearOperator 
 *     BERadDifOpPJac = A == I + gamma*D + gamma*R,
 * where D is a 2x2 block-diagonal diffusion matrix, and R is a 2x2 block matrix with diagonal 
 * blocks.
 * The preconditioner matrix arises in factored form as
 *      P = P_dif * P_react, 
 * where 
 *      P_dif = ( I + gamma*D ),
 *      P_react = ( I + gamma*R ).
 *  The solver is implemented to solve the linear system
 *      A*[E,T] = [bE,bT]
 *  in the form of a stationary linear iteration of the form
 *      [E,T] <- [E,T] + [dE,dT], where P*[dE,dT]=r, for residual r = [bE,bT]-A*[E,T].
 * 
 *  Note that one iteration of the preconditioner is equivalent to returning
 *      [E,T] = P^{-1} * [bE,bT].
 * 
 * The inverse of ( I + gamma*D ) is carried out approximately by applying individual solvers to
 * each of its diagonal blocks. The incoming Database must contain a 'DiffusionBlocks' Database
 * with sufficient information for the corresponding solvers to be created from a Solverfactory.
 * 
 * The incoming operator is assumed to be a AMP::Operator::BERadDifOpPJac. Our operator's d_data
 * member variable is utilized by this class.
 */
class BERadDifOpPJacOpSplitPrec : public AMP::Solver::SolverStrategy {

//
public:

    BERadDifOpPJacOpSplitPrec( std::shared_ptr<AMP::Solver::SolverStrategyParameters> params );

    static std::unique_ptr<AMP::Solver::SolverStrategy> create( std::shared_ptr<AMP::Solver::SolverStrategyParameters> params ) {  
        return std::make_unique<BERadDifOpPJacOpSplitPrec>( params ); };

    std::string type() const override { return "BERadDifOpPJacOpSplitPrec"; };
    
    /** Apply the preconditioner in the form of a stationary linear iteration */
    void apply(std::shared_ptr<const AMP::LinearAlgebra::Vector> bET_, std::shared_ptr< AMP::LinearAlgebra::Vector> ET_) override;
    
    /** Reset the solver. This only resets the solvers used for the diffusion blocks based on the 
     * current state of the operator. I.e., it does not reset the underlying operator, and assumes 
     * that has been reset already via other means.  
     */
    void reset( std::shared_ptr<AMP::Solver::SolverStrategyParameters> params ) override;
    

//
private:
    //! Tolerance for determining if 2x2 system is singular
    static constexpr double SINGULAR_TOL = 1e-15;

    //! Solvers for approximately inverting diagonal diffusion blocks
    std::unique_ptr<AMP::Solver::SolverStrategy> d_difSolverE = nullptr;
    std::unique_ptr<AMP::Solver::SolverStrategy> d_difSolverT = nullptr;


    /** Solve the diffusion system P_dif*[E,T] = [bE, bT] for [E,T] 
     */
    void diffusionSolve(  
        std::shared_ptr<const AMP::LinearAlgebra::Vector> bE,
        std::shared_ptr<const AMP::LinearAlgebra::Vector> bT,
        std::shared_ptr<      AMP::LinearAlgebra::Vector>  E,
        std::shared_ptr<      AMP::LinearAlgebra::Vector>  T ) const; 
    
    /** Solve the reaction system P_react*[E,T] = [bE, bT] for [E,T].
     * Here we directly apply P_react^{-1}, where P_react == I + R_BE
     *      where R_BE = [ diag(r_EE_BE) diag(r_ET_BE) ]
     *                   [ diag(r_TE_BE) diag(r_TT_BE) ] 
     * is a subset of the data stored in our opertor's d_data variable 
     */
    void reactionSolve(  
        std::shared_ptr<const AMP::LinearAlgebra::Vector> bE,
        std::shared_ptr<const AMP::LinearAlgebra::Vector> bT,
        std::shared_ptr<      AMP::LinearAlgebra::Vector>  E,
        std::shared_ptr<      AMP::LinearAlgebra::Vector>  T );

    /** Solve the (non-block) 2x2 linear system
     *      ax+by=e
     *      cx+dy=f 
     * for unknowns x and y. 
     */
    void scalar2x2Solve( double a, double b, double c, double d, double e, double f, double &x, double &y ) const;

    //! Set d_difSolverE and d_difSolverT based on current state of operator
    void setDiffusionSolvers( );
};


} // namespace AMP::Solver


#endif