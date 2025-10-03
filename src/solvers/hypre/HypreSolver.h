#ifndef included_AMP_HypreSolver
#define included_AMP_HypreSolver


#include "AMP/matrices/Matrix.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"

#include "HYPRE_utilities.h"

// Forward declares
struct hypre_Solver_struct;
struct hypre_IJMatrix_struct;
struct hypre_IJVector_struct;
typedef struct hypre_Solver_struct *HYPRE_Solver;
typedef struct hypre_IJMatrix_struct *HYPRE_IJMatrix;
typedef struct hypre_IJVector_struct *HYPRE_IJVector;


namespace AMP::LinearAlgebra {
class HypreMatrixAdaptor;
}

namespace AMP::Solver {


using HypreSolverParameters = SolverStrategyParameters;


/**
 * The HypreSolver is a wrapper to the HYPRE PCG solver.
 * The wrapper at present simply provides an adapter to enable AMP
 * users to use the black box HyprePCG solver.
 */

class HypreSolver : public SolverStrategy
{

public:
    /**
     * Default constructor
     */
    HypreSolver();

    /**
     * Main constructor.
     @param [in] parameters The parameters object contains a database object which must contain the
     following fields in addition to the fields expected by the base class SolverStrategy class:

    */
    explicit HypreSolver( std::shared_ptr<HypreSolverParameters> parameters );

    /**
     * Default destructor
     */
    virtual ~HypreSolver();

    /**
     * Initialize the solution vector and potentially create internal vectors needed for solution
     @param [in] parameters The parameters object
     contains a database object. Refer to the documentation for the constructor to see what fields
     are required.
     This routine assumes that a non-NULL operator of type LinearOperator has been registered with
     the solver.
     The LinearOperator currently is assumed to contain a pointer to an EpetraMatrix object.
     */
    void initialize( std::shared_ptr<const SolverStrategyParameters> parameters ) override;

    /**
     * Register the operator that the solver will use during solves
     @param [in] op shared pointer to the linear operator $A$ for equation \f$A u = f\f$
     */
    void registerOperator( std::shared_ptr<AMP::Operator::Operator> op ) override;

    /**
     * Resets the associated operator internally with new parameters if necessary
     * @param [in] params
     *        OperatorParameters object that is NULL by default
     */
    void resetOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params ) override;

    /**
     * Resets the solver internally with new parameters if necessary
     * @param [in] params
     *        SolverStrategyParameters object that is NULL by default
     * Currently every call to reset destroys the HyprePCG solver object
     * and recreates it based on the parameters object. See constructor for
     * fields required for parameter object.
     */
    void reset( std::shared_ptr<SolverStrategyParameters> params ) override;

    void getFromInput( std::shared_ptr<const AMP::Database> ) {}

    /**
     * Set the desired HYPRE memory location for HYPRE objects
     */
    void setMemoryLocation( HYPRE_MemoryLocation location ) { d_hypre_memory_location = location; }

    /**
     * Set the desired HYPRE execution policy for the solver
     */
    void setExecutionPolicy( HYPRE_ExecutionPolicy policy ) { d_hypre_exec_policy = policy; }

    HYPRE_Solver getHYPRESolver() { return d_solver; }

    /**
     * Solve the system \f$Au = f\f$.
     @param [in] f : shared pointer to right hand side vector
     @param [out] u : shared pointer to approximate computed solution
     */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

protected:
    /**
     * create the internal HYPRE_IJMatrix based on the AMP matrix
     */
    void createHYPREMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix );

    /**
     * create and initialize the internal hypre vectors for rhs and solution
     */
    void createHYPREVectors();

    /**
     *  copy values from amp vector to hypre vector
     */
    void copyToHypre( std::shared_ptr<const AMP::LinearAlgebra::Vector> amp_v,
                      HYPRE_IJVector hypre_v );

    /**
     *  copy values from hypre vector to amp vector
     */
    void copyFromHypre( HYPRE_IJVector hypre_v, std::shared_ptr<AMP::LinearAlgebra::Vector> amp_v );


    void setParameters( void ); //! set parameters based on internally set variables

    void setupHypreMatrixAndRhs();

    void hypreSolve();

    void preSolve( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                   std::shared_ptr<AMP::LinearAlgebra::Vector> u );
    void postSolve( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                    std::shared_ptr<AMP::LinearAlgebra::Vector> u );

    bool d_bMatrixInitialized = false;

    AMP_MPI d_comm;

    std::shared_ptr<AMP::LinearAlgebra::HypreMatrixAdaptor> d_HypreMatrixAdaptor;

    std::shared_ptr<AMP::LinearAlgebra::Vector> d_r;

    HYPRE_IJMatrix d_ijMatrix  = nullptr; //! pointer to HYPRE matrix struct
    HYPRE_IJVector d_hypre_rhs = nullptr; //! pointer to HYPRE representation of rhs
    HYPRE_IJVector d_hypre_sol = nullptr; //! pointer to HYPRE representation of solution
    HYPRE_Solver d_solver      = nullptr; //! pointer to HYPRE solver

    HYPRE_MemoryLocation d_hypre_memory_location;
    HYPRE_ExecutionPolicy d_hypre_exec_policy;

    HYPRE_PtrToSolverFcn d_hypreSolve;
    HYPRE_Int ( *getHypreNumIterations )( HYPRE_Solver solver, HYPRE_Int *num_iterations );
};
} // namespace AMP::Solver

#endif
