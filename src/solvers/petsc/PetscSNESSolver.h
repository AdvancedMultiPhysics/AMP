#ifndef included_AMP_PetscSNESSolver
#define included_AMP_PetscSNESSolver

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscMonitor.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/petsc/PetscHelpers.h"

#include <list>

#ifdef MPICH_SKIP_MPICXX
    #define _FIX_FOR_PETSC_MPI_CXX
    #undef MPICH_SKIP_MPICXX
#endif

#ifdef OMPI_SKIP_MPICXX
    #define _FIX_FOR_PETSC_OMPI_CXX
    #undef OMPI_SKIP_MPICXX
#endif

#undef PETSC_VERSION_DATE_GIT
#undef PETSC_VERSION_GIT
#include "petsc.h"
#include "petscmat.h"
#include "petscsnes.h"

#ifdef _FIX_FOR_PETSC_OMPI_CXX
    #ifndef OMPI_SKIP_MPICXX
        #define OMPI_SKIP_MPICXX
    #endif
#endif

#ifdef _FIX_FOR_PETSC_MPI_CXX
    #ifndef MPICH_SKIP_MPICXX
        #define MPICH_SKIP_MPICXX
    #endif
#endif

namespace AMP::LinearAlgebra {
class PetscVector;
}


namespace AMP::Solver {


/**
 * The PETScSNESSolver is a wrapper to the PETSc SNES solver which provides an implementation
 * of the inexact Newton method.
 */
class PetscSNESSolver : public SolverStrategy
{
public:
    /**
     * default constructor, sets default values for member variables
     */
    PetscSNESSolver();


    /**
     * main constructor
     @param [in] parameters The parameters object
     contains a database objects containing the following fields:

     1. type: string, name: SNESOptions, default value: "",

     2. type: bool, name: usesJacobian, default value: false,
     acceptable values (true, false)

     3. name: MFFDDifferencingStrategy, type:string , default value: MATMFFD_WP,
     acceptable values ()

     4. name: MFFDFunctionDifferencingError, type: double, default value: PETSC_DEFAULT,
     acceptable values ()

     5. name: maximumFunctionEvals, type: integer, default value: none
     acceptable values ()

     6. name: absoluteTolerance, type: double, default value: none
     acceptable values ()

     7. name: relativeTolerance, type: double, default value: none
     acceptable values ()

     8. name: stepTolerance, type: double, default value: none
     acceptable values ()

     9. name: enableLineSearchPreCheck, type: bool, default value: FALSE
     acceptable values ()

     10. name: numberOfLineSearchPreCheckAttempts, type: integer, default value: 5
     acceptable values (non negative integer values)

     11. name: enableMFFDBoundsCheck, type: bool, default value: FALSE
     acceptable values ()

     12. name: operatorComponentToEnableBoundsCheck, type: integer, default value: none
     acceptable values ()
    */
    explicit PetscSNESSolver( std::shared_ptr<SolverStrategyParameters> parameters );

    /**
     * Default destructor.
     */
    virtual ~PetscSNESSolver();

    std::string type() const override { return "PetscSNESSolver"; }

    //! static create routine that is used by SolverFactory
    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> solverStrategyParameters )
    {
        return std::make_unique<PetscSNESSolver>( solverStrategyParameters );
    }

    /**
     * Solve the system \f$Au = 0\f$.
    @param [in] f : shared pointer to right hand side vector
    @param [out] u : shared pointer to approximate computed solution
     */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

    /**
     * Initialize the solution vector by copying the initial guess vector
     * @param [in] initialGuess: shared pointer to the initial guess vector.
     */
    void setInitialGuess( std::shared_ptr<AMP::LinearAlgebra::Vector> initialGuess ) override;

    /**
     * return the PETSc SNES solver object
     */
    SNES getSNESSolver( void ) { return d_SNESSolver; }

    /**
     * Returns boolean to indicate whether the solver expects
     * a Jacobian matrix to be provided or not.
     */
    bool getUsesJacobian( void ) { return d_bUsesJacobian; }

    /**
     * Returns a shared pointer to the PetscKrylovSolver used internally for the linear solves
     */
    std::shared_ptr<PetscKrylovSolver> getKrylovSolver( void ) { return d_pKrylovSolver; }

    std::shared_ptr<SolverStrategy> getNestedSolver( void ) override { return getKrylovSolver(); }

    /**
     * Return a shared pointer to the solution vector
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> getSolution( void ) { return d_pSolutionVector; }

    /**
     * Return a shared pointer to the scratch vector used internally.
     */
    std::shared_ptr<AMP::LinearAlgebra::Vector> getScratchVector( void )
    {
        return d_pScratchVector;
    }

    /**
     * Return the number of line search precheck attempts that were made for the current step
     */
    int getNumberOfLineSearchPreCheckAttempts( void )
    {
        return d_iNumberOfLineSearchPreCheckAttempts;
    }

    /**
     * Return an integer value for the component for which bounds checking is enabled.
     */
    int getBoundsCheckComponent( void ) { return d_operatorComponentToEnableBoundsCheck; }

    //! set the pointer to the user function that does the line search pre-check if provided
    // Note that this will enable the line search functionality also
    void setLineSearchPreCheck( std::function<int( std::shared_ptr<AMP::LinearAlgebra::Vector>,
                                                   std::shared_ptr<AMP::LinearAlgebra::Vector>,
                                                   bool & )> lineSearchPreCheckPtr );

    std::function<int( std::shared_ptr<AMP::LinearAlgebra::Vector>,
                       std::shared_ptr<AMP::LinearAlgebra::Vector>,
                       bool & )>
    getLineSearchPreCheckAdaptor()
    {
        AMP_ASSERT( d_lineSearchPreCheckPtr );
        return d_lineSearchPreCheckPtr;
    } //! pointer to line search function

    void reset( std::shared_ptr<AMP::Solver::SolverStrategyParameters> ) override;

    int getTotalNumberOfLinearIterations( void ) const;

    void printConvergenceStatus( SolverStrategy::SolverStatus status,
                                 std::ostream &os = AMP::pout ) const override;

protected:
private:
    void getFromInput( std::shared_ptr<const AMP::Database> db );

    std::shared_ptr<SolverStrategy>
    createPreconditioner( std::shared_ptr<AMP::Database> pc_solver_db );

    static PetscErrorCode apply( SNES snes, Vec x, Vec f, void *ctx );

    void preApply( std::shared_ptr<const AMP::LinearAlgebra::Vector> v );

    static PetscErrorCode setJacobian( SNES, Vec x, Mat A, Mat, void *ctx );

    static bool isVectorValid( std::shared_ptr<AMP::Operator::Operator> &op,
                               AMP::LinearAlgebra::Vector::shared_ptr &v,
                               const AMP_MPI &comm );

    int defaultLineSearchPreCheck( std::shared_ptr<AMP::LinearAlgebra::Vector> x,
                                   std::shared_ptr<AMP::LinearAlgebra::Vector> y,
                                   bool &changed_y );

    static PetscErrorCode
    wrapperLineSearchPreCheck( SNESLineSearch snes, Vec x, Vec y, PetscBool *changed_y, void *ctx );

    // copies of PETSc routines that are not exposed for Eisenstat-Walker
    static PetscErrorCode KSPPreSolve_SNESEW( KSP ksp, Vec b, Vec x, SNES snes );
    static PetscErrorCode KSPPostSolve_SNESEW( KSP ksp, Vec b, Vec x, SNES snes );

    static PetscErrorCode mffdCheckBounds( void *checkctx, Vec U, Vec a, PetscScalar *h );

    static PetscErrorCode setupPreconditioner( PC pc );
    static PetscErrorCode applyPreconditioner( PC pc, Vec xin, Vec xout );

    void setConvergenceStatus( void );

    void createPetscObjects( std::shared_ptr<const SolverStrategyParameters> params );
    void initializePetscObjects( void );
    void destroyPetscObjects( void );

    // pointer to the line search precheck function
    std::function<int( std::shared_ptr<AMP::LinearAlgebra::Vector>,
                       std::shared_ptr<AMP::LinearAlgebra::Vector>,
                       bool & )>
        d_lineSearchPreCheckPtr;

    bool d_bUsesJacobian              = false;
    bool d_bEnableLineSearchPreCheck  = false;
    bool d_bEnableMFFDBoundsCheck     = false;
    bool d_bPrintNonlinearResiduals   = false;
    bool d_bPrintLinearResiduals      = false;
    bool d_bPetscInterfaceInitialized = false;
    bool d_bDestroyCachedVecs         = false;

    int d_iMaximumFunctionEvals                = PETSC_DEFAULT;
    int d_iNumberOfLineSearchPreCheckAttempts  = 0;
    int d_operatorComponentToEnableBoundsCheck = 0;

    double d_dStepTolerance = PETSC_DEFAULT;

    // strategy to use for MFFD differencing (DS or WP)
    std::string d_sMFFDDifferencingStrategy = MATMFFD_WP;

    // string prefix for SNES options passed on command line or through petsc options
    std::string d_SNESAppendOptionsPrefix = "";
    // error in MFFD approximations
    double d_dMFFDFunctionDifferencingError = PETSC_DEFAULT;

    // The next set of options are specific to forcing
    std::string d_sForcingTermStrategy = "CONSTANT"; //! string is for input
    int d_iForcingTermFlag;                          //! int is for passing choice to PETSc

    double d_dConstantForcingTerm = PETSC_DEFAULT;
    double d_dInitialForcingTerm  = PETSC_DEFAULT;
    double d_dMaximumForcingTerm  = PETSC_DEFAULT;

    // Eisenstat-Walker algo. options
    double d_dEWChoice2Alpha              = PETSC_DEFAULT;
    double d_dEWChoice2Gamma              = PETSC_DEFAULT;
    double d_dEWSafeguardExponent         = PETSC_DEFAULT;
    double d_dEWSafeguardDisableThreshold = PETSC_DEFAULT;

    //! keeps track of linear iteration statistics over solver lifetime
    std::vector<int> d_iLinearIterationHistory;

    AMP_MPI d_comm;

    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pSolutionVector = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pResidualVector = nullptr;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pScratchVector  = nullptr;

    std::shared_ptr<PetscMonitor> d_PetscMonitor = nullptr;

    SNES d_SNESSolver = nullptr;

    Mat d_Jacobian = nullptr;

    SNESConvergedReason d_SNES_completion_code;

    std::shared_ptr<PetscKrylovSolver> d_pKrylovSolver = nullptr;
};
} // namespace AMP::Solver

#endif
