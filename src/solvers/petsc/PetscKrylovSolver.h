#ifndef included_AMP_PetscKrylovSolver
#define included_AMP_PetscKrylovSolver

#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/petsc/PetscMonitor.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/vectors/petsc/PetscHelpers.h"


// Forward declare a few types for PETSc
typedef int PetscErrorCode;
typedef struct _p_SNES *SNES;
typedef struct _p_KSP *KSP;
typedef struct _p_PC *PC;


namespace AMP::Solver {

/**
 * The PetscKrylovSolver class is a wrapper to the PETSc KSP Krylov solver which provides
 * implementations of Krylov
 * methods. Currently
 * the wrapper has only been tested with the GMRES and FGMRES Krylov methods provided by PETSc.
 */
class PetscKrylovSolver : public SolverStrategy
{
public:
    /**
     * default constructor, currently only sets a boolean flag d_bKSPCreatedInternally = false
     */
    PetscKrylovSolver();

    /**
     * main constructor
     @param [in] parameters The parameters object
     contains a database objects containing the following fields:

     1. type: string KSPOptions, optional, default value "", can be used to set KSP solver
     parameters expected by PETSc, preferred way of passing parameters to PETSc

     2. type: string, name : ksp_type, default value "fgmres"
        acceptable values ("fgmres", "gmres")

     3. type: double, name : relative_tolerance, default value of $1.0e-9$, relative tolerance for
    KSP solver
    acceptable values (non-negative real values)

     4. type: double, name : absolute_tolerance, default value of $1.0e-14$, absolute tolerance for
    KSP solver
     acceptable values (non-negative real values)

     5. type: double, name : divergence_tolerance, default value of $1.0e3$, divergence tolerance
    for KSP solver
     acceptable values (non-negative real values)

     6. type: string, name : KSPAppendOptionsPrefix, default value "", used to append options for
    KSP solver
     acceptable values ()

     7. type: integer, name : max_krylov_dimension, default value $20$, maximum dimension of Krylov
    space,
     acceptable values (integer values grater than or equal to 1)
     active only when ksp_type is "fgmres" or "gmres"

     8. type: string, name : gmres_orthogonalization_algorithm, default value "modifiedgramschmidt",
    acceptable values ("modifiedgramschmidt", )
    active only when ksp_type is "fgmres" or "gmres"

     9. type: bool, name : uses_preconditioner, default value false
        acceptable values (false, true),
        side effect: if false sets string pc_type to "none"

     10. type: string, name : pc_type, default "none",
         acceptable values ("none", "shell", see PETSc documentation for acceptable PETSc values)
         active only when uses_preconditioner set to true

     11. type: string, name : pc_side, default value "RIGHT",
     acceptable values ("RIGHT", "LEFT", "SYMMETRIC" )
         active only when uses_preconditioner set to true
     */
    explicit PetscKrylovSolver( std::shared_ptr<SolverStrategyParameters> parameters );

    /**
     * Default destructor. Currently destroys the PETSc KSP object if it was created internally.
     */
    virtual ~PetscKrylovSolver();

    std::string type() const override { return "PetscKrylovSolver"; }

    //! static create routine that is used by SolverFactory
    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> solverStrategyParameters )
    {
        return std::make_unique<PetscKrylovSolver>( solverStrategyParameters );
    }

    /**
     * Solve the system \f$Au = 0\f$.
     * @param [in] f : shared pointer to right hand side vector
     * @param [out] u : shared pointer to approximate computed solution
     */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

    /**
     * returns the internally stored PETSc KSP object
     */
    inline KSP getKrylovSolver( void ) { return d_KrylovSolver; }

    /**
     * sets the PETSc KSP object
     * @param [in] ksp pointer to KSP object
     */
    void setKrylovSolver( KSP *ksp );

    /**
     * Initialize the PetscKrylovSolver. Should not be necessary for the user to call in general.
     * @param parameters
     */
    void initialize( std::shared_ptr<const SolverStrategyParameters> parameters ) override;

    /**
     * Register the operator that the solver will use during solves
     * @param [in] op shared pointer to operator $A()$ for equation \f$A(u) = f\f$
     */
    void registerOperator( std::shared_ptr<AMP::Operator::Operator> op ) override;

    /**
     * Resets the registered operator internally with new parameters if necessary
     * @param parameters    OperatorParameters object that is NULL by default
     */
    void
    resetOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> parameters ) override;

    bool usesPreconditioner( void ) { return d_bUsesPreconditioner; }

    void setZeroInitialGuess( bool use_zero_guess ) override;

protected:
    void getFromInput( std::shared_ptr<AMP::Database> db );
    void initializePreconditioner( std::shared_ptr<const SolverStrategyParameters> parameters );
    std::shared_ptr<AMP::Operator::Operator> createPCOperator();
    void setupPetscMatInterface( std::shared_ptr<AMP::Operator::Operator> op, Mat &mat );

private:
    // static functions to interface with PETSc
    // the signatures of these functions currently vary depending on whether the dev or release
    // release version of PETSc is being used

    static PetscErrorCode setupPreconditioner( PC pc );
    static PetscErrorCode applyPreconditioner( PC pc, Vec r, Vec z );
    static PetscErrorCode matVec( Mat mat, Vec x, Vec y );

    AMP_MPI d_comm;

    std::string d_sKspType;

    double d_dDivergenceTolerance;

    bool d_bKSPCreatedInternally;

    bool d_bUsesPreconditioner = false;
    bool d_bMatrixFree         = false;

    std::string d_sPcType;
    std::string d_KSPAppendOptionsPrefix;

    std::string d_PcSide;

    // FGMRES specific options
    int d_iMaxKrylovDimension;
    std::string d_sGmresOrthogonalizationAlgorithm;

    std::shared_ptr<PetscMonitor> d_PetscMonitor;

    KSP d_KrylovSolver;

    Mat d_Mat;
};
} // namespace AMP::Solver

#endif
