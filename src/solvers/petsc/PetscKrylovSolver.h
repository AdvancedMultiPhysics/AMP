#ifndef included_AMP_PetscKrylovSolver
#define included_AMP_PetscKrylovSolver

#include "utils/AMP_MPI.h"
#include "solvers/SolverStrategy.h"
#include "solvers/petsc/PetscKrylovSolverParameters.h"
#include "solvers/petsc/PetscMonitor.h"


extern "C"{

#ifdef MPICH_SKIP_MPICXX
#define _FIX_FOR_PETSC_MPI_CXX
#undef MPICH_SKIP_MPICXX
#endif

#ifdef OMPI_SKIP_MPICXX
#define _FIX_FOR_PETSC_OMPI_CXX
#undef OMPI_SKIP_MPICXX
#endif

#include "petsc.h"
#include "petscksp.h"

#ifdef _FIX_FOR_PETSC_MPI_CXX
#ifndef MPICH_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#endif
#endif

#ifdef _FIX_FOR_PETSC_OMPI_CXX
#ifndef OMPI_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#endif
#endif

}

namespace AMP {
namespace Solver {

/**
 * The PetscKrylovSolver class is a wrapper to the PETSc KSP Krylov solver which provides implementations of Krylov methods. Currently
 * the wrapper has only been tested with the GMRES and FGMRES Krylov methods provided by PETSc.
 */
class PetscKrylovSolver: public SolverStrategy{
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

     3. type: double, name : relative_tolerance, default value of $1.0e-9$, relative tolerance for KSP solver
	acceptable values (non-negative real values) 

     4. type: double, name : absolute_tolerance, default value of $1.0e-14$, absolute tolerance for KSP solver
     acceptable values (non-negative real values)
     
     5. type: double, name : divergence_tolerance, default value of $1.0e3$, divergence tolerance for KSP solver
     acceptable values (non-negative real values)
     
     6. type: string, name : KSPAppendOptionsPrefix, default value "", used to append options for KSP solver
     acceptable values ()
     
     7. type: integer, name : max_krylov_dimension, default value $20$, maximum dimension of Krylov space,
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
    PetscKrylovSolver(boost::shared_ptr<PetscKrylovSolverParameters> parameters);

    /**
     * Default destructor. Currently destroys the PETSc KSP object if it was created internally. 
     */
    virtual ~PetscKrylovSolver();

    /**
     * Solve the system \f$Au = 0\f$.
     * @param [in] f : shared pointer to right hand side vector
     * @param [out] u : shared pointer to approximate computed solution 
     */
    void solve(boost::shared_ptr<const AMP::LinearAlgebra::Vector>  f,
	     boost::shared_ptr<AMP::LinearAlgebra::Vector>  u);

    /**
     * returns the internally stored PETSc KSP object
     */
    inline KSP getKrylovSolver(void){ return d_KrylovSolver; }

    /**
     * sets the PETSc KSP object
     * @param [in] ksp pointer to KSP object
     */
    void setKrylovSolver(KSP *ksp);

    /**
     * Initialize the PetscKrylovSolver. Should not be necessary for the user to call in general.
     * @param parameters
     */
    void initialize(boost::shared_ptr<SolverStrategyParameters> const parameters);

    /**
     * returns a shared pointer to a preconditioner object. The preconditioner is derived from
     * a SolverStrategy class
     */
    inline boost::shared_ptr<AMP::Solver::SolverStrategy> getPreconditioner(void){ return d_pPreconditioner; }

    /**
     * sets a shared pointer to a preconditioner object. The preconditioner is derived from
     * a SolverStrategy class
     * @param pc shared pointer to preconditioner
     */
    inline void setPreconditioner(boost::shared_ptr<AMP::Solver::SolverStrategy> pc){d_pPreconditioner = pc;}

    /**
     * Register the operator that the solver will use during solves
     * @param [in] op shared pointer to operator $A()$ for equation \f$A(u) = f\f$ 
     */
    void registerOperator(const boost::shared_ptr<AMP::Operator::Operator> op);

    /**
     * Resets the registered operator internally with new parameters if necessary
     * @param parameters    OperatorParameters object that is NULL by default
     */
    void resetOperator(const boost::shared_ptr<AMP::Operator::OperatorParameters> parameters);
  
protected:
  
    void getFromInput(const boost::shared_ptr<AMP::Database>& db);
  
private:
  
    // static functions to interface with PETSc
    // the signatures of these functions currently vary depending on whether the dev or release
    // release version of PETSc is being used
  
#if ( PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR==0 )
    static int setupPreconditioner(void*);
    static PetscErrorCode applyPreconditioner(void* , Vec , Vec );
#elif ( PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR==2 )
    static PetscErrorCode setupPreconditioner(PC pc);
    static PetscErrorCode applyPreconditioner(PC pc, Vec r, Vec z);
#else
    #error Not programmed for this version yet
#endif
  
    AMP_MPI d_comm;
  
    std::string d_sKspType;

    double d_dRelativeTolerance;
    double d_dAbsoluteTolerance;
    double d_dDivergenceTolerance;

    bool d_bKSPCreatedInternally;

    bool d_bUsesPreconditioner;
    std::string d_sPcType;
    std::string d_KSPAppendOptionsPrefix;
  
    PCSide d_PcSide;

    // FGMRES specific options
    int d_iMaxKrylovDimension;
    std::string d_sGmresOrthogonalizationAlgorithm;

    #if ( PETSC_VERSION_MAJOR==3 && PETSC_VERSION_MINOR==0 )
        // The following KSP solver keeps a reference to these vectors around. 
        // By declaring the vectors here, we ensure correct behavior during destruction.
        // This will ensure that the boost::shared_ptr destructor calls VecDestroy on
        // the last reference.
        AMP::LinearAlgebra::Vector::const_shared_ptr  fVecView;
        AMP::LinearAlgebra::Vector::shared_ptr  uVecView;
    #endif

    boost::shared_ptr<PetscMonitor> d_PetscMonitor;

    KSP d_KrylovSolver;

    boost::shared_ptr<AMP::Solver::SolverStrategy> d_pPreconditioner;

};


}
}

#endif