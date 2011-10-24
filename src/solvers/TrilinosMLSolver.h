
#ifndef included_AMP_TrilinosMLSolver
#define included_AMP_TrilinosMLSolver

#include "SolverStrategy.h"
#include "SolverStrategyParameters.h"
#include "ml_MultiLevelPreconditioner.h"
#include "ml_include.h"
#include "MLoptions.h"

namespace AMP {
  namespace Solver {

    typedef SolverStrategyParameters TrilinosMLSolverParameters;

    /**
     * The TrilinosMLSolver is a wrapper to the Trilinos ML solver. ML provides implementations of
     * various algebraic multigrid methods. The wrapper at present simply provides an adaptor
     * to enable AMP users to use the black box ML preconditioner.
     */

    class TrilinosMLSolver: public SolverStrategy {

      public:
        /**
         * Default constructor
         */
        TrilinosMLSolver();

        /**
         * Main constructor.
         @param [in] parameters The parameters object contains a database object which must contain the
         following fields in addition to the fields expected by the base class SolverStrategy class:

         1. name:  print_info_level, type: integer, (optional), default value: 0
         acceptable values (non-negative integer values)

         2. name:  prec_type, type: string, (optional), default value: "MGV"
         acceptable values (see ML manual)

         3. name:  max_levels, type: integer, (optional), default value: 5
         acceptable values (see ML manual)

         4. name:  increasingordecreasing, type: string, (optional), default value: "increasing"
         acceptable values (see ML manual)

         5. name:  aggregation_dampingfactor, type: double, (optional), default value: 4.0/3/0
         acceptable values (see ML manual)

         6. name:  aggregationthreshold, type: double, (optional), default value: 0.0
         acceptable values (see ML manual)

         7. name:  eigen-analysis_type, type: string, (optional), default value: "cg"
         acceptable values (see ML manual)

         8. name:  eigen-analysis_iterations, type: integer, (optional), default value: 10
         acceptable values (see ML manual)

         9. name:  smoother_sweeps, type: integer, (optional), default value: 2
         acceptable values (see ML manual)

         10. name:  smoother_dampingfactor, type: double, (optional), default value: 1.0
         acceptable values (see ML manual)

         11. name:  aggregation_nodes_per_aggregate, type: integer, (optional), default value: none
         acceptable values (see ML manual)

         12. name:  aggregation_nextlevel_aggregates_per_process, type: integer, (optional), default value: none
         acceptable values (see ML manual)

         13. name:  aggregation_damping_factor, type: , (optional), default value: none
         acceptable values (see ML manual)

         14. name:  energy_minimization_enable, type: , (optional), default value: none
         acceptable values (see ML manual)

         15. name:  smoother_preorpost, type: string, (optional), default value: "both"
         acceptable values (see ML manual)

         16. name:  smoothertype, type: string, (optional), default value: "symmetric Gauss-Seidel"
         acceptable values (see ML manual)

         17. name:  coarse_type, type: string, (optional), default value: "Amesos-KLU"
         acceptable values (see ML manual)

         18. name:  PDE_equations, type: integer, (optional), default value: 1
         acceptable values (see ML manual)

         19. name:  coarse_maxsize, type: integer, (optional), default value: 128
         acceptable values (see ML manual)

         20. name: USE_EPETRA, type: bool, (optional), default value: true

         21. name: problem_type, type: string, (optional), default value: "SA" 
         acceptable values "SA" for symmetric and "NSSA" for unsymmetric problems
         */
        TrilinosMLSolver(boost::shared_ptr<TrilinosMLSolverParameters> parameters);

        /**
         * Default destructor
         */
        ~TrilinosMLSolver();

        /**
         * Solve the system \f$Au = f\f$.
         @param [in] f : shared pointer to right hand side vector
         @param [out] u : shared pointer to approximate computed solution 
         */
        void solve(boost::shared_ptr<AMP::LinearAlgebra::Vector>  f,
            boost::shared_ptr<AMP::LinearAlgebra::Vector>  u);

        /**
         * Return a shared pointer to the ML_Epetra::MultiLevelPreconditioner object
         */
        inline const boost::shared_ptr<ML_Epetra::MultiLevelPreconditioner> getMLSolver(void){ return d_mlSolver; }

        /**
         * Initialize the solution vector and potentially create internal vectors needed for solution
         @param [in] parameters The parameters object
         contains a database object. Refer to the documentation for the constructor to see what fields are required.
         This routine assumes that a non-NULL operator of type LinearOperator has been registered with the solver.
         The LinearOperator currently is assumed to contain a pointer to an EpetraMatrix object.
         */
        void initialize(boost::shared_ptr<SolverStrategyParameters> const parameters);

        /**
         * Provide the initial guess for the solver.
         * @param [in] initialGuess: shared pointer to the initial guess vector.
         */
        void setInitialGuess( boost::shared_ptr<AMP::LinearAlgebra::Vector>  initialGuess );

        /**
         * Register the operator that the solver will use during solves
         @param [in] op shared pointer to the linear operator $A$ for equation \f$A u = f\f$ 
         */
        void registerOperator(const boost::shared_ptr<AMP::Operator::Operator> op);

        /**
         * Resets the associated operator internally with new parameters if necessary
         * \param parameters
         *        OperatorParameters object that is NULL by default
         */
        void resetOperator(const boost::shared_ptr<AMP::Operator::OperatorParameters> params);

        /**
         * Resets the solver internally with new parameters if necessary
         * \param parameters
         *        SolverStrategyParameters object that is NULL by default
         * Currently every call to reset destroys the ML preconditioner object
         * and recreates it based on the parameters object. See constructor for
         * fields required for parameter object.
         */
        void reset(boost::shared_ptr<SolverStrategyParameters> );

      protected:

        void getFromInput(const boost::shared_ptr<AMP::Database>& db);

        void convertMLoptionsToTeuchosParameterList();

        void buildML();

      private:

        bool d_bUseEpetra;
        ML* d_ml;
        ML_Aggregate* d_mlAggregate;

        AMP_MPI d_comm;

        bool d_bCreationPhase; /**< set to true if the PC is not ready and false otherwise. */

        double d_dRelativeTolerance;
        double d_dAbsoluteTolerance;
        double d_dDivergenceTolerance;

        boost::shared_ptr<MLoptions> d_mlOptions; 

        boost::shared_ptr<ML_Epetra::MultiLevelPreconditioner> d_mlSolver;
        Teuchos::ParameterList d_MLParameterList;
    };

  }
}

#endif

