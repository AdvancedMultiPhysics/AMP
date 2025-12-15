#ifndef included_AMP_TrilinosMueLuSolver
#define included_AMP_TrilinosMueLuSolver

#include "AMP/matrices/trilinos/ManagedEpetraMatrix.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/trilinos/ml/MLoptions.h"

// Trilinos includes
DISABLE_WARNINGS
#include "MueLu_FactoryManager.hpp"
#include "MueLu_HierarchyManager_fwd.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Xpetra_Matrix.hpp"
#include "Xpetra_Operator_fwd.hpp"
#include "Xpetra_Parameters.hpp"
ENABLE_WARNINGS

#include <vector>


namespace MueLu {
class EpetraOperator;
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
class Hierarchy;
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
class TentativePFactory;
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
class SaPFactory;
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
class TransPFactory;
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
class DirectSolver;
template<typename Scalar, typename LocalOrdinal, typename GlobalOrdinal, typename Node>
class SmootherFactory;

using Scalar        = double;
using LocalOrdinal  = int;
using GlobalOrdinal = int;
using Node          = Xpetra::EpetraNode;
} // namespace MueLu

namespace AMP {

namespace Operator {
class LinearOperator;
}

namespace Solver {


using TrilinosMueLuSolverParameters = SolverStrategyParameters;
using SC                            = MueLu::Scalar;
using LO                            = MueLu::LocalOrdinal;
using GO                            = MueLu::GlobalOrdinal;
using NO                            = MueLu::Node;

/**
 * The TrilinosMueLuSolver is a wrapper to the Trilinos ML solver. ML provides implementations of
 * various algebraic multigrid methods. The wrapper at present simply provides an adaptor
 * to enable AMP users to use the black box ML preconditioner.
 */

class TrilinosMueLuSolver : public SolverStrategy
{

public:
    /**
     * Default constructor
     */
    TrilinosMueLuSolver();

    /**
     * Main constructor.
     @param [in] parameters The parameters object contains a database object which must contain the
     following fields in addition to the fields expected by the base class SolverStrategy class:

    */
    explicit TrilinosMueLuSolver( std::shared_ptr<TrilinosMueLuSolverParameters> parameters );

    /**
     * Default destructor
     */
    virtual ~TrilinosMueLuSolver();

    std::string type() const override { return "TrilinosMueLuSolver"; }

    //! static create routine that is used by SolverFactory
    static std::unique_ptr<SolverStrategy>
    createSolver( std::shared_ptr<SolverStrategyParameters> solverStrategyParameters )
    {
        return std::make_unique<TrilinosMueLuSolver>( solverStrategyParameters );
    }

    /**
     * Solve the system \f$Au = f\f$.
     @param [in] f : shared pointer to right hand side vector
     @param [out] u : shared pointer to approximate computed solution
     */
    void apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                std::shared_ptr<AMP::LinearAlgebra::Vector> u ) override;

    /**
     * Solve the system \f$Au = f\f$.
     @param [in] f : shared pointer to right hand side vector
     @param [out] u : shared pointer to approximate computed solution
     */
    void solveWithHierarchy( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                             std::shared_ptr<AMP::LinearAlgebra::Vector> u );

    /**
     * Return a shared pointer to the ML_Epetra::MultiLevelPreconditioner object
     */
    inline std::shared_ptr<MueLu::EpetraOperator> getMLSolver( void ) { return d_mueluSolver; }

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
     * Currently every call to reset destroys the ML preconditioner object
     * and recreates it based on the parameters object. See constructor for
     * fields required for parameter object.
     */
    void reset( std::shared_ptr<SolverStrategyParameters> params ) override;

protected:
    void reSolveWithLU( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                        std::shared_ptr<AMP::LinearAlgebra::Vector> u );

    void getFromInput( std::shared_ptr<const AMP::Database> db );

    //! build the hierarchy using the defaults constructed by MueLu
    void buildHierarchyFromDefaults( void );

    //! build the hierarchy level by level, potentially customizing each level
    void buildHierarchyByLevel( void );

    //! utility function to extract Xpetra Matrix from AMP LinearOperator
    Teuchos::RCP<Xpetra::Matrix<SC, LO, GO, NO>> getXpetraMatrix();

    Teuchos::RCP<MueLu::TentativePFactory<SC, LO, GO, NO>> getTentativePFactory( void );
    Teuchos::RCP<MueLu::SaPFactory<SC, LO, GO, NO>> getSaPFactory( void );
    Teuchos::RCP<MueLu::TransPFactory<SC, LO, GO, NO>> getRFactory( void );
    Teuchos::RCP<MueLu::SmootherFactory<SC, LO, GO, NO>> getCoarseSolverFactory( void );
    Teuchos::RCP<MueLu::SmootherFactory<SC, LO, GO, NO>> getSmootherFactory( const int level );

    Teuchos::ParameterList &getSmootherParameters( const int level );
    MueLu::MsgType getVerbosityLevel();

private:
    bool d_bUseEpetra      = true;  //! whether we are using Epetra
    bool d_build_hierarchy = false; //! whether to explicitly build the hierarchy
    bool d_build_hierarchy_from_defaults =
        true; //! build the hierarchy using the defaults constructed by MueLu
#if 0
    // ifpack related -- could be used in future
    bool d_construct_partition = false; //! whether to construct user defined partitions
#endif
    bool d_bCreationPhase = false; //! set to true if the solver is yet to be initialized
    bool d_bRobustMode    = false; //! use a direct solver if the MG solve fails to converge

    AMP_MPI d_comm;

    size_t d_maxLevels; //! maximum number of levels

    std::string d_smoother_type; //! key for creating different smoothers

    std::shared_ptr<MueLu::EpetraOperator> d_mueluSolver;

    std::shared_ptr<AMP::LinearAlgebra::ManagedEpetraMatrix> d_matrix;
    Teuchos::ParameterList d_MueLuParameterList;

    //! manager that creates hierarchy
    Teuchos::RCP<MueLu::HierarchyManager<SC, LO, GO, NO>> d_mueluHierarchyManager;

    //! AMG hierarchy
    Teuchos::RCP<MueLu::Hierarchy<SC, LO, GO, NO>> d_mueluHierarchy;

    //! factory manager for MueLu components
    MueLu::FactoryManager<SC, LO, GO, NO> d_factoryManager;
    //! factory manager for MueLu components by level
    std::vector<Teuchos::RCP<MueLu::FactoryManager<SC, LO, GO, NO>>> d_levelFactoryManager;
};
} // namespace Solver
} // namespace AMP

#endif
