#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/trilinos/epetra/EpetraMatrixHelpers.h"
#include "AMP/matrices/trilinos/epetra/ManagedEpetraMatrix.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/operators/trilinos/TrilinosMatrixShellOperator.h"
#include "AMP/vectors/trilinos/epetra/EpetraVector.h"

#include "ProfilerApp.h"

DISABLE_WARNINGS
#include "Epetra_CrsMatrix.h"
#include "Trilinos_version.h"
#include "ml_MultiLevelPreconditioner.h"
#include "ml_include.h"
ENABLE_WARNINGS


namespace AMP::Solver {


/****************************************************************
 * Constructors / Destructor                                     *
 ****************************************************************/
TrilinosMLSolver::TrilinosMLSolver()
{
    d_ml             = nullptr;
    d_mlAggregate    = nullptr;
    d_bCreationPhase = true;
    d_bUseEpetra     = false;
    d_bRobustMode    = false;
}
TrilinosMLSolver::TrilinosMLSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{
    d_ml          = nullptr;
    d_mlAggregate = nullptr;
    AMP_ASSERT( parameters );
    initialize( parameters );
}
TrilinosMLSolver::~TrilinosMLSolver()
{
    if ( d_mlAggregate ) {
        ML_Aggregate_Destroy( &d_mlAggregate );
        d_mlAggregate = nullptr;
    }
    if ( d_ml ) {
        ML_Destroy( &d_ml );
        d_ml = nullptr;
    }
    d_mlSolver.reset();
    d_matrix.reset(); // Need to keep a copy of the matrix alive until after the solver is destroyed
}

void TrilinosMLSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    getFromInput( parameters->d_db );
    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
}

void TrilinosMLSolver::getFromInput( std::shared_ptr<AMP::Database> db )
{
    d_bRobustMode = db->getWithDefault<bool>( "ROBUST_MODE", false );
    d_bUseEpetra  = db->getWithDefault<bool>( "USE_EPETRA", true );
    d_mlOptions.reset( new MLoptions( db ) );
    if ( d_bUseEpetra ) {
        convertMLoptionsToTeuchosParameterList();
    }
}

void TrilinosMLSolver::convertMLoptionsToTeuchosParameterList()
{
    // output level, 0 being silent and 10 verbose
    d_MLParameterList.set( "ML output", std::max( d_iDebugPrintInfoLevel, 0 ) );

    // maximum number of levels
    d_MLParameterList.set( "max levels", d_mlOptions->d_maxLevels );
    d_MLParameterList.set( "prec type", d_mlOptions->d_precType );
    d_MLParameterList.set( "PDE equations", d_mlOptions->d_pdeEquations );
    d_MLParameterList.set( "cycle applications", d_iMaxIterations );

    d_MLParameterList.set( "increasing or decreasing", d_mlOptions->d_increasingDecreasing );
    d_MLParameterList.set( "aggregation: type", d_mlOptions->d_aggregationType );
    d_MLParameterList.set( "aggregation: damping factor", d_mlOptions->d_aggregationDampingFactor );
    d_MLParameterList.set( "aggregation: threshold", d_mlOptions->d_aggregationThreshold );
    d_MLParameterList.set( "aggregation: nodes per aggregate", d_mlOptions->d_nodesPerAggregate );
    d_MLParameterList.set( "aggregation: next-level aggregates per process",
                           d_mlOptions->d_nextLevelAggregatesPerProcess );

    d_MLParameterList.set( "eigen-analysis: type", d_mlOptions->d_eigenAnalysisType );
    d_MLParameterList.set( "eigen-analysis: iterations", d_mlOptions->d_eigenAnalysisIterations );

    d_MLParameterList.set( "smoother: sweeps", d_mlOptions->d_smootherSweeps );
    d_MLParameterList.set( "smoother: damping factor", d_mlOptions->d_smootherDampingFactor );
    d_MLParameterList.set( "smoother: pre or post", d_mlOptions->d_prePost );
    d_MLParameterList.set( "smoother: type", d_mlOptions->d_smootherType );

    d_MLParameterList.set( "energy minimization: enable", d_mlOptions->d_enableEnergyMinimization );

    d_MLParameterList.set( "coarse: type", d_mlOptions->d_coarseType );
    d_MLParameterList.set( "coarse: max size", d_mlOptions->d_coarseMaxSize );

#if TRILINOS_MAJOR_MINOR_VERSION >= 130000
    d_MLParameterList.set( "coarse: split communicator", false );
#endif

    d_MLParameterList.set( "aggregation: aux: enable", d_mlOptions->d_aggregationAuxEnable );
    d_MLParameterList.set( "aggregation: aux: threshold", d_mlOptions->d_aggregationAuxThreshold );

    d_MLParameterList.set( "null space: type", d_mlOptions->d_nullSpaceType );
    d_MLParameterList.set( "null space: dimension", d_mlOptions->d_nullSpaceDimension );
    d_MLParameterList.set( "null space: add default vectors",
                           d_mlOptions->d_nullSpaceAddDefaultVectors );
}


void TrilinosMLSolver::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    d_pOperator = op;
    AMP_INSIST( d_pOperator, "ERROR: TrilinosMLSolver::initialize() operator cannot be NULL" );

    if ( d_bUseEpetra ) {
        // Compute coordinates to give to ML if requested
        if ( d_mlOptions->d_aggregationAuxEnable ) {
            computeCoordinates( op );
            AMP_ASSERT( d_x_values.size() != 0 );
            AMP_ASSERT( d_y_values.size() != 0 );
            AMP_ASSERT( d_z_values.size() != 0 );
            d_MLParameterList.set( "x-coordinates", &d_x_values[0] );
            d_MLParameterList.set( "y-coordinates", &d_y_values[0] );
            d_MLParameterList.set( "z-coordinates", &d_z_values[0] );
        }

        // Compute null space manually if requested
        if ( d_mlOptions->d_nullSpaceType == "pre-computed" ) {
            computeNullSpace( op );
            AMP_ASSERT( d_null_space.size() != 0 );
            d_MLParameterList.set( "null space: vectors", &d_null_space[0] );
        }

        AMP_INSIST( d_mlOptions->d_nullSpaceType == "pre-computed" ?
                        d_mlOptions->d_pdeEquations == 3 :
                        true,
                    "Null space construction only available for mechanics (PDE_equations=3)" );

        auto linearOperator =
            std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
        AMP_INSIST( linearOperator, "linearOperator cannot be NULL" );

        d_matrix = AMP::LinearAlgebra::getEpetraMatrix( linearOperator->getMatrix() );
        AMP_INSIST( d_matrix, "d_matrix cannot be NULL" );

        d_mlSolver.reset( new ML_Epetra::MultiLevelPreconditioner(
            d_matrix->getEpetra_CrsMatrix(), d_MLParameterList, false ) );
    } else {
        auto matShellOperator =
            std::dynamic_pointer_cast<AMP::Operator::TrilinosMatrixShellOperator>( d_pOperator );
        AMP_ASSERT( matShellOperator );

        size_t matSize = matShellOperator->getMatrixSize();
        ML_Create( &d_ml, d_mlOptions->d_maxLevels );

        if ( ( d_mlOptions->d_increasingDecreasing ) == "increasing" ) {
            ML_Init_Amatrix( d_ml, 0, matSize, matSize, d_pOperator.get() );
            ML_Set_Amatrix_Getrow( d_ml,
                                   0,
                                   &( AMP::Operator::TrilinosMatrixShellOperator::getRow ),
                                   nullptr,
                                   matSize );
            ML_Set_Amatrix_Matvec(
                d_ml, 0, &( AMP::Operator::TrilinosMatrixShellOperator::matVec ) );
        } else {
            AMP_ERROR( "The option, increasingordecreasing = \"" +
                       d_mlOptions->d_increasingDecreasing + "\" , is not supported." );
        }
    }
    d_bCreationPhase = true;
}


void TrilinosMLSolver::resetOperator(
    std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    PROFILE( "resetOperator" );
    AMP_INSIST( ( d_pOperator ),
                "ERROR: TrilinosMLSolver::resetOperator() operator cannot be NULL" );
    d_pOperator->reset( params );
    reset( std::shared_ptr<SolverStrategyParameters>() );
}


void TrilinosMLSolver::reset( std::shared_ptr<SolverStrategyParameters> )
{
    PROFILE( "reset" );
    if ( d_mlAggregate ) {
        ML_Aggregate_Destroy( &d_mlAggregate );
        d_mlAggregate = nullptr;
    }
    if ( d_ml ) {
        ML_Destroy( &d_ml );
        d_ml = nullptr;
    }
    d_mlSolver.reset();
    d_matrix.reset(); // Need to keep a copy of the matrix alive until after the solver is destroyed
    registerOperator( d_pOperator );
}


void TrilinosMLSolver::apply( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                              std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "solve" );

    if ( u->getUpdateStatus() != AMP::LinearAlgebra::UpdateState::UNCHANGED ) {
        u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    }

    // in this case we make the assumption we can access a EpetraMat for now
    AMP_INSIST( d_pOperator, "ERROR: TrilinosMLSolver::apply() operator cannot be NULL" );

    auto r = f->clone();
    if ( d_bUseZeroInitialGuess ) {
        u->zero();
        r->copyVector( f );
    } else {
        d_pOperator->residual( f, u, r );
    }

    if ( d_bCreationPhase ) {
        if ( d_bUseEpetra ) {
            d_mlSolver->ComputePreconditioner();
            if ( d_iDebugPrintInfoLevel > 2 ) {
                d_mlSolver->PrintUnused();
            }
        } else {
            buildML();
        }
        d_bCreationPhase = false;
    }

    d_dInitialResidual = d_dResidualNorm = r->L2Norm();
    checkStoppingCriteria( d_dResidualNorm );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "TrilinosMLSolver::apply(), L2 norm of residual before solve "
                  << std::setprecision( 15 ) << d_dInitialResidual << std::endl;
    }

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "TrilinosMLSolver : before solve solution norm: " << std::setprecision( 15 )
                  << u->L2Norm() << std::endl;
    }

    if ( d_bUseEpetra ) {
        // These functions throw exceptions if this cannot be performed.
        AMP_ASSERT( f );

        auto f_epetra             = AMP::LinearAlgebra::EpetraVector::constView( f );
        auto u_epetra             = AMP::LinearAlgebra::EpetraVector::view( u );
        const Epetra_Vector &fVec = f_epetra->getEpetra_Vector();
        Epetra_Vector &uVec       = u_epetra->getEpetra_Vector();

        d_mlSolver->ApplyInverse( fVec, uVec );
    } else {
        auto uArr = new double[u->getLocalSize()];
        auto fArr = new double[f->getLocalSize()];
        u->getRawData( uArr );
        f->getRawData( fArr );
        ML_Iterate( d_ml, uArr, fArr );
        u->putRawData( uArr );
        delete[] uArr;
        delete[] fArr;
    }

    // Check for NaNs in the solution (no communication necessary)
    double localNorm = u->getVectorOperations()->localL2Norm( *u->getVectorData() ).get<double>();
    AMP_INSIST( localNorm == localNorm, "NaNs detected in solution" );

    // we are forced to update the state of u here
    // as Epetra is not going to change the state of a managed vector
    // an example where this will and has caused problems is when the
    // vector is a petsc managed vector being passed back to PETSc
    //    u->getVectorData()->fireDataChange();
    u->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    if ( d_iDebugPrintInfoLevel > 2 ) {
        AMP::pout << "TrilinosMLSolver : after solve solution norm: " << std::setprecision( 15 )
                  << u->L2Norm() << std::endl;
    }

    d_pOperator->residual( f, u, r );
    d_dResidualNorm = r->L2Norm();
    checkStoppingCriteria( d_dResidualNorm );

    if ( d_iDebugPrintInfoLevel > 1 ) {
        AMP::pout << "TrilinosMLSolver::apply(), L2 norm of residual after solve "
                  << std::setprecision( 15 ) << d_dResidualNorm << std::endl;
    }

    if ( d_bRobustMode ) {
        if ( d_dResidualNorm > d_dInitialResidual ) {
            AMP::pout << "Warning: ML was not able to reduce the residual. Using LU instead."
                      << std::endl;
            reSolveWithLU( f, u );
        }
    }
    AMP_DEBUG_ASSERT( u->getUpdateStatus() == AMP::LinearAlgebra::UpdateState::UNCHANGED );
}


void TrilinosMLSolver::reSolveWithLU( std::shared_ptr<const AMP::LinearAlgebra::Vector> f,
                                      std::shared_ptr<AMP::LinearAlgebra::Vector> u )
{
    PROFILE( "reSolveWithLU" );

    if ( !d_bUseEpetra ) {
        AMP_ERROR( "Robust mode can only be used with Epetra matrices." );
    }

    std::shared_ptr<AMP::Operator::LinearOperator> linearOperator =
        std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_INSIST( linearOperator, "linearOperator cannot be NULL" );

    d_matrix = std::dynamic_pointer_cast<AMP::LinearAlgebra::ManagedEpetraMatrix>(
        linearOperator->getMatrix() );
    AMP_INSIST( d_matrix, "d_matrix cannot be NULL" );

    Teuchos::ParameterList tmpMLParameterList;
    tmpMLParameterList.set( "ML output", d_iDebugPrintInfoLevel );
    tmpMLParameterList.set( "max levels", 1 );
    tmpMLParameterList.set( "PDE equations", d_mlOptions->d_pdeEquations );
    tmpMLParameterList.set( "coarse: type", d_mlOptions->d_coarseType );

    d_mlSolver.reset( new ML_Epetra::MultiLevelPreconditioner(
        d_matrix->getEpetra_CrsMatrix(), tmpMLParameterList, false ) );
    d_bCreationPhase = true;

    apply( f, u );

    d_mlSolver.reset( new ML_Epetra::MultiLevelPreconditioner(
        d_matrix->getEpetra_CrsMatrix(), d_MLParameterList, false ) );
    d_bCreationPhase = true;
}


void TrilinosMLSolver::buildML()
{
    ML_Set_MaxIterations( d_ml, d_iMaxIterations );
    ML_Set_PrintLevel( d_iDebugPrintInfoLevel );
    ML_Set_OutputLevel( d_ml, d_iDebugPrintInfoLevel );
    if ( d_iDebugPrintInfoLevel ) {
        ML_Set_ResidualOutputFrequency( d_ml, 1 );
    }

    ML_Aggregate_Create( &d_mlAggregate );

    d_mlAggregate->num_PDE_eqns  = d_mlOptions->d_pdeEquations;
    d_mlAggregate->nullspace_dim = d_mlOptions->d_pdeEquations;

    ML_Aggregate_Set_MaxCoarseSize( d_mlAggregate, ( d_mlOptions->d_coarseMaxSize ) );
    if ( ( d_mlOptions->d_aggregationType ) == "Uncoupled-MIS" ) {
        ML_Aggregate_Set_CoarsenScheme_UncoupledMIS( d_mlAggregate );
    } else {
        AMP_ERROR( "The option, aggregationtype = \"" + d_mlOptions->d_aggregationType +
                   "\" , is not supported." );
    }

    int nlevels = ML_Gen_MGHierarchy_UsingAggregation( d_ml, 0, ML_INCREASING, d_mlAggregate );
    AMP::pout << "Number of actual levels : " << nlevels << std::endl;

    if ( ( d_mlOptions->d_smootherType ) == "symmetric Gauss-Seidel" ) {
        for ( int lev = 0; lev < ( nlevels - 1 ); lev++ ) {
            if ( ( d_mlOptions->d_prePost ) == "pre" ) {
                ML_Gen_Smoother_SymGaussSeidel( d_ml,
                                                lev,
                                                ML_PRESMOOTHER,
                                                ( d_mlOptions->d_smootherSweeps ),
                                                ( d_mlOptions->d_smootherDampingFactor ) );
            } else if ( ( d_mlOptions->d_prePost ) == "post" ) {
                ML_Gen_Smoother_SymGaussSeidel( d_ml,
                                                lev,
                                                ML_POSTSMOOTHER,
                                                ( d_mlOptions->d_smootherSweeps ),
                                                ( d_mlOptions->d_smootherDampingFactor ) );
            } else if ( ( d_mlOptions->d_prePost ) == "both" ) {
                ML_Gen_Smoother_SymGaussSeidel( d_ml,
                                                lev,
                                                ML_BOTH,
                                                ( d_mlOptions->d_smootherSweeps ),
                                                ( d_mlOptions->d_smootherDampingFactor ) );
            } else {
                AMP_ERROR( "The option, smoother_preorpost = \"" + d_mlOptions->d_prePost +
                           "\" , is not supported." );
            }
        }
    } else {
        AMP_ERROR( "The option, smoothertype = \"" + d_mlOptions->d_smootherType +
                   "\" , is not supported." );
    }

    if ( ( d_mlOptions->d_coarseType ) == "Amesos-KLU" ) {
#if TRILINOS_MAJOR_MINOR_VERSION >= 130000
        ML_Gen_Smoother_Amesos( d_ml, ( nlevels - 1 ), ML_AMESOS_KLU, -1, 0.0, 1 );
#else
        ML_Gen_Smoother_Amesos( d_ml, ( nlevels - 1 ), ML_AMESOS_KLU, -1, 0.0 );
#endif
    } else {
        AMP_ERROR( "The option, coarse_type = \"" + d_mlOptions->d_coarseType +
                   "\" , is not supported." );
    }

    if ( ( d_mlOptions->d_precType ) == "MGV" ) {
        ML_Gen_Solver( d_ml, ML_MGV, 0, ( nlevels - 1 ) );
    } else if ( ( d_mlOptions->d_precType ) == "MGW" ) {
        ML_Gen_Solver( d_ml, ML_MGW, 0, ( nlevels - 1 ) );
    } else {
        AMP_ERROR( "The option, prec_type = \"" + d_mlOptions->d_precType +
                   "\" , is not supported." );
    }
}


void TrilinosMLSolver::computeCoordinates( std::shared_ptr<AMP::Operator::Operator> op )
{
    // Get mesh adapter for this operator
    auto myMesh = op->getMesh();

    // Resize vectors to hold node values
    int numNodes = myMesh->numLocalElements( AMP::Mesh::GeomType::Vertex );
    d_x_values.resize( numNodes, 0.0 );
    d_y_values.resize( numNodes, 0.0 );
    d_z_values.resize( numNodes, 0.0 );

    // Get node iterators
    auto thisNode = myMesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto endNode  = thisNode.end();

    int nodeCounter = 0;
    for ( ; thisNode != endNode; ++thisNode ) {
        auto coord = thisNode->coord();
        AMP_INSIST( coord.size() == 3, "Currently only programmed for 3d" );
        d_x_values[nodeCounter] = coord[0];
        d_y_values[nodeCounter] = coord[1];
        d_z_values[nodeCounter] = coord[2];
        nodeCounter++;
    }
}


void TrilinosMLSolver::computeNullSpace( std::shared_ptr<AMP::Operator::Operator> op )
{
    // Get mesh adapter for this operator
    auto myMesh = op->getMesh();
    int numPDE  = d_mlOptions->d_pdeEquations;
    int dimNS   = d_mlOptions->d_nullSpaceDimension;

    AMP_INSIST( d_mlOptions->d_nullSpaceDimension == 6,
                "Null space dimension must be 6 to use computed null space." );

    // Resize vectors to hold node values
    int numNodes  = myMesh->numLocalElements( AMP::Mesh::GeomType::Vertex );
    int vecLength = numPDE * numNodes;
    d_null_space.resize( dimNS * vecLength, 0.0 );

    // Get node iterators
    auto thisNode = myMesh->getIterator( AMP::Mesh::GeomType::Vertex, 0 );
    auto endNode  = thisNode.end();

    int nodeCounter = 0;
    for ( ; thisNode != endNode; ++thisNode ) {
        auto coord = thisNode->coord();
        AMP_INSIST( coord.size() == 3, "Currently only programmed for 3d" );
        double thisX = coord[0];
        double thisY = coord[1];
        double thisZ = coord[2];

        int dof = numPDE * nodeCounter;

        // Constant vector for each PDE
        for ( int i = 0; i < numPDE; ++i ) {
            int offset           = i * vecLength + dof + i;
            d_null_space[offset] = 1.0;
        }

        // Rotation around X
        int offset               = 3 * vecLength + dof;
        d_null_space[offset + 1] = -thisZ;
        d_null_space[offset + 2] = thisY;

        // Rotation around Y
        offset                   = 4 * vecLength + dof;
        d_null_space[offset]     = thisZ;
        d_null_space[offset + 2] = -thisX;

        // Rotation around Z
        offset                   = 5 * vecLength + dof;
        d_null_space[offset]     = -thisY;
        d_null_space[offset + 1] = thisX;

        nodeCounter++;
    }
}


} // namespace AMP::Solver
