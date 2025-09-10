#include "AMP/solvers/hypre/HypreSolver.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/data/hypre/HypreMatrixAdaptor.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"

#include "ProfilerApp.h"

#include <iomanip>
#include <numeric>

DISABLE_WARNINGS
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_config.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "_hypre_parcsr_mv.h"
ENABLE_WARNINGS


namespace AMP::Solver {


/****************************************************************
 * Constructors / Destructor                                     *
 ****************************************************************/
HypreSolver::HypreSolver() : SolverStrategy() {}
HypreSolver::HypreSolver( std::shared_ptr<SolverStrategyParameters> parameters )
    : SolverStrategy( parameters )
{
    AMP_ASSERT( parameters );
    HypreSolver::initialize( parameters );
}

HypreSolver::~HypreSolver()
{
    HYPRE_IJVectorDestroy( d_hypre_rhs );
    HYPRE_IJVectorDestroy( d_hypre_sol );
}

void HypreSolver::initialize( std::shared_ptr<const SolverStrategyParameters> parameters )
{
    AMP_ASSERT( parameters );

    HypreSolver::getFromInput( parameters->d_db );

    if ( d_pOperator ) {
        registerOperator( d_pOperator );
    }
}

void HypreSolver::createHYPREMatrix( std::shared_ptr<AMP::LinearAlgebra::Matrix> matrix )
{
    d_HypreMatrixAdaptor =
        std::make_shared<AMP::LinearAlgebra::HypreMatrixAdaptor>( matrix->getMatrixData() );
    AMP_ASSERT( d_HypreMatrixAdaptor );
    d_ijMatrix = d_HypreMatrixAdaptor->getHypreMatrix();
    if ( d_iDebugPrintInfoLevel > 3 ) {
        HYPRE_IJMatrixPrint( d_ijMatrix, "HypreMatrix" );
    }
}

void HypreSolver::createHYPREVectors()
{
    char hypre_mesg[100];

    auto linearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_INSIST( linearOperator, "linearOperator cannot be NULL" );

    const auto &matrix = linearOperator->getMatrix();
    AMP_INSIST( matrix, "matrix cannot be NULL" );

    const auto myFirstRow = matrix->getLeftDOFManager()->beginDOF();
    const auto myEndRow =
        matrix->getLeftDOFManager()->endDOF(); // check whether endDOF is truly the last -1
    int ierr;

    // create the rhs
    ierr = HYPRE_IJVectorCreate( d_comm.getCommunicator(), myFirstRow, myEndRow - 1, &d_hypre_rhs );
    HYPRE_DescribeError( ierr, hypre_mesg );
    ierr = HYPRE_IJVectorSetObjectType( d_hypre_rhs, HYPRE_PARCSR );
    HYPRE_DescribeError( ierr, hypre_mesg );

    // create the solution vector
    ierr = HYPRE_IJVectorCreate( d_comm.getCommunicator(), myFirstRow, myEndRow - 1, &d_hypre_sol );
    HYPRE_DescribeError( ierr, hypre_mesg );
    ierr = HYPRE_IJVectorSetObjectType( d_hypre_sol, HYPRE_PARCSR );
    HYPRE_DescribeError( ierr, hypre_mesg );
}

static AMP::Utilities::MemoryType getAMPMemorySpace( HYPRE_MemoryLocation memory_location )
{
    if ( memory_location == HYPRE_MEMORY_HOST ) {
        return AMP::Utilities::MemoryType::host;
    } else if ( memory_location == HYPRE_MEMORY_DEVICE ) {
#if defined( HYPRE_USING_DEVICE_MEMORY )
        return AMP::Utilities::MemoryType::device;
#elif defined( HYPRE_USING_UNIFIED_MEMORY )
        return AMP::Utilities::MemoryType::managed;
#else
        AMP_ERROR( "Unable to detect Hypre memory location" );
        return AMP::Utilities::MemoryType::device;
#endif
    } else {
        AMP_ERROR( "Unable to detect Hypre memory location" );
        return AMP::Utilities::MemoryType::host;
    }
}

void HypreSolver::copyToHypre( std::shared_ptr<const AMP::LinearAlgebra::Vector> amp_v,
                               HYPRE_IJVector hypre_v )
{
    char hypre_mesg[100];
    int ierr;

    AMP_INSIST( amp_v, "vector cannot be NULL" );
    AMP_INSIST( amp_v->numberOfDataBlocks() == 1,
                "Copy from AMP vectors with more than one data block to Hypre not implemented" );
    const auto &dofManager = amp_v->getDOFManager();
    AMP_INSIST( dofManager, "DOF_Manager cannot be NULL" );

    const auto nDOFS = dofManager->numLocalDOF();

    ierr = HYPRE_IJVectorInitialize( hypre_v );
    HYPRE_DescribeError( ierr, hypre_mesg );

    HYPRE_Real *vals_p = nullptr;

    if ( amp_v->isType<HYPRE_Real>( 0 ) ) {

        vals_p = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( amp_v )
                     ->getRawDataBlock<HYPRE_Real>();

        auto memType      = AMP::Utilities::getMemoryType( vals_p );
        auto hypreMemType = getAMPMemorySpace( d_memory_location );
        // see if memory spaces are compatible
        if ( memType == hypreMemType ) {
            AMP_ASSERT( vals_p );
            ierr = HYPRE_IJVectorSetValues( hypre_v, nDOFS, nullptr, vals_p );
            HYPRE_DescribeError( ierr, hypre_mesg );
        } else {
            auto compat_amp_v = AMP::LinearAlgebra::createVector( amp_v, hypreMemType );
            compat_amp_v->copyVector( amp_v );
            vals_p = compat_amp_v->getRawDataBlock<HYPRE_Real>();
            ierr   = HYPRE_IJVectorSetValues( hypre_v, nDOFS, nullptr, vals_p );
            HYPRE_DescribeError( ierr, hypre_mesg );
        }

    } else {
        AMP_ERROR( "Copy from AMP to Hypre vectors of different precision not implemented" );
    }

    ierr = HYPRE_IJVectorAssemble( hypre_v );
    HYPRE_DescribeError( ierr, hypre_mesg );
}

void HypreSolver::copyFromHypre( HYPRE_IJVector hypre_v,
                                 std::shared_ptr<AMP::LinearAlgebra::Vector> amp_v )
{
    AMP_INSIST( amp_v, "vector cannot be NULL" );
    AMP_INSIST( amp_v->numberOfDataBlocks() == 1,
                "Copy from Hypre to AMP vectors with more than one data block not "
                "implemented" );
    const auto &dofManager = amp_v->getDOFManager();
    AMP_INSIST( dofManager, "DOF_Manager cannot be NULL" );
    const auto nDOFS = dofManager->numLocalDOF();

    char hypre_mesg[100];
    int ierr;

    auto vals_p = amp_v->getRawDataBlock<HYPRE_Real>();

    if ( amp_v->isType<HYPRE_Real>( 0 ) ) {

        auto memType      = AMP::Utilities::getMemoryType( vals_p );
        auto hypreMemType = getAMPMemorySpace( d_memory_location );
        // see if memory spaces are compatible
        if ( memType == hypreMemType ) {
            ierr = HYPRE_IJVectorGetValues(
                hypre_v, static_cast<HYPRE_Int>( nDOFS ), nullptr, vals_p );
            HYPRE_DescribeError( ierr, hypre_mesg );
            return;
        } else {

            AMP_WARN_ONCE(
                "Hypre not built with support for AMP vector memory, vector will be migrated" );

            auto tmp_amp_v = AMP::LinearAlgebra::createVector( amp_v, hypreMemType );
            AMP_ASSERT( tmp_amp_v );
            vals_p = tmp_amp_v->getRawDataBlock<HYPRE_Real>();
            ierr   = HYPRE_IJVectorGetValues(
                hypre_v, static_cast<HYPRE_Int>( nDOFS ), nullptr, vals_p );
            HYPRE_DescribeError( ierr, hypre_mesg );
            amp_v->copyVector( tmp_amp_v );
        }

    } else {
        AMP_ERROR( "Copy from Hypre to AMP vectors of different precision not implemented" );
    }
}

void HypreSolver::setupHypreMatrixAndRhs()
{
    auto linearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( d_pOperator );
    AMP_INSIST( linearOperator, "linearOperator cannot be NULL" );
    auto matrix = linearOperator->getMatrix();
    // a user can choose to set the Operator without initializing the matrix
    // so check whether a valid matrix exists
    if ( matrix ) {

        // set the comm for this solver based on the comm for the matrix
        const auto &dofManager = matrix->getLeftDOFManager();
        d_comm                 = dofManager->getComm();

        // set the hypre memory and execution spaces from the operator
        if ( linearOperator->getMemoryLocation() > AMP::Utilities::MemoryType::host ) {
            d_memory_location = HYPRE_MEMORY_DEVICE;
            d_exec_policy     = HYPRE_EXEC_DEVICE;
        } else {
            d_memory_location = HYPRE_MEMORY_HOST;
            d_exec_policy     = HYPRE_EXEC_HOST;
        }

        createHYPREMatrix( matrix );
        createHYPREVectors();
        d_bMatrixInitialized = true;
    }
}

void HypreSolver::registerOperator( std::shared_ptr<AMP::Operator::Operator> op )
{
    d_pOperator = op;
    AMP_INSIST( d_pOperator, "ERROR: HypreSolver::registerOperator() operator cannot be NULL" );
    setupHypreMatrixAndRhs();
}

void HypreSolver::resetOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params )
{
    PROFILE( "resetOperator" );
    AMP_INSIST( ( d_pOperator ), "ERROR: HypreSolver::resetOperator() operator cannot be NULL" );
    d_bMatrixInitialized = false;
    d_pOperator->reset( params );
}


void HypreSolver::reset( std::shared_ptr<SolverStrategyParameters> )
{
    PROFILE( "reset" );
    registerOperator( d_pOperator );
}

} // namespace AMP::Solver
