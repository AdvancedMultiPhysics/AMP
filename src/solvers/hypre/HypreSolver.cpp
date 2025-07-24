#include "AMP/solvers/hypre/HypreSolver.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/matrices/Matrix.h"
#include "AMP/matrices/data/hypre/HypreMatrixAdaptor.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/utils/Utilities.h"

#include "ProfilerApp.h"

#include <iomanip>
#include <numeric>

DISABLE_WARNINGS
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
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

void HypreSolver::copyToHypre( std::shared_ptr<const AMP::LinearAlgebra::Vector> amp_v,
                               HYPRE_IJVector hypre_v )
{
    char hypre_mesg[100];
    int ierr;

    AMP_INSIST( amp_v, "vector cannot be NULL" );
    const auto &dofManager = amp_v->getDOFManager();
    AMP_INSIST( dofManager, "DOF_Manager cannot be NULL" );

    const auto nDOFS         = dofManager->numLocalDOF();
    const auto startingIndex = dofManager->beginDOF();

    std::vector<HYPRE_Real> values;
    HYPRE_Real *vals = nullptr;

    if ( amp_v->numberOfDataBlocks() == 1 ) {

        if ( amp_v->isType<HYPRE_Real>( 0 ) ) {
            vals = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( amp_v )
                       ->getRawDataBlock<HYPRE_Real>();
        } else {
            auto block0 = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( amp_v )
                              ->getRawDataBlock<HYPRE_Real>();
            auto memType = AMP::Utilities::getMemoryType( block0 );
            AMP_INSIST( memType < AMP::Utilities::MemoryType::device,
                        "Implemented only for AMP vector memory on host" );
            std::vector<size_t> indices( nDOFS, 0 );
            std::iota( indices.begin(), indices.end(), startingIndex );
            values.resize( nDOFS );
            vals = values.data();
            amp_v->getValuesByGlobalID( nDOFS, indices.data(), vals );
        }


    } else {
        auto block0 = std::const_pointer_cast<AMP::LinearAlgebra::Vector>( amp_v )
                          ->getRawDataBlock<HYPRE_Real>();
        auto memType = AMP::Utilities::getMemoryType( block0 );
        AMP_INSIST( memType < AMP::Utilities::MemoryType::device,
                    "Implemented only for AMP vector memory on host" );

        for ( auto it = amp_v->begin<HYPRE_Real>(); it != amp_v->end<HYPRE_Real>(); ++it ) {
            values.push_back( *it );
        }

        vals = values.data();
    }

    AMP_ASSERT( vals );
    ierr = HYPRE_IJVectorInitialize( hypre_v );
    HYPRE_DescribeError( ierr, hypre_mesg );
    ierr = HYPRE_IJVectorSetValues( hypre_v, nDOFS, nullptr, vals );
    HYPRE_DescribeError( ierr, hypre_mesg );
    ierr = HYPRE_IJVectorAssemble( hypre_v );
    HYPRE_DescribeError( ierr, hypre_mesg );

    // this can be optimized in future so that memory is allocated based on the location
    HYPRE_ParVector par_v;
    HYPRE_IJVectorGetObject( hypre_v, (void **) &par_v );
    hypre_ParVectorMigrate( par_v, d_memory_location );
}

template<typename T>
static void copy_to_amp( std::shared_ptr<AMP::LinearAlgebra::Vector> amp_v, HYPRE_Real *values )
{
    AMP_ASSERT( amp_v && values );
    size_t i = 0;
    for ( auto it = amp_v->begin<T>(); it != amp_v->end<T>(); ++it ) {
        *it = values[i];
        ++i;
    }
}

void HypreSolver::copyFromHypre( HYPRE_IJVector hypre_v,
                                 std::shared_ptr<AMP::LinearAlgebra::Vector> amp_v )
{
    char hypre_mesg[100];

    int ierr;

    AMP_INSIST( amp_v, "vector cannot be NULL" );
    const auto &dofManager = amp_v->getDOFManager();
    AMP_INSIST( dofManager, "DOF_Manager cannot be NULL" );

    const auto nDOFS = dofManager->numLocalDOF();

    auto block0  = amp_v->getRawDataBlock<HYPRE_Real>();
    auto memType = AMP::Utilities::getMemoryType( block0 );

    if ( memType != AMP::Utilities::MemoryType::device ) {

        // this can be optimized in future so that there's less memory movement
        // likewise we should distinguish between managed and host options
        HYPRE_ParVector par_v;
        HYPRE_IJVectorGetObject( hypre_v, (void **) &par_v );
        hypre_ParVectorMigrate( par_v, HYPRE_MEMORY_HOST );
        std::vector<HYPRE_Real> values( nDOFS, 0.0 );
        auto values_p = values.data();
        ierr =
            HYPRE_IJVectorGetValues( hypre_v, static_cast<HYPRE_Int>( nDOFS ), nullptr, values_p );
        HYPRE_DescribeError( ierr, hypre_mesg );

        if ( amp_v->numberOfDataBlocks() == 1 ) {
            const auto startingIndex = dofManager->beginDOF();
            std::vector<size_t> indices( nDOFS, 0 );
            std::iota( indices.begin(), indices.end(), startingIndex );
            amp_v->setLocalValuesByGlobalID( nDOFS, indices.data(), values_p );

        } else {

            if ( amp_v->isType<double>( 0 ) ) {
                copy_to_amp<double>( amp_v, values_p );
            } else if ( amp_v->isType<float>( 0 ) ) {
                copy_to_amp<float>( amp_v, values_p );
            } else {
                AMP_ERROR( "Implemented only for double and float" );
            }
        }

    } else {
        AMP_ERROR( "Not implemented for AMP vector with device memory" );
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
