#include "AMP/time_integrators/sundials/IDATimeIntegrator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/time_integrators/LinearTimeOperator.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/Utilities.h"

#include <iostream>


#ifdef AMP_USE_SUNDIALS
// Note:  sundials 2.4.0 has a memory leak that can cause some tests to fail
extern "C" {
    #include "ida/ida_spils.h"
}


namespace AMP::TimeIntegrator {


/************************************************************************
 *  Constructor.                                                         *
 ************************************************************************/
IDATimeIntegrator::IDATimeIntegrator( std::shared_ptr<TimeIntegratorParameters> parameters )
    : TimeIntegrator( parameters )
{
    initialize( parameters );
}


/************************************************************************
 *  Destructor.                                                          *
 ************************************************************************/
IDATimeIntegrator::~IDATimeIntegrator()
{
    // Delete the time operator
    d_pIDATimeOperator.reset();
    // Call SUNDIALS memory deleter
    IDAFree( &d_ida_mem );
    // Note: There is a memory leak in sundials-2.4.0 in ida_spgmr.c,
    // we need to add "N_VDestroy(yptemp);" to IDASpgmrFree.
    // AMP::pout << "Possible memory leak in IDA:" << std::endl;
}


/************************************************************************
 * Initialize.                                                           *
 ************************************************************************/
void IDATimeIntegrator::initialize( std::shared_ptr<TimeIntegratorParameters> parameters )
{
    #ifdef DEBUG_CHECK_ASSERTIONS
    AMP_ASSERT( parameters.get() != NULL );
    #endif
    getFromInput( parameters->d_db );

    auto params      = std::dynamic_pointer_cast<IDATimeIntegratorParameters>( parameters );
    d_solution_prime = ( params->d_ic_vector_prime )->clone();
    d_solution_prime->copyVector( params->d_ic_vector_prime );

    d_pNestedSolver = params->d_pNestedSolver;

    // reuse the time integrator database, and put additional fields in
    auto timeOperator_db = params->d_db;
    timeOperator_db->putScalar( "CurrentDt", d_current_dt );
    timeOperator_db->putScalar( "CurrentTime", d_current_time );
    timeOperator_db->putScalar( "name", "TimeOperator" );

    // setup the parameter object for the IDATimeOperator
    auto idaTimeOp_Params =
        std::make_shared<AMP::TimeIntegrator::IDATimeOperatorParameters>( params->d_db );

    idaTimeOp_Params->d_pRhsOperator       = parameters->d_operator;
    idaTimeOp_Params->d_pMassOperator      = parameters->d_pMassOperator;
    idaTimeOp_Params->d_pSourceTerm        = parameters->d_pSourceTerm;
    idaTimeOp_Params->d_pAlgebraicVariable = parameters->d_pAlgebraicVariable;
    idaTimeOp_Params->d_Mesh               = ( parameters->d_operator )->getMesh();

    // create the time operator
    auto idaTimeOp     = std::make_shared<AMP::TimeIntegrator::IDATimeOperator>( idaTimeOp_Params );
    d_pIDATimeOperator = idaTimeOp;

    // if we want to create the LinearTimeOperator internally
    if ( d_createLinearOperatorInternally ) {
        if ( d_bLinearRhsOperator && d_bLinearMassOperator ) {
            auto linearTimeOperatorParams =
                std::dynamic_pointer_cast<AMP::TimeIntegrator::TimeOperatorParameters>(
                    idaTimeOp->getParameters( "Jacobian", d_solution_vector ) );
            std::shared_ptr<AMP::Database> linearTimeOperator_db = linearTimeOperatorParams->d_db;
            linearTimeOperator_db->putScalar( "CurrentDt", d_current_dt );
            linearTimeOperator_db->putScalar( "CurrentTime", d_current_time );
            linearTimeOperator_db->putScalar( "name", "TimeOperator" );

            d_pLinearTimeOperator.reset( new LinearTimeOperator( linearTimeOperatorParams ) );
        } else {
            AMP::pout << "ERROR: IDATimeIntegrator::initialize(): creation of linear time "
                         "operators internally is only "
                         "currently supported for linear mass and rhs operators"
                      << std::endl;
        }

        AMP_INSIST( d_pNestedSolver,
                    "ERROR: IDATimeIntegrator::initialize(): creation of linear time "
                    "operators internally is only currently supported with a valid "
                    "non NULL preconditioner " );

        d_pNestedSolver->registerOperator( d_pLinearTimeOperator );
        AMP::pout << " linear op being created internally" << std::endl;
    }

    // Initialize data members from input.
    setupVectors();
    initializeIDA();
}


void IDATimeIntegrator::initializeIDA()
{
    N_Vector id = nullptr;

    auto pSundials_sol = std::dynamic_pointer_cast<AMP::LinearAlgebra::SundialsVector>(
        AMP::LinearAlgebra::SundialsVector::view( d_solution_vector ) );
    auto pSundials_sol_prime = std::dynamic_pointer_cast<AMP::LinearAlgebra::SundialsVector>(
        AMP::LinearAlgebra::SundialsVector::view( d_solution_prime ) );

    id = N_VClone( pSundials_sol->getNVector() );

    d_ida_mem = IDACreate();
    AMP_ASSERT( d_ida_mem != nullptr );

    int ierr = IDASetUserData( d_ida_mem, this );
    AMP_ASSERT( ierr == IDA_SUCCESS );

    N_VConst( 1.0, id );
    ierr = IDASetId( d_ida_mem, id );
    AMP_ASSERT( ierr == IDA_SUCCESS );

    ierr = IDAInit( d_ida_mem,
                    IDAResTrial,
                    d_initial_time,
                    pSundials_sol->getNVector(),
                    pSundials_sol_prime->getNVector() );
    AMP_ASSERT( ierr == IDA_SUCCESS );

    ierr = IDASStolerances( d_ida_mem, d_relative_tolerance, d_absolute_tolerance );
    AMP_ASSERT( ierr == IDA_SUCCESS );

    N_VDestroy( id );
    // set the initial step size
    ierr = IDASetInitStep( d_ida_mem, d_initial_dt );
    AMP_ASSERT( ierr == IDA_SUCCESS );

    // set the max step size
    ierr = IDASetMaxStep( d_ida_mem, d_max_dt );
    AMP_ASSERT( ierr == IDA_SUCCESS );

    // set final time
    ierr = IDASetStopTime( d_ida_mem, d_final_time );
    AMP_ASSERT( ierr == IDA_SUCCESS );


    // ideally, the linear solver type needs to be determined by the user input
    ierr = IDASpgmr( d_ida_mem, 0 );
    AMP_ASSERT( ierr == IDA_SUCCESS );
    if ( d_bUsePreconditioner ) {
        ierr = IDASpilsSetPreconditioner( d_ida_mem, IDAPrecSetup, IDAPrecSolve );
        AMP_ASSERT( ierr == IDASPILS_SUCCESS );
    }

    // ierr = IDASpilsSetMaxRestarts(d_ida_mem, 15);
    ierr = IDASpilsSetMaxRestarts( d_ida_mem, 100 );
    AMP_ASSERT( ierr == IDA_SUCCESS );

    // if we want IDA to calculate consistent IC's
    if ( d_bCallCalcIC ) {
        double tout1 = d_initial_time + d_current_dt;
        // choice of the last argument...?
        ierr = IDACalcIC( d_ida_mem, IDA_YA_YDP_INIT, tout1 );
        AMP_ASSERT( ierr == IDA_SUCCESS );
        ierr = IDAGetConsistentIC(
            d_ida_mem, pSundials_sol->getNVector(), pSundials_sol_prime->getNVector() );
        AMP_ASSERT( ierr == IDA_SUCCESS );
    }

    // For now, just use BDF2
    // ierr = IDASetMaxOrd(d_ida_mem, 2);
    // AMP_ASSERT(ierr==IDA_SUCCESS);

    // d_init_step_size = 0.1;
    // ierr = IDASetInitStep(d_ida_mem, d_init_step_size);
    // AMP_ASSERT(ierr==IDA_SUCCESS);
}


void IDATimeIntegrator::reset( std::shared_ptr<const TimeIntegratorParameters> parameters )
{
    AMP_ASSERT( parameters );
    AMP_ERROR( "Not Finished" );
}


void IDATimeIntegrator::setupVectors()
{
    // clone vectors so they have the same data layout as d_solution_vector
    d_residual = d_solution_vector->clone();
    // Set initial value of vectors to 0.
    d_residual->setToScalar( (double) 0.0 );
}


/************************************************************************
 *  Update internal state to reflect time advanced solution.             *
 ************************************************************************/
void IDATimeIntegrator::updateSolution()
{
    /*
    int retval = IDA_SUCCESS;
    realtype hlast;

    retval = IDAGetLastStep(d_ida_mem, &hlast);
    AMP_ASSERT(retval==IDA_SUCCESS);

    d_current_time += hlast;
    //d_solution_vector->add( d_predictor, d_corrector );
    */
}


/************************************************************************
 * Read input from database.                                             *
 ************************************************************************/
void IDATimeIntegrator::getFromInput( std::shared_ptr<AMP::Database> input_db )
{
    if ( input_db->keyExists( "bLinearMassOperator" ) ) {
        d_bLinearMassOperator = input_db->getScalar<bool>( "bLinearMassOperator" );
    } else {
        AMP_ERROR( d_object_name + " -- Key data `bLinearMassOperator' missing in input." );
    }

    if ( input_db->keyExists( "bLinearRhsOperator" ) ) {
        d_bLinearRhsOperator = input_db->getScalar<bool>( "bLinearRhsOperator" );
    } else {
        AMP_ERROR( d_object_name + " -- Key data `bLinearRhsOperator' missing in input." );
    }

    if ( input_db->keyExists( "linear_solver_type" ) ) {
        d_linear_solver_type = input_db->getScalar<int>( "linear_solver_type" );
    } else {
        AMP_ERROR( d_object_name + " -- Key data `linear_solver_type' missing in input." );
    }

    if ( input_db->keyExists( "relative_tolerance" ) ) {
        d_relative_tolerance = input_db->getScalar<double>( "relative_tolerance" );
    } else {
        AMP_ERROR( d_object_name + " -- Key data `relative_tolerance' missing in input." );
    }

    if ( input_db->keyExists( "absolute_tolerance" ) ) {
        d_absolute_tolerance = input_db->getScalar<double>( "absolute_tolerance" );
    } else {
        AMP_ERROR( d_object_name + " -- Key data `absolute_tolerance' missing in input." );
    }

    d_bCallCalcIC        = input_db->getWithDefault<bool>( "CallCalcIC", true );
    d_bUsePreconditioner = input_db->getWithDefault<bool>( "usePreconditioner", true );

    d_createLinearOperatorInternally =
        input_db->getWithDefault<bool>( "createLinearTimeOperatorInternally", false );
    d_bManufacturedProblem = input_db->getWithDefault<bool>( "bManufacturedProblem", false );
}


double IDATimeIntegrator::getNextDt( const bool /* good_solution */ )
{
    int ierr = IDAGetCurrentStep( d_ida_mem, &d_current_dt );
    AMP_ASSERT( ierr != IDA_SUCCESS );

    return d_current_dt;
}

int IDATimeIntegrator::advanceSolution( const double dt,
                                        const bool /* first_step */,
                                        std::shared_ptr<AMP::LinearAlgebra::Vector>,
                                        std::shared_ptr<AMP::LinearAlgebra::Vector> out )
{
    int retval        = IDA_SUCCESS;
    double hin_actual = 0.0;
    double hlast      = 0.0;
    double hcur       = 0.0;
    long int nniters  = 0;
    long int nncfails = 0;
    long int netfails = 0;
    long int nliters  = 0;
    long int npsolves = 0;

    auto ptr_y = std::dynamic_pointer_cast<AMP::LinearAlgebra::SundialsVector>(
        AMP::LinearAlgebra::SundialsVector::view( d_solution_vector ) );
    auto ptr_ydot = std::dynamic_pointer_cast<AMP::LinearAlgebra::SundialsVector>(
        AMP::LinearAlgebra::SundialsVector::view( d_solution_prime ) );

    // specify some initial step
    hcur = d_current_time + dt;

    AMP::pout << "before IDASolve" << std::endl;
    retval = IDASolve( d_ida_mem,
                       hcur,
                       &d_current_time,
                       ptr_y->getNVector(),
                       ptr_ydot->getNVector(),
                       IDA_ONE_STEP );
    // should be fixed.
    AMP::pout << "after IDASolve" << std::endl;
    retval = IDAGetActualInitStep( d_ida_mem, &hin_actual );
    // cout << "hin_actual = " << hin_actual << endl;

    retval = IDAGetLastStep( d_ida_mem, &hlast );
    AMP::pout << "hlast = " << hlast << std::endl;

    retval       = IDAGetCurrentStep( d_ida_mem, &hcur );
    d_current_dt = hcur;
    AMP::pout << "hcur = " << hcur << std::endl;

    retval = IDAGetNumNonlinSolvIters( d_ida_mem, &nniters );
    AMP::pout << "nniters = " << nniters << std::endl;

    retval = IDAGetNumNonlinSolvConvFails( d_ida_mem, &nncfails );
    AMP::pout << "nncfails = " << nncfails << std::endl;

    retval = IDAGetNumErrTestFails( d_ida_mem, &netfails );
    AMP::pout << "netfails = " << netfails << std::endl;

    retval = IDASpilsGetNumPrecSolves( d_ida_mem, &npsolves );
    AMP::pout << "npsolves = " << npsolves << std::endl;

    retval = IDASpilsGetNumLinIters( d_ida_mem, &nliters );
    AMP::pout << "nliters = " << nliters << std::endl;

    out->copyVector( d_solution_vector );
    return ( retval );
}

void IDATimeIntegrator::updateSourceTerm()
{
    // Ideally, IDATimeIntegrator should call a function which takes d_solution_vector and t and
    // returns the appropriate updated source term.
}
/*
 ************************************************************************
 *                                                                      *
 *  Check whether time advanced solution is acceptable.                 *
 *                                                                      *
 ************************************************************************
 */
bool IDATimeIntegrator::checkNewSolution()
{
    /*
     * Ordinarily we would check the actual error in the solution
     * (proportional to the size of d_corrector) against a specified
     * tolerance.  For now, accept everything.
     */
    return ( true );
}


int IDATimeIntegrator::IDAResTrial(
    realtype /* tt */, N_Vector yy, N_Vector yp, N_Vector rr, void *user_data_in )
{
    auto user_data = reinterpret_cast<IDATimeIntegrator *>( user_data_in );

    std::shared_ptr<AMP::LinearAlgebra::Vector> f;


    auto amp_yy = getAMP( yy );
    auto amp_yp = getAMP( yp );
    amp_yy->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    amp_yp->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    user_data->getIDATimeOperator()->registerIDATimeDerivative( amp_yp );

    auto d_residual_Sundials = user_data->getResidualVector();

    double currentTime = user_data->getCurrentTime();
    user_data->getIDATimeOperator()->registerCurrentTime( currentTime );

    user_data->getIDATimeOperator()->apply( amp_yy, d_residual_Sundials );

    getAMP( rr )->copyVector( d_residual_Sundials );

    return ( IDA_SUCCESS );
}

int IDATimeIntegrator::IDAPrecSetup( realtype tt,
                                     N_Vector yy,
                                     N_Vector /* yp */,
                                     N_Vector /* rr */,
                                     realtype cj,
                                     void *user_data_in,
                                     N_Vector /* tmp1 */,
                                     N_Vector /* tmp2 */,
                                     N_Vector /* tmp3 */ )
{
    auto user_data = reinterpret_cast<IDATimeIntegrator *>( user_data_in );

    int last_order, current_order;
    double last_stepsize, current_stepsize;

    int ierr = IDAGetLastOrder( user_data->getIDAMem(), &last_order );
    AMP_ASSERT( ierr == IDA_SUCCESS );
    ierr = IDAGetCurrentOrder( user_data->getIDAMem(), &current_order );
    AMP_ASSERT( ierr == IDA_SUCCESS );
    ierr = IDAGetLastStep( user_data->getIDAMem(), &last_stepsize );
    AMP_ASSERT( ierr == IDA_SUCCESS );
    ierr = IDAGetCurrentStep( user_data->getIDAMem(), &current_stepsize );
    AMP_ASSERT( ierr == IDA_SUCCESS );

    AMP::pout << "cj = " << cj << std::endl;


    bool OrderHasChanged = FALSE, StepSizeHasChanged = FALSE;
    if ( last_order != current_order ) {
        OrderHasChanged = TRUE;
    }
    if ( last_stepsize != current_stepsize ) {
        StepSizeHasChanged = TRUE;
    }


    if ( OrderHasChanged || StepSizeHasChanged ) {

        auto amp_yy    = getAMP( yy );
        auto jacParams = user_data->getIDATimeOperator()->getParameters( "Jacobian", amp_yy );
        auto &db       = jacParams->d_db;
        auto Overwrite = AMP::Database::Check::Overwrite;
        db->putScalar( "ScalingFactor", cj, {}, Overwrite );
        db->putScalar( "CurrentTime", tt, {}, Overwrite );
        std::shared_ptr<AMP::Solver::SolverStrategy> pSolver = user_data->getPreconditioner();
        // double currentTime = user_data->getCurrentTime();

        // BP:  commenting out these lines that appear to be buggy
        // I will need to fix these lines so that the current time is set through the database
        // object
        // we can have column time operators and in that case the lines below fail
        //          if ( user_data->getBoolManufacturedProblem() ) {
        //          std::shared_ptr<AMP::TimeIntegrator::LinearTimeOperator> pLinearTimeOperator =
        //              std::dynamic_pointer_cast<AMP::TimeIntegrator::LinearTimeOperator> (
        //              pSolver->getOperator() );
        //          AMP_INSIST(pLinearTimeOperator!=NULL, "In IDATimeIntegrator::IDAPrecSetup -
        //          pLinearTimeOperator is
        //          NULL");
        //          pLinearTimeOperator->registerCurrentTime(currentTime);
        pSolver->resetOperator( jacParams );
        //          }
    }

    return ( IDA_SUCCESS );
}


int IDATimeIntegrator::IDAPrecSolve( realtype,
                                     N_Vector,
                                     N_Vector,
                                     N_Vector,
                                     N_Vector rvec,
                                     N_Vector zvec,
                                     realtype,
                                     realtype,
                                     void *user_data_in,
                                     N_Vector )
{
    auto user_data = reinterpret_cast<IDATimeIntegrator *>( user_data_in );

    auto amp_rvec = getAMP( rvec );
    auto amp_zvec = getAMP( zvec );
    amp_rvec->makeConsistent();

    user_data->getPreconditioner()->apply( amp_rvec, amp_zvec );

    return ( IDA_SUCCESS );
}

std::shared_ptr<IDATimeOperator> IDATimeIntegrator::getIDATimeOperator() const
{
    return d_pIDATimeOperator;
}

std::shared_ptr<LinearTimeOperator> IDATimeIntegrator::getLinearTimeOperator() const
{
    return d_pLinearTimeOperator;
}

std::shared_ptr<AMP::LinearAlgebra::Vector> IDATimeIntegrator::getResidualVector() const
{
    return d_residual;
}

std::shared_ptr<AMP::LinearAlgebra::Vector> IDATimeIntegrator::getSourceTerm()
{
    return d_pSourceTerm;
}

} // namespace AMP::TimeIntegrator

#endif
