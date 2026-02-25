#include "AMP/AMP_TPLs.h"
#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearOperator.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/time_integrators/ImplicitIntegrator.h"
#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegratorFactory.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

class UserDiagonalOperator : public AMP::Operator::LinearOperator
{
public:
    explicit UserDiagonalOperator( std::shared_ptr<const AMP::Operator::OperatorParameters> params )
        : LinearOperator( params )
    {
    }
    std::string type() const override { return "UserDiagonalOperator"; }

    // required if we are managing time dependent operator
    void setGamma( AMP::Scalar gamma )
    {
        if ( gamma != d_gamma ) {
            d_gamma         = gamma;
            d_update_matrix = true;
        }
        //        AMP::pout << *d_matrix << std::endl;
    }

    // only required if we are doing multi-physics and scaling of components needed
    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override
    {

        AMP::LinearAlgebra::Vector::const_shared_ptr u;

        reset( {} );

        if ( d_pSolutionScaling ) {

            AMP_ASSERT( d_pFunctionScaling );

            if ( !d_pScratchSolVector ) {
                d_pScratchSolVector = u_in->clone();
            }

            d_pScratchSolVector->multiply( *u_in, *d_pSolutionScaling );
            d_pScratchSolVector->makeConsistent();
            u = d_pScratchSolVector;

        } else {
            u = u_in;
        }

        LinearOperator::apply( u, r );

        if ( d_pFunctionScaling ) {
            r->divide( *r, *d_pFunctionScaling );
        }
    }

    // this updates the matrix when gamma changes
    void reset(
        [[maybe_unused]] std::shared_ptr<const AMP::Operator::OperatorParameters> params ) override
    {
        if ( d_update_matrix ) {
            d_matrix->setScalar( 1.0 + d_gamma );
            d_update_matrix = false;
        }
    }

    // only required if we are doing multi-physics and scaling of components needed
    void setComponentScalings( std::shared_ptr<AMP::LinearAlgebra::Vector> s,
                               std::shared_ptr<AMP::LinearAlgebra::Vector> f )
    {
        d_pSolutionScaling = s;
        d_pFunctionScaling = f;
    }

    AMP::Scalar d_gamma;
    bool d_update_matrix = true;
    // only required if we are doing multi-physics and scaling of components needed
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pSolutionScaling;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pFunctionScaling;
    std::shared_ptr<AMP::LinearAlgebra::Vector> d_pScratchSolVector;
};

void testIntegrator( const std::string &name,
                     const std::string &test,
                     std::shared_ptr<AMP::TimeIntegrator::TimeIntegratorParameters> params,
                     double ans,
                     AMP::UnitTest &ut )
{
    AMP::pout << "Testing " << name << " with " << test << " operator" << std::endl;
    // Create the time integrator
    auto var      = std::make_shared<AMP::LinearAlgebra::Variable>( "x" );
    auto solution = params->d_ic_vector->clone();
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegrator> timeIntegrator =
        AMP::TimeIntegrator::TimeIntegratorFactory::create( params );

    auto op = std::dynamic_pointer_cast<UserDiagonalOperator>( params->d_operator );
    if ( op ) {

        auto implicitIntegrator =
            std::dynamic_pointer_cast<AMP::TimeIntegrator::ImplicitIntegrator>( timeIntegrator );

        // required to set gamma for the user operator at each timestep
        implicitIntegrator->setTimeScalingFunction(
            std::bind( &UserDiagonalOperator::setGamma, &( *op ), std::placeholders::_1 ) );

        // required only if a user operator is being used which is multi-physics
        implicitIntegrator->setComponentScalingFunction(
            std::bind( &UserDiagonalOperator::setComponentScalings,
                       &( *op ),
                       std::placeholders::_1,
                       std::placeholders::_2 ) );
    }

    auto x = solution->clone();
    solution->setToScalar( 1.0 );
    x->copy( *solution );

    // Advance the solution
    double T         = 0;
    double dt        = 0.00025;
    double finalTime = timeIntegrator->getFinalTime();
    timeIntegrator->setInitialDt( dt );
    while ( T < finalTime ) {
        timeIntegrator->advanceSolution( dt, T == 0, solution, x );
        if ( timeIntegrator->checkNewSolution() ) {
            timeIntegrator->updateSolution();
            solution->copyVector( x );
        } else {
            AMP_ERROR( "Solution didn't converge" );
        }
        T += dt;
    }

    // Check the answer
    double ans2 = static_cast<double>( solution->max() );
    if ( AMP::Utilities::approx_equal( ans2, ans, 2.0e-11 ) )
        ut.passes( name + " - " + test );
    else
        ut.failure(
            AMP::Utilities::stringf( "%s - %s (%0.16f)", name.data(), test.data(), ans - ans2 ) );
}

void updateDatabaseWithPC( std::shared_ptr<AMP::Database> db )
{
    AMP_ASSERT( db );
    if ( db->keyExists( "Solver" ) ) {
        auto solver_db = db->getDatabase( "Solver" );
        solver_db->putScalar<bool>( "uses_preconditioner", true );
        auto pc_db = AMP::Database::create(
            "name", "BoomerAMGSolver", "print_info_level", 2, "max_iterations", 1 );
        solver_db->putDatabase( "Preconditioner", std::move( pc_db ) );
    }
}

void updateDatabaseIfImplicit( std::shared_ptr<AMP::Database> db )
{
    AMP_ASSERT( db );
    auto imp_ti = { "CN", "Backward Euler", "BDF1", "BDF2", "BDF3", "BDF4", "BDF5", "BDF6" };
    auto name   = db->getScalar<std::string>( "name" );
    AMP::pout << name << std::endl;
    auto is_implicit = ( std::find( imp_ti.begin(), imp_ti.end(), name ) != imp_ti.end() );
    if ( is_implicit ) {

        //        db->putScalar<std::string>( "name", "ImplicitIntegrator" );
        db->putScalar<bool>( "user_managed_time_operator", true );
        db->putScalar<std::string>( "implicit_integrator", name );
        db->putScalar<std::string>( "solver_name", "Solver" );
        db->putScalar<std::string>( "timestep_selection_strategy", "constant" );
        db->putScalar<bool>( "use_predictor", false );
        // add the next line to turn off component scaling for multi-physics
        //        db->putScalar<bool>( "auto_component_scaling", false );

        auto solver_db = AMP::Database::create( "name",
                                                "CGSolver",
                                                "print_info_level",
                                                2,
                                                "max_iterations",
                                                100,
                                                "absolute_tolerance",
                                                1.0e-14,
                                                "relative_tolerance",
                                                1.0e-16,
                                                "zero_initial_guess",
                                                false );
        db->putDatabase( "Solver", std::move( solver_db ) );
    }
}

void runIntegratorWithUserOperatorTests( const std::string &inputFileName,
                                         const std::string &ti_name,
                                         AMP::UnitTest &ut )
{
    std::string input_file = inputFileName;

    AMP::pout << "runIntegratorWithUserOperatorTests with input " << input_file << " and "
              << ti_name << " integrator" << std::endl;

    // Fill the database from the input file.
    auto input_db = AMP::Database::parseInputFile( input_file );

    auto comm = AMP::AMP_MPI( AMP_COMM_WORLD );

    // Create the Mesh and DOFManager
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mgrParams->setComm( comm );
    auto mesh = AMP::Mesh::MeshFactory::create( mgrParams );
    auto scalarDOFs =
        AMP::Discretization::simpleDOFManager::create( mesh, AMP::Mesh::GeomType::Vertex, 1, 1 );

    // Create variables and vectors
    auto inVar  = std::make_shared<AMP::LinearAlgebra::Variable>( "inputVar" );
    auto outVar = inVar;

    std::shared_ptr<AMP::LinearAlgebra::Vector> inVec, outVec;

    inVec  = AMP::LinearAlgebra::createVector( scalarDOFs, inVar );
    outVec = AMP::LinearAlgebra::createVector( scalarDOFs, outVar );

    // Create the matrix
    auto matrix = AMP::LinearAlgebra::createMatrix(
        inVec, outVec, "CSRMatrix", []( size_t row ) -> std::vector<size_t> {
            return std::vector<size_t>( 1, row );
        } );
    matrix->setIdentity();

    // Create operator to wrap matrix
    auto op_db = std::make_shared<AMP::Database>( "LinearOperator" );

    auto opParams     = std::make_shared<AMP::Operator::OperatorParameters>( op_db );
    auto userOperator = std::make_shared<AMP::Operator::LinearOperator>( opParams );
    userOperator->setMatrix( matrix );
    userOperator->setVariables( inVar, outVar );

    double finalTime = 0.001;
    // Create the vectors
    auto ic = inVec->clone();
    ic->setToScalar( 1.0 );

    // Test creating Create the time integrator
    std::shared_ptr<AMP::Database> db = AMP::Database::create( "name",
                                                               ti_name,
                                                               "initial_time",
                                                               0.0,
                                                               "final_time",
                                                               finalTime,
                                                               "max_integrator_steps",
                                                               1000000,
                                                               "print_info_level",
                                                               2 );
    updateDatabaseIfImplicit( db );

    auto params         = std::make_shared<AMP::TimeIntegrator::TimeIntegratorParameters>( db );
    params->d_ic_vector = ic;

    // Test with a fixed source and null operator
    auto source = ic->clone();
    source->setToScalar( 3.0 );
    params->d_pSourceTerm = source;
    params->d_operator    = userOperator;
    testIntegrator( ti_name, "du/dt=3", params, 1.0 + 3 * finalTime, ut );

    // Test with no source and constant operator
    // testing du/dt = -u

    opParams     = std::make_shared<AMP::Operator::OperatorParameters>( op_db );
    userOperator = std::make_shared<UserDiagonalOperator>( opParams );
    userOperator->setMatrix( matrix );
    userOperator->setVariables( inVar, outVar );

    // Create the vectors
    ic->setToScalar( 1.0 );

    params->d_operator    = userOperator;
    params->d_pSourceTerm = nullptr;
    testIntegrator( ti_name, "du/dt=-u", params, std::exp( -finalTime ), ut );


    // Test with fixed source and constant operator
    // testing du/dt = -u+1
    std::dynamic_pointer_cast<UserDiagonalOperator>( userOperator )->d_gamma = 0.0;
    source->setToScalar( 1.0 );
    params->d_pSourceTerm = source;
    testIntegrator( ti_name, "du/dt=-u+1", params, 1.0, ut );

#ifdef AMP_USE_HYPRE
    // Test with fixed source and const operator, CG w/BoomerAMG
    std::dynamic_pointer_cast<UserDiagonalOperator>( userOperator )->d_gamma = 0.0;
    source->setToScalar( 1.0 );
    params->d_pSourceTerm = nullptr;
    updateDatabaseWithPC( params->d_db );
    testIntegrator( ti_name, "du/dt=-u (PCG)", params, std::exp( -finalTime ), ut );
#endif
}

int main( int argc, char *argv[] )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> files;

    PROFILE_ENABLE();

    if ( argc > 1 ) {

        files.emplace_back( argv[1] );

    } else {
        files.emplace_back( "input_testImplicitIntegrationWithUserOperator" );
    }

    for ( auto &file : files ) {

        // List of integrators
        auto integrators = { "CN", "BDF2", "BDF3", "BDF4", "BDF5" };
        // Run the tests
        for ( auto &ti : integrators )
            runIntegratorWithUserOperatorTests( file, ti, ut );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
