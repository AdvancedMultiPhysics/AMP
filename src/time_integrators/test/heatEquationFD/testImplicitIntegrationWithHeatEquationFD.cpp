#include "AMP/IO/AsciiWriter.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/diffusionFD/DiffusionFD.h"
#include "AMP/operators/diffusionFD/DiffusionRotatedAnisotropicModel.h"
#include "AMP/operators/testHelpers/FDHelper.h"
#include "AMP/solvers/SolverFactory.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/testHelpers/SolverTestParameters.h"
#include "AMP/solvers/testHelpers/testSolverHelpers.h"
#include "AMP/time_integrators/ImplicitIntegrator.h"
#include "AMP/time_integrators/TimeIntegrator.h"
#include "AMP/time_integrators/TimeIntegratorFactory.h"
#include "AMP/time_integrators/TimeIntegratorParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Constants.h"
#include "AMP/utils/UnitTest.h"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

constexpr double pi = AMP::Constants::pi;


/** This test applies implicit time integration (in the form of BDF) to a heat equation of the form
 *      u_t - div ( D * grad u ) = s,   (x, t) in Omega \times [0,T],
 *      u(x,0) = u_0(x),     t = 0, x in Omega
 *      u(x,t) = ubnd(x),    t > 0, x in DOmega.
 *
 * where the spatially-constant diffusion tensor D represents anisotropies and rotations.
 * Discretization of the spatial term div ( D * grad u ) is provided by an
 * AMP::Operator::DiffusionFD, which is a finite-difference discretization.
 *
 * A manufactured solution is implemented to measure the accuracy of the discretization (the
 * spatial discretization is second order); the test can be run in 1D, 2D, or 3D.
 * The implementation of the spatial boundary conditions assumes BDF time integration is being
 * used. The Implementation also assumes that the spatial boundary conditons are time independent.
 *
 * The operator provided to the time integrator is a user-managed time operator
 *
 * Note the use of non-symmetric Krylov solvers (e.g., BiCGSTAB) in the tests: The spatial operator
 * is not symmetric because Dirichlet DOFs are (non-symmetrically) included in the system.
 */


/** A manufactured solution of the form
 *    u = ubnd(x) + f(t)*uzero(x),
 * for the PDE
 *    u'(t) - L*u = s(t),
 * where L is a spatial differential operator. The function uzero(x) vanishes on the domain of the
 * spatial domain, such that u = ubnd on the boundary.
 *
 * Plugging in the solution ansatz, we have
 *    s(t) = [f'(t)*uzero - f(t)*L*uzero] - L*ubnd
 *
 * The spatial operator is assumed to be of the form
 *    1D: L*u = [d_cxx*u_xx]
 *    2D: L*u = [[d_cxx*u_xx] + d_cyy*u_yy + d_cyx*u_yx]
 *    3D: L*u = [[[d_cxx*u_xx] + d_cyy*u_yy + d_cyx*u_yx] + d_czz*u_zz + d_czx*u_zx + d_czy*u_zy]
 *
 * The coefficients c must be provided in the incoming "Lcoefficients_db" Database
 * Information about the geometry must be provided in the incoming "mesh_db" Database. The
 * geometry is assumed to correspond to an AMP::Mesh::Boxmesh with a "cube" generator.
 *
 * The expressions in the ubnd, Lubnd, uzero, and Luzero functions for this class were generated
 * symbolically with the helper script ../testHelpers/ManufacturedHeatEquationModel-helper.py
 */
class ManufacturedHeatEquationModel
{

private:
    mutable double d_currentTime = 0.0;
    size_t d_dim;

    //! Lower extend of domain
    std::vector<double> d_xmin;
    //! Length of domain
    std::vector<double> d_Lx;
    //! Coefficients defining L
    double d_cxx, d_cyy, d_cyx, d_czz, d_czx, d_czy;
    //! Coefficients that uzero depends on in the form of sin( K*x + PHI )
    std::vector<double> d_K;   // frequency
    std::vector<double> d_PHI; // phase shift

public:
    double getCurrentTime() const { return d_currentTime; };
    void setCurrentTime( double time ) { d_currentTime = time; };

    ManufacturedHeatEquationModel( std::shared_ptr<AMP::Database> mesh_db,
                                   std::shared_ptr<AMP::Database> Lcoefficients_db )
    {

        // Unpack coefficients of L
        d_cxx = Lcoefficients_db->getScalar<double>( "cxx" );
        // Note that the below coeffs don't exist in 1D
        d_cyy = Lcoefficients_db->getWithDefault<double>( "cyy", 0.0 );
        d_czz = Lcoefficients_db->getWithDefault<double>( "czz", 0.0 );
        d_cyx = Lcoefficients_db->getWithDefault<double>( "cyx", 0.0 );
        d_czx = Lcoefficients_db->getWithDefault<double>( "czx", 0.0 );
        d_czy = Lcoefficients_db->getWithDefault<double>( "czy", 0.0 );

        AMP_INSIST( mesh_db->getScalar<std::string>( "MeshType" ) == "AMP" &&
                        mesh_db->getScalar<std::string>( "Generator" ) == "cube",
                    "Mesh database must describe an 'AMP' 'cube' mesh" );

        // Compute domain-based constants
        d_dim      = mesh_db->getScalar<size_t>( "dim" );
        auto range = mesh_db->getArray<double>( "Range" );
        for ( size_t dim = 0; dim < d_dim; dim++ ) {
            d_xmin.push_back( range[2 * dim] );
            auto xmax = range[2 * dim + 1];
            d_Lx.push_back( xmax - d_xmin[dim] );
        }

        // Compute constants that uzero depends on
        for ( size_t dim = 0; dim < d_dim; dim++ ) {
            d_K.push_back( 2 * pi / d_Lx[dim] ); //
            d_PHI.push_back( 2 * pi / d_Lx[dim] * -d_xmin[dim] );
        }
    }

    ~ManufacturedHeatEquationModel() {}

    double exactSolution( const AMP::Mesh::Point &point ) const
    {
        return f() * uzero( point ) + ubnd( point );
    };
    //! s(t) = [f'(t)*uzero - f(t)*L*uzero]  - L*ubnd
    double sourceTerm( const AMP::Mesh::Point &point ) const
    {
        return ( fprime() * uzero( point ) - f() * Luzero( point ) ) - Lubnd( point );
    };
    double initialCondition( const AMP::Mesh::Point &point ) const
    {
        auto temp     = d_currentTime;
        d_currentTime = 0.0;
        auto u0       = exactSolution( point );
        d_currentTime = temp;
        return u0;
    };

private:
    double ubnd( const AMP::Mesh::Point &point ) const
    {
        auto x   = point.x();
        auto y   = point.y();
        auto z   = point.z();
        double u = 0.0;
        if ( d_dim == 1 ) {
            u = std::cos( x );
        } else if ( d_dim == 2 ) {
            u = std::cos( x ) * std::cos( y );
        } else if ( d_dim == 3 ) {
            u = std::cos( x ) * std::cos( y ) * std::cos( z );
        }
        return u;
    };
    double Lubnd( const AMP::Mesh::Point &point ) const
    {
        auto x    = point.x();
        auto y    = point.y();
        auto z    = point.z();
        double Lu = 0.0;
        if ( d_dim == 1 ) {
            Lu = -d_cxx * std::cos( x );
        } else if ( d_dim == 2 ) {
            Lu = -d_cxx * std::cos( x ) * std::cos( y ) + d_cyx * std::sin( x ) * std::sin( y ) -
                 d_cyy * std::cos( x ) * std::cos( y );
        } else if ( d_dim == 3 ) {
            Lu = -d_cxx * std::cos( x ) * std::cos( y ) * std::cos( z ) +
                 d_cyx * std::sin( x ) * std::sin( y ) * std::cos( z ) -
                 d_cyy * std::cos( x ) * std::cos( y ) * std::cos( z ) +
                 d_czx * std::sin( x ) * std::sin( z ) * std::cos( y ) +
                 d_czy * std::sin( y ) * std::sin( z ) * std::cos( x ) -
                 d_czz * std::cos( x ) * std::cos( y ) * std::cos( z );
        }
        return Lu;
    };
    // Constants K and PHI in each dimension are such that u in that dimension is the
    // lowest-frequency sine function
    double uzero( const AMP::Mesh::Point &point ) const
    {
        auto x   = point.x();
        auto y   = point.y();
        auto z   = point.z();
        double u = 0.0;
        if ( d_dim == 1 ) {
            u = std::sin( d_K[0] * x + d_PHI[0] );
        } else if ( d_dim == 2 ) {
            u = std::sin( d_K[0] * x + d_PHI[0] ) * std::sin( d_K[1] * y + d_PHI[1] );
        } else if ( d_dim == 3 ) {
            u = std::sin( d_K[0] * x + d_PHI[0] ) * std::sin( d_K[1] * y + d_PHI[1] ) *
                std::sin( d_K[2] * z + d_PHI[2] );
        }
        return u;
    };
    double Luzero( const AMP::Mesh::Point &point ) const
    {
        auto x    = point.x();
        auto y    = point.y();
        auto z    = point.z();
        double Lu = 0.0;
        if ( d_dim == 1 ) {
            Lu = -std::pow( d_K[0], 2 ) * d_cxx * std::sin( d_K[0] * x + d_PHI[0] );
        } else if ( d_dim == 2 ) {
            Lu = -std::pow( d_K[0], 2 ) * d_cxx * std::sin( d_K[0] * x + d_PHI[0] ) *
                     std::sin( d_K[1] * y + d_PHI[1] ) +
                 d_K[0] * d_K[1] * d_cyx * std::cos( d_K[0] * x + d_PHI[0] ) *
                     std::cos( d_K[1] * y + d_PHI[1] ) -
                 std::pow( d_K[1], 2 ) * d_cyy * std::sin( d_K[0] * x + d_PHI[0] ) *
                     std::sin( d_K[1] * y + d_PHI[1] );
        } else if ( d_dim == 3 ) {
            Lu = -std::pow( d_K[0], 2 ) * d_cxx * std::sin( d_K[0] * x + d_PHI[0] ) *
                     std::sin( d_K[1] * y + d_PHI[1] ) * std::sin( d_K[2] * z + d_PHI[2] ) +
                 d_K[0] * d_K[1] * d_cyx * std::sin( d_K[2] * z + d_PHI[2] ) *
                     std::cos( d_K[0] * x + d_PHI[0] ) * std::cos( d_K[1] * y + d_PHI[1] ) +
                 d_K[0] * d_K[2] * d_czx * std::sin( d_K[1] * y + d_PHI[1] ) *
                     std::cos( d_K[0] * x + d_PHI[0] ) * std::cos( d_K[2] * z + d_PHI[2] ) -
                 std::pow( d_K[1], 2 ) * d_cyy * std::sin( d_K[0] * x + d_PHI[0] ) *
                     std::sin( d_K[1] * y + d_PHI[1] ) * std::sin( d_K[2] * z + d_PHI[2] ) +
                 d_K[1] * d_K[2] * d_czy * std::sin( d_K[0] * x + d_PHI[0] ) *
                     std::cos( d_K[1] * y + d_PHI[1] ) * std::cos( d_K[2] * z + d_PHI[2] ) -
                 std::pow( d_K[2], 2 ) * d_czz * std::sin( d_K[0] * x + d_PHI[0] ) *
                     std::sin( d_K[1] * y + d_PHI[1] ) * std::sin( d_K[2] * z + d_PHI[2] );
        }
        return Lu;
    };
    double f() const
    {
        auto t = d_currentTime;
        return std::exp( -t );
    };
    double fprime() const
    {
        auto t = d_currentTime;
        return -std::exp( -t );
    };
};


/******************************************************
 *        Class implementing a BDF operator           *
 ******************************************************/
/** Implements the linear operator
 *      I + gamma*L,
 * where L is a DiffusionFDOperator. This linear operator arises during the BDF discretization of
 * the linear ODEs
 *      u'(t) + L*u = s(t).
 *
 * The operator I + gamma*L is stored as a matrix by overwriting the matrix stored in L.
 * This class is an example of a user-managed time operator
 */
class BDFDiffusionFDOp : public AMP::Operator::LinearOperator
{

private:
    //! Absolute tolerance used to determine if an incoming gamma is different from the existing one
    static constexpr double GAMMA_DIFTOL = 1e-14;

public:
    //! Time step size (potentially scaled by BDF method)
    double d_gamma = -1.0;
    //! Underlying operator we wrap as a BDF operator
    std::shared_ptr<AMP::Operator::DiffusionFDOperator> d_L = nullptr;
    //! Incoming parameters
    std::shared_ptr<const AMP::Operator::OperatorParameters> d_params = nullptr;

    BDFDiffusionFDOp( std::shared_ptr<const AMP::Operator::OperatorParameters> params )
        : AMP::Operator::LinearOperator( params ),
          d_L( std::make_shared<AMP::Operator::DiffusionFDOperator>( params ) ),
          d_params( params )
    {
    }

    //! Destructor
    virtual ~BDFDiffusionFDOp() {}

    //! Set base class's matrix as A == I + gamma*L, by overwriting L's matrix
    void resetMatrix()
    {
        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "BDFDiffusionFDOp::resetMatrix()" << std::endl;
        }

        // Create a new DiffusionFDOperator because we've overwritten the current one's matrix to
        // store our matrix
        d_L = std::make_shared<AMP::Operator::DiffusionFDOperator>( d_params );
        std::shared_ptr<AMP::LinearAlgebra::Matrix> L_matrix = d_L->getMatrix();
        AMP_INSIST( L_matrix, "L matrix is null" );

        auto A_matrix = L_matrix;
        A_matrix->scale( d_gamma ); // A <- gamma*A

        // A <- A + I
        auto DOFMan = A_matrix->getRightDOFManager();
        for ( auto dof = DOFMan->beginDOF(); dof != DOFMan->endDOF(); dof++ ) {
            A_matrix->addValueByGlobalID( dof, dof, 1.0 );
        }

        // Set our matrix
        this->setMatrix( A_matrix );
    }

    std::string type() const override { return "BDFDiffusionFDOp"; }

    //! Set the time-step size of the operator
    void setGamma( AMP::Scalar gamma_ )
    {
        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "BDFDiffusionFDOp::setGamma()" << std::endl;
        }

        double gamma = double( gamma_ );
        // Check if gamma has changed from previous value; if so, reset the matrix
        bool gammaChanged = !( AMP::Utilities::approx_equal( gamma, d_gamma, GAMMA_DIFTOL ) );
        if ( gammaChanged ) {
            d_gamma = gamma;
            resetMatrix();
        }
    }

    void apply( AMP::LinearAlgebra::Vector::const_shared_ptr u_in,
                AMP::LinearAlgebra::Vector::shared_ptr r ) override
    {
        if ( d_iDebugPrintInfoLevel > 1 ) {
            AMP::pout << "BDFDiffusionFDOp::apply()" << std::endl;
        }
        AMP_INSIST( this->getMatrix(), "Matrix is null" );
        LinearOperator::apply( u_in, r );
    }
};


void driver( AMP::AMP_MPI comm, AMP::UnitTest *ut, const std::string &inputFileName )
{

    // Input and output file names
    std::string input_file = inputFileName;
    std::string log_file   = "output_" + inputFileName;

    AMP::logOnlyNodeZero( log_file );
    AMP::plog << "Running driver with input " << input_file << std::endl;

    auto input_db = AMP::Database::parseInputFile( input_file );
    AMP_INSIST( input_db, "No input database was found" );
    AMP::plog << "Input database:" << std::endl;
    AMP::plog << "---------------" << std::endl;
    input_db->print( AMP::plog );

    // Unpack databases from the input file
    auto RACoefficients_db = input_db->getDatabase( "RACoefficients" );
    auto mesh_db           = input_db->getDatabase( "Mesh" );
    auto ti_db             = input_db->getDatabase( "TimeIntegrator" );

    AMP_INSIST( RACoefficients_db, "A ''RACoefficients'' database must be provided" );
    AMP_INSIST( mesh_db, "A ''Mesh'' database must be provided" );
    AMP_INSIST( ti_db, "A ''TimeIntegrator'' database must be provided" );


    /****************************************************************
     * Create a mesh                                                 *
     ****************************************************************/
    // Create MeshParameters
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( comm );
    std::shared_ptr<AMP::Mesh::BoxMesh> mesh = AMP::Mesh::BoxMesh::generate( meshParams );


    /*******************************************************************
     * Create diffusion and heat equation models                       *
     *******************************************************************/
    auto myRADiffusionModel =
        std::make_shared<AMP::Operator::ManufacturedRotatedAnisotropicDiffusionModel>(
            RACoefficients_db );

    auto myHeatModel =
        std::make_shared<ManufacturedHeatEquationModel>( mesh_db, myRADiffusionModel->d_c_db );


    /****************************************************************
     * Create BDFDiffusionFDOp                                       *
     ****************************************************************/
    const auto Op_db = std::make_shared<AMP::Database>( "linearOperatorDB" );
    Op_db->putScalar<int>( "print_info_level", 0 );
    Op_db->putScalar<std::string>( "name", "DiffusionFDOperator" );
    Op_db->putDatabase( "DiffusionCoefficients", myRADiffusionModel->d_c_db->cloneDatabase() );
    auto OpParams    = std::make_shared<AMP::Operator::OperatorParameters>( Op_db );
    OpParams->d_name = "DiffusionFDOperator";
    OpParams->d_Mesh = mesh;

    auto myBEDiffusionOp = std::make_shared<BDFDiffusionFDOp>( OpParams );
    // Get underlying diffusion operator
    auto myDiffusionOp = myBEDiffusionOp->d_L;


    /****************************************************************
     * Set up required vectors                                       *
     ****************************************************************/
    // 1. Create initial condition vector
    auto icFun = [&]( const AMP::Mesh::Point &point ) {
        return myHeatModel->initialCondition( point );
    };

    auto icVec = myDiffusionOp->createOutputVector();
    myDiffusionOp->fillVectorWithFunction( icVec, icFun );
    icVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // 2. Set up things required to create solution-independent source term containing Dirichlet BCs
    // Vector holding boundary data and PDE source term
    std::shared_ptr<AMP::LinearAlgebra::Vector> ODESolIndepSourceVec;

    // Create a Dirichlet boundary-condition function which just returns the initial condition at
    // the given boundary node. That is, we have time-independent Dirichlet boundary conditions that
    // satisfy the initial condition
    auto boundaryFun = [&]( const AMP::Mesh::Point &point, int ) { return icFun( point ); };

    // Get wrapper around heat equation source term
    auto PDESourceFun = [&]( const AMP::Mesh::Point &point ) {
        return myHeatModel->sourceTerm( point );
    };

    // 3. Auxiliary vectors
    // Create vectors to hold current and new solution (when integrating)
    auto uNumOldVec = icVec->clone();
    uNumOldVec->copyVector( icVec );
    auto uNumNewVec = icVec->clone();
    uNumNewVec->copyVector( icVec );

    // Vector holding manufactured exact solution
    auto uManVec   = icVec->clone();
    auto uexactFun = [&]( const AMP::Mesh::Point &point ) {
        return myHeatModel->exactSolution( point );
    };


    /****************************************************************
     * Set up implicit time integrator                               *
     ****************************************************************/
    // Ensure BDF integrator is being used
    auto bdf_ti  = { "Backward Euler", "BDF1", "BDF2", "BDF3", "BDF4", "BDF5", "BDF6" };
    auto ti_name = ti_db->getScalar<std::string>( "name" );
    auto is_bdf  = ( std::find( bdf_ti.begin(), bdf_ti.end(), ti_name ) != bdf_ti.end() );
    AMP_INSIST( is_bdf, "Implementation assumes BDF integrator" );

    // Ensure user is managing time operator (since the operator we use is such an operator)
    AMP_INSIST( ti_db->getWithDefault<bool>( "user_managed_time_operator", false ),
                "Test requires 'user_managed_time_operator=TRUE'" );

    // Put LinearSolver database in TimeIntegrator database if it's not already
    if ( !ti_db->keyExists( "LinearSolver" ) ) {
        auto linearSolver_db = input_db->getDatabase( "LinearSolver" );
        AMP_INSIST( linearSolver_db, "A ''LinearSolver'' database must be provided" );
        ti_db->putDatabase( "LinearSolver", linearSolver_db->cloneDatabase() );
    }

    // Parameters for time integrator
    auto tiParams = std::make_shared<AMP::TimeIntegrator::TimeIntegratorParameters>( ti_db );
    tiParams->d_ic_vector = icVec;
    tiParams->d_operator  = myBEDiffusionOp;

    // Create timeIntegrator from factory
    std::shared_ptr<AMP::TimeIntegrator::TimeIntegrator> timeIntegrator =
        AMP::TimeIntegrator::TimeIntegratorFactory::create( tiParams );

    // Cast to implicit integrator
    auto implicitIntegrator =
        std::dynamic_pointer_cast<AMP::TimeIntegrator::ImplicitIntegrator>( timeIntegrator );

    // Tell implicitIntegrator how to tell our operator what the time step is
    implicitIntegrator->setTimeScalingFunction(
        std::bind( &BDFDiffusionFDOp::setGamma, &( *myBEDiffusionOp ), std::placeholders::_1 ) );

    // Integrate!
    double finalTime = timeIntegrator->getFinalTime();
    double T         = 0.0;

    // Set the initial time step
    double dt = -1.0;
    if ( ti_db->keyExists( "initial_dt" ) ) {
        dt = ti_db->getScalar<double>( "initial_dt" );
    } else {
        dt = myDiffusionOp->getMeshSize()[0]; // Just set dt = hx
    }

    timeIntegrator->setInitialDt( dt );
    bool prematureHalt = false;

    // Step from T to T + dt, so long as T is smaller than the final time.
    while ( T < finalTime ) {

        // Try to advance the solution with the current dt; if that fails (for whatever reason)
        // we'll try again with a different dt
        bool good_solution = false;
        while ( !good_solution ) {

            // Set the solution-independent source term in the ODEs, and point the ti source vector
            // to it; note that this approach only works for implicit multistep methods
            myHeatModel->setCurrentTime( T + dt ); // Set manufactured solution to new time to
                                                   // ensure source term is sampled at the new time.
            ODESolIndepSourceVec = myDiffusionOp->createRHSVector( PDESourceFun, boundaryFun );
            timeIntegrator->setSourceTerm( ODESolIndepSourceVec );

            // Attempt to advance the solution with the current dt
            timeIntegrator->advanceSolution( dt, T == 0.0, uNumOldVec, uNumNewVec );

            // Check the computed solution (returns true if it is acceptable, and false otherwise)
            good_solution = timeIntegrator->checkNewSolution();

            // If step succeeded, update solution, time, step counter, etc.
            if ( good_solution ) {
                timeIntegrator->updateSolution();
                uNumOldVec->copyVector( uNumNewVec );
                T += dt;
            }

            /** Note that ImplicitIntegrator's getNextDt() will call BDF's
             * integratorSpecificGetNextDt(), and parse it the solver_retcode
             * Return the next time increment through which to advance the solution.
             * The good_solution is the value returned by a call to checkNewSolution(),
             * which determines whether the computed solution is acceptable or not.
             */
            double dt_next = implicitIntegrator->getNextDt( good_solution );

            // Set dt for the next step
            dt = dt_next;
        }


        /* Compare numerical solution with manufactured solution */
        AMP::plog << "----------------------------------------" << std::endl;
        AMP::plog << "Manufactured discretization error norms:" << std::endl;
        myDiffusionOp->fillVectorWithFunction( uManVec, uexactFun );
        auto e = uNumNewVec->clone();
        e->axpy( -1.0, *uNumNewVec, *uManVec );
        auto enorms = getDiscreteNorms( myDiffusionOp->getMeshSize(), e );
        AMP::plog.precision( 3 );
        AMP::plog << "||e|| = (" << enorms[0] << ", " << enorms[1] << ", " << enorms[2] << ")"
                  << std::endl;
        AMP::plog << "----------------------------------------" << std::endl;

        // Drop out if we've exceeded max steps
        if ( !timeIntegrator->stepsRemaining() ) {
            prematureHalt = true;
            AMP_WARNING( "max_integrator_steps has been reached, dropping out of loop now..." );
            break;
        }
    }

    if ( prematureHalt ) {
        ut->failure( "testImplicitIntegrationWithHeatEquationFD fails with: " + inputFileName +
                     " due to max_integrator_steps reached" );
    } else {
        if ( ut->NumFailLocal() == 0 ) {
            ut->passes( "testImplicitIntegrationWithHeatEquationFD passes with: " + inputFileName );
        } else {
            ut->failure( "testImplicitIntegrationWithHeatEquationFD fails with: " + inputFileName );
        }
    }
}
// end of driver()


/** The input file must contain the following databases:
 *
 * Mesh : Describes parameters required to build a "cube" BoxMesh
 * RACoefficients : Provides parameters required to build a RotatedAnisotropicDiffusionModel
 * TimeIntegrator : Provides parameters required to build an AMP BDF time integrator from a time
 * integrator factory.
 */
int main( int argc, char **argv )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    // Create a global communicator
    AMP::AMP_MPI comm( AMP_COMM_WORLD );

    std::vector<std::string> exeNames;
    exeNames.emplace_back( "input_testImplicitIntegrationWithHeatEquationFD-1D-Diagonal-BiCGSTAB" );
    exeNames.emplace_back( "input_testImplicitIntegrationWithHeatEquationFD-2D-Diagonal-BiCGSTAB" );
    exeNames.emplace_back( "input_testImplicitIntegrationWithHeatEquationFD-3D-Diagonal-BiCGSTAB" );
#ifdef AMP_USE_HYPRE
    exeNames.emplace_back(
        "input_testImplicitIntegrationWithHeatEquationFD-1D-BoomerAMG-BiCGSTAB" );
    exeNames.emplace_back(
        "input_testImplicitIntegrationWithHeatEquationFD-2D-BoomerAMG-BiCGSTAB" );
    exeNames.emplace_back(
        "input_testImplicitIntegrationWithHeatEquationFD-3D-BoomerAMG-BiCGSTAB" );
#endif

    for ( auto &exeName : exeNames ) {
        PROFILE_ENABLE();

        driver( comm, &ut, exeName );

        // build unique profile name to avoid collisions
        std::ostringstream ss;
        ss << exeName << std::setw( 3 ) << std::setfill( '0' )
           << AMP::AMPManager::getCommWorld().getSize();
        PROFILE_SAVE( ss.str() );
    }
    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
