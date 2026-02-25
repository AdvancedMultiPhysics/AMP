#include "AMP/solvers/testHelpers/testSolverHelpers.h"
#include "AMP/AMP_TPLs.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/solvers/SolverStrategy.h"
#include "AMP/utils/Database.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#ifdef AMP_USE_LIBMESH
    #include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#endif

#include <fstream>
#include <functional>
#include <iomanip>

// Check the solution of the form: T = a + b*z + c*z*z
bool checkAnalyticalSolution( const std::string &exeName,
                              std::function<double( double, double, double )> fun,
                              const AMP::Mesh::MeshIterator &iterator,
                              std::shared_ptr<const AMP::LinearAlgebra::Vector> vec )
{
    // Serial execution
    bool passes = true;
    auto DOFmap = vec->getDOFManager();
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    for ( int i = 0; i < globalComm.getSize(); ++i ) {
        if ( globalComm.getRank() == i ) {
            std::string filename = "data_" + exeName;
            int rank             = globalComm.getRank();
            int nranks           = globalComm.getSize();
            auto omode           = std::ios_base::out;
            if ( rank > 0 )
                omode |= std::ios_base::app;
            std::ofstream file( filename.c_str(), omode );
            if ( rank == 0 ) {
                file << "(* x y z analytic calculated relative-error *)" << std::endl;
                file << "results={" << std::endl;
            }
            file.precision( 14 );

            size_t numNodes = 0, iNode = 0;
            for ( auto it = iterator.begin(); it != iterator.end(); ++it )
                numNodes++;

            double mse = 0.0;
            for ( auto it = iterator.begin(); it != iterator.end(); ++it ) {
                std::vector<size_t> gid;
                DOFmap->getDOFs( it->globalID(), gid );
                double cal = vec->getValueByGlobalID( gid[0] );
                double x   = ( it->coord() )[0];
                double y   = ( it->coord() )[1];
                double z   = ( it->coord() )[2];
                double sol = fun( x, y, z );
                double err =
                    fabs( cal - sol ) * 2. / ( cal + sol + std::numeric_limits<double>::epsilon() );
                mse += ( sol - cal ) * ( sol - cal );
                file << "{" << x << "," << y << "," << z << "," << sol << "," << cal << "," << err
                     << "}";
                if ( iNode < numNodes - 1 )
                    file << "," << std::endl;
                if ( fabs( cal - sol ) > cal * 1e-3 )
                    passes = false;
                iNode++;
            }

            if ( rank == nranks - 1 ) {
                file << "};" << std::endl;
                mse /= ( 1. * iNode );
                mse = std::sqrt( mse );
                file << "l2err = {" << iNode << "," << mse << "};\n";
            }
            file.close();
        }
        globalComm.barrier();
    }
    return passes;
}

std::shared_ptr<AMP::Mesh::Mesh> createMesh( std::shared_ptr<AMP::Database> input_db )
{
    AMP_INSIST( input_db && input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db   = input_db->getDatabase( "Mesh" );
    auto mgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    auto comm      = AMP::AMP_MPI( AMP_COMM_WORLD );
    mgrParams->setComm( comm );
    return AMP::Mesh::MeshFactory::create( mgrParams );
}

std::pair<std::shared_ptr<AMP::Discretization::DOFManager>,
          std::shared_ptr<AMP::Discretization::DOFManager>>
getDofMaps( std::shared_ptr<const AMP::Mesh::Mesh> mesh )
{
    // Create a DOF manager for a nodal vector
    constexpr int DOFsPerNode          = 1;
    constexpr int DOFsPerElement       = 8;
    constexpr int nodalGhostWidth      = 1;
    constexpr int gaussPointGhostWidth = 1;
    bool split                         = true;
    auto nodalDofMap                   = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    auto gaussPointDofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Cell, gaussPointGhostWidth, DOFsPerElement, split );
    return std::make_pair( nodalDofMap, gaussPointDofMap );
}

#ifdef AMP_USE_LIBMESH
std::shared_ptr<AMP::LinearAlgebra::Vector>
constructNeutronicsPowerSource( std::shared_ptr<AMP::Database> input_db,
                                std::shared_ptr<AMP::Mesh::Mesh> mesh )
{

    auto [nodalDofMap, gaussPointDofMap] = getDofMaps( mesh );

    // CREATE THE NEUTRONICS SOURCE
    AMP_INSIST( input_db->keyExists( "NeutronicsOperator" ),
                "Key ''NeutronicsOperator'' is missing!" );
    auto neutronicsOp_db  = input_db->getDatabase( "NeutronicsOperator" );
    auto neutronicsParams = std::make_shared<AMP::Operator::OperatorParameters>( neutronicsOp_db );
    auto neutronicsOperator = std::make_shared<AMP::Operator::NeutronicsRhs>( neutronicsParams );

    auto SpecificPowerVec =
        AMP::LinearAlgebra::createVector( gaussPointDofMap,
                                          neutronicsOperator->getOutputVariable(),
                                          true,
                                          neutronicsOperator->getMemoryLocation() );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    neutronicsOperator->apply( nullVec, SpecificPowerVec );

    // Integrate Nuclear Source over Density * Volume
    AMP_INSIST( input_db->keyExists( "VolumeIntegralOperator" ), "key missing!" );
    auto sourceOperator = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            mesh, "VolumeIntegralOperator", input_db ) );

    // Create the power (heat source) vector.
    auto PowerInWattsVec = AMP::LinearAlgebra::createVector( nodalDofMap,
                                                             sourceOperator->getOutputVariable(),
                                                             true,
                                                             sourceOperator->getMemoryLocation() );
    PowerInWattsVec->zero();

    // convert the vector of specific power to power for a given basis.
    sourceOperator->apply( SpecificPowerVec, PowerInWattsVec );

    return PowerInWattsVec;
}
#else
std::shared_ptr<AMP::LinearAlgebra::Vector>
constructNeutronicsPowerSource( [[maybe_unused]] std::shared_ptr<AMP::Database> input_db,
                                [[maybe_unused]] std::shared_ptr<AMP::Mesh::Mesh> mesh )
{
    AMP_ERROR( "Required LibMesh to be enabled at present" );
    return nullptr;
}
#endif

// Function to get the "solution" convergence rate and iteration count for the
// given input
std::tuple<int, double, double, bool>
get_regression_solution( std::shared_ptr<const AMP::Database> input_db )
{
    auto db        = input_db->getDatabase( "Reference" );
    auto its       = db->getWithDefault<int>( "iterations", 0 );
    auto res_norm  = db->getWithDefault<double>( "res_l2_norm", -1.0 );
    auto tolerance = db->getWithDefault<double>( "tolerance", 0.0 );
    auto strict    = db->getWithDefault<bool>( "strict", false );
    return std::tuple<int, double, double, bool>( its, res_norm, tolerance, strict );
}

// Test for validity of solution by referencing map above and following
// the rules laid out there.
// Unrecognized input files just check if convergence reason is ok
void checkConvergence( AMP::Solver::SolverStrategy *solver,
                       std::shared_ptr<const AMP::Database> input_db,
                       const std::string &inputFile,
                       AMP::UnitTest &ut )
{
    const auto final_norm = static_cast<double>( solver->getResidualNorm() );
    const int iter        = solver->getIterations();
    const auto calc_norm  = std::fmax( static_cast<double>( solver->getAbsoluteTolerance() ),
                                      static_cast<double>( solver->getInitialResidual() ) *
                                          static_cast<double>( solver->getRelativeTolerance() ) );

    // Accept solution only if converged on absolute or relative tolerance
    const auto convReason = solver->getConvergenceStatus();
    const bool accept =
        convReason == AMP::Solver::SolverStrategy::SolverStatus::ConvergedOnRelTol ||
        convReason == AMP::Solver::SolverStrategy::SolverStatus::ConvergedOnAbsTol;

    if ( input_db->keyExists( "Reference" ) ) {
        // Get the reference information
        auto [ref_iter, ref_norm, ref_tol, strict] = get_regression_solution( input_db );
        // override reference norm if needed
        ref_norm = ref_norm > 0.0 ? ref_norm : calc_norm;
        // set pass/fail for each condition
        const bool pass_iters  = strict ? iter == ref_iter : iter <= ref_iter;
        const double norm_diff = final_norm - ref_norm;
        const bool pass_norm   = strict ? std::fabs( norm_diff ) <= ref_tol : norm_diff <= ref_tol;
        // Report passing or dump out information and fail
        if ( pass_iters && pass_norm && accept ) {
            ut.passes( "Passes convergence rate test for " + inputFile );
        } else if ( strict ) {
            AMP::pout << "FAILED: " << inputFile << std::endl;
            AMP::pout << "Iterations: " << iter << ", residual norm: " << std::setprecision( 15 )
                      << final_norm << std::endl;
            AMP::pout << "Expected: Iterations: " << ref_iter
                      << ", residual norm: " << std::setprecision( 15 ) << ref_norm << std::endl;
            AMP::pout << "Difference ( computed - reference ), iterations: " << ( iter - ref_iter )
                      << ", L2 norms: " << norm_diff << ", tolerance: " << ref_tol << std::endl;
            AMP::pout << "  Solver finished with status: " << solver->getConvergenceStatusString()
                      << std::endl;
            ut.failure( "FAILED: convergence rate test for " + inputFile );
        } else {
            AMP::pout << "FAILED: " << inputFile << std::endl;
            AMP::pout << "Iterations: " << iter << ", residual norm: " << std::setprecision( 15 )
                      << final_norm << std::endl;
            AMP::pout << "Maximum: Iterations: " << ref_iter
                      << ", residual norm: " << std::setprecision( 15 ) << ref_norm << std::endl;
            AMP::pout << "Difference ( computed - reference ): " << norm_diff
                      << ", tolerance: " << ref_tol << std::endl;
            AMP::pout << "  Solver finished with status: " << solver->getConvergenceStatusString()
                      << std::endl;
            ut.failure( "FAILED: convergence rate test for " + inputFile );
        }
    } else {
        if ( accept ) {
            ut.passes( "Solver has converged for " + inputFile );
        } else {
            AMP::pout << "Solver has NOT converged for " << inputFile << std::endl;
            AMP::pout << "  Solver finished with status: " << solver->getConvergenceStatusString()
                      << std::endl;
            ut.failure( "Solver has NOT converged for " + inputFile );
        }
    }
}
