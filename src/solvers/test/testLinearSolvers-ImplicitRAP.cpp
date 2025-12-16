#include <fstream>
#include <iomanip>

#include "AMP/operators/diffusionFD/DiffusionFD.h"
#include "AMP/operators/diffusionFD/DiffusionRotatedAnisotropicModel.h"
#include "AMP/operators/testHelpers/FDHelper.h"

#include "AMP/matrices/CSRVisit.h"
#include "AMP/solvers/amg/Aggregation.h"

#include "AMP/mesh/MeshParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Memory.h"
#include "AMP/utils/UnitTest.h"

std::shared_ptr<AMP::Operator::LinearOperator>
implicit_RAP( std::shared_ptr<AMP::Operator::LinearOperator> op,
              AMP::Solver::AMG::Aggregator &aggregator )
{
    auto [R, Ac, P] = AMP::Solver::AMG::aggregator_coarsen( op, aggregator );

    return Ac;
}

std::shared_ptr<AMP::Operator::LinearOperator>
explicit_RAP( std::shared_ptr<AMP::Operator::LinearOperator> op,
              AMP::Solver::AMG::Aggregator &aggregator )
{
    auto A  = op->getMatrix();
    auto P  = aggregator.getAggregateMatrix( A );
    auto R  = P->transpose();
    auto AP = AMP::LinearAlgebra::Matrix::matMatMult( A, P );
    auto Ac = AMP::LinearAlgebra::Matrix::matMatMult( R, AP );

    auto db     = std::make_shared<AMP::Database>( "coarse" );
    auto params = std::make_shared<AMP::Operator::OperatorParameters>( db );
    auto Acop   = std::make_shared<AMP::Operator::LinearOperator>( params );
    Acop->setMatrix( Ac );

    return Acop;
}

void driver( AMP::AMP_MPI comm,
             AMP::UnitTest &ut,
             const std::string &input_fname,
             const std::string &mem_loc_label,
             const std::string &accel_backend_label )
{
    std::string logfile{ "output_" + input_fname };
    AMP::logOnlyNodeZero( logfile );

    AMP::pout << "Running driver with input " << input_fname << '\n';

    auto input_db = AMP::Database::parseInputFile( input_fname );
    AMP::plog << "Input database:" << '\n';
    AMP::plog << "---------------" << '\n';
    input_db->print( AMP::plog );

    auto ra_coeff_db = input_db->getDatabase( "RACoefficients" );
    auto mesh_db     = input_db->getDatabase( "Mesh" );

    AMP_INSIST( ra_coeff_db, "''RACoefficients'' dabase must be provided" );
    AMP_INSIST( mesh_db, "''Mesh'' database must be provided" );

    auto mesh_params = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    mesh_params->setComm( comm );
    std::shared_ptr<AMP::Mesh::BoxMesh> mesh = AMP::Mesh::BoxMesh::generate( mesh_params );

    AMP::plog << "--------------------------------------------------------------------------------"
              << std::endl;
    AMP::plog << "Building " << static_cast<int>( mesh->getDim() )
              << "D Poisson problem on mesh with "
              << mesh->numGlobalElements( AMP::Mesh::GeomType::Vertex ) << " total DOFs across "
              << mesh->getComm().getSize() << " ranks" << std::endl;
    AMP::plog << "--------------------------------------------------------------------------------"
              << std::endl;

    auto ra_diffusion_model =
        std::make_shared<AMP::Operator::ManufacturedRotatedAnisotropicDiffusionModel>(
            ra_coeff_db );

    const auto op_db = std::make_shared<AMP::Database>( "linearOperatorDB" );
    op_db->putScalar<int>( "print_info_level", 0 );
    op_db->putScalar<std::string>( "name", "DiffusionFDOperator" );
    op_db->putDatabase( "DiffusionCoefficients", ra_diffusion_model->d_c_db->cloneDatabase() );

    auto op_params    = std::make_shared<AMP::Operator::OperatorParameters>( op_db );
    op_params->d_name = "DiffusionFDOperator";
    op_params->d_Mesh = mesh;

    auto poisson_op_host = std::make_shared<AMP::Operator::DiffusionFDOperator>( op_params );

    auto mem_loc    = AMP::Utilities::memoryLocationFromString( mem_loc_label );
    auto backend    = AMP::Utilities::backendFromString( accel_backend_label );
    auto poisson_op = [&]() -> std::shared_ptr<AMP::Operator::LinearOperator> {
        if ( mem_loc == AMP::Utilities::MemoryType::host ) {
            poisson_op_host->getMatrix()->setBackend( backend );
            return poisson_op_host;
        }

        auto inVar  = poisson_op_host->getInputVariable();
        auto outVar = poisson_op_host->getOutputVariable();

        // Create operator to wrap matrix
        auto op_db = std::make_shared<AMP::Database>( "LinearOperator" );
        op_db->putScalar<std::string>( "AccelerationBackend", accel_backend_label );
        op_db->putScalar<std::string>( "MemoryLocation", mem_loc_label );

        auto op_params       = std::make_shared<AMP::Operator::OperatorParameters>( op_db );
        auto poisson_op      = std::make_shared<AMP::Operator::LinearOperator>( op_params );
        auto matrix          = poisson_op_host->getMatrix();
        auto migrated_matrix = AMP::LinearAlgebra::createMatrix( matrix, mem_loc, backend );
        poisson_op->setMatrix( migrated_matrix );
        poisson_op->setVariables( inVar, outVar );
        return poisson_op;
    }();

    AMP::Solver::AMG::PairwiseCoarsenSettings coarsen_settings;
    coarsen_settings.pairwise_passes    = 3;
    coarsen_settings.checkdd            = true;
    coarsen_settings.strength_threshold = 0.25;

    AMP::Solver::AMG::PairwiseAggregator aggregator{ coarsen_settings };

    auto Ac_implicit_op = implicit_RAP( poisson_op, aggregator );
    auto Ac_explicit_op = explicit_RAP( poisson_op, aggregator );

    auto Ac_explicit = Ac_explicit_op->getMatrix();
    auto Ac_implicit = Ac_implicit_op->getMatrix();

    Ac_implicit->axpy( -1, *Ac_explicit );

    auto mnorm = Ac_implicit->LinfNorm();
    AMP::pout << mnorm << std::endl;

    if ( mnorm < 1e-10 )
        ut.passes( "Passes RAP difference test." );
    else
        ut.failure( "FAILED: Implicit and Explicit RAP difference exceeded tolerance." );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    PROFILE_ENABLE();
    AMP::AMP_MPI comm( AMP_COMM_WORLD );


    if ( argc > 2 ) {
        driver( comm, ut, argv[2], argv[1], "serial" );
    }

    std::ostringstream ss;
    ss << "testLinearSolvers-ImplicitRAP_r" << std::setw( 3 ) << std::setfill( '0' )
       << comm.getSize();
    PROFILE_SAVE( ss.str() );

    ut.report();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
