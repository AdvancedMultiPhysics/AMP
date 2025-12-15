#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionTransportModel.h"
#include "AMP/operators/libmesh/MassLinearElement.h"
#include "AMP/operators/libmesh/MassLinearFEOperator.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/time_integrators/sundials/IDATimeIntegrator.h"
#include "AMP/time_integrators/sundials/IDATimeOperator.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Variable.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include <memory>
#include <string>


static void IDATimeIntegratorTest( AMP::UnitTest *ut )
{
    std::weak_ptr<AMP::Mesh::Mesh> mesh_ref;
    std::string input_file = "input_testIDA-LinearBVPOperator-1";
    std::string log_file   = "output_testIDA-LinearBVPOperator-1";
    AMP::logOnlyNodeZero( log_file );

    {
        // Read the input file

        auto input_db = AMP::Database::parseInputFile( input_file );

        // Get the Mesh database and create the mesh parameters
        auto database = input_db->getDatabase( "Mesh" );
        auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
        params->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );

        // Create the meshes from the input database
        auto mesh = AMP::Mesh::MeshFactory::create( params );
        mesh_ref  = std::weak_ptr<AMP::Mesh::Mesh>( mesh );

        // Create a DOF manager for a nodal vector
        int DOFsPerNode = 1;
        // int DOFsPerElement = 8;
        int nodalGhostWidth = 1;
        // int gaussPointGhostWidth = 1;
        bool split       = true;
        auto nodalDofMap = AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );

        // create a linear BVP operator
        std::shared_ptr<AMP::Operator::LinearBVPOperator> linearPCOperator;
        auto IDARhsOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator( mesh, "LinearOperator", input_db ) );

        // create a mass linear BVP operator
        auto massOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                mesh, "MassLinearOperator", input_db ) );

        // create vectors for initial conditions (IC) and time derivative at IC
        auto outputVar             = IDARhsOperator->getOutputVariable();
        auto initialCondition      = AMP::LinearAlgebra::createVector( nodalDofMap, outputVar );
        auto initialConditionPrime = AMP::LinearAlgebra::createVector( nodalDofMap, outputVar );
        initialCondition->zero();
        initialConditionPrime->zero();

        // get the ida database
        AMP_INSIST( input_db->keyExists( "IDATimeIntegrator" ),
                    "Key ''IDATimeIntegrator'' is missing!" );
        auto ida_db      = input_db->getDatabase( "IDATimeIntegrator" );
        auto pcSolver_db = ida_db->getDatabase( "Preconditioner" );
        auto pcSolverParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( pcSolver_db );
        auto pcSolver = std::make_shared<AMP::Solver::TrilinosMLSolver>( pcSolverParams );

        // create the IDA time integrator
        auto time_Params =
            std::make_shared<AMP::TimeIntegrator::IDATimeIntegratorParameters>( ida_db );
        time_Params->d_pMassOperator   = massOperator;
        time_Params->d_operator        = IDARhsOperator;
        time_Params->d_pNestedSolver   = pcSolver;
        time_Params->d_ic_vector       = initialCondition;
        time_Params->d_ic_vector_prime = initialConditionPrime;
        time_Params->d_name            = "IDATimeIntegratorParameters";
        auto pIDATimeIntegrator =
            std::make_shared<AMP::TimeIntegrator::IDATimeIntegrator>( time_Params );
        if ( pIDATimeIntegrator.get() == nullptr ) {
            ut->failure( "Testing IDATimeIntegrator's constructor" );
        } else {
            ut->passes( "Tested IDATimeIntegrator's constructor" );
        }
    }

    // Check that everything was destroyed
    if ( mesh_ref.expired() )
        ut->passes( "IDATimeIntegrator delete" );
    else
        ut->failure( "IDATimeIntegrator delete" );
}


// Main
int testIDATimeIntegrator( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    IDATimeIntegratorTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
