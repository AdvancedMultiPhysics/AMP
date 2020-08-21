#include "AMP/ampmesh/Mesh.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/materials/Material.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NeutronicsRhs.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/diffusion/FickSoretNonlinearFEOperator.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/hypre/BoomerAMGSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolverParameters.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolverParameters.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/PIO.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/utils/Writer.h"
#include "AMP/vectors/VectorBuilder.h"
#include <memory>

#include <iostream>
#include <string>


struct null_deleter {
    void operator()( void const * ) const {}
};


void fickSoretTest( AMP::UnitTest *ut, std::string exeName, std::vector<double> &results )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::PIO::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    //--------------------------------------------------
    //   Create the Mesh.
    //--------------------------------------------------
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    std::shared_ptr<AMP::Database> mesh_db = input_db->getDatabase( "Mesh" );
    std::shared_ptr<AMP::Mesh::MeshParameters> mgrParams(
        new AMP::Mesh::MeshParameters( mesh_db ) );
    mgrParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    std::shared_ptr<AMP::Mesh::Mesh> meshAdapter = AMP::Mesh::Mesh::buildMesh( mgrParams );
    //--------------------------------------------------

    //--------------------------------------------------
    // Create a DOF manager for a nodal vector
    //--------------------------------------------------
    int DOFsPerNode     = 1;
    int nodalGhostWidth = 1;
    bool split          = true;
    AMP::Discretization::DOFManager::shared_ptr nodalDofMap =
        AMP::Discretization::simpleDOFManager::create(
            meshAdapter, AMP::Mesh::GeomType::Vertex, nodalGhostWidth, DOFsPerNode, split );
    //--------------------------------------------------

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // create a nonlinear BVP operator for nonlinear Fick-Soret diffusion
    AMP_INSIST( input_db->keyExists( "testNonlinearFickSoretBVPOperator" ), "key missing!" );

    // Create nonlinear FickSoret BVP operator and access volume nonlinear FickSoret operator
    std::shared_ptr<AMP::Operator::ElementPhysicsModel> elementPhysicsModel;
    std::shared_ptr<AMP::Operator::Operator> nlinBVPOperator =
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "testNonlinearFickSoretBVPOperator", input_db, elementPhysicsModel );
    std::shared_ptr<AMP::Operator::NonlinearBVPOperator> nlinBVPOp =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>( nlinBVPOperator );
    std::shared_ptr<AMP::Operator::FickSoretNonlinearFEOperator> nlinOp =
        std::dynamic_pointer_cast<AMP::Operator::FickSoretNonlinearFEOperator>(
            nlinBVPOp->getVolumeOperator() );
    std::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperator> fickOp =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nlinOp->getFickOperator() );
    std::shared_ptr<AMP::Operator::DiffusionNonlinearFEOperator> soretOp =
        std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
            nlinOp->getSoretOperator() );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // use the linear BVP operator to create a Fick linear operator with bc's
    AMP_INSIST( input_db->keyExists( "testLinearFickBVPOperator" ), "key missing!" );

    std::shared_ptr<AMP::Operator::Operator> linBVPOperator =
        AMP::Operator::OperatorBuilder::createOperator(
            meshAdapter, "testLinearFickBVPOperator", input_db, elementPhysicsModel );
    std::shared_ptr<AMP::Operator::LinearBVPOperator> linBVPOp =
        std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>( linBVPOperator );
    // std::shared_ptr<AMP::Operator::DiffusionLinearFEOperator> linOp =
    //         std::dynamic_pointer_cast<AMP::Operator::DiffusionLinearFEOperator>(linBVPOp->getVolumeOperator());

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // Set up input and output variables
    // AMP::LinearAlgebra::Variable::shared_ptr
    // tVar(fickOp->getInputVariable(AMP::Operator::Diffusion::TEMPERATURE));
    // AMP::LinearAlgebra::Variable::shared_ptr
    // cVar(fickOp->getInputVariable(AMP::Operator::Diffusion::CONCENTRATION));
    AMP::LinearAlgebra::Variable::shared_ptr tVar( new AMP::LinearAlgebra::Variable( "temp" ) );
    AMP::LinearAlgebra::Variable::shared_ptr cVar( fickOp->getOutputVariable() );
    std::shared_ptr<AMP::LinearAlgebra::Variable> fsOutVar( nlinBVPOp->getOutputVariable() );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // create solution, rhs, and residual vectors

    AMP::LinearAlgebra::Vector::shared_ptr solVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, cVar );
    AMP::LinearAlgebra::Vector::shared_ptr rhsVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, fsOutVar );
    AMP::LinearAlgebra::Vector::shared_ptr resVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, fsOutVar );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // create parameters for reset test and reset fick and soret operators

    AMP::LinearAlgebra::Vector::shared_ptr tVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, tVar );

    fickOp->setVector( 0, tVec );
    soretOp->setVector( 0, tVec );

    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> fickFrozen  = fickOp->getFrozen();
    std::vector<AMP::LinearAlgebra::Vector::shared_ptr> soretFrozen = soretOp->getFrozen();

    double lenscale = input_db->getDouble( "LengthScale" );
    soretFrozen[AMP::Operator::Diffusion::TEMPERATURE]->setToScalar(
        300. ); // Fill in manufactured solution
    int zeroGhostWidth = 0;
    AMP::Mesh::MeshIterator iterator =
        meshAdapter->getIterator( AMP::Mesh::GeomType::Vertex, zeroGhostWidth );
    for ( ; iterator != iterator.end(); iterator++ ) {
        double x, y;
        std::valarray<double> poly( 10 );
        x = ( iterator->coord() )[0];
        y = ( iterator->coord() )[1];
        std::vector<size_t> gid;
        nodalDofMap->getDOFs( iterator->globalID(), gid );
        double value =
            300. + 450 * ( 1. - ( x * x / lenscale / lenscale + y * y / lenscale / lenscale ) );
        fickFrozen[AMP::Operator::Diffusion::TEMPERATURE]->setValueByGlobalID( gid[0], value );
        soretFrozen[AMP::Operator::Diffusion::TEMPERATURE]->setValueByGlobalID( gid[0], value );
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // Initial guess

    double initialValue = input_db->getDouble( "InitialValue" );
    solVec->setToScalar( initialValue );
    double initialGuessNorm = solVec->L2Norm();
    std::cout << "initial guess norm = " << initialGuessNorm << "\n";

    nlinBVPOp->modifyInitialSolutionVector( solVec );

    initialGuessNorm = solVec->L2Norm();
    std::cout << "initial guess norm  after apply = " << initialGuessNorm << "\n";

    rhsVec->setToScalar( 0.0 );
    nlinBVPOp->modifyRHSvector( rhsVec );

    //----------------------------------------------------------------------------------------------------------------------------------------------/

    std::shared_ptr<AMP::Database> nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );
    std::shared_ptr<AMP::Database> linearSolver_db =
        nonlinearSolver_db->getDatabase( "LinearSolver" );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // initialize the nonlinear solver
    std::shared_ptr<AMP::Solver::PetscSNESSolverParameters> nonlinearSolverParams(
        new AMP::Solver::PetscSNESSolverParameters( nonlinearSolver_db ) );

    // change the next line to get the correct communicator out
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = nlinBVPOp;
    nonlinearSolverParams->d_pInitialGuess = solVec;

    std::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearSolver(
        new AMP::Solver::PetscSNESSolver( nonlinearSolverParams ) );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    std::shared_ptr<AMP::Database> fickPreconditioner_db =
        linearSolver_db->getDatabase( "Preconditioner" );
    std::shared_ptr<AMP::Solver::SolverStrategyParameters> fickPreconditionerParams(
        new AMP::Solver::SolverStrategyParameters( fickPreconditioner_db ) );
    fickPreconditionerParams->d_pOperator = linBVPOp;
    auto linearFickPreconditioner =
        std::make_shared<AMP::Solver::BoomerAMGSolver>( fickPreconditionerParams );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // register the preconditioner with the Jacobian free Krylov solver
    std::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver =
        nonlinearSolver->getKrylovSolver();

    linearSolver->setPreconditioner( linearFickPreconditioner );

    nlinBVPOp->residual( rhsVec, solVec, resVec );
    double initialResidualNorm = resVec->L2Norm();

    AMP::pout << "Initial Residual Norm: " << initialResidualNorm << std::endl;

    nonlinearSolver->setZeroInitialGuess( false );

    nonlinearSolver->solve( rhsVec, solVec );

    nlinBVPOp->residual( rhsVec, solVec, resVec );

    double finalResidualNorm = resVec->L2Norm();

    std::cout << "Final Residual Norm: " << finalResidualNorm << std::endl;

    solVec->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );
    resVec->makeConsistent( AMP::LinearAlgebra::VectorData::ScatterType::CONSISTENT_SET );

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // evaluate and register material coefficients for graphical output

    AMP::LinearAlgebra::Variable::shared_ptr fickCoeffVar(
        new AMP::LinearAlgebra::Variable( "FickCoefficient" ) );
    AMP::LinearAlgebra::Variable::shared_ptr soretCoeffVar(
        new AMP::LinearAlgebra::Variable( "SoretCoefficient" ) );
    AMP::LinearAlgebra::Vector::shared_ptr fickCoeffVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, fickCoeffVar );
    AMP::LinearAlgebra::Vector::shared_ptr soretCoeffVec =
        AMP::LinearAlgebra::createVector( nodalDofMap, soretCoeffVar );
    std::shared_ptr<AMP::Operator::DiffusionTransportModel> fickModel = fickOp->getTransportModel();
    std::shared_ptr<AMP::Operator::DiffusionTransportModel> soretModel =
        soretOp->getTransportModel();

    {
        int zeroGhostWidth = 0;
        AMP::Mesh::MeshIterator iterator =
            meshAdapter->getIterator( AMP::Mesh::GeomType::Vertex, zeroGhostWidth );
        size_t nnodes = fickCoeffVec->getLocalSize(), node;
        std::vector<size_t> gids( nnodes );
        std::vector<double> temp( nnodes ), conc( nnodes ), fickCoeff( nnodes ),
            soretCoeff( nnodes ), burn( nnodes );
        for ( node = 0; iterator != iterator.end(); iterator++ ) {
            std::vector<size_t> gid;
            nodalDofMap->getDOFs( iterator->globalID(), gid );
            gids[node] = gid[0];
            node++;
        }
        AMP_INSIST( node == nnodes, "invalid count" );
        fickFrozen[AMP::Operator::Diffusion::TEMPERATURE]->getValuesByGlobalID(
            nnodes, &gids[0], &temp[0] );
        solVec->getValuesByGlobalID( nnodes, &gids[0], &conc[0] );
        // this is  used to plot the fick and soret coefficnets used.  commenting it out till
        // someone finds out.
        // This is kevin - i found out because the vector wasn't used when silo is not enabled.
        std::map<std::string, std::shared_ptr<std::vector<double>>> args;
        args.insert( std::make_pair(
            "temperature", std::shared_ptr<std::vector<double>>( &temp, null_deleter() ) ) );
        args.insert( std::make_pair(
            "concentration", std::shared_ptr<std::vector<double>>( &conc, null_deleter() ) ) );
        args.insert( std::make_pair(
            "burnup", std::shared_ptr<std::vector<double>>( &burn, null_deleter() ) ) );
        fickModel->getTransport( fickCoeff, args );
        soretModel->getTransport( soretCoeff, args );
        fickCoeffVec->setValuesByGlobalID( nnodes, &gids[0], &fickCoeff[0] );
        soretCoeffVec->setValuesByGlobalID( nnodes, &gids[0], &soretCoeff[0] );
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // write graphical output

#ifdef USE_EXT_SILO
    AMP::Utilities::Writer::shared_ptr siloWriter = AMP::Utilities::Writer::buildWriter( "Silo" );
    siloWriter->registerMesh( meshAdapter );

    siloWriter->registerVector( solVec, meshAdapter, AMP::Mesh::GeomType::Vertex, "Solution" );
    siloWriter->registerVector( resVec, meshAdapter, AMP::Mesh::GeomType::Vertex, "Residual" );
    siloWriter->registerVector( fickFrozen[AMP::Operator::Diffusion::TEMPERATURE],
                                meshAdapter,
                                AMP::Mesh::GeomType::Vertex,
                                "Temperature" );
    siloWriter->registerVector(
        fickCoeffVec, meshAdapter, AMP::Mesh::GeomType::Vertex, "FickCoefficient" );
    siloWriter->registerVector(
        soretCoeffVec, meshAdapter, AMP::Mesh::GeomType::Vertex, "ThermalDiffusionCoefficient" );

    siloWriter->writeFile( exeName, 0 );
#endif

    //----------------------------------------------------------------------------------------------------------------------------------------------//
    // store result
    {
        int zeroGhostWidth = 0;
        AMP::Mesh::MeshIterator iterator =
            meshAdapter->getIterator( AMP::Mesh::GeomType::Vertex, zeroGhostWidth );
        iterator        = iterator.begin();
        size_t numNodes = 0;
        for ( ; iterator != iterator.end(); iterator++ )
            numNodes++;
        results.resize( numNodes );

        iterator     = iterator.begin();
        size_t iNode = 0;
        for ( ; iterator != iterator.end(); iterator++ ) {
            std::vector<size_t> gid;
            nodalDofMap->getDOFs( iterator->globalID(), gid );
            results[iNode] = solVec->getValueByGlobalID( gid[0] );
            iNode++;
        }
    }

    if ( finalResidualNorm > 1.0e-08 ) {
        ut->failure( exeName );
    } else {
        ut->passes( "PetscSNES Solver successfully solves a nonlinear mechanics equation with "
                    "Jacobian provided, "
                    "FGMRES for Krylov" );
    }
    ut->passes( exeName );
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<double> results;
    fickSoretTest( &ut, "testPetscSNESSolver-NonlinearFickSoret-cylinder-OxMSRZC09-1", results );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
