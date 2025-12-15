#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/libmesh/initializeLibMesh.h"
#include "AMP/mesh/libmesh/libmeshMesh.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include <memory>

DISABLE_WARNINGS
#include "libmesh/mesh_communication.h"
#undef PETSC_VERSION_GIT
#undef PETSC_VERSION_DATE_GIT
#include "petsc.h"
#include "petscksp.h"
ENABLE_WARNINGS

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );


    AMP::AMP_MPI globalComm = AMP::AMP_MPI( AMP_COMM_WORLD );
    auto input_db           = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    int numMeshes = input_db->getScalar<int>( "NumberOfMeshFiles" );
    [[maybe_unused]] auto libmeshInit =
        std::make_shared<AMP::Mesh::initializeLibMesh>( globalComm );

    auto ml_db   = input_db->getDatabase( "ML_Solver" );
    auto lu_db   = input_db->getDatabase( "LU_Solver" );
    auto cg_db   = input_db->getDatabase( "CG_Solver" );
    auto rich_db = input_db->getDatabase( "Richardson_Solver" );

    for ( int meshId = 1; meshId <= numMeshes; meshId++ ) {
        std::cout << "Working on mesh " << meshId << std::endl;

        auto meshFileKey = AMP::Utilities::stringf( "mesh%d", meshId );

        auto meshFile = input_db->getString( meshFileKey );

        auto mesh = AMP::Mesh::MeshWriters::readBinaryTestMeshLibMesh( meshFile, AMP_COMM_WORLD );

        auto bvpOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator( mesh, "BVPOperator", input_db ) );

        auto dispVar = bvpOperator->getOutputVariable();

        auto loadOperator = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
            AMP::Operator::OperatorBuilder::createOperator( mesh, "LoadOperator", input_db ) );
        loadOperator->setVariable( dispVar );

        auto NodalVectorDOF = AMP::Discretization::simpleDOFManager::create(
            mesh, AMP::Mesh::GeomType::Vertex, 1, 3 );

        AMP::LinearAlgebra::Vector::shared_ptr nullVec;
        auto solVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );
        auto rhsVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );
        auto resVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, dispVar );

        rhsVec->zero();
        loadOperator->apply( nullVec, rhsVec );

        solVec->zero();
        resVec->zero();

        size_t numDofs = solVec->getGlobalSize();

        if ( globalComm.getRank() == 0 ) {
            std::cout << "Solving using LU" << std::endl;
        }
        globalComm.barrier();
        double luStartTime = AMP::AMP_MPI::time();

        auto luParams         = std::make_shared<AMP::Solver::TrilinosMLSolverParameters>( lu_db );
        luParams->d_pOperator = bvpOperator;
        auto luPC             = std::make_shared<AMP::Solver::TrilinosMLSolver>( luParams );

        auto richParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( rich_db );
        richParams->d_pOperator     = bvpOperator;
        richParams->d_comm          = globalComm;
        richParams->d_pNestedSolver = luPC;
        auto richSolver = std::make_shared<AMP::Solver::PetscKrylovSolver>( richParams );
        richSolver->setZeroInitialGuess( true );

        richSolver->apply( rhsVec, solVec );

        globalComm.barrier();
        double luEndTime = AMP::AMP_MPI::time();

        PetscInt richIters = 0;
        KSP richKsp        = richSolver->getKrylovSolver();
        KSPGetIterationNumber( richKsp, &richIters );
        AMP_INSIST( richIters <= 1, "Should not need more than 1 LU-Richardson iteration." );

        KSPConvergedReason richReason;
        KSPGetConvergedReason( richKsp, &richReason );
        AMP_INSIST(
            ( ( richReason == KSP_CONVERGED_RTOL ) || ( richReason == KSP_CONVERGED_ATOL ) ),
            "KSP did not converge properly." );

        richSolver.reset();
        luPC.reset();

        solVec->zero();
        resVec->zero();

        if ( globalComm.getRank() == 0 ) {
            std::cout << "Solving using ML" << std::endl;
        }
        globalComm.barrier();
        double mlStartTime = AMP::AMP_MPI::time();

        auto mlParams         = std::make_shared<AMP::Solver::TrilinosMLSolverParameters>( ml_db );
        mlParams->d_pOperator = bvpOperator;
        auto mlPC             = std::make_shared<AMP::Solver::TrilinosMLSolver>( mlParams );

        auto cgParams         = std::make_shared<AMP::Solver::SolverStrategyParameters>( cg_db );
        cgParams->d_pOperator = bvpOperator;
        cgParams->d_comm      = globalComm;
        cgParams->d_pNestedSolver = mlPC;
        auto cgSolver             = std::make_shared<AMP::Solver::PetscKrylovSolver>( cgParams );
        cgSolver->setZeroInitialGuess( true );

        cgSolver->apply( rhsVec, solVec );

        globalComm.barrier();
        double mlEndTime = AMP::AMP_MPI::time();

        PetscInt cgIters = 0;
        KSP cgKsp        = cgSolver->getKrylovSolver();
        KSPGetIterationNumber( cgKsp, &cgIters );

        KSPConvergedReason cgReason;
        KSPGetConvergedReason( cgKsp, &cgReason );
        AMP_INSIST( ( ( cgReason == KSP_CONVERGED_RTOL ) || ( cgReason == KSP_CONVERGED_ATOL ) ),
                    "KSP did not converge properly." );

        cgSolver.reset();
        mlPC.reset();

        if ( globalComm.getRank() == 0 ) {
            std::cout << "Result: " << numDofs << " & " << globalComm.getSize() << " & " << cgIters
                      << " & " << ( luEndTime - luStartTime ) << " & "
                      << ( mlEndTime - mlStartTime ) << " \\\\ " << std::endl;
        }

    } // end for meshId

    ut->passes( exeName );
}

int testMechElasticLUvsML( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName = "testMechElasticLUvsML";

    myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
