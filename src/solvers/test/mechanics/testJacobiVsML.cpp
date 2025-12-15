#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>


static void myTest( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Read the mesh
    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( AMP::AMP_MPI( AMP_COMM_WORLD ) );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    std::cout << "Mesh has " << mesh->numLocalElements( AMP::Mesh::GeomType::Vertex ) << " nodes."
              << std::endl;

    auto bvpOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "MechanicsBVPOperator", input_db ) );

    auto displacementVariable = bvpOperator->getOutputVariable();

    auto dirichletVecOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator( mesh, "Load_Boundary", input_db ) );
    // This has an in-place apply. So, it has an empty input variable and
    // the output variable is the same as what it is operating on.
    dirichletVecOp->setVariable( displacementVariable );

    auto dofMap = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, 1, 3, true );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto mechSolVec = AMP::LinearAlgebra::createVector( dofMap, displacementVariable, true );
    auto mechRhsVec = mechSolVec->clone();
    auto mechResVec = mechSolVec->clone();

    mechRhsVec->zero();
    mechResVec->zero();

    dirichletVecOp->apply( nullVec, mechRhsVec );

    for ( int type = 1; type < 4; type++ ) {
        if ( type == 0 ) {
            std::cout << "Solving using CG algorithm (Own Implementation)..." << std::endl;

            auto linearSolver_db = input_db->getDatabase( "CGsolver" );

            int maxIters = linearSolver_db->getScalar<int>( "max_iterations" );

            auto matOutVec = mechSolVec->clone();
            auto pVec      = mechSolVec->clone();

            mechSolVec->zero();

            bvpOperator->apply( mechSolVec, matOutVec );

            mechResVec->subtract( *mechRhsVec, *matOutVec );

            pVec->copyVector( mechResVec );

            for ( int iter = 0; iter <= maxIters; iter++ ) {
                std::cout << "Iter = " << iter << " ResNorm2 = " << std::setprecision( 15 )
                          << mechResVec->L2Norm() << std::endl;

                bvpOperator->apply( pVec, matOutVec );

                std::cout << "CG-Iter = " << iter << " MatOutNorm2 = " << std::setprecision( 15 )
                          << matOutVec->L2Norm() << std::endl;

                double resOldDot = static_cast<double>( mechResVec->dot( *mechResVec ) );

                double alphaDenom = static_cast<double>( matOutVec->dot( *pVec ) );

                double alpha = resOldDot / alphaDenom;

                mechSolVec->axpy( alpha, *pVec, *mechSolVec );

                mechResVec->axpy( -alpha, *matOutVec, *mechResVec );

                double resNewDot = static_cast<double>( mechResVec->dot( *mechResVec ) );

                double beta = resNewDot / resOldDot;

                std::cout << "Iter = " << iter << " resOldDot = " << std::setprecision( 15 )
                          << resOldDot << " alphaDenom = " << std::setprecision( 15 ) << alphaDenom
                          << " alpha = " << std::setprecision( 15 ) << alpha
                          << " resNewDot = " << std::setprecision( 15 ) << resNewDot
                          << " beta = " << std::setprecision( 15 ) << beta << std::endl
                          << std::endl;

                pVec->axpy( beta, *pVec, *mechResVec );
            }

            std::cout << std::endl << std::endl;
        } else if ( type == 1 ) {
            std::cout << "Solving using CG algorithm (Petsc Implementation)..." << std::endl;

            auto linearSolver_db = input_db->getDatabase( "CGsolver" );

            // initialize the linear solver
            auto linearSolverParams =
                std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db );
            linearSolverParams->d_pOperator = bvpOperator;
            linearSolverParams->d_comm      = globalComm;
            auto linearSolver =
                std::make_shared<AMP::Solver::PetscKrylovSolver>( linearSolverParams );

            linearSolver->apply( mechRhsVec, mechSolVec );

            std::cout << std::endl << std::endl;
        } else if ( type == 2 ) {
            std::cout << "Solving using Jacobi preconditioned CG algorithm..." << std::endl;

            auto linearSolver_db = input_db->getDatabase( "JacobiCGsolver" );

            // initialize the linear solver
            auto linearSolverParams =
                std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db );
            linearSolverParams->d_pOperator = bvpOperator;
            linearSolverParams->d_comm      = globalComm;
            auto linearSolver =
                std::make_shared<AMP::Solver::PetscKrylovSolver>( linearSolverParams );

            linearSolver->apply( mechRhsVec, mechSolVec );

            std::cout << std::endl << std::endl;
        } else {
            std::cout << "Solving using ML preconditioned CG algorithm..." << std::endl;

            auto linearSolver_db = input_db->getDatabase( "MLCGsolver" );

            // ---- first initialize the preconditioner
            auto pcSolver_db = linearSolver_db->getDatabase( "MLsolver" );
            auto pcSolverParams =
                std::make_shared<AMP::Solver::TrilinosMLSolverParameters>( pcSolver_db );
            pcSolverParams->d_pOperator = bvpOperator;
            auto pcSolver = std::make_shared<AMP::Solver::TrilinosMLSolver>( pcSolverParams );

            // initialize the linear solver
            auto linearSolverParams =
                std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db );
            linearSolverParams->d_pOperator     = bvpOperator;
            linearSolverParams->d_comm          = globalComm;
            linearSolverParams->d_pNestedSolver = pcSolver;
            auto linearSolver =
                std::make_shared<AMP::Solver::PetscKrylovSolver>( linearSolverParams );

            linearSolver->apply( mechRhsVec, mechSolVec );

            std::cout << std::endl << std::endl;
        }
    }

    ut->passes( exeName );
}


int testJacobiVsML( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::string exeName = "testJacobiVsML";

    myTest( &ut, exeName );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
