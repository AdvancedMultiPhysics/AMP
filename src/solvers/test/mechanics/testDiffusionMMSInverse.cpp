#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/OperatorFactory.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrectionParameters.h"
#include "AMP/operators/boundary/libmesh/NeumannVectorCorrection.h"
#include "AMP/operators/diffusion/DiffusionLinearElement.h"
#include "AMP/operators/diffusion/DiffusionLinearFEOperator.h"
#include "AMP/operators/diffusion/DiffusionNonlinearElement.h"
#include "AMP/operators/diffusion/DiffusionNonlinearFEOperator.h"
#include "AMP/operators/libmesh/MassDensityModel.h"
#include "AMP/operators/libmesh/MassLinearFEOperator.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/ManufacturedSolution.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"

#include "../../../operators/test/applyTests.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>


static void inverseTest1( AMP::UnitTest *ut, const std::string &exeName )
{
    // Tests diffusion Dirchlet BVP operator for temperature

    // Initialization
    std::string input_file = exeName;
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    // Read the input file
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    // Get the Mesh database and create the mesh parameters
    auto database = input_db->getDatabase( "Mesh" );
    auto params   = std::make_shared<AMP::Mesh::MeshParameters>( database );
    params->setComm( globalComm );

    // Create the meshes from the input database
    auto mesh = AMP::Mesh::MeshFactory::create( params );

    // Create nonlinear diffusion BVP operator and access volume nonlinear Diffusion operator
    auto nlinBVPOperator =
        AMP::Operator::OperatorBuilder::createOperator( mesh, "NonlinearBVPOperator", input_db );
    auto nlinBVPOp =
        std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>( nlinBVPOperator );
    auto nlinOp = std::dynamic_pointer_cast<AMP::Operator::DiffusionNonlinearFEOperator>(
        nlinBVPOp->getVolumeOperator() );

    // Get source mass operator
    auto sourceOperator = AMP::Operator::OperatorBuilder::createOperator(
        mesh, "ManufacturedSourceOperator", input_db );
    auto sourceOp =
        std::dynamic_pointer_cast<AMP::Operator::MassLinearFEOperator>( sourceOperator );

    auto densityModel = sourceOp->getDensityModel();
    auto mfgSolution  = densityModel->getManufacturedSolution();

    // Set up input and output vectors
    auto solVar  = nlinOp->getOutputVariable();
    auto rhsVar  = nlinOp->getOutputVariable();
    auto resVar  = nlinOp->getOutputVariable();
    auto bndVar  = nlinOp->getOutputVariable();
    auto inpVar  = sourceOp->getInputVariable();
    auto srcVar  = sourceOp->getOutputVariable();
    auto workVar = std::make_shared<AMP::LinearAlgebra::Variable>( "work" );

    auto Vertex = AMP::Mesh::GeomType::Vertex;
    auto DOF    = AMP::Discretization::simpleDOFManager::create( mesh, Vertex, 1, 1, true );

    auto solVec  = AMP::LinearAlgebra::createVector( DOF, solVar );
    auto rhsVec  = AMP::LinearAlgebra::createVector( DOF, rhsVar );
    auto resVec  = AMP::LinearAlgebra::createVector( DOF, resVar );
    auto bndVec  = AMP::LinearAlgebra::createVector( DOF, bndVar );
    auto inpVec  = AMP::LinearAlgebra::createVector( DOF, inpVar );
    auto srcVec  = AMP::LinearAlgebra::createVector( DOF, srcVar );
    auto workVec = AMP::LinearAlgebra::createVector( DOF, workVar );

    resVec->zero();
    srcVec->zero();
    inpVec->zero();

    // Fill in manufactured solution in mesh interior
    const double Pi     = 3.1415926535898;
    std::string mfgName = mfgSolution->get_name();
    bool isCylindrical  = mfgName.find( "Cylindrical" ) < mfgName.size();
    std::vector<size_t> gid;
    for ( auto &elem : mesh->getIterator( Vertex, 0 ) ) {
        auto coord = elem.coord();
        double x   = coord[0];
        double y   = coord[1];
        double z   = coord[2];
        std::array<double, 10> poly;
        if ( isCylindrical ) {
            double th = 0.;
            double r  = std::sqrt( x * x + y * y );
            if ( r > 0 ) {
                th = acos( x / r );
                if ( y < 0. )
                    th = 2 * Pi - th;
            }
            poly = mfgSolution->evaluate( r, th, z );
        } else {
            poly = mfgSolution->evaluate( x, y, z );
        }
        DOF->getDOFs( elem.globalID(), gid );
        std::vector<double> srcVal( 1 ), dumT( 1 ), dumU( 1 ), dumB( 1 );
        std::vector<libMesh::Point> point( 1, libMesh::Point( x, y, z ) );
        densityModel->getDensityManufactured( srcVal, dumT, dumU, dumB, point );
        inpVec->setValuesByGlobalID( 1, &gid[0], &srcVal[0] );
    }
    inpVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Fill in manufactured solution on mesh boundary
    for ( int j = 0; j <= 8; j++ ) {
        for ( auto &elem : mesh->getBoundaryIDIterator( Vertex, j, 0 ) ) {
            auto coord = elem.coord();
            double x   = coord[0];
            double y   = coord[1];
            double z   = coord[2];
            std::array<double, 10> poly;
            if ( isCylindrical ) {
                double th = 0.;
                double r  = std::sqrt( x * x + y * y );
                if ( r > 0 ) {
                    th = acos( x / r );
                    if ( y < 0. )
                        th = 2 * Pi - th;
                }
                poly = mfgSolution->evaluate( r, th, z );
            } else {
                poly = mfgSolution->evaluate( x, y, z );
            }
            DOF->getDOFs( elem.globalID(), gid );
            AMP_ASSERT( gid.size() == 1 );
            bndVec->setValuesByGlobalID( 1, &gid[0], &poly[0] );
        }
    }

    // Set boundary values for manufactured solution for sinusoid, gaussian, etc. (non constant BC)
    auto dirichletOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        nlinBVPOp->getBoundaryOperator() );
    if ( dirichletOp ) {
        dirichletOp->setVariable( bndVar );
        dirichletOp->setDirichletValues( bndVec );
    }

    // Evaluate manufactured solution as an FE source
    sourceOp->apply( inpVec, rhsVec );

    // Reset solution vector to initial value and print out norm
    solVec->setToScalar( 0.1 );

    // Set up initial guess
    nlinBVPOp->modifyInitialSolutionVector( solVec );

    // Set up solver
    auto nonlinearSolver_db = input_db->getDatabase( "NonlinearSolver" );
    auto linearSolver_db    = nonlinearSolver_db->getDatabase( "LinearSolver" );
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );
    nonlinearSolverParams->d_comm          = globalComm;
    nonlinearSolverParams->d_pOperator     = nlinBVPOp;
    nonlinearSolverParams->d_pInitialGuess = solVec;
    auto nonlinearSolver = std::make_shared<AMP::Solver::PetscSNESSolver>( nonlinearSolverParams );

    // Set up preconditioner
    auto preconditioner_db = linearSolver_db->getDatabase( "Preconditioner" );
    auto preconditionerParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( preconditioner_db );
    auto pc_params                    = nlinBVPOp->getParameters( "Jacobian", solVec );
    preconditionerParams->d_pOperator = AMP::Operator::OperatorFactory::create( pc_params );
    auto preconditioner = std::make_shared<AMP::Solver::TrilinosMLSolver>( preconditionerParams );

    // Register the preconditioner with the Jacobian free Krylov solver
    auto linearSolver = nonlinearSolver->getKrylovSolver();
    linearSolver->setNestedSolver( preconditioner );

    // Get initial residual
    nlinBVPOp->residual( rhsVec, solVec, resVec );
    AMP::pout << "Initial Residual Norm: " << resVec->L2Norm() << std::endl;

    // Run solver
    nonlinearSolver->setZeroInitialGuess( false );
    nonlinearSolver->apply( rhsVec, solVec );

    // Get final residual
    nlinBVPOp->residual( rhsVec, solVec, resVec );
    std::cout << "Final Residual Norm: " << resVec->L2Norm() << std::endl;

    // Final communication
    solVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    resVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // Output Mathematica form (requires serial execution)
    for ( int i = 0; i < globalComm.getSize(); i++ ) {
        if ( globalComm.getRank() == i ) {
            auto filename = "data_" + exeName;
            int rank      = globalComm.getRank();
            int nranks    = globalComm.getSize();
            auto omode    = std::ios_base::out;
            if ( rank > 0 )
                omode |= std::ios_base::app;
            std::ofstream file( filename.c_str(), omode );
            if ( rank == 0 ) {
                file << "(* x y z solution solution fe-source fe-operator error *)" << std::endl;
                file << "results={" << std::endl;
            }

            auto iterator   = mesh->getIterator( Vertex, 0 );
            size_t numNodes = iterator.size();
            size_t iNode    = 0;
            double l2err    = 0.;
            for ( ; iterator != iterator.end(); ++iterator ) {
                double x, y, z;
                auto coord = iterator->coord();
                x          = coord[0];
                y          = coord[1];
                z          = coord[2];
                DOF->getDOFs( iterator->globalID(), gid );
                double res = resVec->getValueByGlobalID( gid[0] );
                double sol = solVec->getValueByGlobalID( gid[0] );
                double src = srcVec->getValueByGlobalID( gid[0] );
                double err = res / ( src + .5 * res + std::numeric_limits<double>::epsilon() );
                auto poly  = mfgSolution->evaluate( x, y, z );
                double val = poly[0];
                workVec->setValuesByGlobalID( 1, &gid[0], &err );

                file << "{" << x << "," << y << "," << z << "," << val << "," << sol << "," << src
                     << "," << res + src << "," << err << "}";
                if ( iNode < numNodes - 1 )
                    file << "," << std::endl;

                l2err += ( res * res );
                iNode++;
            }

            if ( rank == nranks - 1 ) {
                file << "};" << std::endl;
                file << "nodes = " << numNodes << "; l2err = " << l2err << ";" << std::endl;
            }

            file.close();
        }
        globalComm.barrier();
    }

    ut->passes( exeName );
    std::cout.flush();
}

int testDiffusionMMSInverse( int argc, char *argv[] )
{

    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    if ( argc > 1 ) {
        for ( int i = 1; i < argc; ++i )
            inverseTest1( &ut, argv[i] );
    } else {
        inverseTest1( &ut, "inputDiffusionMMSInverse" );
    }

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
