#include "AMP/IO/PIO.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/CustomConstraintsEliminationOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletMatrixCorrection.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/mechanics/IsotropicElasticModel.h"
#include "AMP/operators/mechanics/MechanicsLinearElement.h"
#include "AMP/operators/mechanics/MechanicsLinearFEOperator.h"
#include "AMP/operators/petsc/PetscMatrixShellOperator.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/ConstraintsEliminationSolver.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
// #include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/vectors/VectorBuilder.h"

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>


static void myTest( AMP::UnitTest *ut )
{
    std::string exeName( "testConstraintsEliminationOperator" );
    std::string input_file = "input_" + exeName;
    std::string log_file   = "output_" + exeName;

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    AMP_INSIST( input_db->keyExists( "Mesh" ), "Key ''Mesh'' is missing!" );
    auto mesh_db    = input_db->getDatabase( "Mesh" );
    auto meshParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshParams->setComm( globalComm );
    auto mesh = AMP::Mesh::MeshFactory::create( meshParams );

    int const dofsPerNode = 3;
    int const gostWidth   = 1;
    bool const split      = true;
    auto dofMap           = AMP::Discretization::simpleDOFManager::create(
        mesh, AMP::Mesh::GeomType::Vertex, gostWidth, dofsPerNode, split );

    AMP::LinearAlgebra::Vector::shared_ptr vec1;
    AMP::LinearAlgebra::Vector::shared_ptr vec2;

    for ( int dummy = 0; dummy < 2; ++dummy ) {
        auto colOp             = std::make_shared<AMP::Operator::ColumnOperator>();
        auto linearSolver_db   = input_db->getDatabase( "LinearSolver" );
        auto preconditioner_db = linearSolver_db->getDatabase( "Preconditioner" );
        auto preconditionerParams =
            std::make_shared<AMP::Solver::ColumnSolverParameters>( preconditioner_db );
        preconditionerParams->d_pOperator = colOp;
        auto colPre = std::make_shared<AMP::Solver::ColumnSolver>( preconditionerParams );

        auto bvpOp = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                mesh, ( dummy ? "DummyBVPOperator" : "BVPOperator" ), input_db ) );
        colOp->append( bvpOp );

        auto solver_db    = preconditioner_db->getDatabase( "Solver" );
        auto solverParams = std::make_shared<AMP::Solver::SolverStrategyParameters>( solver_db );
        solverParams->d_pOperator = bvpOp;
        solverParams->d_comm      = globalComm;
        auto solver = std::make_shared<AMP::Solver::PetscKrylovSolver>( solverParams );
        colPre->append( solver );

        std::shared_ptr<AMP::Operator::CustomConstraintsEliminationOperator> dirOp;
        if ( dummy ) {
            auto dummyOperator_db = std::make_shared<AMP::Database>( "DummyOperator" );
            dummyOperator_db->putScalar( "InputVariable", "displacement" );
            dummyOperator_db->putScalar( "OutputVariable", "displacement" );
            auto dummyOperatorParams =
                std::make_shared<AMP::Operator::OperatorParameters>( dummyOperator_db );
            dirOp = std::make_shared<AMP::Operator::CustomConstraintsEliminationOperator>(
                dummyOperatorParams );
            std::vector<size_t> slaveIndices;
            std::vector<double> slaveValues;
            int const boundaryID = 2;
            AMP::Mesh::MeshIterator it =
                mesh->getBoundaryIDIterator( AMP::Mesh::GeomType::Vertex, boundaryID );
            size_t const numDOFs = it.size() * dofsPerNode;
            slaveIndices.resize( numDOFs );
            slaveValues.resize( numDOFs );
            AMP::Mesh::MeshIterator it_begin = it.begin();
            AMP::Mesh::MeshIterator it_end   = it.end();
            std::vector<size_t> dofIndices;
            double dirichletValues[3] = { 2.0, 1.0, 0.0 };
            size_t p                  = 0;
            for ( it = it_begin; it != it_end; ++it ) {
                dofMap->getDOFs( it->globalID(), dofIndices );
                AMP_ASSERT( static_cast<int>( dofIndices.size() ) == dofsPerNode );
                std::copy( &( dofIndices[0] ),
                           &( dofIndices[0] ) + dofsPerNode,
                           &( slaveIndices[0] ) + p );
                std::copy( &( dirichletValues[0] ),
                           &( dirichletValues[0] ) + dofsPerNode,
                           &( slaveValues[0] ) + p );
                p += dofsPerNode;
            } // end for
            AMP_ASSERT( p == numDOFs );
            dirOp->initialize( slaveIndices, slaveValues );
            // colOp->append(dirOp);

            // auto dummySolver_db = preconditioner_db->getDatabase("ContactPreconditioner");
            auto dummySolver_db = std::make_shared<AMP::Database>( "DummySolver" );
            dummySolver_db->putScalar( "print_info_level", 1 );
            dummySolver_db->putScalar( "max_iterations", 1 );
            dummySolver_db->putScalar( "max_error", 1.0e-16 );
            auto dummySolverParams =
                std::make_shared<AMP::Solver::ConstraintsEliminationSolverParameters>(
                    dummySolver_db );
            dummySolverParams->d_pOperator = dirOp;
            auto dirSolver =
                std::make_shared<AMP::Solver::ConstraintsEliminationSolver>( dummySolverParams );
            colPre->append( dirSolver );
        } // end if

        auto loadOp = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
            AMP::Operator::OperatorBuilder::createOperator( mesh, "LoadOperator", input_db ) );
        auto var = bvpOp->getOutputVariable();
        loadOp->setVariable( var );

        auto shell_db = std::make_shared<AMP::Database>( "MatrixShellOperator" );
        shell_db->putScalar( "name", "MatShellOperator" );
        shell_db->putScalar( "print_info_level", 1 );
        auto shellParams = std::make_shared<AMP::Operator::OperatorParameters>( shell_db );
        auto shellOp     = std::make_shared<AMP::Operator::PetscMatrixShellOperator>( shellParams );
        int const numLocalNodes = mesh->numLocalElements( AMP::Mesh::GeomType::Vertex );
        int const matLocalSize  = dofsPerNode * numLocalNodes;
        AMP_ASSERT( matLocalSize == static_cast<int>( dofMap->numLocalDOF() ) );
        shellOp->setComm( globalComm );
        shellOp->setMatLocalRowSize( matLocalSize );
        shellOp->setMatLocalColumnSize( matLocalSize );
        shellOp->setOperator( colOp );

        AMP::LinearAlgebra::Vector::shared_ptr nullVec;
        auto solVec = AMP::LinearAlgebra::createVector( dofMap, var, split );
        auto rhsVec = solVec->clone();
        auto resVec = solVec->clone();
        auto dirVec = solVec->clone();
        auto corVec = solVec->clone();

        solVec->zero();
        rhsVec->zero();
        loadOp->apply( nullVec, rhsVec );
        bvpOp->modifyRHSvector( rhsVec );
        if ( dummy ) {
            dirVec->zero();
            dirOp->addShiftToSlave( dirVec );
            dirVec->makeConsistent();
            colOp->apply( dirVec, corVec );
            corVec->scale( -1.0 );
            colOp->append( dirOp );
            rhsVec->add( *rhsVec, *corVec );
            dirOp->addSlaveToMaster( rhsVec );
            dirOp->setSlaveToZero( rhsVec );
            dirOp->copyMasterToSlave( solVec );
        }
        solVec->makeConsistent();

        auto linearSolverParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( linearSolver_db );
        linearSolverParams->d_pOperator     = shellOp;
        linearSolverParams->d_comm          = globalComm;
        linearSolverParams->d_pNestedSolver = colPre;
        auto linearSolver = std::make_shared<AMP::Solver::PetscKrylovSolver>( linearSolverParams );
        linearSolver->setInitialGuess( solVec );

        linearSolver->apply( rhsVec, solVec );

        if ( dummy ) {
            dirOp->copyMasterToSlave( solVec );
            dirOp->addShiftToSlave( solVec );
        }

        if ( dummy ) {
            vec1 = solVec;
        } else {
            vec2 = solVec;
        }
    }

    vec1->subtract( *vec1, *vec2 );
    double solutionL2Norm = static_cast<double>( vec2->L2Norm() );
    double errorL2Norm    = static_cast<double>( vec1->L2Norm() );
    std::cout << "solution L2 norm = " << std::setprecision( 15 ) << solutionL2Norm << "\n";
    std::cout << "error L2 norm = " << std::setprecision( 15 ) << errorL2Norm << "\n";
    std::cout << "relative error = " << errorL2Norm / solutionL2Norm << "\n";
    AMP_ASSERT( errorL2Norm / solutionL2Norm < 1.0e-12 );

    ut->passes( exeName );
}

int testConstraintsEliminationOperator( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    myTest( &ut );

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
