#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/MultiMesh.h"
#include "AMP/mesh/libmesh/initializeLibMesh.h"
#include "AMP/mesh/libmesh/libmeshMesh.h"
#include "AMP/mesh/testHelpers/meshWriters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/trilinos/ml/TrilinosMatrixShellOperator.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/Vector.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/trilinos/epetra/EpetraVector.h"
#include "libmesh/mesh_communication.h"

DISABLE_WARNINGS
#include "Epetra_CrsMatrix.h"
#include "Trilinos_version.h"
#include "ml_MultiLevelPreconditioner.h"
#include "ml_include.h"
#include <Trilinos_version.h>
ENABLE_WARNINGS

#include <iostream>
#include <string>


void myGetRow2( void *object, int row, std::vector<size_t> &cols, std::vector<double> &values )
{
    auto *op = reinterpret_cast<AMP::Operator::ColumnOperator *>( object );
    auto mat = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op->getOperator( 0 ) )
                   ->getMatrix();
    mat->getRowByGlobalID( row, cols, values );
}

void myGetRow3( void *object, int row, std::vector<size_t> &cols, std::vector<double> &values )
{
    auto *op      = reinterpret_cast<AMP::Operator::ColumnOperator *>( object );
    auto firstMat = std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op->getOperator( 0 ) )
                        ->getMatrix();
    auto secondMat =
        std::dynamic_pointer_cast<AMP::Operator::LinearOperator>( op->getOperator( 1 ) )
            ->getMatrix();
    size_t firstMatNumGlobalRows    = firstMat->numGlobalRows();
    size_t firstMatNumGlobalColumns = firstMat->numGlobalColumns();
    if ( row < (int) firstMatNumGlobalRows ) {
        firstMat->getRowByGlobalID( row, cols, values );
    } else {
        secondMat->getRowByGlobalID( row - firstMatNumGlobalRows, cols, values );
        for ( auto &col : cols ) {
            col += firstMatNumGlobalColumns;
        } // end for j
    }     // end if
}

int myMatVec( ML_Operator *data, int in_length, double in[], int out_length, double out[] )
{

    auto *op = reinterpret_cast<AMP::Operator::LinearOperator *>( ML_Get_MyMatvecData( data ) );
    auto mat = op->getMatrix();

    auto inVec  = mat->createInputVector();
    auto outVec = mat->createOutputVector();

    AMP_ASSERT( in_length == (int) inVec->getLocalSize() );
    AMP_ASSERT( out_length == (int) outVec->getLocalSize() );

    inVec->putRawData( in );

    mat->mult( inVec, outVec );

    outVec->getRawData( out );

    return 0;
}


int myGetRow( ML_Operator *data,
              int N_requested_rows,
              int requested_rows[],
              int allocated_space,
              int columns[],
              double values[],
              int row_lengths[] )
{

    auto *op = reinterpret_cast<AMP::Operator::LinearOperator *>( ML_Get_MyGetrowData( data ) );
    auto mat = op->getMatrix();

    int spaceRequired = 0;
    int cnt           = 0;
    for ( int i = 0; i < N_requested_rows; i++ ) {
        int row = requested_rows[i];
        std::vector<size_t> cols;
        std::vector<double> vals;

        mat->getRowByGlobalID( row, cols, vals );
        spaceRequired += cols.size();

        if ( allocated_space >= spaceRequired ) {
            for ( size_t j = 0; j < cols.size(); j++ ) {
                columns[cnt] = cols[j];
                values[cnt]  = vals[j];
                cnt++;
            }
            row_lengths[i] = cols.size();
        } else {
            return 0;
        }
    }

    return 1;
}


void myTest( AMP::UnitTest *ut, const std::string &exeName, int type )
{
    std::string input_file = "input_" + exeName;
    auto log_file          = "output_" + exeName + "_" + std::to_string( type );

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );

    auto mesh_file = input_db->getString( "mesh_file" );
    auto fusedMeshAdapter =
        AMP::Mesh::MeshWriters::readTestMeshLibMesh( mesh_file, AMP_COMM_WORLD );
    auto fusedOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            fusedMeshAdapter, "BVPOperator", input_db ) );

    auto fusedVar = fusedOperator->getOutputVariable();

    auto loadOperator = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator(
            fusedMeshAdapter, "LoadOperator", input_db ) );
    loadOperator->setVariable( fusedVar );

    auto NodalVectorDOF = AMP::Discretization::simpleDOFManager::create(
        fusedMeshAdapter, AMP::Mesh::GeomType::Vertex, 1, 3 );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto fusedSolVec = AMP::LinearAlgebra::createVector( NodalVectorDOF, fusedVar );
    auto fusedRhsVec = fusedSolVec->clone();
    auto fusedResVec = fusedSolVec->clone();

    fusedRhsVec->zero();
    loadOperator->apply( nullVec, fusedRhsVec );

    auto mlSolver_db = input_db->getDatabase( "MLoptions" );

    std::cout << std::endl;

    const size_t localSize = fusedSolVec->getLocalSize();

    // Matrix-based
    if ( type == 0 ) {
        ML_set_random_seed( 123456 );
        std::cout << "Matrix-Based ML: " << std::endl;
        fusedSolVec->zero();

        auto mlSolverParams =
            std::make_shared<AMP::Solver::TrilinosMLSolverParameters>( mlSolver_db );
        auto mlSolver = std::make_shared<AMP::Solver::TrilinosMLSolver>( mlSolverParams );

        auto mat = fusedOperator->getMatrix();

        auto matCopy = mat->clone();
        matCopy->zero();
        matCopy->axpy( 1.0, mat );

        mat->zero();
        mlSolver->registerOperator( fusedOperator );

        mat->axpy( 1.0, matCopy );

        mlSolver->apply( fusedRhsVec, fusedSolVec );
        std::cout << std::endl;
    }

    // Matrix-Free-1
    if ( type == 1 ) {
        ML_set_random_seed( 123456 );
        std::cout << "Matrix-Free ML Type-1: " << std::endl;
        fusedSolVec->zero();

        ML_Comm *MLcomm;
        ML_Comm_Create( &MLcomm );
        ML_Comm_Set_UsrComm( MLcomm, globalComm.getCommunicator() );

        ML_Operator *ml_op = ML_Operator_Create( MLcomm );
        ML_Operator_Set_ApplyFuncData(
            ml_op, localSize, localSize, fusedOperator.get(), localSize, myMatVec, 0 );
        ML_Operator_Set_Getrow( ml_op, localSize, myGetRow );

        Teuchos::ParameterList paramsList;
        ML_Epetra::SetDefaults( "SA", paramsList );
        paramsList.set( "ML output", mlSolver_db->getScalar<int>( "print_info_level" ) );
        paramsList.set( "PDE equations", mlSolver_db->getScalar<int>( "PDE_equations" ) );
        paramsList.set( "cycle applications", mlSolver_db->getScalar<int>( "max_iterations" ) );
        paramsList.set( "max levels", mlSolver_db->getScalar<int>( "max_levels" ) );

        auto mlSolver = std::make_shared<ML_Epetra::MultiLevelPreconditioner>( ml_op, paramsList );

        const ML_Aggregate *agg_obj = mlSolver->GetML_Aggregate();
        ML_Aggregate_Print( const_cast<ML_Aggregate *>( agg_obj ) );

        auto f_epetra = AMP::LinearAlgebra::EpetraVector::view( fusedRhsVec );
        auto u_epetra = AMP::LinearAlgebra::EpetraVector::view( fusedSolVec );

        Epetra_Vector &fVec = f_epetra->getEpetra_Vector();
        Epetra_Vector &uVec = u_epetra->getEpetra_Vector();

        fusedOperator->residual( fusedRhsVec, fusedSolVec, fusedResVec );
        std::cout << "MatFree-1: L2 norm of residual before solve " << std::setprecision( 15 )
                  << fusedResVec->L2Norm() << std::endl;

        mlSolver->ApplyInverse( fVec, uVec );

        fusedSolVec->getVectorData()->fireDataChange();

        double solution_norm = static_cast<double>( fusedSolVec->L2Norm() );
        std::cout << "MatFree-1:  solution norm: " << std::setprecision( 15 ) << solution_norm
                  << std::endl;

        fusedOperator->residual( fusedRhsVec, fusedSolVec, fusedResVec );
        std::cout << "MatFree-1: L2 norm of residual after solve " << std::setprecision( 15 )
                  << fusedResVec->L2Norm() << std::endl;

        ML_Operator_Destroy( &ml_op );

        ML_Comm_Destroy( &MLcomm );
    }

    // Matrix-free-2
    if ( type == 2 ) {
        ML_set_random_seed( 123456 );
        std::cout << "Matrix-Free ML Type-2: " << std::endl;
        fusedSolVec->zero();

        int numGrids = mlSolver_db->getScalar<int>( "max_levels" );
        int numPDEs  = mlSolver_db->getScalar<int>( "PDE_equations" );

        ML *ml_object;
        ML_Create( &ml_object, numGrids );

        ML_Init_Amatrix( ml_object, 0, localSize, localSize, fusedOperator.get() );
        ML_Set_Amatrix_Getrow( ml_object, 0, &myGetRow, nullptr, localSize );
        ML_Set_Amatrix_Matvec( ml_object, 0, &myMatVec );
        ML_Set_MaxIterations( ml_object, 1 + mlSolver_db->getScalar<int>( "max_iterations" ) );
        ML_Set_ResidualOutputFrequency( ml_object, 1 );
        ML_Set_PrintLevel( mlSolver_db->getScalar<int>( "print_info_level" ) );
        ML_Set_OutputLevel( ml_object, mlSolver_db->getScalar<int>( "print_info_level" ) );

        ML_Aggregate *agg_object;
        ML_Aggregate_Create( &agg_object );
        agg_object->num_PDE_eqns  = numPDEs;
        agg_object->nullspace_dim = numPDEs;
        ML_Aggregate_Set_MaxCoarseSize( agg_object, 128 );
        ML_Aggregate_Set_CoarsenScheme_UncoupledMIS( agg_object );

        int nlevels =
            ML_Gen_MGHierarchy_UsingAggregation( ml_object, 0, ML_INCREASING, agg_object );
        std::cout << "Number of actual levels : " << nlevels << std::endl;

        for ( int lev = 0; lev < ( nlevels - 1 ); lev++ ) {
            ML_Gen_Smoother_SymGaussSeidel( ml_object, lev, ML_BOTH, 2, 1.0 );
        }

#if TRILINOS_MAJOR_MINOR_VERSION >= 130000
        ML_Gen_Smoother_Amesos( ml_object, ( nlevels - 1 ), ML_AMESOS_KLU, -1, 0.0, 1 );
#else
        ML_Gen_Smoother_Amesos( ml_object, ( nlevels - 1 ), ML_AMESOS_KLU, -1, 0.0 );
#endif

        ML_Gen_Solver( ml_object, ML_MGV, 0, ( nlevels - 1 ) );


        fusedOperator->residual( fusedRhsVec, fusedSolVec, fusedResVec );
        std::cout << "MatFree-2: L2 norm of residual before solve " << std::setprecision( 15 )
                  << fusedResVec->L2Norm() << std::endl;

        auto solArr = new double[fusedSolVec->getLocalSize()];
        auto rhsArr = new double[fusedRhsVec->getLocalSize()];
        fusedSolVec->getRawData( solArr );
        fusedRhsVec->getRawData( rhsArr );
        ML_Iterate( ml_object, solArr, rhsArr );
        fusedSolVec->putRawData( solArr );
        fusedRhsVec->putRawData( rhsArr );
        delete[] solArr;
        solArr = nullptr;
        delete[] rhsArr;
        rhsArr = nullptr;

        fusedSolVec->getVectorData()->fireDataChange();

        double solution_norm = static_cast<double>( fusedSolVec->L2Norm() );
        std::cout << "MatFree-2:  solution norm: " << std::setprecision( 15 ) << solution_norm
                  << std::endl;

        fusedOperator->residual( fusedRhsVec, fusedSolVec, fusedResVec );
        std::cout << "MatFree-2: L2 norm of residual after solve " << std::setprecision( 15 )
                  << fusedResVec->L2Norm() << std::endl;

        ML_Aggregate_Destroy( &agg_object );
        ML_Destroy( &ml_object );
        std::cout << std::endl;
    }

    // matrix-free-3 using TrilinosMatrixShellOperator and customized getRow()
    if ( type == 3 ) {
        auto columnOperator = std::make_shared<AMP::Operator::ColumnOperator>();
        columnOperator->append( fusedOperator );

        auto matrixShellDatabase = std::make_shared<AMP::Database>( "MatrixShellOperator" );
        matrixShellDatabase->putScalar( "name", "MatShellOperator" );
        matrixShellDatabase->putScalar( "print_info_level", 1 );
        auto matrixShellParams =
            std::make_shared<AMP::Operator::OperatorParameters>( matrixShellDatabase );
        auto trilinosMatrixShellOperator =
            std::make_shared<AMP::Operator::TrilinosMatrixShellOperator>( matrixShellParams );
        trilinosMatrixShellOperator->setNodalDofMap( NodalVectorDOF );
        trilinosMatrixShellOperator->setGetRow( &myGetRow2 );
        // trilinosMatrixShellOperator->setOperator(fusedOperator);
        trilinosMatrixShellOperator->setOperator( columnOperator );

        mlSolver_db->putScalar( "USE_EPETRA", false );
        auto mlSolverParams =
            std::make_shared<AMP::Solver::TrilinosMLSolverParameters>( mlSolver_db );
        mlSolverParams->d_pOperator = trilinosMatrixShellOperator;
        auto mlSolver = std::make_shared<AMP::Solver::TrilinosMLSolver>( mlSolverParams );

        std::cout << "MatFree-3: L2 norm of residual before solve " << std::setprecision( 15 )
                  << fusedResVec->L2Norm() << std::endl;

        mlSolver->apply( fusedRhsVec, fusedSolVec );

        std::cout << "MatFree-3:  solution norm: " << std::setprecision( 15 )
                  << fusedSolVec->L2Norm() << std::endl;
        fusedOperator->residual( fusedRhsVec, fusedSolVec, fusedResVec );
        std::cout << "MatFree-3: L2 norm of residual after solve " << std::setprecision( 15 )
                  << fusedResVec->L2Norm() << std::endl;

        std::cout << std::endl;
    }

    ut->passes( exeName );
}

void myTest2( AMP::UnitTest *ut, const std::string &exeName, bool useTwoMeshes )
{
    std::string input_file = "input_" + exeName;
    int type               = 4;
    if ( useTwoMeshes )
        type = 5;
    auto log_file = "output" + exeName + "_" + std::to_string( type );

    AMP::logOnlyNodeZero( log_file );
    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );

    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );
    std::string mesh_file = input_db->getString( "mesh_file" );

    auto firstMesh  = AMP::Mesh::MeshWriters::readTestMeshLibMesh( mesh_file, AMP_COMM_WORLD );
    auto secondMesh = AMP::Mesh::MeshWriters::readTestMeshLibMesh( mesh_file, AMP_COMM_WORLD );
    firstMesh->setName( "Mesh_1" );
    firstMesh->setName( "Mesh_2" );

    std::vector<std::shared_ptr<AMP::Mesh::Mesh>> vectorOfMeshes;
    vectorOfMeshes.push_back( firstMesh );
    if ( useTwoMeshes )
        vectorOfMeshes.push_back( secondMesh );

    auto fusedMeshes =
        std::make_shared<AMP::Mesh::MultiMesh>( "MultiMesh", globalComm, vectorOfMeshes );

    auto firstFusedOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            fusedMeshes->getMeshes()[0], "BVPOperator", input_db ) );
    auto firstFusedVar     = firstFusedOperator->getOutputVariable();
    auto firstLoadOperator = std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
        AMP::Operator::OperatorBuilder::createOperator(
            fusedMeshes->getMeshes()[0], "LoadOperator", input_db ) );
    firstLoadOperator->setVariable( firstFusedVar );

    auto fusedColumnOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    fusedColumnOperator->append( firstFusedOperator );

    auto loadColumnOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    loadColumnOperator->append( firstLoadOperator );
    if ( useTwoMeshes ) {
        auto secondFusedOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
            AMP::Operator::OperatorBuilder::createOperator(
                fusedMeshes->getMeshes()[1], "BVPOperator", input_db ) );
        auto secondFusedVar = secondFusedOperator->getOutputVariable();
        auto secondLoadOperator =
            std::dynamic_pointer_cast<AMP::Operator::DirichletVectorCorrection>(
                AMP::Operator::OperatorBuilder::createOperator(
                    fusedMeshes->getMeshes()[1], "LoadOperator", input_db ) );
        secondLoadOperator->setVariable( secondFusedVar );
        fusedColumnOperator->append( secondFusedOperator );
        loadColumnOperator->append( secondLoadOperator );
    } // end if


    auto dofManager = AMP::Discretization::simpleDOFManager::create(
        fusedMeshes, AMP::Mesh::GeomType::Vertex, 1, 3 );

    auto fusedColumnVar = fusedColumnOperator->getOutputVariable();
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto fusedColumnSolVec = AMP::LinearAlgebra::createVector( dofManager, fusedColumnVar );
    auto fusedColumnRhsVec = fusedColumnSolVec->clone();
    auto fusedColumnResVec = fusedColumnSolVec->clone();

    fusedColumnRhsVec->zero();
    loadColumnOperator->apply( nullVec, fusedColumnRhsVec );

    std::cout << std::endl;

    auto matrixShellDatabase = std::make_shared<AMP::Database>( "MatrixShellOperator" );
    matrixShellDatabase->putScalar( "name", "MatShellOperator" );
    matrixShellDatabase->putScalar( "print_info_level", 1 );
    auto matrixShellParams =
        std::make_shared<AMP::Operator::OperatorParameters>( matrixShellDatabase );
    auto trilinosMatrixShellOperator =
        std::make_shared<AMP::Operator::TrilinosMatrixShellOperator>( matrixShellParams );
    trilinosMatrixShellOperator->setNodalDofMap( dofManager );
    if ( !useTwoMeshes ) {
        trilinosMatrixShellOperator->setGetRow( &myGetRow2 );
    } else {
        trilinosMatrixShellOperator->setGetRow( &myGetRow3 );
    } // end if
    trilinosMatrixShellOperator->setOperator( fusedColumnOperator );


    auto mlSolver_db = input_db->getDatabase( "MLoptions" );
    mlSolver_db->putScalar( "USE_EPETRA", false );
    auto mlSolverParams = std::make_shared<AMP::Solver::TrilinosMLSolverParameters>( mlSolver_db );
    mlSolverParams->d_pOperator = trilinosMatrixShellOperator;
    auto mlSolver               = std::make_shared<AMP::Solver::TrilinosMLSolver>( mlSolverParams );

    std::cout << "MatFree-4: L2 norm of residual before solve " << std::setprecision( 15 )
              << fusedColumnResVec->L2Norm() << std::endl;

    mlSolver->apply( fusedColumnRhsVec, fusedColumnSolVec );

    std::cout << "MatFree-4:  solution norm: " << std::setprecision( 15 )
              << fusedColumnSolVec->L2Norm() << std::endl;
    fusedColumnOperator->residual( fusedColumnRhsVec, fusedColumnSolVec, fusedColumnResVec );
    std::cout << "MatFree-4: L2 norm of residual after solve " << std::setprecision( 15 )
              << fusedColumnResVec->L2Norm() << std::endl;

    std::cout << std::endl;

    ut->passes( exeName );
}


void loopMyTest( AMP::UnitTest *ut, const std::string &exeName )
{
    myTest2( ut, exeName, false );
    myTest2( ut, exeName, true );
    for ( int type = 0; type < 4; type++ ) {
        myTest( ut, exeName, type );
    }
}


int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;

    std::vector<std::string> exeNames;

    if ( argc == 1 ) {
        exeNames.emplace_back( "testMatrixFreeML-1" );
    } else {
        for ( int i = 1; i < argc; i++ ) {
            auto inpName = AMP::Utilities::stringf( "testMatrixFreeML-%s", argv[i] );
            exeNames.emplace_back( inpName );
        } // end for i
    }

    for ( auto &exeName : exeNames )
        loopMyTest( &ut, exeName );

    ut.report();
    ut.reset();
    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}
