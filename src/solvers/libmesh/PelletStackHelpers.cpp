#include "AMP/solvers/libmesh/PelletStackHelpers.h"

#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/matrices/MatrixBuilder.h"
#include "AMP/operators/CoupledOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/DirichletVectorCorrection.h"
#include "AMP/operators/libmesh/PelletStackOperator.h"
#include "AMP/operators/map/AsyncMapColumnOperator.h"
#include "AMP/operators/map/NodeToNodeMap.h"
#include "AMP/operators/mechanics/MechanicsNonlinearFEOperator.h"
#include "AMP/vectors/VectorBuilder.h"

#ifdef AMP_USE_PETSC
    #include "AMP/solvers/petsc/PetscKrylovSolver.h"
#endif

#ifdef AMP_USE_TRILINOS
    #include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#endif


namespace AMP::Operator::PelletMechanics {


std::shared_ptr<AsyncMapColumnOperator> createMaps( std::shared_ptr<AMP::Mesh::Mesh> manager,
                                                    std::shared_ptr<AMP::Database> global_input_db )
{
    std::shared_ptr<AsyncMapColumnOperator> n2nmaps;
    if ( global_input_db->keyExists( "MechanicsNodeToNodeMaps" ) ) {
        auto map_db = global_input_db->getDatabase( "MechanicsNodeToNodeMaps" );
        n2nmaps     = AsyncMapColumnOperator::build<NodeToNodeMap>( manager, map_db );
    }
    return n2nmaps;
}
std::shared_ptr<PelletStackOperator>
createStackOperator( std::shared_ptr<AMP::Mesh::Mesh> manager,
                     std::shared_ptr<AsyncMapColumnOperator> n2nmaps,
                     std::shared_ptr<AMP::Database> global_input_db )
{
    auto pelletStackOp_db = global_input_db->getDatabase( "PelletStackOperator" );
    auto pelletMeshes     = manager->Subset( "PelletMeshes" );
    auto pelletMeshIDs    = pelletMeshes->getBaseMeshIDs();
    int totalNumPellets   = pelletMeshIDs.size();
    pelletStackOp_db->putScalar( "TOTAL_NUMBER_OF_PELLETS", totalNumPellets );
    auto pelletStackOpParams = std::make_shared<PelletStackOperatorParameters>( pelletStackOp_db );
    pelletStackOpParams->d_pelletStackComm = pelletMeshes->getComm();
    pelletStackOpParams->d_n2nMaps         = n2nmaps;
    pelletStackOpParams->d_Mesh            = pelletMeshes;
    return std::make_shared<PelletStackOperator>( pelletStackOpParams );
}

std::shared_ptr<ColumnOperator>
createNonlinearColumnOperator( std::shared_ptr<AMP::Operator::PelletStackOperator> pelletStackOp,
                               std::shared_ptr<AMP::Database> global_input_db )
{
    auto localPelletIds          = pelletStackOp->getLocalPelletIds();
    auto localMeshes             = pelletStackOp->getLocalMeshes();
    auto nonlinearColumnOperator = std::make_shared<ColumnOperator>();
    for ( unsigned int id = 0; id < localPelletIds.size(); id++ ) {
        std::string prefix = "";
        if ( localPelletIds[id] == 0 )
            prefix = "Bottom";
        auto mesh = localMeshes[id];
        auto nonlinOperator =
            std::dynamic_pointer_cast<NonlinearBVPOperator>( OperatorBuilder::createOperator(
                mesh, prefix + "PelletMechanicsNonlinearBVPOperator", global_input_db ) );
        nonlinearColumnOperator->append( nonlinOperator );
    } // end for id
    return nonlinearColumnOperator;
}


std::shared_ptr<AMP::Operator::CoupledOperator>
createCoupledOperator( std::shared_ptr<AMP::Operator::AsyncMapColumnOperator> n2nmaps,
                       std::shared_ptr<AMP::Operator::ColumnOperator> nonlinearColumnOperator )
{
    std::shared_ptr<AMP::Database> emptyDb( new AMP::Database );
    auto coupledOpParams = std::make_shared<AMP::Operator::CoupledOperatorParameters>( emptyDb );
    coupledOpParams->d_MapOperator = n2nmaps;
    coupledOpParams->d_BVPOperator = nonlinearColumnOperator;
    return std::make_shared<AMP::Operator::CoupledOperator>( coupledOpParams );
}


std::shared_ptr<ColumnOperator>
createLinearColumnOperator( std::shared_ptr<ColumnOperator> nonlinearColumnOperator )
{

    return std::make_shared<ColumnOperator>(
        nonlinearColumnOperator->getParameters( "Jacobian", nullptr ) );
}


void setFrozenVectorForMaps( std::shared_ptr<AMP::Mesh::Mesh> manager,
                             std::shared_ptr<CoupledOperator> coupledOp )
{
    auto n2nmaps = std::dynamic_pointer_cast<AsyncMapColumnOperator>( coupledOp->getOperator( 2 ) );
    auto nonlinearColumnOperator =
        std::dynamic_pointer_cast<ColumnOperator>( coupledOp->getOperator( 3 ) );
    auto dispVar         = nonlinearColumnOperator->getOutputVariable();
    auto nodal3VectorDOF = AMP::Discretization::simpleDOFManager::create(
        manager, AMP::Mesh::GeomType::Vertex, 1, 3, true );
    auto dirichletValues = AMP::LinearAlgebra::createVector( nodal3VectorDOF, dispVar, true );
    if ( n2nmaps ) {
        n2nmaps->setVector( dirichletValues );
    }
    for ( unsigned int id = 0; id < nonlinearColumnOperator->getNumberOfOperators(); id++ ) {
        auto dirichletOp = std::dynamic_pointer_cast<DirichletVectorCorrection>(
            std::dynamic_pointer_cast<NonlinearBVPOperator>(
                nonlinearColumnOperator->getOperator( id ) )
                ->getBoundaryOperator() );
        dirichletOp->setDirichletValues( dirichletValues );
    } // end for id
}

void createVectors( std::shared_ptr<AMP::Mesh::Mesh> manager,
                    std::shared_ptr<CoupledOperator> coupledOp,
                    AMP::LinearAlgebra::Vector::shared_ptr &solVec,
                    AMP::LinearAlgebra::Vector::shared_ptr &rhsVec,
                    AMP::LinearAlgebra::Vector::shared_ptr &scaledRhsVec )
{
    auto nonlinearColumnOperator =
        std::dynamic_pointer_cast<ColumnOperator>( coupledOp->getOperator( 3 ) );
    auto dispVar         = nonlinearColumnOperator->getOutputVariable();
    auto nodal3VectorDOF = AMP::Discretization::simpleDOFManager::create(
        manager, AMP::Mesh::GeomType::Vertex, 1, 3, true );
    solVec       = AMP::LinearAlgebra::createVector( nodal3VectorDOF, dispVar, true );
    rhsVec       = AMP::LinearAlgebra::createVector( nodal3VectorDOF, dispVar, true );
    scaledRhsVec = rhsVec->clone();
}

void buildPointLoadRHS( std::shared_ptr<AMP::Database> global_input_db,
                        std::shared_ptr<CoupledOperator> coupledOp,
                        AMP::LinearAlgebra::Vector::shared_ptr rhsVec )
{
    AMP::LinearAlgebra::Vector::shared_ptr nullVec;
    auto nonlinearColumnOperator =
        std::dynamic_pointer_cast<ColumnOperator>( coupledOp->getOperator( 3 ) );
    rhsVec->zero();
    for ( unsigned int id = 0; id < nonlinearColumnOperator->getNumberOfOperators(); id++ ) {
        auto currOp = nonlinearColumnOperator->getOperator( id );
        auto mesh   = currOp->getMesh();
        auto loadOp = std::dynamic_pointer_cast<DirichletVectorCorrection>(
            OperatorBuilder::createOperator( mesh, "PointLoad", global_input_db ) );
        loadOp->setVariable( currOp->getOutputVariable() );
        loadOp->apply( nullVec, rhsVec );
    } // end for id
}

void applyBoundaryCorrections( std::shared_ptr<CoupledOperator> coupledOp,
                               AMP::LinearAlgebra::Vector::shared_ptr solVec,
                               AMP::LinearAlgebra::Vector::shared_ptr rhsVec )
{
    auto nonlinearColumnOperator =
        std::dynamic_pointer_cast<ColumnOperator>( coupledOp->getOperator( 3 ) );
    for ( unsigned int id = 0; id < nonlinearColumnOperator->getNumberOfOperators(); id++ ) {
        auto nonlinOperator = std::dynamic_pointer_cast<NonlinearBVPOperator>(
            nonlinearColumnOperator->getOperator( id ) );
        nonlinOperator->modifyInitialSolutionVector( solVec );
        nonlinOperator->modifyRHSvector( rhsVec );
    } // end for id
}

void createTemperatureVectors( std::shared_ptr<AMP::Mesh::Mesh> manager,
                               AMP::LinearAlgebra::Vector::shared_ptr &initialTemperatureVec,
                               AMP::LinearAlgebra::Vector::shared_ptr &finalTemperatureVec )
{
    auto tempVar        = std::make_shared<AMP::LinearAlgebra::Variable>( "temperature" );
    auto nodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
        manager, AMP::Mesh::GeomType::Vertex, 1, 1, true );
    initialTemperatureVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, tempVar, true );
    finalTemperatureVec   = initialTemperatureVec->clone();
}

void setReferenceTemperature( std::shared_ptr<CoupledOperator> coupledOp,
                              AMP::LinearAlgebra::Vector::shared_ptr initialTemperatureVec )
{
    auto nonlinearColumnOperator =
        std::dynamic_pointer_cast<ColumnOperator>( coupledOp->getOperator( 3 ) );
    for ( unsigned int id = 0; id < nonlinearColumnOperator->getNumberOfOperators(); id++ ) {
        auto bvpOp = std::dynamic_pointer_cast<NonlinearBVPOperator>(
            nonlinearColumnOperator->getOperator( id ) );
        auto mechOp =
            std::dynamic_pointer_cast<MechanicsNonlinearFEOperator>( bvpOp->getVolumeOperator() );
        mechOp->setReferenceTemperature( initialTemperatureVec );
    } // end for id
}

void setFinalTemperature( std::shared_ptr<CoupledOperator> coupledOp,
                          AMP::LinearAlgebra::Vector::shared_ptr finalTemperatureVec )
{
    auto nonlinearColumnOperator =
        std::dynamic_pointer_cast<ColumnOperator>( coupledOp->getOperator( 3 ) );
    for ( unsigned int id = 0; id < nonlinearColumnOperator->getNumberOfOperators(); id++ ) {
        auto bvpOp = std::dynamic_pointer_cast<NonlinearBVPOperator>(
            nonlinearColumnOperator->getOperator( id ) );
        auto mechOp =
            std::dynamic_pointer_cast<MechanicsNonlinearFEOperator>( bvpOp->getVolumeOperator() );
        mechOp->setVector( Mechanics::TEMPERATURE, finalTemperatureVec );
    } // end for id
}

void buildColumnSolver( std::shared_ptr<AMP::Database> columnSolver_db,
                        std::shared_ptr<ColumnOperator> linearColumnOperator,
                        std::shared_ptr<AMP::Solver::ColumnSolver> &columnSolver )
{
    auto ikspSolver_db = columnSolver_db->getDatabase( "KrylovSolver" );
    auto mlSolver_db   = ikspSolver_db->getDatabase( "MLSolver" );
    auto columnSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( columnSolver_db );
    columnSolverParams->d_pOperator = linearColumnOperator;
    columnSolver.reset( new AMP::Solver::ColumnSolver( columnSolverParams ) );
    for ( unsigned int id = 0; id < linearColumnOperator->getNumberOfOperators(); id++ ) {
        auto currOp = linearColumnOperator->getOperator( id );
        auto mlSolverParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( mlSolver_db );
        mlSolverParams->d_pOperator = currOp;
#ifdef AMP_USE_TRILINOS
        auto mlSolver = std::make_shared<AMP::Solver::TrilinosMLSolver>( mlSolverParams );
#else
        AMP_ERROR( "helperBuildColumnSolver: trilinos required" );
#endif

#ifdef AMP_USE_PETSC
        auto ikspSolverParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( ikspSolver_db );
        ikspSolverParams->d_pOperator = currOp;
        ikspSolverParams->d_comm      = ( currOp->getMesh() )->getComm();

    #ifdef AMP_USE_TRILINOS
        ikspSolverParams->d_pNestedSolver = mlSolver;
    #endif
        auto ikspSolver = std::make_shared<AMP::Solver::PetscKrylovSolver>( ikspSolverParams );
        columnSolver->append( ikspSolver );
#else
        AMP_ERROR( "petsc required" );
#endif

    } // end for id
}

std::shared_ptr<AMP::Solver::SolverStrategy>
buildStackSolver( std::shared_ptr<AMP::Database> pelletStackSolver_db,
                  std::shared_ptr<PelletStackOperator> pelletStackOp,
                  std::shared_ptr<ColumnOperator> linearColumnOperator )
{
    auto columnSolver_db = pelletStackSolver_db->getDatabase( "ColumnSolver" );
    std::shared_ptr<AMP::Solver::ColumnSolver> columnSolver;
    buildColumnSolver( columnSolver_db, linearColumnOperator, columnSolver );
    auto pelletStackSolverParams =
        std::make_shared<AMP::Solver::PelletStackMechanicsSolverParameters>( pelletStackSolver_db );
    pelletStackSolverParams->d_columnSolver = columnSolver;
    pelletStackSolverParams->d_pOperator    = pelletStackOp;
    return std::make_shared<AMP::Solver::PelletStackMechanicsSolver>( pelletStackSolverParams );
}

void resetNonlinearOperator( std::shared_ptr<CoupledOperator> coupledOp )
{
    auto nonlinearColumnOperator =
        std::dynamic_pointer_cast<ColumnOperator>( coupledOp->getOperator( 3 ) );
    auto tmp_db    = std::make_shared<AMP::Database>( "Dummy" );
    auto tmpParams = std::make_shared<MechanicsNonlinearFEOperatorParameters>( tmp_db );
    for ( unsigned int id = 0; id < nonlinearColumnOperator->getNumberOfOperators(); id++ ) {
        auto nonlinOperator = std::dynamic_pointer_cast<NonlinearBVPOperator>(
            nonlinearColumnOperator->getOperator( id ) );
        ( nonlinOperator->getVolumeOperator() )->reset( tmpParams );
    } // end for id
}


} // namespace AMP::Operator::PelletMechanics
