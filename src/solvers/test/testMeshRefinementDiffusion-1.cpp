#include "AMP/IO/PIO.h"
#include "AMP/discretization/DOF_Manager.h"
#include "AMP/discretization/simpleDOF_Manager.h"
#include "AMP/mesh/Mesh.h"
#include "AMP/mesh/MeshFactory.h"
#include "AMP/mesh/MeshParameters.h"
#include "AMP/operators/BVPOperatorParameters.h"
#include "AMP/operators/ColumnOperator.h"
#include "AMP/operators/CoupledOperator.h"
#include "AMP/operators/LinearBVPOperator.h"
#include "AMP/operators/NonlinearBVPOperator.h"
#include "AMP/operators/OperatorBuilder.h"
#include "AMP/operators/boundary/ColumnBoundaryOperator.h"
#include "AMP/operators/boundary/libmesh/RobinVectorCorrection.h"
#include "AMP/operators/libmesh/VolumeIntegralOperator.h"
#include "AMP/operators/map/AsyncMapColumnOperator.h"
#include "AMP/operators/map/ScalarZAxisMap.h"
#include "AMP/solvers/ColumnSolver.h"
#include "AMP/solvers/SolverStrategyParameters.h"
#include "AMP/solvers/petsc/PetscKrylovSolver.h"
#include "AMP/solvers/petsc/PetscSNESSolver.h"
#include "AMP/solvers/trilinos/ml/TrilinosMLSolver.h"
#include "AMP/utils/AMPManager.h"
#include "AMP/utils/AMP_MPI.h"
#include "AMP/utils/Database.h"
#include "AMP/utils/UnitTest.h"
#include "AMP/utils/Utilities.h"
#include "AMP/vectors/VectorBuilder.h"
#include "AMP/vectors/VectorSelector.h"

// Libmesh headers
DISABLE_WARNINGS
#include "libmesh/auto_ptr.h"
#include "libmesh/cell_hex8.h"
#include "libmesh/elem.h"
#include "libmesh/enum_fe_family.h"
#include "libmesh/enum_order.h"
#include "libmesh/enum_quadrature_type.h"
#include "libmesh/fe_base.h"
#include "libmesh/fe_type.h"
#include "libmesh/node.h"
#include "libmesh/quadrature.h"
#include "libmesh/string_to_enum.h"
ENABLE_WARNINGS

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>


// Manufactured Solution
static inline double fun( double x, double y, double )
{
    return 800 +
           ( ( 0.00004 - 20 * std::pow( x, 2 ) - 20 * std::pow( y, 2 ) ) * std::pow( 10, 6 ) );
}
static inline double __dTdn__( double, double, double, double ) { return 0; }
static inline double __FsnK__() { return 80000000; }


void calculateManufacturedSolution(
    std::shared_ptr<AMP::Mesh::Mesh> mesh,
    AMP::LinearAlgebra::Vector::shared_ptr manufacturedSolution,
    AMP::LinearAlgebra::Vector::shared_ptr manufacturedNormalGradient )
{
    //  CALCULATE THE MANFACTURED SOLUTION
    auto dof_map       = manufacturedSolution->getDOFManager();
    auto bottomAdapter = mesh->Subset( "Bottom" );
    if ( bottomAdapter ) {
        auto el     = bottomAdapter->getIterator( AMP::Mesh::GeomType::Cell, 0 );
        auto end_el = el.end();

        for ( ; el != end_el; ++el ) {
            auto d_currNodes = el->getElements( AMP::Mesh::GeomType::Vertex );

            std::vector<AMP::Mesh::MeshElementID> globalIDs( d_currNodes.size() );

            std::vector<size_t> d_dofIndices;
            for ( unsigned int j = 0; j < d_currNodes.size(); j++ ) {
                globalIDs[j] = d_currNodes[j]->globalID();
            } // end of j
            dof_map->getDOFs( globalIDs, d_dofIndices );

            for ( unsigned int j = 0; j < d_currNodes.size(); j++ ) {
                auto pt     = d_currNodes[j]->coord();
                double val1 = fun( pt[0], pt[1], pt[2] );
                // double val2 = fun(pt[0],pt[1],pt[2]-20); // not used.
                double val3 = __dTdn__( pt[0], pt[1], pt[2], 1 );

                manufacturedSolution->setValuesByGlobalID( 1, &d_dofIndices[j], &val1 );
                manufacturedNormalGradient->setValuesByGlobalID( 1, &d_dofIndices[j], &val3 );
            } // end for node
        }
    }

    manufacturedSolution->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


void calculateSources( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                       std::shared_ptr<AMP::Discretization::DOFManager> gaussPointDOF,
                       std::shared_ptr<AMP::LinearAlgebra::Vector> manufacturedRHS )
{
    // Compute the source on the gauss point

    auto el     = mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    auto end_el = el.end();

    auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
    auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );

    auto d_feType = std::make_shared<libMesh::FEType>( feTypeOrder, feFamily );
    std::shared_ptr<libMesh::FEBase> d_fe(
        ( libMesh::FEBase::build( 3, ( *d_feType ) ) ).release() );

    auto qruleOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "SECOND" );
    std::shared_ptr<libMesh::QBase> d_qrule(
        ( libMesh::QBase::build( "QGAUSS", 3, qruleOrder ) ).release() );

    d_fe->attach_quadrature_rule( d_qrule.get() );

    for ( ; el != end_el; ++el ) {

        std::vector<size_t> d_gaussPtIndices;
        gaussPointDOF->getDOFs( el->globalID(), d_gaussPtIndices );

        for ( auto &d_gaussPtIndice : d_gaussPtIndices ) {
            double manufacturedAtGauss1;
            manufacturedAtGauss1 = __FsnK__();

            manufacturedRHS->setValuesByGlobalID( 1, &d_gaussPtIndice, &manufacturedAtGauss1 );
        }
    }

    manufacturedRHS->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
}


void computeL2Norm( std::shared_ptr<AMP::Mesh::Mesh> mesh,
                    const AMP::AMP_MPI &globalComm,
                    AMP::LinearAlgebra::Vector::shared_ptr TemperatureVec,
                    double *discretizationErrorNorm2 )
{
    // CALCULATE THE L2Norm OF (U-Uh)
    auto el                        = mesh->getIterator( AMP::Mesh::GeomType::Cell, 0 );
    AMP::Mesh::MeshIterator end_el = el.end();

    auto dof_map = TemperatureVec->getDOFManager();

    auto feTypeOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "FIRST" );
    auto feFamily    = libMesh::Utility::string_to_enum<libMeshEnums::FEFamily>( "LAGRANGE" );

    auto d_feType = std::make_shared<libMesh::FEType>( feTypeOrder, feFamily );
    std::shared_ptr<libMesh::FEBase> d_fe(
        ( libMesh::FEBase::build( 3, ( *d_feType ) ) ).release() );

    auto qruleOrder = libMesh::Utility::string_to_enum<libMeshEnums::Order>( "SECOND" );
    std::shared_ptr<libMesh::QBase> d_qrule(
        ( libMesh::QBase::build( "QGAUSS", 3, qruleOrder ) ).release() );

    d_fe->attach_quadrature_rule( d_qrule.get() );

    for ( ; el != end_el; ++el ) {

        auto d_currNodes = el->getElements( AMP::Mesh::GeomType::Vertex );

        std::vector<size_t> bndGlobalIds;
        std::vector<AMP::Mesh::MeshElementID> globalIDs( d_currNodes.size() );
        for ( unsigned int j = 0; j < d_currNodes.size(); j++ ) {
            globalIDs[j] = d_currNodes[j]->globalID();
        } // end of j
        dof_map->getDOFs( globalIDs, bndGlobalIds );

        auto d_currElemPtr = new libMesh::Hex8;
        for ( size_t j = 0; j < d_currNodes.size(); j++ ) {
            auto pt                      = d_currNodes[j]->coord();
            d_currElemPtr->set_node( j ) = new libMesh::Node( pt[0], pt[1], pt[2], j );
        } // end for j

        d_fe->reinit( d_currElemPtr );

        const auto &JxW  = d_fe->get_JxW();
        auto coordinates = d_fe->get_xyz();
        const auto &phi  = d_fe->get_phi();

        std::vector<double> computedAtGauss( d_qrule->n_points(), 0.0 );
        for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
            for ( unsigned int j = 0; j < bndGlobalIds.size(); j++ ) {
                double computedAtNode = TemperatureVec->getValueByGlobalID( bndGlobalIds[j] );
                computedAtGauss[qp] += computedAtNode * phi[j][qp];
            } // end for j
        } // end for qp

        for ( unsigned int qp = 0; qp < d_qrule->n_points(); qp++ ) {
            double px                  = coordinates[qp]( 0 );
            double py                  = coordinates[qp]( 1 );
            double pz                  = coordinates[qp]( 2 );
            double manufacturedAtGauss = fun( px, py, pz );

            *discretizationErrorNorm2 +=
                JxW[qp] * std::pow( ( computedAtGauss[qp] - manufacturedAtGauss ), 2 );
        }
    }
    *discretizationErrorNorm2 = globalComm.sumReduce( *discretizationErrorNorm2 );
}

void createThermalOperators(
    std::shared_ptr<AMP::Database> global_input_db,
    std::shared_ptr<AMP::Mesh::Mesh> manager,
    std::shared_ptr<AMP::Operator::ColumnOperator> &nonlinearColumnOperator,
    std::shared_ptr<AMP::Operator::ColumnOperator> &linearColumnOperator )
{
    AMP::pout << "Entering createThermalOperators" << std::endl;

    nonlinearColumnOperator = std::make_shared<AMP::Operator::ColumnOperator>();
    linearColumnOperator    = std::make_shared<AMP::Operator::ColumnOperator>();

    auto bottomAdapter = manager->Subset( "Bottom" );

    //   CREATE THE NONLINEAR THERMAL OPERATOR 1
    AMP_INSIST( global_input_db->keyExists( "BottomNonlinearThermalOperator" ), "key missing!" );
    auto thermalNonlinearOperator = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            bottomAdapter, "BottomNonlinearThermalOperator", global_input_db ) );
    nonlinearColumnOperator->append( thermalNonlinearOperator );

    //-------------------------------------
    //   CREATE THE LINEAR THERMAL OPERATOR 1 ----
    //-------------------------------------
    AMP_INSIST( global_input_db->keyExists( "BottomLinearThermalOperator" ), "key missing!" );
    auto thermalLinearOperator = std::dynamic_pointer_cast<AMP::Operator::LinearBVPOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            bottomAdapter, "BottomLinearThermalOperator", global_input_db ) );
    linearColumnOperator->append( thermalLinearOperator );

    AMP::pout << "Leaving createThermalOperators" << std::endl;
}

void createThermalSolvers( std::shared_ptr<AMP::Database> &global_input_db,
                           AMP::LinearAlgebra::Vector::shared_ptr &globalSolVec,
                           std::shared_ptr<AMP::Operator::ColumnOperator> &nonlinearOperator,
                           std::shared_ptr<AMP::Operator::Operator> &linearOperator,
                           std::shared_ptr<AMP::Solver::PetscSNESSolver> &nonlinearSolver,
                           std::shared_ptr<AMP::Solver::PetscKrylovSolver> &linearSolver )
{
    // initialize the nonlinear solver
    auto nonlinearSolver_db = global_input_db->getDatabase( "NonlinearThermalSolver" );
    auto nonlinearSolverParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( nonlinearSolver_db );

    // change the next line to get the correct communicator out
    nonlinearSolverParams->d_comm          = AMP::AMP_MPI( AMP_COMM_WORLD );
    nonlinearSolverParams->d_pOperator     = nonlinearOperator;
    nonlinearSolverParams->d_pInitialGuess = globalSolVec;
    nonlinearSolver.reset( new AMP::Solver::PetscSNESSolver( nonlinearSolverParams ) );

    // initialize the column preconditioner which is a diagonal block preconditioner
    auto linearSolver_db         = nonlinearSolver_db->getDatabase( "LinearSolver" );
    auto columnPreconditioner_db = linearSolver_db->getDatabase( "Preconditioner" );
    auto columnPreconditionerParams =
        std::make_shared<AMP::Solver::SolverStrategyParameters>( columnPreconditioner_db );
    auto linearColumnOperator =
        std::dynamic_pointer_cast<AMP::Operator::ColumnOperator>( linearOperator );
    AMP_ASSERT( linearColumnOperator );
    columnPreconditionerParams->d_pOperator = linearColumnOperator;
    auto columnPreconditioner =
        std::make_shared<AMP::Solver::ColumnSolver>( columnPreconditionerParams );

    auto trilinosPreconditioner_db =
        columnPreconditioner_db->getDatabase( "TrilinosPreconditioner" );
    for ( unsigned int id = 0; id != linearColumnOperator->getNumberOfOperators(); id++ ) {
        auto trilinosPreconditionerParams =
            std::make_shared<AMP::Solver::SolverStrategyParameters>( trilinosPreconditioner_db );
        trilinosPreconditionerParams->d_pOperator = linearColumnOperator->getOperator( id );
        auto trilinosPreconditioner =
            std::make_shared<AMP::Solver::TrilinosMLSolver>( trilinosPreconditionerParams );
        columnPreconditioner->append( trilinosPreconditioner );
    }

    //--------------------------------------------------------------------//
    // register the preconditioner with the Jacobian free Krylov solver
    linearSolver = nonlinearSolver->getKrylovSolver();
    linearSolver->setNestedSolver( columnPreconditioner );
}

void createThermalMaps( std::shared_ptr<AMP::Database> input_db,
                        std::shared_ptr<AMP::Mesh::Mesh> manager,
                        AMP::LinearAlgebra::Vector::shared_ptr &thermalMapVec,
                        std::shared_ptr<AMP::Operator::AsyncMapColumnOperator> &mapsColumn )
{
    auto map_db = input_db->getDatabase( "MeshToMeshMaps" );

    mapsColumn = AMP::Operator::AsyncMapColumnOperator::build<AMP::Operator::ScalarZAxisMap>(
        manager, map_db );
    mapsColumn->setVector( thermalMapVec );
}

void registerMapswithThermalOperator(
    std::shared_ptr<AMP::Database>,
    std::shared_ptr<AMP::Operator::ColumnOperator> &nonlinearThermalColumnOperator,
    AMP::LinearAlgebra::Vector::shared_ptr &thermalMapVec )
{
    auto curBVPop = std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
        nonlinearThermalColumnOperator->getOperator( 0 ) );
    auto curBCcol = std::dynamic_pointer_cast<AMP::Operator::ColumnBoundaryOperator>(
        curBVPop->getBoundaryOperator() );
    auto gapBC = std::dynamic_pointer_cast<AMP::Operator::RobinVectorCorrection>(
        curBCcol->getBoundaryOperator( 0 ) );
    gapBC->setVariableFlux( thermalMapVec );
    gapBC->reset( gapBC->getOperatorParameters() );
}

//       Main Program
void myTest( AMP::UnitTest *ut,
             std::shared_ptr<AMP::Database> input_db,
             const AMP::AMP_MPI &globalComm )
{
    auto mesh_db       = input_db->getDatabase( "Mesh" );
    auto meshmgrParams = std::make_shared<AMP::Mesh::MeshParameters>( mesh_db );
    meshmgrParams->setComm( globalComm );
    auto manager = AMP::Mesh::MeshFactory::create( meshmgrParams );

    // create the nonlinear and linear thermal operators
    std::shared_ptr<AMP::Operator::ColumnOperator> nonlinearThermalColumnOperator;
    std::shared_ptr<AMP::Operator::ColumnOperator> linearThermalColumnOperator;

    createThermalOperators(
        input_db, manager, nonlinearThermalColumnOperator, linearThermalColumnOperator );

    AMP_ASSERT( nonlinearThermalColumnOperator );
    AMP_ASSERT( linearThermalColumnOperator );

    auto outputVar = std::make_shared<AMP::LinearAlgebra::Variable>( "Temperature" );

    auto nodalScalarDOF = AMP::Discretization::simpleDOFManager::create(
        manager, AMP::Mesh::GeomType::Vertex, 1, 1, true );

    //  create solution, rhs, and  residual vectors
    auto TemperatureVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, outputVar, true );
    auto ResidualVec    = AMP::LinearAlgebra::createVector( nodalScalarDOF, outputVar, true );

    AMP::LinearAlgebra::Vector::shared_ptr nullVec;

    auto manufacturedSolution = AMP::LinearAlgebra::createVector( nodalScalarDOF, outputVar, true );
    auto manufacturedNormalGradient =
        AMP::LinearAlgebra::createVector( nodalScalarDOF, outputVar, true );
    auto solutionError = AMP::LinearAlgebra::createVector( nodalScalarDOF, outputVar, true );
    auto thermMapVec   = AMP::LinearAlgebra::createVector( nodalScalarDOF, outputVar, true );

    int DOFsPerElement = 8;
    auto gaussPointDOF = AMP::Discretization::simpleDOFManager::create(
        manager, AMP::Mesh::GeomType::Cell, 1, DOFsPerElement, true );

    AMP::pout << "Creating gauss Vectors " << std::endl;

    auto manuSourceVar =
        std::make_shared<AMP::LinearAlgebra::Variable>( "SpecificPowerInWattsPerGram" );
    auto manufacturedRHS = AMP::LinearAlgebra::createVector( gaussPointDOF, manuSourceVar, true );

    AMP::pout << "Calculating Manufactured Solution and Sources " << std::endl;

    calculateManufacturedSolution( manager, manufacturedSolution, manufacturedNormalGradient );

    calculateSources( manager, gaussPointDOF, manufacturedRHS );

    std::shared_ptr<AMP::Operator::AsyncMapColumnOperator> mapsColumn;
    //  createThermalMaps( input_db,  manager, thermMapVec , mapsColumn);

    thermMapVec->copyVector( manufacturedNormalGradient );
    thermMapVec->scale( 1 / 20. );
    thermMapVec->add( *thermMapVec, *manufacturedSolution );

    registerMapswithThermalOperator( input_db, nonlinearThermalColumnOperator, thermMapVec );

    TemperatureVec->copyVector( manufacturedSolution );
    std::cout << "Max value of manufactured solution : " << manufacturedSolution->max()
              << std::endl;
    std::cout << "Min value of manufactured solution : " << manufacturedSolution->min()
              << std::endl;
    std::cout << "Max value of manufactured RHS : " << manufacturedRHS->max()
              << " normal gradient: " << manufacturedNormalGradient->max() << std::endl;
    std::cout << "Min value of manufactured RHS : " << manufacturedRHS->min()
              << " normal gradient: " << manufacturedNormalGradient->min() << std::endl;


    // OPERATOR APPLY TO CALCULATE MANUFACTURED RHS

    auto bottomAdapter = manager->Subset( "Bottom" );
    // auto topAdapter = manager->Subset( "Top" );

    auto tmp_db       = std::make_shared<AMP::Database>( "Dummy" );
    auto columnParams = std::make_shared<AMP::Operator::OperatorParameters>( tmp_db );
    auto volumeIntegralColumnOperator =
        std::make_shared<AMP::Operator::ColumnOperator>( columnParams );

    auto sourceOperator1 = std::dynamic_pointer_cast<AMP::Operator::VolumeIntegralOperator>(
        AMP::Operator::OperatorBuilder::createOperator(
            bottomAdapter, "BottomVolumeIntegralOperator", input_db ) );
    volumeIntegralColumnOperator->append( sourceOperator1 );

    auto rhsVar           = std::make_shared<AMP::LinearAlgebra::Variable>( "Temperature" );
    auto integratedRHSVec = AMP::LinearAlgebra::createVector( nodalScalarDOF, rhsVar, true );
    integratedRHSVec->zero();

    volumeIntegralColumnOperator->apply( manufacturedRHS, integratedRHSVec );

    integratedRHSVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    // modify the RHS to take into account boundary conditions
    for ( int i = 0; i < 1; i++ ) {
        auto nonlinearThermalOperator = ( std::dynamic_pointer_cast<AMP::Operator::ColumnOperator>(
                                              nonlinearThermalColumnOperator ) )
                                            ->getOperator( i );
        ( std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
              nonlinearThermalOperator ) )
            ->modifyInitialSolutionVector( TemperatureVec );
        ( std::dynamic_pointer_cast<AMP::Operator::NonlinearBVPOperator>(
              nonlinearThermalOperator ) )
            ->modifyRHSvector( integratedRHSVec );
    }
    nonlinearThermalColumnOperator->residual( integratedRHSVec, TemperatureVec, ResidualVec );
    ResidualVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );
    double initialResidualNorm = static_cast<double>( ResidualVec->L2Norm() );

    AMP::pout << "Initial Residual Norm: " << initialResidualNorm << std::endl;

    std::shared_ptr<AMP::Solver::PetscSNESSolver> nonlinearThermalSolver;
    std::shared_ptr<AMP::Solver::PetscKrylovSolver> linearThermalSolver;
    auto linearThermalOperator =
        std::dynamic_pointer_cast<AMP::Operator::Operator>( linearThermalColumnOperator );
    // createThermalSolvers(input_db, TemperatureVec ,  nonlinearThermalCoupledOperator,
    // linearThermalOperator,
    // nonlinearThermalSolver, linearThermalSolver);
    createThermalSolvers( input_db,
                          TemperatureVec,
                          nonlinearThermalColumnOperator,
                          linearThermalOperator,
                          nonlinearThermalSolver,
                          linearThermalSolver );

    nonlinearThermalSolver->setZeroInitialGuess( false );

    nonlinearThermalSolver->apply( integratedRHSVec, TemperatureVec );

    //  nonlinearThermalCoupledOperator->apply(integratedRHSVec, TemperatureVec, ResidualVec, 1.0,
    //  -1.0);
    nonlinearThermalColumnOperator->residual( integratedRHSVec, TemperatureVec, ResidualVec );
    solutionError->subtract( *TemperatureVec, *manufacturedSolution );

    std::cout << "Max of ||U-Uh|| : " << solutionError->max()
              << " Min of ||U-Uh|| : " << solutionError->min() << std::endl;

    TemperatureVec->makeConsistent( AMP::LinearAlgebra::ScatterType::CONSISTENT_SET );

    double discretizationErrorNorm2;
    double TotalNorm2 = 0;

    discretizationErrorNorm2 = 0;
    AMP::LinearAlgebra::VS_Mesh meshSelector1( bottomAdapter );
    computeL2Norm( bottomAdapter,
                   globalComm,
                   TemperatureVec->select( meshSelector1 ),
                   &discretizationErrorNorm2 );
    TotalNorm2 += discretizationErrorNorm2;
    std::cout << "Discretized error norm ^2 for Mesh  1: " << discretizationErrorNorm2 << std::endl;

    std::cout << "Discretized error norm for ||U-Uh|| : " << std::sqrt( TotalNorm2 ) << std::endl;

    std::cout << "Max of U : " << TemperatureVec->max() << " Min of U : " << TemperatureVec->min()
              << std::endl;

    ut->passes( "Ran to completion" );
}

void multiMeshLoop( AMP::UnitTest *ut, const std::string &exeName )
{
    std::string input_file = "input_" + exeName;


    AMP::AMP_MPI globalComm( AMP_COMM_WORLD );
    auto input_db = AMP::Database::parseInputFile( input_file );
    input_db->print( AMP::plog );


    std::string str1 = "butterfly_pellet_1x.e";

    auto mesh_db       = input_db->getDatabase( "Mesh" );
    auto bottomMesh_db = mesh_db->getDatabase( "Mesh_1" );

    bottomMesh_db->putScalar( "FileName", str1 );

    myTest( ut, input_db, globalComm );
}

int main( int argc, char *argv[] )
{
    AMP::AMPManager::startup( argc, argv );
    AMP::UnitTest ut;
    std::vector<std::string> exeNames;
    exeNames.emplace_back( "testMeshRefinementDiffusion-1" );

    for ( auto name : exeNames )
        multiMeshLoop( &ut, name );

    ut.report();
    int num_failed = ut.NumFailGlobal();

    AMP::AMPManager::shutdown();
    return num_failed;
}
