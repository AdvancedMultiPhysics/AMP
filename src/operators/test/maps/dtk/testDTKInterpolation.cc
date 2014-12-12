
#include <utils/UnitTest.h>
#include <utils/Utilities.h>
#include <utils/shared_ptr.h>
#include <utils/Database.h>
#include <utils/InputDatabase.h>
#include <utils/InputManager.h>
#include <utils/AMP_MPI.h>
#include <utils/AMPManager.h>
#include <utils/PIO.h>

#include <vectors/VectorBuilder.h>

#include <discretization/simpleDOF_Manager.h>

#include <ampmesh/Mesh.h>

#include <operators/map/dtk/DTKAMPVectorHelpers.h>
#include <operators/map/dtk/DTKAMPMeshManager.h>

#include <DTK_ConsistentInterpolationOperator.hpp>
#include <DTK_BasicEntityPredicates.hpp>

#include <iostream>
#include <string>
#include <cstdlib>

double testFunction1( const std::vector<double>& coords )
{
    return coords[0] + coords[1] + coords[2] + 2.0;
}

void myTest(AMP::UnitTest *ut)
{
    std::string exeName("testDTKInterpolation");
    std::string log_file = "output_" + exeName;
    std::string msgPrefix;
    AMP::PIO::logOnlyNodeZero(log_file);

    // load the source mesh
    AMP::pout<<"Loading the source mesh"<<std::endl;
    boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
    AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

    std::string input_file = "input_" + exeName;
    AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
    input_db->printClassData(AMP::plog);

    boost::shared_ptr<AMP::Database> sourceMeshDatabase = input_db->getDatabase("SourceMesh");
    boost::shared_ptr<AMP::Mesh::MeshParameters> sourceMeshParams(new AMP::Mesh::MeshParameters(sourceMeshDatabase));
    sourceMeshParams->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));
    AMP::Mesh::Mesh::shared_ptr sourceMesh = AMP::Mesh::Mesh::buildMesh(sourceMeshParams);
    std::size_t const numVerticesOnSourceMesh = sourceMesh->numGlobalElements(AMP::Mesh::Vertex);
    std::size_t const numElementsOnSourceMesh = sourceMesh->numGlobalElements(AMP::Mesh::Volume);
    AMP::pout<<"source mesh contains "<<numVerticesOnSourceMesh<<" vertices\n";
    AMP::pout<<"source mesh contains "<<numElementsOnSourceMesh<<" elements\n";

    // build source vector
    AMP::pout<<"Building the source vector"<<std::endl;
    bool const split = true;
    int const ghostWidth = 1;
    int const dofsPerNode = 1;
    AMP::Discretization::DOFManager::shared_ptr sourceDofManager = AMP::Discretization::simpleDOFManager::create(sourceMesh, AMP::Mesh::Vertex, ghostWidth, dofsPerNode);
    AMP::LinearAlgebra::Variable::shared_ptr variable(new AMP::LinearAlgebra::Variable("dummy"));
    AMP::LinearAlgebra::Vector::shared_ptr sourceVector = AMP::LinearAlgebra::createVector(sourceDofManager, variable, split);
    // and fill it
    std::vector<std::size_t> dofIndices;
    double value;
    AMP::pout<<"Filling source vector"<<std::endl;
    AMP::Mesh::MeshIterator sourceMeshIterator = sourceMesh->getIterator(AMP::Mesh::Vertex);
    for ( sourceMeshIterator = sourceMeshIterator.begin();
          sourceMeshIterator != sourceMeshIterator.end();
          ++sourceMeshIterator ) {
        sourceDofManager->getDOFs(sourceMeshIterator->globalID(), dofIndices);
        AMP_ASSERT(dofIndices.size() == 1);
        value = testFunction1(sourceMeshIterator->coord());
        sourceVector->setLocalValueByGlobalID(dofIndices[0], value);
    }

    // load the target mesh
    AMP::pout<<"Loading the target mesh"<<std::endl;
    boost::shared_ptr<AMP::Database> targetMeshDatabase = input_db->getDatabase("TargetMesh");
    boost::shared_ptr<AMP::Mesh::MeshParameters> targetMeshParams(new AMP::Mesh::MeshParameters(targetMeshDatabase));
    targetMeshParams->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));
    AMP::Mesh::Mesh::shared_ptr targetMesh = AMP::Mesh::Mesh::buildMesh(targetMeshParams);
    std::size_t const numVerticesOnTargetMesh = targetMesh->numGlobalElements(AMP::Mesh::Vertex);
    std::size_t const numElementsOnTargetMesh = targetMesh->numGlobalElements(AMP::Mesh::Volume);
    AMP::pout<<"target mesh contains "<<numVerticesOnTargetMesh<<" vertices\n";
    AMP::pout<<"target mesh contains "<<numElementsOnTargetMesh<<" elements\n";

    AMP::pout<<"Building the target vector"<<std::endl;
    AMP::Discretization::DOFManager::shared_ptr targetDofManager = AMP::Discretization::simpleDOFManager::create(targetMesh, AMP::Mesh::Vertex, ghostWidth, dofsPerNode);
    AMP::LinearAlgebra::Vector::shared_ptr targetVector = AMP::LinearAlgebra::createVector(targetDofManager, variable, split);

    // setup dtk objects.
    AMP::pout<<"Setting up dtk"<<std::endl;
    DataTransferKit::SelectAllPredicate select_all_pred;
    AMP::Operator::DTKAMPMeshManager dtk_source_mesh( sourceMesh, sourceDofManager, DataTransferKit::ENTITY_TYPE_VOLUME, select_all_pred.getFunction() );
    AMP::Operator::DTKAMPMeshManager dtk_target_mesh( targetMesh, targetDofManager, DataTransferKit::ENTITY_TYPE_NODE, select_all_pred.getFunction() );
    Teuchos::RCP<Tpetra::Vector<double,int,std::size_t> > dtk_source_vector =
        AMP::Operator::DTKAMPVectorHelpers::pullTpetraVectorFromAMPVector( sourceVector );
    Teuchos::RCP<Tpetra::Vector<double,int,std::size_t> > dtk_target_vector =
        AMP::Operator::DTKAMPVectorHelpers::pullTpetraVectorFromAMPVector( targetVector );
    
    // create dtk map operator.
    Teuchos::RCP<Teuchos::ParameterList> dtk_parameters = Teuchos::parameterList();
    Teuchos::RCP<DataTransferKit::MapOperator<double> > dtk_operator = Teuchos::rcp( new DataTransferKit::ConsistentInterpolationOperator<double>() );
    dtk_operator->setup( dtk_source_vector->getMap(), dtk_source_mesh.functionSpace(), 
    	 dtk_target_vector->getMap(), dtk_target_mesh.functionSpace(),
    	 dtk_parameters );
    
    // apply the map.
    AMP::pout<<"Apply dtk operator"<<std::endl;
    dtk_operator->apply( *dtk_source_vector, *dtk_target_vector );
    AMP::Operator::DTKAMPVectorHelpers::pushTpetraVectorToAMPVector( *dtk_target_vector, targetVector );

    // checking the answer
    AMP::pout<<"Check answer"<<std::endl;
    AMP::pout<<"source vector l2 norm = "<<sourceVector->L2Norm()<<std::endl;
    AMP::pout<<"target vector l2 norm = "<<targetVector->L2Norm()<<std::endl;
    double const atol = 1.0e-14;
    double const rtol = 1.0e-14;
    double const tol = atol + rtol * targetVector->L2Norm();
    AMP::Mesh::MeshIterator targetMeshIterator = targetMesh->getIterator(AMP::Mesh::Vertex);
    for ( targetMeshIterator = targetMeshIterator.begin();
          targetMeshIterator != targetMeshIterator.end();
          ++targetMeshIterator ) {
        targetDofManager->getDOFs(targetMeshIterator->globalID(), dofIndices);
        AMP_ASSERT(dofIndices.size() == 1);
        value = testFunction1(targetMeshIterator->coord());
        targetVector->addLocalValueByGlobalID(dofIndices[0], - value);
    }
    AMP::pout<<"error l2 norm = "<<targetVector->L2Norm()<<std::endl;
    AMP_ASSERT( targetVector->L2Norm() < tol );

    ut->passes( exeName );
}


int main(int argc, char *argv[])
{
    AMP::AMPManagerProperties startup_properties;
    startup_properties.use_MPI_Abort = false;
    AMP::AMPManager::startup(argc,argv,startup_properties);
    AMP::UnitTest ut;

    myTest(&ut);

    ut.report();

    int num_failed = ut.NumFailGlobal();
    AMP::AMPManager::shutdown();
    return num_failed;
}   


