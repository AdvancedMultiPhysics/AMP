
#include "mpi.h"

#include "utils/AMPManager.h"
#include "utils/UnitTest.h"
#include "utils/Utilities.h"
#include "utils/Database.h"
#include "utils/InputDatabase.h"
#include "utils/InputManager.h"
#include "utils/AMP_MPI.h"
#include "utils/PIO.h"

#include "discretization/DOF_Manager.h"
#include "discretization/simpleDOF_Manager.h"
#include "vectors/Variable.h"
#include "vectors/Vector.h"
#include "vectors/VectorBuilder.h"


#include "ampmesh/Mesh.h"
#include "ampmesh/DendroSearch.h"


double dummyFunction(const std::vector<double> &xyz) {
  AMP_ASSERT(xyz.size() == 3);
  double x = xyz[0], y = xyz[1], z = xyz[2];
  return (1.0 + x) * (2.0 - y) * (3.0 + z);
}

void myTest(AMP::UnitTest *ut, std::string exeName) {
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName; 

  ot::RegisterEvents();

  AMP::PIO::logOnlyNodeZero(log_file);
  AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

  int rank = globalComm.getRank();
  int npes = globalComm.getSize();

  // Load the input file
  globalComm.barrier();
  double inpReadBeginTime = MPI_Wtime();
  boost::shared_ptr<AMP::InputDatabase> input_db(new AMP::InputDatabase("input_db"));
  AMP::InputManager::getManager()->parseInputFile(input_file, input_db);
  input_db->printClassData(AMP::plog);
  globalComm.barrier();
  double inpReadEndTime = MPI_Wtime();
  if(!rank) {
    std::cout<<"Finished parsing the input file in "<<(inpReadEndTime - inpReadBeginTime)<<" seconds."<<std::endl;
  }

  // Load the mesh
  globalComm.barrier();
  double meshBeginTime = MPI_Wtime();
  AMP_INSIST(input_db->keyExists("Mesh"), "Key ''Mesh'' is missing!");
  boost::shared_ptr<AMP::Database> mesh_db = input_db->getDatabase("Mesh");
  boost::shared_ptr<AMP::Mesh::MeshParameters> meshParams(new AMP::Mesh::MeshParameters(mesh_db));
  meshParams->setComm(AMP::AMP_MPI(AMP_COMM_WORLD));
  AMP::Mesh::Mesh::shared_ptr meshAdapter = AMP::Mesh::Mesh::buildMesh(meshParams);
  globalComm.barrier();
  double meshEndTime = MPI_Wtime();
  if(!rank) {
    std::cout<<"Finished reading the mesh in "<<(meshEndTime - meshBeginTime)<<" seconds."<<std::endl;
  }

  // Create a vector field
  int DOFsPerNode = 1;
  int nodalGhostWidth = 1;
  bool split = true;
  AMP::Discretization::DOFManager::shared_ptr DOFs = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, nodalGhostWidth, DOFsPerNode, split);
  AMP::LinearAlgebra::Variable::shared_ptr dummyVariable(new AMP::LinearAlgebra::Variable("Dummy"));
  AMP::LinearAlgebra::Vector::shared_ptr dummyVector = createVector(DOFs, dummyVariable, split);

  AMP::Mesh::MeshIterator node = meshAdapter->getIterator(AMP::Mesh::Vertex, 0);
  AMP::Mesh::MeshIterator end_node = node.end();
  for ( ; node != end_node; ++node) {
    std::vector<size_t> globalID;
    DOFs->getDOFs(node->globalID(), globalID); 
    AMP_ASSERT(globalID.size() == 1);
    dummyVector->setValueByGlobalID(globalID.front(), dummyFunction(node->coord()));
  }
  
  double minCoords[3];
  double maxCoords[3];
  std::vector<double> box = meshAdapter->getBoundingBox();
  for(int i=0; i<meshAdapter->getDim(); ++i) {
    minCoords[i] = box[2*i+0];
    maxCoords[i] = box[2*i+1];
//    ScalingFactor[i] = 1.0/(1.0e-10 + maxCoords[i] - minCoords[i]);
  }

  int totalNumPts = input_db->getInteger("TotalNumberOfPoints");
  int avgNumPts = totalNumPts/npes;
  int extraNumPts = totalNumPts%npes;

  int numLocalPts = avgNumPts;
  if(rank < extraNumPts) {
    numLocalPts++;
  }

  // Generate Random points in [min, max]
  const unsigned int seed = (0x1234567 + (24135*rank));
  srand48(seed);

  std::vector<double> pts(3*numLocalPts);
  for(int i = 0; i < numLocalPts; ++i) {
    double x = ((maxCoords[0] - minCoords[0])*drand48()) + minCoords[0];
    double y = ((maxCoords[1] - minCoords[1])*drand48()) + minCoords[1];
    double z = ((maxCoords[2] - minCoords[2])*drand48()) + minCoords[2];
    pts[3*i] = x;
    pts[(3*i) + 1] = y;
    pts[(3*i) + 2] = z;
  }//end i
  if(!rank) {
    std::cout<<"Finished generating "<<totalNumPts <<" random points for search!"<<std::endl;
  }



  DendroSearch dendroSearch(globalComm, meshAdapter);
  std::vector<double> interpolatedData = dendroSearch.interpolate(dummyVector, numLocalPts, pts);
  AMP_ASSERT(interpolatedData.size() == numLocalPts);




  std::vector<double> interpolationError(numLocalPts);
  for (unsigned int i = 0; i < numLocalPts; ++i) {
    interpolationError[i] = interpolatedData[i] - dummyFunction(std::vector<double>(&(pts[3*i]), &(pts[3*i+3])));
  } // end for i
  double localErrorSquaredNorm = std::inner_product(interpolationError.begin(), interpolationError.end(), interpolationError.begin(), 0.0);
  if(!rank) {
    std::cout<<"Finished computing the local squared norm of the interpolation error."<<std::endl;
  }
  globalComm.barrier();


  double globalErrorSquaredNorm = -1.0;
  MPI_Allreduce(&localErrorSquaredNorm, &globalErrorSquaredNorm, 1, MPI_DOUBLE, MPI_SUM, globalComm.getCommunicator());
  if(!rank) {
    std::cout<<"Global error norm is "<<sqrt(globalErrorSquaredNorm)<<std::endl;
  }

  AMP_ASSERT(sqrt(globalErrorSquaredNorm) < 1.0e-15);



  ut->passes(exeName);
}


int main(int argc, char *argv[])
{
  AMP::AMPManager::startup(argc, argv);
  AMP::UnitTest ut;

  std::string exeName = "testDendroSearch";

  try {
    myTest(&ut, exeName);
  } catch (std::exception &err) {
    std::cout << "ERROR: While testing "<<argv[0] << err.what() << std::endl;
    ut.failure("ERROR: While testing");
  } catch( ... ) {
    std::cout << "ERROR: While testing "<<argv[0] << "An unknown exception was thrown." << std::endl;
    ut.failure("ERROR: While testing");
  }

  ut.report();
  int num_failed = ut.NumFailGlobal();

  AMP::AMPManager::shutdown();
  return num_failed;
}  


