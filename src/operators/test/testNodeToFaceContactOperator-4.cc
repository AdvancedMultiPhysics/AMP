
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

#include "externVars.h"

#include "ampmesh/libmesh/libMesh.h"
#include "ampmesh/Mesh.h"
#include "utils/Writer.h"

#include "operators/OperatorBuilder.h"
#include "operators/LinearBVPOperator.h"
#include "operators/ColumnOperator.h"
#include "operators/PetscMatrixShellOperator.h"
#include "operators/boundary/DirichletVectorCorrection.h"
#include "operators/mechanics/MechanicsModelParameters.h"
#include "operators/mechanics/MechanicsMaterialModel.h"
#include "operators/mechanics/MechanicsLinearFEOperator.h"
#include "operators/mechanics/ConstructLinearMechanicsRHSVector.h"
#include "operators/contact/NodeToFaceContactOperator.h"

#include "solvers/ColumnSolver.h"
#include "solvers/PetscKrylovSolver.h"
#include "solvers/ConstraintsEliminationSolver.h"

#include "utils/ReadTestMesh.h"

#include <set>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include "ampmesh/latex_visualization_tools.h"
#include "ampmesh/euclidean_geometry_tools.h"

void rotateMesh(AMP::Mesh::Mesh::shared_ptr mesh) {
  AMP::Discretization::DOFManager::shared_ptr dofManager = AMP::Discretization::simpleDOFManager::create(mesh, AMP::Mesh::Vertex, 0, 3, false);
  AMP::LinearAlgebra::Variable::shared_ptr dispVar(new AMP::LinearAlgebra::Variable("disp"));
  AMP::LinearAlgebra::Vector::shared_ptr dispVec = AMP::LinearAlgebra::createVector(dofManager, dispVar, false);
  dispVec->zero();
  AMP::Mesh::MeshIterator meshIterator = mesh->getIterator(AMP::Mesh::Vertex);
  AMP::Mesh::MeshIterator meshIterator_begin = meshIterator.begin();
  AMP::Mesh::MeshIterator meshIterator_end = meshIterator.end();
  for (meshIterator = meshIterator_begin; meshIterator != meshIterator_end; ++meshIterator) {
    std::vector<double> oldVertexCoord = meshIterator->coord();
    std::vector<size_t> dofIndices;
    dofManager->getDOFs(meshIterator->globalID(), dofIndices);
    AMP_ASSERT(dofIndices.size() == 3);
    std::vector<double> newVertexCoord(oldVertexCoord);
    // 15 degrees around the z axis
//    rotate_points(2, M_PI / 12.0, 1, &(newVertexCoord[0]));
    rotate_points(1, M_PI / 3.0, 1, &(newVertexCoord[0]));
    std::vector<double> vertexDisp(3, 0.0);
    make_vector_from_two_points(&(oldVertexCoord[0]), &(newVertexCoord[0]), &(vertexDisp[0]));  
    dispVec->setLocalValuesByGlobalID(3, &(dofIndices[0]), &(vertexDisp[0]));
  } // end for
  mesh->displaceMesh(dispVec);
}

struct dummyClass {
  bool operator() (std::pair<AMP::Mesh::MeshElementID, size_t> const & lhs, 
                   std::pair<AMP::Mesh::MeshElementID, size_t> const & rhs) {
    return lhs.first < rhs.first;
  }
};

void computeStressTensor(AMP::Mesh::Mesh::shared_ptr mesh, AMP::LinearAlgebra::Vector::shared_ptr displacementField, 
    AMP::LinearAlgebra::Vector::shared_ptr sigma_xx, AMP::LinearAlgebra::Vector::shared_ptr sigma_yy, AMP::LinearAlgebra::Vector::shared_ptr sigma_zz, 
    AMP::LinearAlgebra::Vector::shared_ptr sigma_yz, AMP::LinearAlgebra::Vector::shared_ptr sigma_xz, AMP::LinearAlgebra::Vector::shared_ptr sigma_xy, 
    AMP::LinearAlgebra::Vector::shared_ptr sigma_eff, double YoungsModulus, double PoissonsRatio, 
    double referenceTemperature = 273, double thermalExpansionCoefficient = 2.0e-6, 
    AMP::LinearAlgebra::Vector::shared_ptr temperatureField = AMP::LinearAlgebra::Vector::shared_ptr()) { 

  sigma_eff->setToScalar(-1.0);

  double constitutiveMatrix[36];
  compute_constitutive_matrix(YoungsModulus, PoissonsRatio, constitutiveMatrix);
/*
  double constitutiveMatrix[6][6];

  for(size_t i = 0; i < 6; ++i) {
    for(size_t j = 0; j < 6; ++j) {
      constitutiveMatrix[i][j] = 0.0;
    } // end for j
  } // end for i

  double E = YoungsModulus;
  double nu = PoissonsRatio;
  double K = E / (3.0 * (1.0 - (2.0 * nu)));
  double G = E / (2.0 * (1.0 + nu));

  for(size_t i = 0; i < 3; ++i) {
    constitutiveMatrix[i][i] += (2.0 * G);
  } // end for i

  for(size_t i = 3; i < 6; ++i) {
    constitutiveMatrix[i][i] += G;
  } // end for i

  for(size_t i = 0; i < 3; ++i) {
    for(size_t j = 0; j < 3; ++j) {
      constitutiveMatrix[i][j] += (K - ((2.0 * G) / 3.0));
    } // end for j
  } // end for i
*/

  double verticesLocalCoordinates[24] = {
      -1.0, -1.0, -1.0,
      +1.0, -1.0, -1.0,
      +1.0, +1.0, -1.0,
      -1.0, +1.0, -1.0,
      -1.0, -1.0, +1.0,
      +1.0, -1.0, +1.0,
      +1.0, +1.0, +1.0,
      -1.0, +1.0, +1.0
  };

  std::set<std::pair<AMP::Mesh::MeshElementID, size_t>, dummyClass> verticesGlobalIDsAndCount;

  size_t numLocalVertices = mesh->numLocalElements(AMP::Mesh::Vertex);
  size_t countVertices = 0;
  std::vector<size_t> verticesInHowManyVolumeElements(numLocalVertices, 1);
  std::vector<size_t> verticesDOFIndex(numLocalVertices);
  AMP::Mesh::MeshIterator meshIterator = mesh->getIterator(AMP::Mesh::Volume);
  AMP::Mesh::MeshIterator meshIterator_begin = meshIterator.begin();
  AMP::Mesh::MeshIterator meshIterator_end = meshIterator.end();
  for (meshIterator = meshIterator_begin; meshIterator != meshIterator_end; ++meshIterator) {
    std::vector<AMP::Mesh::MeshElement> volumeVertices = meshIterator->getElements(AMP::Mesh::Vertex);
    AMP_ASSERT(volumeVertices.size() == 8);
    std::vector<AMP::Mesh::MeshElementID> volumeVerticesGlobalIDs(8);
    double volumeVerticesCoordinates[24];
    for (size_t v = 0; v < 8; ++v) {
      std::vector<double> vertexCoord = volumeVertices[v].coord();
      std::copy(vertexCoord.begin(), vertexCoord.end(), &(volumeVerticesCoordinates[3*v]));
      volumeVerticesGlobalIDs[v] = volumeVertices[v].globalID();
    } // end for v
    hex8_element_t volumeElement(volumeVerticesCoordinates);

    std::vector<size_t> displacementIndices;
    displacementField->getDOFManager()->getDOFs(volumeVerticesGlobalIDs, displacementIndices);
    AMP_ASSERT(displacementIndices.size() == 24);
    double displacementValues[24];
    displacementField->getValuesByGlobalID(24, &(displacementIndices[0]), &(displacementValues[0]));

    double temperatureValues[8];
    if (temperatureField.get() != NULL) {
      std::vector<size_t> temperatureIndices;
      temperatureField->getDOFManager()->getDOFs(volumeVerticesGlobalIDs, temperatureIndices);
      AMP_ASSERT(temperatureIndices.size() == 8);
      temperatureField->getValuesByGlobalID(8, &(temperatureIndices[0]), &(temperatureValues[0]));
    } // end if

    double strainTensor[6];
    double stressTensor[6];
    for (size_t v = 0; v < 8; ++v) {
      volumeElement.compute_strain_tensor(&(verticesLocalCoordinates[3*v]), displacementValues, strainTensor);
      if (temperatureField.get() != NULL) {
        for (size_t i = 0; i < 3; ++i) {
          strainTensor[i] -= (thermalExpansionCoefficient * (temperatureValues[v] - referenceTemperature));
        } // end for i
      } // end if
      compute_stress_tensor(constitutiveMatrix, strainTensor, stressTensor);
/*
      for (size_t i = 0; i < 6; ++i) {
        stressTensor[i] = 0.0;
        for (size_t j = 0; j < 6; ++j) {
          stressTensor[i] += (constitutiveMatrix[i][j] * strainTensor[j]);
        } // end for j
      } // end for i
*/

     double vonMisesStress = std::sqrt(
         0.5 * (
           std::pow(stressTensor[0] - stressTensor[1], 2) + 
           std::pow(stressTensor[1] - stressTensor[2], 2) + 
           std::pow(stressTensor[2] - stressTensor[0], 2)
         ) + 3.0 * (
             std::pow(stressTensor[3], 2) + 
             std::pow(stressTensor[4], 2) + 
             std::pow(stressTensor[5], 2)
         ) 
     );

      

      std::vector<size_t> indices;
      sigma_xx->getDOFManager()->getDOFs(volumeVerticesGlobalIDs[v], indices);
      AMP_ASSERT(indices.size() == 1);
      // good job buddy
      std::pair<std::set<std::pair<AMP::Mesh::MeshElementID, size_t>, dummyClass>::iterator, bool> dummy = verticesGlobalIDsAndCount.insert(std::pair<AMP::Mesh::MeshElementID, size_t>(volumeVerticesGlobalIDs[v], countVertices));
      if (dummy.second) {
        verticesDOFIndex[countVertices] = indices[0];
        ++countVertices;
        sigma_xx->setLocalValueByGlobalID(indices[0], stressTensor[0]);
        sigma_yy->setLocalValueByGlobalID(indices[0], stressTensor[1]);
        sigma_zz->setLocalValueByGlobalID(indices[0], stressTensor[2]);
        sigma_yz->setLocalValueByGlobalID(indices[0], stressTensor[3]);
        sigma_xz->setLocalValueByGlobalID(indices[0], stressTensor[4]);
        sigma_xy->setLocalValueByGlobalID(indices[0], stressTensor[5]);
        sigma_eff->setLocalValueByGlobalID(indices[0], vonMisesStress);
      } else {
        // sigh...
        ++verticesInHowManyVolumeElements[dummy.first->second];
        sigma_xx->addLocalValueByGlobalID(indices[0], stressTensor[0]);
        sigma_yy->addLocalValueByGlobalID(indices[0], stressTensor[1]);
        sigma_zz->addLocalValueByGlobalID(indices[0], stressTensor[2]);
        sigma_yz->addLocalValueByGlobalID(indices[0], stressTensor[3]);
        sigma_xz->addLocalValueByGlobalID(indices[0], stressTensor[4]);
        sigma_xy->addLocalValueByGlobalID(indices[0], stressTensor[5]);
        sigma_eff->addLocalValueByGlobalID(indices[0], vonMisesStress);

/*        double deltaStressTensor[6];
        deltaStressTensor[0] = sigma_xx->getLocalValueByGlobalID(indices[0]) - stressTensor[0];
        deltaStressTensor[1] = sigma_yy->getLocalValueByGlobalID(indices[0]) - stressTensor[1];
        deltaStressTensor[2] = sigma_zz->getLocalValueByGlobalID(indices[0]) - stressTensor[2];
        deltaStressTensor[3] = sigma_yz->getLocalValueByGlobalID(indices[0]) - stressTensor[3];
        deltaStressTensor[4] = sigma_xz->getLocalValueByGlobalID(indices[0]) - stressTensor[4];
        deltaStressTensor[5] = sigma_xy->getLocalValueByGlobalID(indices[0]) - stressTensor[5];
        double deltaVonMisesStress = sigma_eff->getLocalValueByGlobalID(indices[0]) - vonMisesStress;
        std::cout<<"xx  "<<deltaStressTensor[0]<<"  "<<100.0*deltaStressTensor[0]/std::abs(stressTensor[0])<<" %  |  ";
        std::cout<<"yy  "<<deltaStressTensor[1]<<"  "<<100.0*deltaStressTensor[1]/std::abs(stressTensor[1])<<" %  |  ";
        std::cout<<"zz  "<<deltaStressTensor[2]<<"  "<<100.0*deltaStressTensor[2]/std::abs(stressTensor[2])<<" %  |  ";
        std::cout<<"yz  "<<deltaStressTensor[3]<<"  "<<100.0*deltaStressTensor[3]/std::abs(stressTensor[3])<<" %  |  ";
        std::cout<<"xz  "<<deltaStressTensor[4]<<"  "<<100.0*deltaStressTensor[4]/std::abs(stressTensor[4])<<" %  |  ";
        std::cout<<"xy  "<<deltaStressTensor[5]<<"  "<<100.0*deltaStressTensor[5]/std::abs(stressTensor[5])<<" %  |  ";
        std::cout<<"eff  "<<deltaVonMisesStress<<"  "<<100.0*deltaVonMisesStress/std::abs(vonMisesStress)<<" % \n";
*/
      } // end if
      
    } // end for v
  } // end for
  AMP_ASSERT(verticesGlobalIDsAndCount.size() == numLocalVertices);
  AMP_ASSERT(countVertices == numLocalVertices);
  AMP_ASSERT(find(verticesInHowManyVolumeElements.begin(), verticesInHowManyVolumeElements.end(), 0) == verticesInHowManyVolumeElements.end());

  for (size_t v = 0; v < numLocalVertices; ++v) {
    if (verticesInHowManyVolumeElements[v] > 1) {
//      AMP_ASSERT(verticesInHowManyVolumeElements[v] < 9);
      sigma_xx->setLocalValueByGlobalID(verticesDOFIndex[v], sigma_xx->getLocalValueByGlobalID(verticesDOFIndex[v]) / static_cast<double>(verticesInHowManyVolumeElements[v]));
      sigma_yy->setLocalValueByGlobalID(verticesDOFIndex[v], sigma_yy->getLocalValueByGlobalID(verticesDOFIndex[v]) / static_cast<double>(verticesInHowManyVolumeElements[v]));
      sigma_zz->setLocalValueByGlobalID(verticesDOFIndex[v], sigma_zz->getLocalValueByGlobalID(verticesDOFIndex[v]) / static_cast<double>(verticesInHowManyVolumeElements[v]));
      sigma_yz->setLocalValueByGlobalID(verticesDOFIndex[v], sigma_yz->getLocalValueByGlobalID(verticesDOFIndex[v]) / static_cast<double>(verticesInHowManyVolumeElements[v]));
      sigma_xz->setLocalValueByGlobalID(verticesDOFIndex[v], sigma_xz->getLocalValueByGlobalID(verticesDOFIndex[v]) / static_cast<double>(verticesInHowManyVolumeElements[v]));
      sigma_xy->setLocalValueByGlobalID(verticesDOFIndex[v], sigma_xy->getLocalValueByGlobalID(verticesDOFIndex[v]) / static_cast<double>(verticesInHowManyVolumeElements[v]));
      sigma_eff->setLocalValueByGlobalID(verticesDOFIndex[v], sigma_eff->getLocalValueByGlobalID(verticesDOFIndex[v]) / static_cast<double>(verticesInHowManyVolumeElements[v]));
    } //end if
  } // end for v
  
} 
 



void drawVerticesOnBoundaryID(AMP::Mesh::Mesh::shared_ptr meshAdapter, int boundaryID, std::ostream &os, double const * point_of_view, const std::string & option = "") {
  AMP::Mesh::MeshIterator boundaryIterator = meshAdapter->getBoundaryIDIterator(AMP::Mesh::Vertex, boundaryID);
  AMP::Mesh::MeshIterator boundaryIterator_begin = boundaryIterator.begin(), 
      boundaryIterator_end = boundaryIterator.end();
  std::vector<double> vertexCoordinates;

  os<<std::setprecision(6)<<std::fixed;

  for (boundaryIterator = boundaryIterator_begin; boundaryIterator != boundaryIterator_end; ++boundaryIterator) {
    vertexCoordinates = boundaryIterator->coord();
    AMP_ASSERT( vertexCoordinates.size() == 3 );
    draw_point(&(vertexCoordinates[0]), option, os);
  } // end for
}

void drawFacesOnBoundaryID(AMP::Mesh::Mesh::shared_ptr meshAdapter, int boundaryID, std::ostream &os, double const * point_of_view, const std::string & option = "") {
  AMP::Mesh::MeshIterator boundaryIterator = meshAdapter->getBoundaryIDIterator(AMP::Mesh::Face, boundaryID);
  AMP::Mesh::MeshIterator boundaryIterator_begin = boundaryIterator.begin(), 
      boundaryIterator_end = boundaryIterator.end();
  std::vector<AMP::Mesh::MeshElement> faceVertices;
  std::vector<double> faceVertexCoordinates;
  double faceData[12];
  double const * faceDataPtr[4] = { faceData, faceData+3, faceData+6, faceData+9 };

  os<<std::setprecision(6)<<std::fixed;

  for (boundaryIterator = boundaryIterator_begin; boundaryIterator != boundaryIterator_end; ++boundaryIterator) {
    faceVertices = boundaryIterator->getElements(AMP::Mesh::Vertex);
    AMP_ASSERT( faceVertices.size() == 4 );
    for (size_t i = 0; i < 4; ++i) {
      faceVertexCoordinates = faceVertices[i].coord();
      AMP_ASSERT( faceVertexCoordinates.size() == 3 );
      std::copy(faceVertexCoordinates.begin(), faceVertexCoordinates.end(), faceData+3*i);
    } // end for i
    triangle_t t(faceDataPtr[0], faceDataPtr[1], faceDataPtr[2]);

//    if (compute_scalar_product(point_of_view, t.get_normal()) > 0.0) {
    if (true) {
      os<<"\\draw["<<option<<"]\n";
      write_face(faceDataPtr, os);
    } // end if
  } // end for
}

void myPCG(AMP::LinearAlgebra::Vector::shared_ptr rhs, AMP::LinearAlgebra::Vector::shared_ptr sol, 
    AMP::Operator::Operator::shared_ptr op, boost::shared_ptr<AMP::Solver::SolverStrategy> pre,
    size_t maxIters, double relTol, double absTol, bool verbose = false, std::ostream &os = std::cout) {
  AMP::LinearAlgebra::Vector::shared_ptr res = sol->cloneVector();
  AMP::LinearAlgebra::Vector::shared_ptr dir = sol->cloneVector();
  AMP::LinearAlgebra::Vector::shared_ptr ext = sol->cloneVector();
  AMP::LinearAlgebra::Vector::shared_ptr oldSol = sol->cloneVector();
  AMP::LinearAlgebra::Vector::shared_ptr oldRes = sol->cloneVector();
  AMP::LinearAlgebra::Vector::shared_ptr oldDir = sol->cloneVector();
  AMP::LinearAlgebra::Vector::shared_ptr matVec = sol->cloneVector();
  AMP::LinearAlgebra::Vector::shared_ptr nullVec;

  op->apply(nullVec, sol, matVec, 1.0, 0.0);
  oldRes->subtract(rhs, matVec);
  pre->solve(oldRes, ext);
  oldDir->copyVector(ext);
  oldSol->copyVector(sol);
  double initialResNorm = oldRes->L2Norm();
  AMP::AMP_MPI comm = sol->getComm();
  int rank = comm.getRank();
  verbose = verbose && !rank;
  double tol = absTol + relTol * initialResNorm;
  if (verbose) { os<<std::setprecision(15)<<"  iter=0  itialResNorm="<<initialResNorm<<"\n"; }
  for (size_t iter = 0; iter < maxIters; ++iter) {
    if (verbose) { os<<"  iter="<<iter+1<<"  "; }
    op->apply(nullVec, oldDir, matVec, 1.0, 0.0);
    double extDOToldRes = ext->dot(oldRes);
    double oldDirDOTmatVec = oldDir->dot(matVec);
    double alpha = extDOToldRes / oldDirDOTmatVec;
    if (verbose) { os<<"alpha="<<alpha<<"  "; }
    if (verbose) { os<<"oldDirDOTmatVec="<<oldDirDOTmatVec<<"  "; }
    sol->axpy(alpha, oldDir, oldSol);
    res->axpy(-alpha, matVec, oldRes);
    double resNorm = res->L2Norm();
    if (verbose) { os<<"resNorm="<<resNorm<<"  "; }
    if (resNorm < tol) { os<<"\n"; break; }
    pre->solve(res, ext);
    double extDOTres = ext->dot(res);
    double beta = extDOTres / extDOToldRes;
    if (verbose) { os<<"beta="<<beta<<"\n"; }
    dir->axpy(beta, oldDir, ext);
    oldSol->copyVector(sol);
    oldRes->copyVector(res);
    oldDir->copyVector(dir);
  } // end for
}

void sideExperiment() {
  double sp[24] = {
      -1.0, -1.0, -1.0,
      +1.0, -1.0, -1.0,
      +1.0, +1.0, -1.0,
      -1.0, +1.0, -1.0,
      -1.0, -1.0, +1.0,
      +1.0, -1.0, +1.0,
      +1.0, +1.0, +1.0,
      -1.0, +1.0, +1.0
  };
  std::transform(sp, sp+24, sp, std::bind1st(std::multiplies<double>(), 1.0e-2));
  hex8_element_t e(sp);
  std::transform(sp, sp+24, sp, std::bind1st(std::multiplies<double>(), 1.0e+2));

  double c[6][6];

  for(size_t i = 0; i < 6; ++i) {
    for(size_t j = 0; j < 6; ++j) {
      c[i][j] = 0.0;
    } // end for j
  } // end for i

  double E = 1.0e6;
  double nu = 0.3;
  double K = E / (3.0 * (1.0 - (2.0 * nu)));
  double G = E / (2.0 * (1.0 + nu));

  for(size_t i = 0; i < 3; ++i) {
    c[i][i] += (2.0 * G);
  } // end for i

  for(size_t i = 3; i < 6; ++i) {
    c[i][i] += G;
  } // end for i

  for(size_t i = 0; i < 3; ++i) {
    for(size_t j = 0; j < 3; ++j) {
      c[i][j] += (K - ((2.0 * G) / 3.0));
    } // end for j
  } // end for i

  double u[24];
  std::copy(sp, sp+24, u);
  double sf[3] = { 1.0, 1.0, 1.0 };
  std::transform(sf, sf+3, sf, std::bind1st(std::multiplies<double>(), 1.0e-10));
  scale_points(sf, 8, u);
  double lc[3] = { 0.0, 0.0, -1.0 };
  double eps[6], sig[6];
  e.compute_strain_tensor(lc, u, eps);
   for (size_t i = 0; i < 6; ++i) {
    sig[i] = 0.0;
    for (size_t j = 0; j < 6; ++j) {
      sig[i] += (c[i][j] * eps[j]);
    } // end for j
  } // end for i
  double n[3] = { 0.0, 0.0, -1.0 };
  double t[3];
  compute_traction(sig, n, t);
  std::cout<<compute_scalar_product(t, n) <<std::endl;
  abort();
  
 

}

void too_young_too_dumb(AMP::Mesh::Mesh::shared_ptr mesh, AMP::LinearAlgebra::Vector::shared_ptr vector) {
  AMP::Mesh::MeshIterator meshIterator = mesh->getIterator(AMP::Mesh::Volume);
  std::vector<AMP::Mesh::MeshElement> vertices = meshIterator->getElements(AMP::Mesh::Vertex); 
  std::vector<AMP::Mesh::MeshElementID> verticesGlobalIDs(8);
  AMP_ASSERT(vertices.size() == 8); 
  std::vector<size_t> dofIndices;
  double const values[3] = { 1.0, 2.0, 3.0 };
  for (size_t i = 0; i < 8; ++i) {
    verticesGlobalIDs[i] = vertices[i].globalID();
    vector->getDOFManager()->getDOFs(verticesGlobalIDs[i], dofIndices);
    AMP_ASSERT(dofIndices.size() == 3);
    vector->setLocalValuesByGlobalID(3, &(dofIndices[0]), values);
  } // end for i
  double surprise[24];
  vector->getDOFManager()->getDOFs(verticesGlobalIDs, dofIndices);
  AMP_ASSERT(dofIndices.size() == 24);
  vector->getLocalValuesByGlobalID(24, &(dofIndices[0]), surprise);
  for (size_t i = 0; i < 24; ++i) {
    std::cout<<surprise[i]<<"  ";
  } // end for i
  std::cout<<std::endl;
  abort();
}

void why_cant_we_be_friend(AMP::Mesh::Mesh::shared_ptr mesh, AMP::LinearAlgebra::Vector::shared_ptr displacementVector) {
  double gaussianQuadrature[24];
  double gamma = std::sqrt(3.0) / 3.0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        gaussianQuadrature[3*(4*i+2*j+k)+0] = gamma * std::pow(-1.0, i);
        gaussianQuadrature[3*(4*i+2*j+k)+1] = gamma * std::pow(-1.0, j);
        gaussianQuadrature[3*(4*i+2*j+k)+2] = gamma * std::pow(-1.0, k);
      } // end for k
    } // end for j
  } // end for i
  double constitutiveMatrix[36];
  compute_constitutive_matrix(1.0e6, 0.3, constitutiveMatrix);
  double strainTensor[6];
  double stressTensor[6];
  AMP::Mesh::MeshIterator meshIterator = mesh->getIterator(AMP::Mesh::Volume);
  std::vector<AMP::Mesh::MeshElement> vertices = meshIterator->getElements(AMP::Mesh::Vertex);
  AMP_ASSERT(vertices.size() == 8);
  std::vector<AMP::Mesh::MeshElementID> verticesGlobalIDs(8);
  double verticesCoordinates[24];
  for (size_t v = 0; v < 8; ++v) {
    std::vector<double> vertexCoord = vertices[v].coord();
    std::copy(vertexCoord.begin(), vertexCoord.end(), &(verticesCoordinates[3*v]));
    verticesGlobalIDs[v] = vertices[v].globalID();
  } // end for v
  std::vector<size_t> dofIndices;
  displacementVector->getDOFManager()->getDOFs(verticesGlobalIDs, dofIndices);
  double displacementValues[24];
  hex8_element_t volumel(verticesCoordinates);
  displacementVector->getLocalValuesByGlobalID(24, &(dofIndices[0]), &(displacementValues[0]));
  double localCoordinates[3] = { 0.0, 0.0, 0.0 };
  for (size_t q = 0; q < 8; ++q) {
    volumel.compute_strain_tensor(&(gaussianQuadrature[3*q]), displacementValues, strainTensor);
    for (size_t i = 0; i < 3; ++i) {
      std::cout<<gaussianQuadrature[3*q+i]<<"  ";
    } // end for i
    std::cout<<"|  ";
    for (size_t i = 0; i < 6; ++i) {
      std::cout<<strainTensor[i]<<"  ";
    } // end for i
    std::cout<<"|  ";
    compute_stress_tensor(constitutiveMatrix, strainTensor, stressTensor);
    for (size_t i = 0; i < 6; ++i) {
      std::cout<<stressTensor[i]<<"  ";
    } // end for i
    std::cout<<"\n";
  } // end for q

  for (double x = -1.0; x <= 1.0; x += 0.2) {
    localCoordinates[0] = x;
    for (size_t i = 0; i < 3; ++i) {
      std::cout<<localCoordinates[i]<<"  ";
    } // end for i
    std::cout<<"|  ";
    volumel.compute_strain_tensor(localCoordinates, displacementValues, strainTensor);
    for (size_t i = 0; i < 6; ++i) {
      std::cout<<strainTensor[i]<<"  ";
    } // end for i
    std::cout<<"|  ";
    compute_stress_tensor(constitutiveMatrix, strainTensor, stressTensor);
    for (size_t i = 0; i < 6; ++i) {
      std::cout<<stressTensor[i]<<"  ";
    } // end for i
    std::cout<<"\n";
  } // end for x 
}


void myTest(AMP::UnitTest *ut, std::string exeName) {
  std::string input_file = "input_" + exeName;
  std::string log_file = "output_" + exeName; 

  AMP::PIO::logOnlyNodeZero(log_file);
  AMP::AMP_MPI globalComm(AMP_COMM_WORLD);

#ifdef USE_EXT_SILO
  AMP::Utilities::Writer::shared_ptr siloWriter = AMP::Utilities::Writer::buildWriter("Silo");
  siloWriter->setDecomposition(1);
#endif

//  int npes = globalComm.getSize();
  int rank = globalComm.getRank();
  std::fstream fout;
  std::string fileName = "debug_driver_" + boost::lexical_cast<std::string>(rank);
  fout.open(fileName.c_str(), std::fstream::out);

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

  // Load the meshes
  globalComm.barrier();
  double meshBeginTime = MPI_Wtime();

  AMP_INSIST(input_db->keyExists("Mesh"), "Key ''Mesh'' is missing!");
  boost::shared_ptr<AMP::Database> mesh_db = input_db->getDatabase("Mesh");
  boost::shared_ptr<AMP::Mesh::MeshParameters> meshParams(new AMP::Mesh::MeshParameters(mesh_db));
  meshParams->setComm(globalComm);
  AMP::Mesh::Mesh::shared_ptr meshAdapter = AMP::Mesh::Mesh::buildMesh(meshParams);

  globalComm.barrier();
  double meshEndTime = MPI_Wtime();
  if(!rank) {
    std::cout<<"Finished reading the mesh in "<<(meshEndTime - meshBeginTime)<<" seconds."<<std::endl;
  }

  // Create a DOF manager
  int dofsPerNode = 3;
  int nodalGhostWidth = 1;
  bool split = true;
  AMP::Discretization::DOFManager::shared_ptr dispDofManager = AMP::Discretization::simpleDOFManager::create(meshAdapter,
      AMP::Mesh::Vertex, nodalGhostWidth, dofsPerNode, split);

  // Build a column operator and a column preconditioner
  boost::shared_ptr<AMP::Operator::OperatorParameters> emptyParams;
  boost::shared_ptr<AMP::Operator::ColumnOperator> columnOperator(new AMP::Operator::ColumnOperator(emptyParams));

  boost::shared_ptr<AMP::Database> linearSolver_db = input_db->getDatabase("LinearSolver"); 
  boost::shared_ptr<AMP::Database> columnPreconditioner_db = linearSolver_db->getDatabase("Preconditioner");
  boost::shared_ptr<AMP::Solver::ColumnSolverParameters> columnPreconditionerParams(new AMP::Solver::ColumnSolverParameters(columnPreconditioner_db));
  columnPreconditionerParams->d_pOperator = columnOperator;
  boost::shared_ptr<AMP::Solver::ColumnSolver> columnPreconditioner(new AMP::Solver::ColumnSolver(columnPreconditionerParams));

  // Get the mechanics material model for the contact operator
  boost::shared_ptr<AMP::Database> model_db = input_db->getDatabase("MechanicsMaterialModel");
  boost::shared_ptr<AMP::Operator::MechanicsModelParameters> mechanicsMaterialModelParams(new AMP::Operator::MechanicsModelParameters(model_db));
  boost::shared_ptr<AMP::Operator::MechanicsMaterialModel> masterMechanicsMaterialModel(new AMP::Operator::MechanicsMaterialModel(mechanicsMaterialModelParams));

  // Build the contact operator
  AMP_INSIST(input_db->keyExists("ContactOperator"), "Key ''ContactOperator'' is missing!");
  boost::shared_ptr<AMP::Database> contact_db = input_db->getDatabase("ContactOperator");
  boost::shared_ptr<AMP::Operator::ContactOperatorParameters> 
      contactOperatorParams( new AMP::Operator::ContactOperatorParameters(contact_db) );
  contactOperatorParams->d_DOFsPerNode = dofsPerNode;
  contactOperatorParams->d_DOFManager = dispDofManager;
  contactOperatorParams->d_GlobalComm = globalComm;
  contactOperatorParams->d_Mesh = meshAdapter;
  contactOperatorParams->d_MasterMechanicsMaterialModel = masterMechanicsMaterialModel;
  contactOperatorParams->reset(); // got segfault at constructor since d_Mesh was pointing to NULL

  boost::shared_ptr<AMP::Operator::NodeToFaceContactOperator> 
      contactOperator( new AMP::Operator::NodeToFaceContactOperator(contactOperatorParams) );

  contactOperator->initialize();
  
  boost::shared_ptr<AMP::Operator::LinearBVPOperator> masterBVPOperator;

  // Build the master and slave operators
  AMP::Mesh::MeshID masterMeshID = contactOperator->getMasterMeshID();
  AMP::Mesh::Mesh::shared_ptr masterMeshAdapter = meshAdapter->Subset(masterMeshID);

  // NB: need to rotate the mesh before building mechanics op 
//  rotateMesh(meshAdapter);
//  rotateMesh(masterMeshAdapter);

  if (masterMeshAdapter.get() != NULL) {
    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> masterElementPhysicsModel;
    masterBVPOperator = boost::dynamic_pointer_cast<
        AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(masterMeshAdapter,
                                                                                         "MasterBVPOperator",
                                                                                         input_db,
                                                                                         masterElementPhysicsModel));
    columnOperator->append(masterBVPOperator);

    boost::shared_ptr<AMP::Database> masterSolver_db = columnPreconditioner_db->getDatabase("MasterSolver"); 
    boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> masterSolverParams(new AMP::Solver::PetscKrylovSolverParameters(masterSolver_db));
    masterSolverParams->d_pOperator = masterBVPOperator;
    masterSolverParams->d_comm = masterMeshAdapter->getComm();
//    masterSolverParams->d_comm = globalComm;
    boost::shared_ptr<AMP::Solver::PetscKrylovSolver> masterSolver(new AMP::Solver::PetscKrylovSolver(masterSolverParams));
    columnPreconditioner->append(masterSolver);
  } // end if

  boost::shared_ptr<AMP::Operator::LinearBVPOperator> slaveBVPOperator;

  AMP::Mesh::MeshID slaveMeshID = contactOperator->getSlaveMeshID();
  AMP::Mesh::Mesh::shared_ptr slaveMeshAdapter = meshAdapter->Subset(slaveMeshID);
  if (slaveMeshAdapter.get() != NULL) {
    boost::shared_ptr<AMP::Operator::ElementPhysicsModel> slaveElementPhysicsModel;
    slaveBVPOperator = boost::dynamic_pointer_cast<
        AMP::Operator::LinearBVPOperator>(AMP::Operator::OperatorBuilder::createOperator(slaveMeshAdapter,
                                                                                         "SlaveBVPOperator",
                                                                                         input_db,
                                                                                         slaveElementPhysicsModel));
    columnOperator->append(slaveBVPOperator);

    boost::shared_ptr<AMP::Database> slaveSolver_db = columnPreconditioner_db->getDatabase("SlaveSolver"); 
    boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> slaveSolverParams(new AMP::Solver::PetscKrylovSolverParameters(slaveSolver_db));
    slaveSolverParams->d_pOperator = slaveBVPOperator;
    slaveSolverParams->d_comm = slaveMeshAdapter->getComm();
    boost::shared_ptr<AMP::Solver::PetscKrylovSolver> slaveSolver(new AMP::Solver::PetscKrylovSolver(slaveSolverParams));
    columnPreconditioner->append(slaveSolver);
  } // end if

  boost::shared_ptr<AMP::Database> contactPreconditioner_db = columnPreconditioner_db->getDatabase("ContactPreconditioner"); 
  boost::shared_ptr<AMP::Solver::ConstraintsEliminationSolverParameters> contactPreconditionerParams(new 
      AMP::Solver::ConstraintsEliminationSolverParameters(contactPreconditioner_db));
  contactPreconditionerParams->d_pOperator = contactOperator;
  boost::shared_ptr<AMP::Solver::ConstraintsEliminationSolver> contactPreconditioner(new AMP::Solver::ConstraintsEliminationSolver(contactPreconditionerParams));
  columnPreconditioner->append(contactPreconditioner);

  // Items for computing the RHS correction due to thermal expansion
  boost::shared_ptr<AMP::Database> temperatureRhs_db = input_db->getDatabase("TemperatureRHSVectorCorrection");
  AMP::LinearAlgebra::Variable::shared_ptr tempVar(new AMP::LinearAlgebra::Variable("temperature"));
  AMP::LinearAlgebra::Variable::shared_ptr dispVar = columnOperator->getOutputVariable();
  AMP::Discretization::DOFManager::shared_ptr tempDofManager = AMP::Discretization::simpleDOFManager::create(meshAdapter, AMP::Mesh::Vertex, nodalGhostWidth, 1 , split);
  AMP::LinearAlgebra::Vector::shared_ptr tempVec = AMP::LinearAlgebra::createVector(tempDofManager, tempVar, split);
  AMP::LinearAlgebra::Vector::shared_ptr refTempVec = tempVec->cloneVector();

  AMP::LinearAlgebra::Vector::shared_ptr sigma_xx = AMP::LinearAlgebra::createVector(tempDofManager, AMP::LinearAlgebra::Variable::shared_ptr(new AMP::LinearAlgebra::Variable("sigma_xx")), split);
  AMP::LinearAlgebra::Vector::shared_ptr sigma_yy = AMP::LinearAlgebra::createVector(tempDofManager, AMP::LinearAlgebra::Variable::shared_ptr(new AMP::LinearAlgebra::Variable("sigma_yy")), split);
  AMP::LinearAlgebra::Vector::shared_ptr sigma_zz = AMP::LinearAlgebra::createVector(tempDofManager, AMP::LinearAlgebra::Variable::shared_ptr(new AMP::LinearAlgebra::Variable("sigma_zz")), split);
  AMP::LinearAlgebra::Vector::shared_ptr sigma_yz = AMP::LinearAlgebra::createVector(tempDofManager, AMP::LinearAlgebra::Variable::shared_ptr(new AMP::LinearAlgebra::Variable("sigma_yz")), split);
  AMP::LinearAlgebra::Vector::shared_ptr sigma_xz = AMP::LinearAlgebra::createVector(tempDofManager, AMP::LinearAlgebra::Variable::shared_ptr(new AMP::LinearAlgebra::Variable("sigma_xz")), split);
  AMP::LinearAlgebra::Vector::shared_ptr sigma_xy = AMP::LinearAlgebra::createVector(tempDofManager, AMP::LinearAlgebra::Variable::shared_ptr(new AMP::LinearAlgebra::Variable("sigma_xy")), split);
  AMP::LinearAlgebra::Vector::shared_ptr sigma_eff = AMP::LinearAlgebra::createVector(tempDofManager, AMP::LinearAlgebra::Variable::shared_ptr(new AMP::LinearAlgebra::Variable("sigma_eff")), split);

  tempVec->setToScalar(500.0);
//AMP::Mesh::MeshIterator meshIterator = meshAdapter->getIterator(AMP::Mesh::Vertex),
AMP::Mesh::MeshIterator meshIterator = slaveMeshAdapter->getIterator(AMP::Mesh::Vertex),
    meshIterator_begin = meshIterator.begin(),
    meshIterator_end = meshIterator.end();
std::vector<double> vertexCoord;
std::vector<size_t> DOFsIndices;
double temperatureOuterRadius = input_db->getDouble("TemperatureOuterRadius"); 
double heatGenerationRate = input_db->getDouble("HeatGenerationRate");
double outerRadius = input_db->getDouble("OuterRadius");
double outerRadiusSquared = outerRadius * outerRadius;
double thermalConductivity = input_db->getDouble("ThermalConductivity");
double temperatureCenterLine = temperatureOuterRadius + heatGenerationRate * outerRadiusSquared / (4.0 * thermalConductivity);
double referenceTemperature = input_db->getDouble("ReferenceTemperature");
if (!rank) { std::cout<<"temperatureCenterLine="<<temperatureCenterLine<<"\n"; }
  refTempVec->setToScalar(referenceTemperature);
  tempVec->setToScalar(referenceTemperature);
for (meshIterator = meshIterator_begin; meshIterator != meshIterator_end; ++meshIterator) {
  vertexCoord = meshIterator->coord();
//rotate_points(1, M_PI / -3.0, 1, &(vertexCoord[0]));
  double radiusSquared = vertexCoord[0]*vertexCoord[0] + vertexCoord[1]*vertexCoord[1];
  double temperature = temperatureCenterLine - heatGenerationRate * radiusSquared / (4.0 * thermalConductivity);
  tempDofManager->getDOFs(meshIterator->globalID(), DOFsIndices);
  AMP_ASSERT(DOFsIndices.size() == 1);
  tempVec->setLocalValuesByGlobalID(1, &(DOFsIndices[0]), &temperature);
} // end for
boost::shared_ptr<AMP::Database> tmp_db = temperatureRhs_db->getDatabase("RhsMaterialModel");
double thermalExpansionCoefficient = tmp_db->getDouble("THERMAL_EXPANSION_COEFFICIENT");
  contactOperator->uglyHack(tempVec, tempDofManager, thermalExpansionCoefficient, referenceTemperature);
//  refTempVec->setToScalar(300.0);

  AMP::LinearAlgebra::Vector::shared_ptr nullVec;
  AMP::LinearAlgebra::Variable::shared_ptr columnVar = columnOperator->getOutputVariable();
  AMP::LinearAlgebra::Vector::shared_ptr columnSolVec = AMP::LinearAlgebra::createVector(dispDofManager, columnVar, split);
  AMP::LinearAlgebra::Vector::shared_ptr columnRhsVec = AMP::LinearAlgebra::createVector(dispDofManager, columnVar, split);
  columnSolVec->zero();
  columnRhsVec->zero();

  AMP::LinearAlgebra::Vector::shared_ptr activeSetVec = sigma_eff->cloneVector();
  AMP::LinearAlgebra::Vector::shared_ptr suckItVec = sigma_eff->cloneVector();
  AMP::LinearAlgebra::Vector::shared_ptr surfaceTractionVec = columnSolVec->cloneVector();

  computeStressTensor(meshAdapter, columnSolVec, 
      sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy,
      sigma_eff, 1.0e6, 0.3,
      referenceTemperature, thermalExpansionCoefficient, tempVec);

  bool skipDisplaceMesh = true;
  contactOperator->updateActiveSet(nullVec, skipDisplaceMesh);

  
  AMP::LinearAlgebra::Vector::shared_ptr contactShiftVec = createVector(dispDofManager, columnVar, split);
  contactShiftVec->zero();

  AMP::LinearAlgebra::Vector::shared_ptr oldSolVec = columnSolVec->cloneVector();
  oldSolVec->zero();

#ifdef USE_EXT_SILO
  {
    siloWriter->registerVector(columnSolVec, meshAdapter, AMP::Mesh::Vertex, "Solution");
    siloWriter->registerVector(tempVec, meshAdapter, AMP::Mesh::Vertex, "Temperature");
    siloWriter->registerVector(sigma_eff, meshAdapter, AMP::Mesh::Vertex, "vonMises");
    siloWriter->registerVector(sigma_xx, meshAdapter, AMP::Mesh::Vertex, "sigma_xx");
    siloWriter->registerVector(sigma_yy, meshAdapter, AMP::Mesh::Vertex, "sigma_yy");
    siloWriter->registerVector(sigma_zz, meshAdapter, AMP::Mesh::Vertex, "sigma_zz");
    siloWriter->registerVector(sigma_yz, meshAdapter, AMP::Mesh::Vertex, "sigma_yz");
    siloWriter->registerVector(sigma_xz, meshAdapter, AMP::Mesh::Vertex, "sigma_xz");
    siloWriter->registerVector(sigma_xy, meshAdapter, AMP::Mesh::Vertex, "sigma_xy");
    siloWriter->registerVector(activeSetVec, meshAdapter, AMP::Mesh::Vertex, "Contact");
    siloWriter->registerVector(oldSolVec, meshAdapter, AMP::Mesh::Vertex, "Error");
    siloWriter->registerVector(surfaceTractionVec, meshAdapter, AMP::Mesh::Vertex, "Traction");
    siloWriter->registerVector(suckItVec, meshAdapter, AMP::Mesh::Vertex, "Suction");
    siloWriter->registerVector(contactShiftVec, meshAdapter, AMP::Mesh::Vertex, "Shift");
    char outFileName[256];
    sprintf(outFileName, "TOTO_%d", 0);
    siloWriter->writeFile(outFileName, 0);
  }
#endif
  oldSolVec->copyVector(columnSolVec);

  columnSolVec->zero();
  columnOperator->append(contactOperator);

{
AMP::Mesh::MeshIterator meshIterator;
std::vector<int> boundaryIDs;

std::cout<<"MASTER\n";
meshIterator = masterMeshAdapter->getIterator(AMP::Mesh::Vertex);
std::cout<<"VERTICES "<<meshIterator.size()<<"\n";
meshIterator = masterMeshAdapter->getIterator(AMP::Mesh::Volume);
std::cout<<"ELEMENTS "<<meshIterator.size()<<"\n";
boundaryIDs = masterMeshAdapter->getBoundaryIDs();
for (std::vector<int>::const_iterator boundaryIDsIterator = boundaryIDs.begin(); boundaryIDsIterator != boundaryIDs.end(); ++boundaryIDsIterator) {
std::cout<<"BOUNDARY "<<*boundaryIDsIterator<<"\n";
meshIterator = masterMeshAdapter->getBoundaryIDIterator(AMP::Mesh::Vertex, *boundaryIDsIterator);
std::cout<<"VERTICES "<<meshIterator.size()<<"\n";
meshIterator = masterMeshAdapter->getBoundaryIDIterator(AMP::Mesh::Volume, *boundaryIDsIterator);
std::cout<<"ELEMENTS "<<meshIterator.size()<<"\n";
meshIterator = masterMeshAdapter->getBoundaryIDIterator(AMP::Mesh::Face, *boundaryIDsIterator);
std::cout<<"FACES "<<meshIterator.size()<<"\n";
} // end for

std::cout<<"SLAVE\n";
meshIterator = slaveMeshAdapter->getIterator(AMP::Mesh::Vertex);
std::cout<<"VERTICES "<<meshIterator.size()<<"\n";
meshIterator = slaveMeshAdapter->getIterator(AMP::Mesh::Volume);
std::cout<<"ELEMENTS "<<meshIterator.size()<<"\n";
boundaryIDs = slaveMeshAdapter->getBoundaryIDs();
for (std::vector<int>::const_iterator boundaryIDsIterator = boundaryIDs.begin(); boundaryIDsIterator != boundaryIDs.end(); ++boundaryIDsIterator) {
std::cout<<"BOUNDARY "<<*boundaryIDsIterator<<"\n";
meshIterator = slaveMeshAdapter->getBoundaryIDIterator(AMP::Mesh::Vertex, *boundaryIDsIterator);
std::cout<<"VERTICES "<<meshIterator.size()<<"\n";
meshIterator = slaveMeshAdapter->getBoundaryIDIterator(AMP::Mesh::Volume, *boundaryIDsIterator);
std::cout<<"ELEMENTS "<<meshIterator.size()<<"\n";
meshIterator = slaveMeshAdapter->getBoundaryIDIterator(AMP::Mesh::Face, *boundaryIDsIterator);
std::cout<<"FACES "<<meshIterator.size()<<"\n";
} // end for

double pov[] = { 1.0, 1.0, 1.0 };
std::fstream fout;
fout.open("uhntissuhntiss", std::fstream::out);
drawFacesOnBoundaryID(masterMeshAdapter, 8, fout, pov, "");
fout.close();
}

  // Build a matrix shell operator to use the column operator with the petsc krylov solvers
  boost::shared_ptr<AMP::Database> matrixShellDatabase = input_db->getDatabase("MatrixShellOperator");
  boost::shared_ptr<AMP::Operator::OperatorParameters> matrixShellParams(new AMP::Operator::OperatorParameters(matrixShellDatabase));
  boost::shared_ptr<AMP::Operator::PetscMatrixShellOperator> matrixShellOperator(new AMP::Operator::PetscMatrixShellOperator(matrixShellParams));

  int numMasterLocalNodes = 0;
  int numSlaveLocalNodes = 0;
  if (masterMeshAdapter.get() != NULL) { numMasterLocalNodes = masterMeshAdapter->numLocalElements(AMP::Mesh::Vertex); }
  if (slaveMeshAdapter.get() != NULL) { numSlaveLocalNodes = slaveMeshAdapter->numLocalElements(AMP::Mesh::Vertex); }
  int matLocalSize = dofsPerNode * (numMasterLocalNodes + numSlaveLocalNodes);
  AMP_ASSERT( matLocalSize == static_cast<int>(dispDofManager->numLocalDOF()) );
  matrixShellOperator->setComm(globalComm);
  matrixShellOperator->setMatLocalRowSize(matLocalSize);
  matrixShellOperator->setMatLocalColumnSize(matLocalSize);
  matrixShellOperator->setOperator(columnOperator); 

  boost::shared_ptr<AMP::Solver::PetscKrylovSolverParameters> linearSolverParams(new AMP::Solver::PetscKrylovSolverParameters(linearSolver_db));
  linearSolverParams->d_pOperator = matrixShellOperator;
  linearSolverParams->d_comm = globalComm;
  linearSolverParams->d_pPreconditioner = columnPreconditioner;
  boost::shared_ptr<AMP::Solver::PetscKrylovSolver> linearSolver(new AMP::Solver::PetscKrylovSolver(linearSolverParams));
//  linearSolver->setZeroInitialGuess(true);
  linearSolver->setInitialGuess(columnSolVec);

  AMP::LinearAlgebra::Vector::shared_ptr fullThermalLoadingTempMinusRefTempVec = tempVec->cloneVector();
  fullThermalLoadingTempMinusRefTempVec->subtract(tempVec, refTempVec);

size_t const maxThermalLoadingIterations = input_db->getIntegerWithDefault("maxThermalLoadingIterations", 5);
for (size_t thermalLoadingIteration = 0; thermalLoadingIteration < maxThermalLoadingIterations; ++thermalLoadingIteration) {
  if (!rank) { std::cout<<"THERMAL LOADING "<<thermalLoadingIteration+1<<"/"<<maxThermalLoadingIterations<<"\n"; }
  double scalingFactor = static_cast<double>(thermalLoadingIteration+1) / static_cast<double>(maxThermalLoadingIterations);
  tempVec->axpy(scalingFactor, fullThermalLoadingTempMinusRefTempVec, refTempVec);

  size_t const maxActiveSetIterations = input_db->getIntegerWithDefault("maxActiveSetIterations", 5);
  for (size_t activeSetIteration = 0; activeSetIteration < maxActiveSetIterations; ++activeSetIteration) {
    if (!rank) { std::cout<<"ACTIVE SET ITERATION #"<<activeSetIteration+1<<std::endl; }

    columnSolVec->zero();
    columnRhsVec->zero();

    // compute thermal load f
    computeTemperatureRhsVector(meshAdapter, temperatureRhs_db, tempVar, dispVar, tempVec, refTempVec, columnRhsVec);

    // apply dirichlet rhs correction on f
    if (masterBVPOperator.get() != NULL) {
      masterBVPOperator->modifyRHSvector(columnRhsVec);
    } // end if
    if (slaveBVPOperator.get() != NULL) {
      slaveBVPOperator->modifyRHSvector(columnRhsVec);
    } // end if

    // get d
//    AMP::LinearAlgebra::Vector::shared_ptr contactShiftVec = createVector(dispDofManager, columnVar, split);
    contactShiftVec->zero();
    contactOperator->addShiftToSlave(contactShiftVec);
//  contactOperator->addShiftToSlave(columnSolVec);

    // compute - Kd
    AMP::LinearAlgebra::Vector::shared_ptr rhsCorrectionVec = createVector(dispDofManager, columnVar, split);
    rhsCorrectionVec->zero();
    masterBVPOperator->apply(nullVec, contactShiftVec, rhsCorrectionVec, -1.0, 0.0);
    slaveBVPOperator->apply(nullVec, contactShiftVec, rhsCorrectionVec, -1.0, 0.0);
//  columnOperator->apply(nullVec, columnSolVec, rhsCorrectionVec, -1.0, 0.0);
//  columnOperator->append(contactOperator);

    // f = f - Kd
    columnRhsVec->add(columnRhsVec, rhsCorrectionVec);

    // f^m = f^m + C^T f^s
    // f^s = 0
    contactOperator->addSlaveToMaster(columnRhsVec);
    contactOperator->setSlaveToZero(columnRhsVec);

    // u_s = C u_m
    contactOperator->copyMasterToSlave(columnSolVec);


    linearSolver->solve(columnRhsVec, columnSolVec);

    // u^s = C u^m + d
    contactOperator->copyMasterToSlave(columnSolVec);
    contactOperator->addShiftToSlave(columnSolVec);

    computeStressTensor(meshAdapter, columnSolVec, 
        sigma_xx, sigma_yy, sigma_zz, sigma_yz, sigma_xz, sigma_xy,
        sigma_eff, 1.0e6, 0.3,
        referenceTemperature, thermalExpansionCoefficient, tempVec);

    std::vector<AMP::Mesh::MeshElementID> const * pointerToActiveSet;
    contactOperator->getActiveSet(pointerToActiveSet);
    size_t const sizeOfActiveSetBeforeUpdate = pointerToActiveSet->size();

    std::vector<size_t> activeSetTempDOFsIndicesBeforeUpdate;
    tempDofManager->getDOFs(*pointerToActiveSet, activeSetTempDOFsIndicesBeforeUpdate);
    AMP_ASSERT( activeSetTempDOFsIndicesBeforeUpdate.size() == sizeOfActiveSetBeforeUpdate );
    std::vector<double> valuesForActiveSet(sizeOfActiveSetBeforeUpdate, 2.0); 
    activeSetVec->setToScalar(-1.0);
    activeSetVec->setLocalValuesByGlobalID(sizeOfActiveSetBeforeUpdate, &(activeSetTempDOFsIndicesBeforeUpdate[0]), &(valuesForActiveSet[0]));

    std::vector<size_t> activeSetDispDOFsIndicesBeforeUpdate;
    dispDofManager->getDOFs(*pointerToActiveSet, activeSetDispDOFsIndicesBeforeUpdate);
    AMP_ASSERT( activeSetDispDOFsIndicesBeforeUpdate.size() == 3*sizeOfActiveSetBeforeUpdate );
    
#ifdef USE_EXT_SILO
{
    meshAdapter->displaceMesh(columnSolVec);
    char outFileName[256];
    sprintf(outFileName, "TOTO_%d", 0);
    siloWriter->writeFile(outFileName, (activeSetIteration+1)+(thermalLoadingIteration)*maxActiveSetIterations);
    columnSolVec->scale(-1.0);
    meshAdapter->displaceMesh(columnSolVec);
    columnSolVec->scale(-1.0);
}
#endif

    size_t nChangesInActiveSet = contactOperator->updateActiveSet(columnSolVec);

    size_t const sizeOfActiveSetAfterUpdate = pointerToActiveSet->size();
//    std::vector<size_t> activeSetDOFsIndicesAfterUpdate;
//    tempDofManager->getDOFs(*pointerToActiveSet, activeSetDOFsIndicesAfterUpdate);
//    AMP_ASSERT( activeSetDOFsIndicesAfterUpdate.size() == sizeOfActiveSetAfterUpdate );
//    std::vector<double> valuesForActiveSet(pointerToActiveSet->size(), 2.0); 
//    activeSetVec->setToScalar(-1.0);
//    activeSetVec->setLocalValuesByGlobalID(sizeOfActiveSetAfterUpdate, &(activeSetDOFsIndicesAfterUpdate[0]), &(valuesForActiveSet[0]));

    std::vector<double> const * slaveVerticesNormalVector;
    std::vector<double> const * slaveVerticesSurfaceTraction;
    contactOperator->getSlaveVerticesNormalVectorAndSurfaceTraction(slaveVerticesNormalVector, slaveVerticesSurfaceTraction);
std::cout<<slaveVerticesSurfaceTraction->size()<<"  "<<slaveVerticesNormalVector->size()<<"  "<<sizeOfActiveSetBeforeUpdate<<std::endl;
    AMP_ASSERT( slaveVerticesSurfaceTraction->size() == 3*sizeOfActiveSetBeforeUpdate);
    AMP_ASSERT( slaveVerticesNormalVector->size() == 3*sizeOfActiveSetBeforeUpdate);
    surfaceTractionVec->zero();
    surfaceTractionVec->setLocalValuesByGlobalID(3*sizeOfActiveSetBeforeUpdate, &(activeSetDispDOFsIndicesBeforeUpdate[0]), &((*slaveVerticesSurfaceTraction)[0]));

    std::vector<double> surfaceTractionDOTnormalVector(sizeOfActiveSetBeforeUpdate);
    for (size_t kk = 0; kk < sizeOfActiveSetBeforeUpdate; ++kk) {
      surfaceTractionDOTnormalVector[kk] = - compute_scalar_product(&((*slaveVerticesSurfaceTraction)[3*kk]), &((*slaveVerticesNormalVector)[3*kk]));
    } // end for kk
    suckItVec->zero();
    suckItVec->setLocalValuesByGlobalID(sizeOfActiveSetBeforeUpdate, &(activeSetTempDOFsIndicesBeforeUpdate[0]), &(surfaceTractionDOTnormalVector[0]));
    
    
//why_cant_we_be_friend(masterMeshAdapter, columnSolVec);

oldSolVec->subtract(columnSolVec, oldSolVec);
#ifdef USE_EXT_SILO
    meshAdapter->displaceMesh(columnSolVec);
/*    siloWriter->registerVector(columnSolVec, meshAdapter, AMP::Mesh::Vertex, "Solution");
    siloWriter->registerVector(tempVec, meshAdapter, AMP::Mesh::Vertex, "Temperature");
    siloWriter->registerVector(sigma_eff, meshAdapter, AMP::Mesh::Vertex, "vonMises");
    siloWriter->registerVector(activeSetVec, meshAdapter, AMP::Mesh::Vertex, "Contact");
    siloWriter->registerVector(oldSolVec, meshAdapter, AMP::Mesh::Vertex, "Error");
    siloWriter->registerVector(surfaceTractionVec, meshAdapter, AMP::Mesh::Vertex, "Traction");
*/
    char outFileName[256];
    sprintf(outFileName, "TOTO_%d", 0);
    siloWriter->writeFile(outFileName, (activeSetIteration+1)+(thermalLoadingIteration)*maxActiveSetIterations);
    columnSolVec->scale(-1.0);
    meshAdapter->displaceMesh(columnSolVec);
    columnSolVec->scale(-1.0);
#endif
//    for (std::vector<AMP::Mesh::MeshElementID>::iterator activeSetIterator = pointerToActiveSet->begin(); activeSetIterator != pointerToActiveSet->end(); ++activeSetIterator) {
//    } // end for
    if (!rank) { std::cout<<nChangesInActiveSet<<" CHANGES IN ACTIVE SET\n"; }
double errL1Norm = oldSolVec->L1Norm();
double solL1Norm = columnSolVec->L1Norm();
double relErrL1Norm = errL1Norm / solL1Norm;
double errL2Norm = oldSolVec->L2Norm();
double solL2Norm = columnSolVec->L2Norm();
double relErrL2Norm = errL2Norm / solL2Norm;
    if (!rank) { std::cout<<"ERROR L1 NORM "<<errL1Norm<<" ("<<100.0*relErrL1Norm<<"%)    "; }
    if (!rank) { std::cout<<"ERROR L2 NORM "<<errL2Norm<<" ("<<100.0*relErrL2Norm<<"%)  \n"; }
oldSolVec->copyVector(columnSolVec);

    if (nChangesInActiveSet == 0) { break; }
//    AMP_ASSERT( activeSetIteration != maxActiveSetIterations - 1 );
    if ( activeSetIteration == maxActiveSetIterations - 1 ) {
      if (!rank) { std::cout<<"!!!!!! ACTIVE SET ITERATIONS DID NOT CONVERGE !!!!!!!!\n"; }
    } // end if
  } // end for

} // end for

  meshAdapter->displaceMesh(columnSolVec);

#ifdef USE_EXT_SILO
  siloWriter->registerVector(columnSolVec, meshAdapter, AMP::Mesh::Vertex, "Solution");
  char outFileName[256];
  sprintf(outFileName, "MPC_%d", 0);
  siloWriter->writeFile(outFileName, 0);
#endif
  fout.close();

  ut->passes(exeName);
}


int main(int argc, char *argv[])
{
  AMP::AMPManager::startup(argc, argv);
  AMP::AMP_MPI globalComm(AMP_COMM_WORLD);
  AMP::UnitTest ut;

  std::vector<std::string> exeNames; 
  exeNames.push_back("testNodeToFaceContactOperator-4");

  try {
    for (size_t i = 0; i < exeNames.size(); ++i) { 
      myTest(&ut, exeNames[i]); 
    } // end for
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



